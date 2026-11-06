"""Portable trainable V4.1 components. See sources/deepseek-v41/LICENSE (MIT).

Equations follow the pinned DeepSeek reference; hardware-specific kernels are
replaced with ordinary PyTorch, with fp32 master weights and explicit compute dtype.
"""
import math
import re
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.gpt import Linear


def pack_fp4(x):
    """E2M1 nibbles with one E4M3 scale per 16 channels (4.5 bits/value)."""
    if x.size(-1) % 16:
        raise ValueError("FP4 channels must be a multiple of 16")
    groups = x.detach().float().unflatten(-1, (-1, 16))
    scales = (groups.abs().amax(-1) / 6).clamp(2**-9, 448).to(torch.float8_e4m3fn)
    scaled = groups / scales.float().unsqueeze(-1)
    boundaries = x.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32)
    codes = torch.bucketize(scaled.abs().contiguous(), boundaries).to(torch.uint8)
    codes = (codes | ((scaled < 0).to(torch.uint8) * 8)).flatten(-2)
    return codes[..., ::2] | (codes[..., 1::2] << 4), scales


def unpack_fp4(packed, scales):
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).flatten(-2).long()
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    values = levels[codes & 7] * torch.where(codes < 8, 1., -1.)
    return (values.unflatten(-1, (-1, 16)) * scales.float().unsqueeze(-1)).flatten(-2)


def fp4_ste(x):
    restored = unpack_fp4(*pack_fp4(x)).to(x.dtype)
    return x + (restored - x).detach()


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return (F.rms_norm(x.float(), (x.size(-1),), eps=self.eps) * self.weight).to(x.dtype)


def rope(x, positions, rope_dim, theta, inverse=False):
    """Out-of-place partial adjacent-pair RoPE; safe for autograd and shared KV."""
    freqs = theta ** (-torch.arange(0, rope_dim, 2, device=x.device).float() / rope_dim)
    angles = positions.float().unsqueeze(-1) * freqs
    if x.ndim == 4:
        angles = angles[:, None, :]
    a, b = x[..., -rope_dim::2].float(), x[..., -rope_dim + 1::2].float()
    sin = angles.sin() * (-1 if inverse else 1)
    tail = torch.stack([a * angles.cos() - b * sin, a * sin + b * angles.cos()], -1).flatten(-2)
    return torch.cat([x[..., :-rope_dim], tail.to(x.dtype)], -1)


class SinglePassMHC(nn.Module):
    def __init__(self, dim, streams, iterations=10):
        super().__init__()
        self.streams, self.iterations = streams, iterations
        self.fn = nn.Parameter(torch.empty(streams * (streams + 2), dim * streams))
        self.base = nn.Parameter(torch.empty(streams * (streams + 2)))
        self.scale = nn.Parameter(torch.empty(3))

    def coefficients(self, x):
        n = self.streams
        flat = x.flatten(-2).float()
        scores = F.linear(F.rms_norm(flat, (flat.size(-1),)), self.fn)
        pre = torch.sigmoid(scores[..., :n] * self.scale[0] + self.base[:n]) + 1e-6
        post = 2 * torch.sigmoid(scores[..., n:2*n] * self.scale[1] + self.base[n:2*n])
        comb = (scores[..., 2*n:] * self.scale[2] + self.base[2*n:]).unflatten(-1, (n, n)).softmax(-1) + 1e-6
        for _ in range(self.iterations):
            comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
            comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
        return pre, post, comb

    @staticmethod
    def read(x, previous_pre):
        return (x.float() * previous_pre.unsqueeze(-1)).sum(-2).to(x.dtype)

    @staticmethod
    def write(x, output, post, comb):
        residual = torch.einsum("btij,btid->btjd", comb, x.float())
        return (residual + post.unsqueeze(-1) * output.float().unsqueeze(-2)).to(x.dtype)

    @torch.no_grad()
    def initialize(self):
        nn.init.normal_(self.fn, std=0.01 / math.sqrt(self.fn.size(-1)))
        self.base.zero_(); self.scale.fill_(1.)
        n = self.streams
        self.base[:n].fill_(math.log(1 / (n - 1)) if n > 1 else math.log(1e6))
        self.base[2*n:].view(n, n).copy_(torch.eye(n, device=self.base.device) * 4)


class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate, self.up, self.down = Linear(dim, hidden, bias=False), Linear(dim, hidden, bias=False), Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down((F.silu(self.gate(x).float()) * self.up(x).float()).to(x.dtype))


class RoutedMoE(nn.Module):
    def __init__(self, dim, hidden, experts, topk, bias_rate=.001):
        super().__init__()
        self.topk, self.bias_rate = topk, bias_rate
        self.router = Linear(dim, experts, bias=False)
        self.shared = Expert(dim, hidden)
        self.experts = nn.ModuleList([Expert(dim, hidden) for _ in range(experts)])
        self.register_buffer("bias", torch.zeros(experts))
        self.register_buffer("counts", torch.zeros(experts), persistent=False)

    def forward(self, x):
        flat = x.flatten(0, 1)
        scores = F.softplus(self.router(flat.float())).sqrt()
        indices = (scores + self.bias).topk(self.topk, -1).indices
        weights = scores.gather(-1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        if self.training:
            with torch.no_grad():
                self.counts.add_(torch.bincount(indices.flatten(), minlength=self.bias.numel()))
        out = self.shared(flat)
        for i, expert in enumerate(self.experts):
            row, choice = torch.where(indices == i)
            # Empty dispatch still creates zero gradients for every expert, keeping Muon safe.
            contribution = expert(flat[row]) * weights[row, choice, None].to(flat.dtype)
            out = out.index_add(0, row, contribution)
        return out.view_as(x)

    @torch.no_grad()
    def balance(self):
        counts = self.counts.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
        if counts.sum() > 0:
            self.bias.add_(self.bias_rate * (counts.mean() - counts).sign())
        self.counts.zero_()


def next_prime(number):
    while True:
        number += 1
        if all(number % i for i in range(2, math.isqrt(number) + 1)):
            return number


def build_token_map(tokenizer):
    """Tokenizer-agnostic version of the reference's normalization-based compression."""
    keys, mapping = {}, []
    for token in range(tokenizer.get_vocab_size()):
        text = tokenizer.decode([token])
        if "\ufffd" in text:
            key = ("partial_utf8", token)
        else:
            normalized = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
            normalized = "".join(c for c in normalized if not unicodedata.combining(c)).lower()
            normalized = re.sub(r"[ \t\r\n]+", " ", normalized)
            key = " " if normalized == " " else (normalized.strip() or text)
        if key not in keys:
            keys[key] = len(keys)
        mapping.append(keys[key])
    return torch.tensor(mapping, dtype=torch.long)


class Engram(nn.Module):
    """2/3/4-gram multi-head memory with stream-specific normalized signed-sqrt gates.

    Token IDs use an identity compression map unless set_token_map is called.
    No causal convolution, matching V4.1. Tables are ordinary trainable embeddings.
    """
    def __init__(self, dim, streams, memory_dim, buckets, heads=2, vocab_size=0, hash_seed=0):
        super().__init__()
        self.streams, self.heads = streams, heads
        self.hash_seed = hash_seed
        sizes = []
        for _ in range(heads * 3):
            buckets = next_prime(buckets); sizes.append(buckets)
        self.tables = nn.ModuleList([nn.Embedding(size, memory_dim) for size in sizes])
        self.project = Linear(heads * 3 * memory_dim, dim * (streams + 1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(streams, dim))
        self.k_weight = nn.Parameter(torch.ones(streams, dim))
        self.register_buffer("token_map", torch.arange(vocab_size, dtype=torch.long))

    def set_token_map(self, mapping):
        self.token_map = mapping.to(self.q_weight.device, dtype=torch.long)

    def forward(self, x, ids, history=None):
        prefix = ids[:, :0] if history is None else history
        tokens = torch.cat([prefix, ids], 1)
        if self.token_map.numel():
            tokens = self.token_map[tokens]
        n_prefix = prefix.size(1)
        positions = torch.arange(n_prefix, tokens.size(1), device=ids.device)
        embeddings = []
        for order in (2, 3, 4):
            for head in range(self.heads):
                table = self.tables[(order - 2) * self.heads + head]
                hashed = torch.zeros_like(ids)
                for shift in range(order):
                    source = tokens[:, (positions - shift).clamp_min(0)]
                    source = torch.where(positions >= shift, source, 0)
                    hashed = (hashed * (131 + 2 * head + 32 * self.hash_seed) + source) % table.num_embeddings
                embeddings.append(table(hashed).to(x.dtype))
        key, value = self.project(torch.cat(embeddings, -1)).split([self.streams * x.size(-1), x.size(-1)], -1)
        key = key.unflatten(-1, (self.streams, x.size(-1))).float()
        q, k = F.rms_norm(x.float(), (x.size(-1),)), F.rms_norm(key, (key.size(-1),))
        dot = (q * k * self.q_weight * self.k_weight).sum(-1) / math.sqrt(x.size(-1))
        gate = torch.sigmoid(torch.sign(dot) * dot.abs().clamp_min(1e-6).sqrt())
        return (x.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
