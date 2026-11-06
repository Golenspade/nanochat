"""A scaled, trainable V4.1-style text backbone (portable reference kernels).

Structural source: supplied report and pinned MIT reference in sources/deepseek-v41.
This implements CED explicitly; it does not load the released giant checkpoint.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import Linear
from nanochat.v41_components import RMSNorm, SinglePassMHC, RoutedMoE, Engram, rope, fp4_ste, build_token_map
from nanochat.v41_cache import V41Cache


@dataclass
class V41Config:
    architecture: str = "deepseek_v41"
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 8
    n_embd: int = 256
    n_head: int = 4
    head_dim: int = 64
    rope_dim: int = 16
    q_rank: int = 128
    o_rank: int = 128
    o_groups: int = 2
    window_size: int = 32
    local_only_layers: int = 1
    encoder_pattern: str = "FIRR"
    decoder_pattern: str = "FIRR"
    encoder_compression: int = 2
    index_dim: int = 32
    index_heads: int = 2
    index_topk: int = 8
    candidate_blocks: int = 4
    candidate_block_size: int = 4
    index_aux_weight: float = .01
    hc_mult: int = 2
    n_experts: int = 8
    experts_per_token: int = 2
    expert_dim: int = 128
    enable_engram: bool = True
    engram_dim: int = 16
    engram_buckets: int = 257
    fp4_cache: bool = False
    window_pattern: str = "L"  # pipeline metadata compatibility; V41 uses window_size

    @property
    def n_kv_head(self):
        return 1  # one shared latent, not one KV vector per query head

    def __post_init__(self):
        if self.architecture != "deepseek_v41" or self.n_layer < 4 or self.n_layer % 2:
            raise ValueError("V41 requires an even depth >= 4")
        if not 0 <= self.local_only_layers < self.n_layer // 2:
            raise ValueError("Encoder needs at least one global layer")
        for pattern in (self.encoder_pattern, self.decoder_pattern):
            if not pattern or pattern[0] != "F" or any(c not in "FIR" for c in pattern):
                raise ValueError("CSA2 patterns start with F and contain F/I/R")
        if "F" in self.decoder_pattern[1:]:
            raise ValueError("Decoder uses one global producer followed by Reindex/Reuse")
        if min(self.window_size, self.encoder_compression, self.index_topk, self.candidate_blocks, self.candidate_block_size, self.n_head, self.hc_mult) < 1:
            raise ValueError("Attention sizes must be positive")
        if self.head_dim % 16 or self.rope_dim % 2 or not 0 < self.rope_dim <= min(self.head_dim, self.index_dim):
            raise ValueError("head_dim must align to 16; even rope_dim must fit main/index heads")
        if not 1 <= self.experts_per_token <= self.n_experts:
            raise ValueError("Invalid expert top-k")
        if self.o_groups < 1 or self.n_head % self.o_groups:
            raise ValueError("Output groups must divide attention heads")

    def layer_modes(self):
        half = self.n_layer // 2
        encoder = ["S"] * self.local_only_layers
        encoder += [self.encoder_pattern[i % len(self.encoder_pattern)] for i in range(half - len(encoder))]
        tail = self.decoder_pattern[1:] or "R"
        return tuple(encoder + ["F"] + [tail[i % len(tail)] for i in range(half - 1)])

    def sources(self):
        return {i: self.encoder_compression if i < self.n_layer // 2 else 1
                for i, mode in enumerate(self.layer_modes()) if mode == "F"}


def gather(values, indices):
    batch = torch.arange(values.size(0), device=values.device)[:, None, None]
    return values[batch, indices.clamp(0, max(values.size(1) - 1, 0))]


class CSA2(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config, self.layer = config, layer
        c = config
        self.mode = c.layer_modes()[layer]
        self.q_a, self.q_norm = Linear(c.n_embd, c.q_rank, bias=False), RMSNorm(c.q_rank)
        self.q_b = Linear(c.q_rank, c.n_head * c.head_dim, bias=False)
        self.local_kv, self.local_norm = Linear(c.n_embd, c.head_dim, bias=False), RMSNorm(c.head_dim)
        self.o_a = Linear(c.n_head * c.head_dim // c.o_groups, c.o_groups * c.o_rank, bias=False)
        self.o_b = Linear(c.o_groups * c.o_rank, c.n_embd, bias=False)
        self.sink = nn.Parameter(torch.zeros(c.n_head))
        self.ratio = c.encoder_compression if layer < c.n_layer // 2 else 1
        if self.mode == "F":
            self.compressor = Linear(c.n_embd, c.head_dim, bias=False)
            self.compress_gate = Linear(c.n_embd, c.head_dim, bias=False) if self.ratio > 1 else None
            self.compress_norm = RMSNorm(c.head_dim)
            self.index_k = Linear(c.head_dim, c.index_dim, bias=False)
            self.index_norm = RMSNorm(c.index_dim)
        if self.mode in "FI":
            self.index_q = Linear(c.q_rank, c.index_heads * c.index_dim, bias=False)
            self.index_weights = Linear(c.n_embd, c.index_heads, bias=False)

    def produce(self, source, cache):
        c, ratio = self.config, self.ratio
        values = self.compressor(source.float() if ratio > 1 else source)
        scores = self.compress_gate(source.float()) if ratio > 1 else torch.zeros_like(values)
        first_entry = 0 if cache is None else cache.global_counts[self.layer]
        if cache is not None and self.layer in cache.pending:
            old_v, old_s = cache.pending[self.layer]
            values, scores = torch.cat([old_v, values], 1), torch.cat([old_s, scores], 1)
        complete = values.size(1) // ratio * ratio
        if cache is not None:
            cache.pending[self.layer] = values[:, complete:].clone(), scores[:, complete:].clone()
        shape = (source.size(0), complete // ratio, ratio, c.head_dim)
        values = values[:, :complete].reshape(shape)
        scores = scores[:, :complete].reshape(shape)
        latent = self.compress_norm((values * scores.softmax(2)).sum(2).to(source.dtype))
        positions = torch.arange(first_entry, first_entry + latent.size(1), device=source.device) * ratio
        index = rope(self.index_norm(self.index_k(latent)), positions, c.rope_dim, 40000)
        main = rope(latent, positions, c.rope_dim, 40000)
        if c.fp4_cache:
            main = fp4_ste(main)
        if cache is not None:
            cache.append_global(self.layer, main, index)
            main, index = cache.global_values(self.layer)
        return main, index

    def select(self, x, qr, q, positions, state):
        c = self.config
        main, keys = state["main"], state["index"]
        count = main.size(1)
        shape = (*x.shape[:2], 0)
        if count == 0:
            state["selected"] = torch.empty(shape, dtype=torch.long, device=x.device)
            return x.new_zeros(())
        iq = self.index_q(qr).unflatten(-1, (c.index_heads, c.index_dim))
        iq = rope(iq, positions, c.rope_dim, 40000)
        pool = state.get("pool") if self.layer >= c.n_layer // 2 and self.mode == "I" else None
        if pool is None:
            candidates = torch.arange(count, device=x.device).expand(*x.shape[:2], -1)
            ik = keys[:, None].expand(-1, x.size(1), -1, -1)
        else:
            candidates, ik = pool, gather(keys, pool)
        visible = (candidates >= 0) & ((candidates + 1) * self.ratio - 1 <= positions[:, None])
        weights = self.index_weights(x).float() / math.sqrt(c.index_heads * c.index_dim)
        scores = (torch.einsum("bthd,btpd->bthp", iq.float(), ik.float()).relu() * weights.unsqueeze(-1)).sum(2)
        scores = scores.masked_fill(~visible, -torch.inf)
        if self.layer == c.n_layer // 2:
            bs = c.candidate_block_size
            blocks = F.pad(scores, (0, -count % bs), value=-torch.inf).unflatten(-1, (-1, bs)).amax(-1)
            chosen = blocks.argsort(dim=-1, descending=True, stable=True)[..., :c.candidate_blocks]
            pool = (chosen.unsqueeze(-1) * bs + torch.arange(bs, device=x.device)).flatten(-2)
            state["pool"] = pool.masked_fill(pool >= count, -1)
        # Stable positional tie-breaking is required for prefix/chunk equivalence:
        # ReLU index scores often tie at zero, and topk's tie order depends on width.
        selected = scores.argsort(dim=-1, descending=True, stable=True)[..., :c.index_topk]
        chosen = candidates.gather(-1, selected)
        state["selected"] = chosen.masked_fill(~torch.isfinite(scores.gather(-1, selected)), -1)
        if not self.training:
            return x.new_zeros(())
        # Train the otherwise nondifferentiable indexer against detached main attention.
        teacher_keys = gather(main, candidates).float()
        teacher = torch.einsum("bthd,btpd->bthp", q.float().detach(), teacher_keys.detach()).mean(2) / math.sqrt(c.head_dim)
        teacher = teacher.masked_fill(~visible, -1e30).softmax(-1) * visible
        student = scores.masked_fill(~visible, -1e30).log_softmax(-1)
        loss = F.kl_div(student, teacher, reduction="none").sum(-1)
        return (loss * visible.any(-1)).mean()

    def forward(self, x, positions, state, cache=None, global_source=None):
        c = self.config
        theta = 10000 if self.mode == "S" else 40000
        qr = self.q_norm(self.q_a(x))
        q = rope(self.q_b(qr).unflatten(-1, (c.n_head, c.head_dim)), positions, c.rope_dim, theta)
        local = rope(self.local_norm(self.local_kv(x)), positions, c.rope_dim, theta)
        if cache is not None:
            local = cache.append_local(self.layer, local)
        origin = positions[0] - (local.size(1) - x.size(1))
        local_ids = positions[:, None] - c.window_size + 1 + torch.arange(c.window_size, device=x.device)
        local_ids = (local_ids - origin).expand(x.size(0), -1, -1)
        valid = (local_ids >= 0) & (local_ids < local.size(1))
        values = gather(local, local_ids)
        auxiliary = x.new_zeros(())
        if self.mode == "F":
            main, index = self.produce(x if global_source is None else global_source, cache)
            state.clear(); state.update(main=main, index=index, owner=self.layer)
        if self.mode in "FI":
            auxiliary = self.select(x, qr, q, positions, state)
        if self.mode != "S" and state["main"].size(1):
            selection = state["selected"]
            values = torch.cat([values, gather(state["main"], selection)], 2)
            valid = torch.cat([valid, selection >= 0], -1)
        logits = torch.einsum("bthd,btkd->bthk", q.float(), values.float()) / math.sqrt(c.head_dim)
        logits = logits.masked_fill(~valid.unsqueeze(2), -torch.inf)
        sink = self.sink[None, None, :, None].expand(*logits.shape[:-1], 1)
        probability = torch.cat([logits, sink], -1).softmax(-1)[..., :-1]
        output = torch.einsum("bthk,btkd->bthd", probability, values.float()).to(x.dtype)
        output = rope(output, positions, c.rope_dim, theta, inverse=True)
        grouped = output.reshape(*x.shape[:2], c.o_groups, -1)
        weights = self.o_a.weight.to(x.dtype).view(c.o_groups, c.o_rank, -1)
        projected = torch.einsum("btgd,grd->btgr", grouped, weights).flatten(-2)
        return self.o_b(projected), auxiliary


class V41Block(nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        self.attn_mix, self.ffn_mix = SinglePassMHC(c.n_embd, c.hc_mult), SinglePassMHC(c.n_embd, c.hc_mult)
        self.attn_norm, self.ffn_norm = RMSNorm(c.n_embd), RMSNorm(c.n_embd)
        self.attn = CSA2(c, layer)
        self.ffn = RoutedMoE(c.n_embd, c.expert_dim, c.n_experts, c.experts_per_token)

    def forward(self, x, pre, positions, state, cache=None, global_source=None):
        next_pre, post, comb = self.attn_mix.coefficients(x)
        output, aux = self.attn(self.attn_norm(self.attn_mix.read(x, pre)), positions, state, cache, global_source)
        x = self.attn_mix.write(x, output, post, comb)
        final_pre, post, comb = self.ffn_mix.coefficients(x)
        output = self.ffn(self.ffn_norm(self.ffn_mix.read(x, next_pre)))
        return self.ffn_mix.write(x, output, post, comb), final_pre, aux


class DeepSeekV41(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([V41Block(config, i) for i in range(config.n_layer)])
        layers = sorted({1, max(2, config.n_layer * 14 // 40)}) if config.enable_engram else []
        self.engram = nn.ModuleDict({str(i): Engram(config.n_embd, config.hc_mult, config.engram_dim, config.engram_buckets, vocab_size=config.vocab_size, hash_seed=i) for i in layers})
        self.norm, self.lm_head = RMSNorm(config.n_embd), Linear(config.n_embd, config.vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, RMSNorm):
                module.weight.fill_(1)
            elif isinstance(module, SinglePassMHC):
                module.initialize()
            elif isinstance(module, CSA2):
                module.sink.zero_()
            elif isinstance(module, RoutedMoE):
                module.bias.zero_(); module.counts.zero_()
            elif isinstance(module, Engram):
                module.q_weight.fill_(1); module.k_weight.fill_(1)
                module.token_map.copy_(torch.arange(module.token_map.numel(), device=module.token_map.device))

    def compress_engram_tokenizer(self, tokenizer):
        mapping = build_token_map(tokenizer)
        if mapping.numel() != self.config.vocab_size:
            raise ValueError("Engram tokenizer vocabulary mismatch")
        for memory in self.engram.values():
            memory.set_token_map(mapping)

    def get_device(self):
        return self.embedding.weight.device

    def create_kv_cache(self, batch_size, seq_len, bounded_replay=False):
        return V41Cache(self.config, batch_size, seq_len, self.get_device(), COMPUTE_DTYPE, bounded_replay)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean", return_hidden=False):
        c = self.config
        start = 0 if kv_cache is None else kv_cache.get_pos()
        if kv_cache is not None:
            if torch.is_grad_enabled() or kv_cache.config != c:
                raise ValueError("Request cache is inference-only and must match the model config")
            if start + idx.size(1) > kv_cache.max_seq_len:
                raise ValueError("Cache capacity exceeded")
        history = None if kv_cache is None else kv_cache.history
        x = self.embedding(idx).to(COMPUTE_DTYPE).unsqueeze(2).expand(-1, -1, c.hc_mult, -1)
        pre = x.new_zeros(*idx.shape, c.hc_mult, dtype=torch.float32); pre[..., 0] = 1
        positions = torch.arange(start, start + idx.size(1), device=idx.device)
        state, auxiliary, encoder = {}, x.new_zeros(()), None
        active_ids, active_history, features = idx, history, []
        for i, block in enumerate(self.blocks):
            if i == c.n_layer // 2:
                encoder = SinglePassMHC.read(x, pre)
                if kv_cache is not None and kv_cache.bounded_replay and start == 0:
                    tail = min(c.window_size, idx.size(1))
                    prefix = idx[:, :-tail] if history is None else torch.cat([history, idx[:, :-tail]], 1)
                    active_history, active_ids = prefix[:, -3:], idx[:, -tail:]
                    x, pre, positions = x[:, -tail:], pre[:, -tail:], positions[-tail:]
            if str(i) in self.engram:
                x = self.engram[str(i)](x, active_ids, active_history)
            if return_hidden and i >= c.n_layer - 3:
                features.append(x.mean(2))
            source = encoder if i >= c.n_layer // 2 else None
            x, pre, loss = block(x, pre, positions, state, kv_cache, source)
            auxiliary = auxiliary + loss
        logits = self.lm_head(self.norm(SinglePassMHC.read(x, pre))).float()
        if kv_cache is not None:
            previous = idx[:, :0] if history is None else history
            kv_cache.history = torch.cat([previous, idx], 1)[:, -3:].clone()
            kv_cache.pos += idx.size(1)
        if targets is None:
            return (logits, torch.cat([h[:, -logits.size(1):] for h in features], -1)) if return_hidden else logits
        loss = F.cross_entropy(logits.reshape(-1, c.vocab_size), targets[:, -logits.size(1):].reshape(-1), ignore_index=-1, reduction=loss_reduction)
        return loss + c.index_aux_weight * auxiliary if self.training and loss_reduction == "mean" else loss

    @torch.no_grad()
    def post_optimizer_step(self):
        for block in self.blocks:
            block.ffn.balance()

    def setup_optimizer(self, unembedding_lr=.004, embedding_lr=.2, matrix_lr=.02, weight_decay=.01, scalar_lr=.5):
        # Portable training baseline. The report's head-wise Muon/Engram Sinkhorn updates
        # are not claimed by this optimizer; preserve explicit groups for later ablations.
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise NotImplementedError("V41 reference optimizer is single-GPU; distributed gradient reduction is not implemented")
        embeddings = list(self.embedding.parameters()) + [p for m in self.engram.values() for table in m.tables for p in table.parameters()]
        embedding_ids = {id(p) for p in embeddings}
        other = [p for p in self.parameters() if id(p) not in embedding_ids and p is not self.lm_head.weight]
        groups = [dict(params=embeddings, lr=embedding_lr * .1, kind="adamw"), dict(params=[self.lm_head.weight], lr=unembedding_lr, kind="adamw"), dict(params=other, lr=matrix_lr * .1, kind="adamw")]
        opt = torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=weight_decay)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        return opt

    def num_scaling_params(self):
        total = sum(p.numel() for p in self.parameters())
        routed = sum(p.numel() for b in self.blocks for p in b.ffn.experts.parameters())
        wte = self.embedding.weight.numel()
        engram = sum(p.numel() for p in self.engram.parameters())
        head = self.lm_head.weight.numel()
        active = total - routed + routed * self.config.experts_per_token // self.config.n_experts
        return dict(total=total, active_total=active, wte=wte, value_embeds=engram, lm_head=head,
                    transformer_matrices=active - wte - engram - head, scalars=0)

    def estimate_flops(self):
        c = self.config
        active = self.num_scaling_params()
        return 6 * (active["active_total"] - active["wte"] - active["value_embeds"]) + 12 * c.n_layer * c.n_head * c.head_dim * (c.window_size + c.index_topk)

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1., top_k=None, seed=42):
        ids = torch.tensor([tokens], device=self.get_device())
        cache = self.create_kv_cache(1, len(tokens) + max_tokens)
        rng = torch.Generator(device=self.get_device()).manual_seed(seed)
        logits = self(ids, kv_cache=cache)[:, -1]
        for i in range(max_tokens):
            if temperature == 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                if top_k:
                    cutoff = logits.topk(min(top_k, logits.size(-1)), -1).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, -torch.inf)
                next_id = torch.multinomial((logits / temperature).softmax(-1), 1, generator=rng)
            yield next_id.item()
            if i + 1 < max_tokens:
                logits = self(next_id, kv_cache=cache)[:, -1]
