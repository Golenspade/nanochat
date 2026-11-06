"""Post-backbone DSpark-style forward: parallel drafts + Markov/confidence heads.

Adapted from the pinned MIT inference reference. This is a trainable portable
drafter, not a claim of calibrated speculative-decoding acceleration. The caller
must verify draft tokens against the backbone before emitting them.
"""
import math
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.gpt import Linear
from nanochat.deepseek_v41 import CSA2
from nanochat.v41_components import RMSNorm, SinglePassMHC, RoutedMoE, rope


class DraftBlock(nn.Module):
    def __init__(self, c, window):
        super().__init__()
        self.c, self.window = c, window
        self.attn = CSA2(c, 0)  # local latent projections only
        self.a_mix, self.f_mix = SinglePassMHC(c.n_embd, c.hc_mult), SinglePassMHC(c.n_embd, c.hc_mult)
        self.a_norm, self.f_norm = RMSNorm(c.n_embd), RMSNorm(c.n_embd)
        self.ffn = RoutedMoE(c.n_embd, c.expert_dim, c.n_experts, c.experts_per_token)

    def forward(self, x, pre, context, start_pos):
        c, a = self.c, self.attn
        following, post, comb = self.a_mix.coefficients(x)
        z = self.a_norm(self.a_mix.read(x, pre))
        positions = torch.arange(start_pos, start_pos + z.size(1), device=z.device)
        context_pos = torch.arange(start_pos - context.size(1), start_pos, device=z.device)
        q = rope(a.q_b(a.q_norm(a.q_a(z))).unflatten(-1, (c.n_head, c.head_dim)), positions, c.rope_dim, 10000)
        context_kv = rope(a.local_norm(a.local_kv(context)), context_pos, c.rope_dim, 10000)
        draft_kv = rope(a.local_norm(a.local_kv(z)), positions, c.rope_dim, 10000)
        values = torch.cat([context_kv, draft_kv], 1)
        logits = torch.einsum("bthd,bkd->bthk", q.float(), values.float()) / math.sqrt(c.head_dim)
        valid_context = context_pos >= positions[:, None] - self.window + 1
        mask = torch.cat([valid_context, torch.ones(z.size(1), z.size(1), dtype=torch.bool, device=z.device)], -1)
        logits = logits.masked_fill(~mask[None, :, None], -torch.inf)
        sink = a.sink[None, None, :, None].expand(*logits.shape[:-1], 1)
        weights = torch.cat([logits, sink], -1).softmax(-1)[..., :-1]
        out = torch.einsum("bthk,bkd->bthd", weights, values.float()).to(z.dtype)
        out = rope(out, positions, c.rope_dim, 10000, inverse=True)
        grouped = out.reshape(*z.shape[:2], c.o_groups, -1)
        out = torch.einsum("btgd,grd->btgr", grouped, a.o_a.weight.to(z.dtype).view(c.o_groups, c.o_rank, -1)).flatten(-2)
        x = self.a_mix.write(x, a.o_b(out), post, comb)
        final, post, comb = self.f_mix.coefficients(x)
        out = self.ffn(self.f_norm(self.f_mix.read(x, following)))
        return self.f_mix.write(x, out, post, comb), final


class DSparkDrafter(nn.Module):
    def __init__(self, backbone, block_size=5, window=128, markov_rank=32):
        super().__init__()
        self.backbone_ref = weakref.ref(backbone)
        c = backbone.config
        self.block_size, self.window, self.c = block_size, window, c
        self.context = Linear(c.n_embd * 3, c.n_embd, bias=False)
        self.context_norm = RMSNorm(c.n_embd)
        self.noise = nn.Parameter(torch.empty(c.n_embd))
        self.blocks = nn.ModuleList([DraftBlock(c, window) for _ in range(3)])
        self.norm = RMSNorm(c.n_embd)
        self.markov = nn.Embedding(c.vocab_size, markov_rank)
        self.markov_head = Linear(markov_rank, c.vocab_size, bias=False)
        self.confidence = Linear(c.n_embd + markov_rank, 1, bias=False)

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=.02)
            elif isinstance(module, RMSNorm): module.weight.fill_(1)
            elif isinstance(module, SinglePassMHC): module.initialize()
            elif isinstance(module, CSA2): module.sink.zero_()
            elif isinstance(module, RoutedMoE): module.bias.zero_(); module.counts.zero_()
        nn.init.normal_(self.noise, std=.02)

    def forward(self, prefix_features, seed_token, start_pos, teacher_tokens=None):
        backbone = self.backbone_ref()
        if backbone is None:
            raise RuntimeError("DSpark backbone has been released")
        context = self.context_norm(self.context(prefix_features[:, -self.window:].detach()))
        batch = prefix_features.size(0)
        hidden = self.noise.to(context.dtype).expand(batch, self.block_size, -1).clone()
        hidden[:, 0] = F.embedding(seed_token, backbone.embedding.weight.detach()).to(context.dtype)
        hidden = hidden.unsqueeze(2).expand(-1, -1, self.c.hc_mult, -1)
        pre = hidden.new_zeros(batch, self.block_size, self.c.hc_mult, dtype=torch.float32); pre[..., 0] = 1
        for block in self.blocks:
            hidden, pre = block(hidden, pre, context, start_pos)
        hidden = self.norm(SinglePassMHC.read(hidden, pre))
        base = F.linear(hidden, backbone.lm_head.weight.detach().to(hidden.dtype)).float()
        previous, outputs, logits, confidence = seed_token, [], [], []
        for i in range(self.block_size):
            memory = self.markov(previous).to(hidden.dtype)
            scores = base[:, i] + self.markov_head(memory).float()
            token = scores.argmax(-1)
            outputs.append(token); logits.append(scores)
            confidence.append(self.confidence(torch.cat([hidden[:, i], memory], -1)).squeeze(-1))
            previous = token if teacher_tokens is None else teacher_tokens[:, i]
        return dict(tokens=torch.stack(outputs, 1), logits=torch.stack(logits, 1), confidence=torch.stack(confidence, 1))

    def training_loss(self, prefix_features, seed_token, start_pos, targets, teacher_logits):
        result = self(prefix_features, seed_token, start_pos, teacher_tokens=targets)
        ce = F.cross_entropy(result["logits"].flatten(0, 1), targets.flatten())
        acceptance = (result["tokens"] == teacher_logits.detach().argmax(-1)).float()
        confidence = F.binary_cross_entropy_with_logits(result["confidence"].float(), acceptance)
        return ce + .1 * confidence

    @staticmethod
    def verification_length(confidence, draft_ms, verify_ms):
        """Confidence scheduling given caller-profiled costs for lengths 1..K.

        Optimize expected accepted draft prefix per measured total time. This is
        deliberately not a fabricated throughput curve or an unverified speed claim.
        """
        survival = confidence.sigmoid().cumprod(-1).cumsum(-1)
        costs = torch.as_tensor(verify_ms, device=confidence.device) + draft_ms
        if costs.numel() != confidence.size(-1) or (costs <= 0).any():
            raise ValueError("Need positive profiled verification cost for every draft length")
        return (survival / costs).argmax(-1) + 1
