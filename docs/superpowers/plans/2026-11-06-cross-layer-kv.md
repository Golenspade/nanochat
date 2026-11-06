# Cross-Layer KV Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in cross-layer KV reuse inspired by DeepSeek-V4.1-Flash section 2.3.1, with physically smaller inference caches.

**Architecture:** Tile an F/R pattern: F produces KV, R attends to the most recent F's KV with its own query. Carry differentiable KV through training blocks; map logical layers to shared physical cache slots during inference. This is a dense adaptation, not a full CSA2 implementation.

**Tech Stack:** Python, PyTorch 2.9.1, existing FA3/SDPA interface, argparse, pytest.

## Global Constraints

- Use the existing COMPUTE_DTYPE; do not introduce autocast.
- Default F preserves existing parameter names, initialization, and checkpoint behavior.
- Preserve the user's dirty work by implementing in a separate worktree based on origin/chinese-support.
- Do not claim model quality, GPU throughput, full CSA2, or FP4 compression without evidence.
- Execute inline as part of the user's implementation-and-PR request.

### Task 1: Shared KV model and cache

**Files:** Modify `nanochat/gpt.py`, `nanochat/engine.py`, `nanochat/flash_attention.py`; create `tests/test_kv_sharing.py`.

**Interfaces:** `GPTConfig.kv_share_pattern: str = "F"`; `GPTConfig.kv_cache_layer_map() -> tuple[int, ...]`; `KVCache(..., layer_mapping=None)`.

- [x] Test tiled patterns and reject empty, invalid, or R-first patterns.

```python
assert GPTConfig(n_layer=5, kv_share_pattern="FR").kv_cache_layer_map() == (0, 0, 1, 1, 2)
```

- [x] Implement producer-only K/V and value-embedding parameters, differentiable shared KV, and independent queries/windows for every layer. Default F must retain all old state keys.

```python
# Consumers receive the producer's post-RoPE, post-QK-norm key and gated value.
y, shared_kv = self.attn(norm(x), ve, cos_sin, window_size, kv_cache, shared_kv)
x = x + y
```

- [x] Allocate cache tensors using the number of unique slots. Owners insert once; consumers read without modifying cache or advancing its cursor. Advance once per complete model forward.

```python
slot = self.layer_mapping[layer_idx]
return self.k_cache[slot], self.v_cache[slot]
```

- [x] Verify cached prefill, one-token/chunk decode, cache cloning, causality, gradients, and byte counts against full-sequence inference. Run `python -m pytest tests/test_kv_sharing.py tests/test_engine.py tests/test_attention_fallback.py -v`.

### Task 2: Training, compatibility, and review

**Files:** Modify `scripts/base_train.py` and `README.md`; create `docs/deepseek-kv-reuse.md`; extend `tests/test_kv_sharing.py`.

**Interfaces:** argparse `--kv-share-pattern F|FR|FRR...`; existing `asdict(GPTConfig)` metadata carries the new option through SFT/RL and inference.

- [x] Wire the CLI option into `GPTConfig`, and reject resume metadata with a different sharing pattern before loading weights.

```python
parser.add_argument("--kv-share-pattern", type=str, default="F", help="KV sharing pattern: F=produce, R=reuse; tiled across layers")
```

- [x] Test old metadata without the new field, a new shared checkpoint round trip, optimizer parameter coverage, Engine sampling, and compiled training gradients.
- [x] Document report section/page provenance, differences from CSA2, cache formula `2 * slots * batch * capacity * kv_heads * head_dim * element_size`, fresh-training requirements, and controlled d12 experiments.
- [x] Run the non-slow test suite and review the complete diff: 61 passed, 12 FA3/Hopper skips. Compare default F with the original model: state keys, seeded weights, and full-sequence logits are bitwise identical.
- [x] Commit only scoped files, push the branch, and open a PR targeting `chinese-support`. Verified PR [#1](https://github.com/Golenspade/nanochat/pull/1): open, not merged, head `codex/deepseek-cross-layer-kv`, base `chinese-support`.

### Task 3: User-requested Colab GPU validation

- [x] Verify Colab CLI/auth, inspect existing session load, and create a separate A100 test session without interrupting the existing training.
- [x] Add `dev/prepare_kv_corpus.py` and `dev/benchmark_kv_sharing.py` for a bounded, reproducible real-data comparison.
- [x] Run F/FR/FRR at two seeds with the same 1,048,576-token budget, independent validation data, CUDA Inductor and MuonAdamW; measure actual allocated memory and eager inference at 1K/4K contexts.
- [x] Download and validate all six result JSONs, archive hash and corpus/commit provenance; release the dedicated GPU session.
- [x] Document measurements, reproduction commands and limits in `docs/kv-sharing-a100-results.md` with raw records in `dev/benchmarks/kv-sharing-a100-2026-11-06/`.
