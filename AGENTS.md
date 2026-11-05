# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

nanochat is a minimal, full-stack experimental harness for training LLMs. It is designed to run end-to-end on a single GPU node: tokenizer training, base pretraining, supervised finetuning (SFT), reinforcement learning (RL), evaluation, and web serving. The codebase prioritizes minimalism and hackability over configurability.

The current development focus is the "GPT-2 speedrun" — training a model that outperforms OpenAI's 2019 GPT-2 (1.6B params, ~$43K to train) in under 3 hours on an 8xH100 node for ~$48.

## Development Environment

Dependencies are managed with **uv**.

```bash
# Install dependencies (GPU or CPU)
uv sync --extra gpu
uv sync --extra cpu

# For development (adds pytest, matplotlib, transformers, etc.)
uv sync --extra gpu --group dev

# Run tests
python -m pytest tests/test_engine.py -v
python -m pytest tests/test_attention_fallback.py -v

# Skip slow tests
python -m pytest tests/ -v -m "not slow"
```

All scripts use the virtual environment at `.venv/`. The `runs/speedrun.sh` script handles full setup automatically.

## Common Commands

### Full Pipeline

```bash
# GPT-2 grade model (~3 hours on 8xH100)
bash runs/speedrun.sh

# CPU/MPS demo (educational only)
bash runs/runcpu.sh

# Scaling laws experiments
bash runs/scaling_laws.sh

# Miniseries of compute-optimal models
bash runs/miniseries.sh
```

### Individual Stages

Scripts use standard `argparse`. For distributed training, prefix with `torchrun`:

```bash
# Tokenizer
python -m nanochat.dataset -n 8              # download data shards
python -m scripts.tok_train                  # train tokenizer (default vocab 32768)
python -m scripts.tok_eval                   # evaluate compression

# Base pretraining (single GPU or distributed)
python -m scripts.base_train --depth=12 --device-batch-size=32
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 --fp8 --run=myrun

# SFT (loads from base model automatically, no midtraining stage needed)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run=myrun

# RL
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=myrun

# Evaluation
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# Inference
python -m scripts.chat_cli -p "Why is the sky blue?"
python -m scripts.chat_web --num-gpus 1 --source sft

# Reporting
python -m nanochat.report reset
python -m nanochat.report generate   # produces report.md
```

### Quick Research Iteration

For rapid experimentation (~5 min runs), the d12 scale is the standard:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 --run="d12" --model-tag="d12" \
    --core-metric-every=999999 --sample-every=-1 --save-every=-1
```

## High-Level Architecture

### Precision / Compute Dtype

nanochat does **not** use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`).

- `Linear` layers (custom subclass in `gpt.py`) cast their weights to match input dtype during forward pass
- Master weights stay in fp32 for optimizer precision
- Embeddings are stored directly in `COMPUTE_DTYPE` to save memory
- Default: bf16 on CUDA SM80+ (A100/H100), fp32 on older GPUs / CPU / MPS
- Override via `NANOCHAT_DTYPE` environment variable: `bfloat16`, `float16`, `float32`
- fp16 training automatically enables `GradScaler` in training scripts

### Model Architecture

The GPT model (`nanochat/gpt.py`) is a decoder-only Transformer with these specific choices:

- **Rotary embeddings** (base theta = 100000), no absolute positional embeddings
- **QK norm** before attention, with a 1.2x scaling factor on Q and K
- **Untied weights** (separate token embedding and lm_head)
- **ReLU^2 activation** in the MLP
- **RMSNorm** after embedding, before attention, before MLP — no learnable params
- **No bias** in any linear layer
- **Group Query Attention (GQA)** via `n_kv_head`
- **Logits softcap** of 15 (`softcap * tanh(logits / softcap)`)
- **Padded vocab size** (rounded up to nearest 64 for DDP/tensor core efficiency)
- **Sliding window attention** (`window_pattern` string, e.g. `"SSSL"` = short/short/short/long, tiled across layers; final layer always full context)

Additional architectural features (more recent additions):

- **Custom `Linear` layer**: replaces `nn.Linear`, casts weights to input dtype in forward (replaces autocast)
- **Value embeddings (ResFormer)**: alternating layers get per-token value embeddings mixed into V with a learned gate
- **Smear**: mixes previous token's embedding into current position (cheap bigram info) via learned gate
- **Backout**: subtracts cached mid-layer residual before final norm to remove low-level features
- **Per-layer learnable scalars**: `resid_lambdas` (residual scaling) and `x0_lambdas` (input embedding blending), initialized with layer-dependent values

Model dimensions are derived from a single `depth` parameter:
- `n_layer = depth`
- `n_embd` is nudged up to nearest multiple of `head_dim` (default 128): `model_dim = ceil(depth * aspect_ratio / head_dim) * head_dim`
- `n_head = model_dim // head_dim`
- `n_kv_head = n_head` by default (1:1 GQA ratio)

The `__init__` runs on meta device (shapes only, no data). All actual initialization happens in `init_weights()`.

### Attention: Flash Attention 3 + SDPA Fallback

`nanochat/flash_attention.py` provides a unified interface that automatically uses:

- **Flash Attention 3** on Hopper GPUs (SM90, H100) with bf16
- **PyTorch SDPA** fallback on all other hardware (Blackwell, Ampere, CPU, MPS)

The tensor format is `(B, T, H, D)` — FA3's native layout. For inference, `flash_attn_with_kvcache` handles cache updates in-place.

If FA3 is unavailable and you use a non-trivial `window_pattern`, training efficiency drops significantly because SDPA has no native sliding window support.

### Training Pipeline

The pipeline has three stages (midtraining was removed and merged into SFT):

| Stage | Script | Output Dir | Notes |
|-------|--------|------------|-------|
| Base | `scripts/base_train.py` | `base_checkpoints/d<N>/` | Pretraining on ClimbMix |
| SFT | `scripts/chat_sft.py` | `chatsft_checkpoints/d<N>/` | Conversation, tool use, MC, math |
| RL | `scripts/chat_rl.py` | `chatrl_checkpoints/d<N>/` | Optional, currently only GSM8K |

`checkpoint_manager.py` provides `load_model(source, device, phase)` which maps source strings (`"base"`, `"sft"`, `"rl"`) to these directories and automatically loads the largest model tag and latest step.

**Important backward-compatibility detail**: `build_model()` patches missing keys from old checkpoints (e.g. `resid_lambdas`, `x0_lambdas`, `window_pattern`) with sensible defaults.

### Optimizer

`nanochat/optim.py` provides a **single combined optimizer** (`MuonAdamW` / `DistMuonAdamW`) that handles both AdamW and Muon parameters in one object:

- **Muon** (with Polar Express orthogonalization, not Newton-Schulz) for 2D matrix params (grouped by shape and stacked for efficiency)
- **AdamW** (fused kernel) for embeddings, lm_head, scalars, value embeddings
- **DistMuonAdamW** adds ZeRO-2 style distributed communication with 3-phase async overlap

Param groups have a `kind` field (`'adamw'` or `'muon'`). The optimizer uses `@torch.compile(dynamic=False, fullgraph=True)` fused kernels for both AdamW and Muon steps.

### Scaling Laws Automation

`base_train.py` automatically determines training hyperparameters from a single `depth` dial:

1. **Training horizon**: `target_tokens = target_param_data_ratio * num_scaling_params` (default ratio is 12, not Chinchilla's 20)
2. **Batch size**: follows Power Lines paper (`B ∝ D^0.383`), referenced to d12's measured optimal batch size of 524,288
3. **Learning rates**: scaled by `sqrt(B / B_ref)`
4. **Weight decay**: scaled to keep `T_epoch = B / (η · λ · D)` constant per the T_epoch framework

This means `--depth` is typically the only hyperparameter you need to change.

### Data Loading

- **Base training**: `nanochat/dataloader.py` streams from parquet shards, tokenizes on the fly, uses BOS-aligned best-fit packing (`bos_bestfit` variants). Supports approximate resume via state dict.
- **SFT**: `chat_sft.py` defines its own generator using best-fit-pad packing. Conversations are packed into rows with BOS alignment; unfilled positions are padded (not cropped), and padding targets are masked with `-1` (ignore index).

### FP8 Training

`nanochat/fp8.py` provides a ~150-line minimal FP8 training implementation (alternative to torchao's ~2000 lines). It wraps matmuls in a custom `autograd.Function` that quantizes to `float8_e4m3fn`/`float8_e5m2` and calls `torch._scaled_mm`. Enabled with `--fp8` flag in `base_train.py` (requires H100+). Evaluation automatically disables FP8 and falls back to BF16 via the `disable_fp8()` context manager.

### KV Cache and Inference

`nanochat/engine.py` provides the `Engine` class for efficient batched inference:

- KV cache format is `(n_layers, B, T, H, D)` — FA3 native layout
- Position tracked via `cache_seqlens` int32 tensor (per batch element)
- Single prefill at batch size 1, then clones KV cache for parallel generation
- Supports tool use: when model emits `<|python_start|>` ... `<|python_end|>`, engine evaluates via `use_calculator()` and forces result back
- `generate()` yields `(token_column, token_masks)` tuples; `generate_batch()` returns full sequences

### Tasks / Datasets

All datasets for SFT and evaluation live in `tasks/`. Each task subclasses `Task` and implements `get_example(index)` → conversation dict (OpenAI messages format) and `evaluate(problem, completion)` → metric.

`TaskMixture` interleaves multiple tasks; pass a task multiple times to oversample. `render_mc()` in `tasks/common.py` is the standard multiple-choice formatting.

### Compute Initialization Pattern

Every script follows this pattern:

```python
from nanochat.common import compute_init, compute_cleanup, autodetect_device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
# ... script logic ...
compute_cleanup()
```

`compute_init()` handles CUDA/MPS/CPU autodetection, seeding, float32 matmul precision, and DDP setup. `print0()` prints only from rank 0.

Scripts also commonly use `get_peak_flops()` for MFU calculation and manually manage the Python garbage collector (`gc.disable()` after step 1, with periodic manual collection).

### Reporting

`nanochat/report.py` generates a markdown report card (`report.md`) aggregating metrics from all pipeline stages. Scripts call `get_report().log(section=..., data=...)` which writes section files; `python -m nanochat.report generate` assembles them.

## Important Conventions

- **Use argparse** for CLI arguments in new scripts. The old `configurator.py` system was removed.
- **Keep `orig_model`** (uncompiled) alongside `model` (compiled) for evaluation/sampling, because compiled models may recompile on changing input shapes.
- **Checkpoint keys** prefixed with `_orig_mod.` (from torch.compile) are stripped in `build_model()`.
- When modifying the tokenizer split pattern or special tokens, check if the external `rustbpe` package (now a pip dependency, not local source) needs corresponding changes.
- The `COMPUTE_DTYPE` global is the source of truth for precision; do not introduce `torch.amp.autocast`.
