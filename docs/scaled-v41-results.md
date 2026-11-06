# Scaled V4.1 A100 acceptance - 2026-11-06

The separate V4.1 text backbone has completed real training, checkpoint reload, normal generation, BF16/FP4 cache execution, bounded replay and frozen-backbone draft training. This establishes a working **scaled architecture prototype**, not a useful assistant or quality-equivalent reproduction of the released model.

## Runtime and data

- Dedicated NVIDIA A100-SXM4-40GB, PyTorch 2.9.1+cu128, BF16 compute / fp32 master weights, portable PyTorch gathered attention and eager expert dispatch. The original A100-80GB training was kept separate.
- Short acceptance code: `8d0f510edea0743a697d0ed3ab6d1593daf8ec06`. Wider candidate/replay code: `5cb951f6b166f991bfeae880910992cea429cbe9`. Subsequent changes document results and CLI behavior.
- Identical frozen ClimbMix corpus from the earlier experiment, SHA-256 `9ae6fb133c2eb1aa26c407ac5e44659f75983d47b8f238d41feafff26ac21ea1`. Original train and validation shards are `shard_00000` / `shard_06542`. No Chinese stream.
- The original BOS-packed 1024-token rows are resegmented to the smaller acceptance sequence length. Not every new short row begins at BOS. The existing 49,152-token tokenizer is reused; Engram normalization maps are built and saved in the checkpoints.
- Optimizer: grouped AdamW, LR 0.0006 for all three groups in this runner, weight decay 0.01, 16-step warmup, final-quarter decay, gradient clipping at 1.0. Router correction biases update after each optimizer step. Training loss includes the 0.01 indexer objective; evaluation BPB excludes it.

## Training results

All runs use eight layers, two mHC streams, 4 routed experts / top-2 plus one shared expert, local window 16, encoder compression 2, decoder compression 1, sparse top-8, and two Engram placements unless disabled. Both decoder global and index caches share the encoder-derived source.

| Run | Width | Total / active params | Training tokens | Validation tokens | Initial BPB | Final BPB | Training time | Peak allocated memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 128 | 14.12M / 13.72M | 131,072 | 2,048 | 2.95766 | 2.14055 | 121.65 s | 399.08 MiB |
| FP4 QAT/cache | 128 | 14.12M / 13.72M | 131,072 | 2,048 | 2.95768 | 2.12869 | 124.21 s | 399.08 MiB |
| No Engram | 128 | 13.99M / 13.60M | 131,072 | 2,048 | 2.95474 | 2.11597 | 115.86 s | 393.33 MiB |
| Wider candidate | 256 | 30.80M / 29.23M | 1,048,576 | 32,768 | 3.17760 | **2.01037** | 249.21 s | **1.377 GiB** |

Each run used seed 42. The 128-wide checks used batch 2 x length 128 for 512 steps. The 256-wide candidate used batch 4 x length 256 for 1024 steps, consuming the complete frozen training corpus once. Every optimizer step checked finite loss and gradient norm. The full repository suite on the final A100 code passed **77 tests**, with **12 FA3/Hopper-only skips**.

These are acceptance runs, not an isolated architecture-quality comparison. Removing Engram changes parameter count and initialization; the candidate also changes width, batch, sequence length, token budget and validation subset. The no-Engram result is lower in this short check, so these data do **not** establish an Engram benefit. FP4's slightly lower BPB is likewise not evidence that quantization improves quality. Neither table is directly comparable with the earlier dense-reuse BPB table, whose validation set and training recipe differ.

Eager reference execution is dominated by many small kernels and host dispatch. No training-throughput improvement over nanochat or the production DeepSeek kernels is claimed. The vocabulary embedding/head dominate these small parameter counts, so the total/active ratio is far less sparse than the released giant model.

## Cache, replay and draft checks

- Every saved model was reconstructed from metadata and weights; GPU logits after reload were bitwise identical to the pre-save model.
- Full-sequence versus cached prefill/token/chunk maximum absolute BF16 logit differences were 0.03125 (Full), 0.0625 (FP4), 0.03125 (No Engram), 0.0625 (candidate). CPU tests use tight floating-point tolerances for exact algorithmic equivalence. BF16 results are reported numerically, not called bitwise equal.
- At prompt length 32, capacity 128, head dimension 32 and local window 16, allocated cache storage was **26,624 bytes BF16** versus **17,792 bytes FP4**, including local latents, global/index storage and partial-group buffers. This is about 33% total storage reduction. Main latents use actual packed nibbles and E4M3 scales; index/local storage remains BF16.
- Decoder bounded replay computed 16 decoder positions for the 32-token prompt. Candidate mean absolute logit difference from full decoder computation was **0.04333**. This is an intentional approximation, not a quality-equivalence result.
- Candidate prefix-hit replay retained 24 prefix tokens, appended 8 uncached tokens, and replayed only 16 old tokens: **24 encoder positions and 16 decoder positions** were processed. All existing global prefix entries remained unchanged, and the final cache position was 32. Its mean absolute logit difference from full computation was **0.07332**.
- The Full run trained DSpark-style draft/Markov/confidence heads for 32 steps; the candidate trained them for 64. Candidate draft loss went from 11.4054 to 8.0870. Backbone gradients remained absent. No acceptance-rate, verified speculative-generator or speed claim is made.

The candidate's actual greedy continuation was:

```text
 the same time. The most important time, the same time, the same time, and the same time, and the
```

Repetition remains obvious. A million training tokens, one seed and a small held-out slice do not establish broad language quality, Chinese/chat capability, long-context retrieval, SFT/RL readiness or usefulness as an assistant. Longer training and matched controls remain research work; the architecture/software acceptance is complete.

## Artifacts and reproduction

Raw machine-readable records:

- [Full](../dev/benchmarks/v41-scaled-a100-2026-11-06/full.json)
- [FP4](../dev/benchmarks/v41-scaled-a100-2026-11-06/fp4.json)
- [No Engram](../dev/benchmarks/v41-scaled-a100-2026-11-06/no-engram.json)
- [Wider candidate](../dev/benchmarks/v41-scaled-a100-2026-11-06/candidate.json)

With the frozen corpus and existing tokenizer directory:

```bash
python -m dev.train_v41 --corpus /path/nanochat_kv_corpus.pt \
    --tokenizer-dir /path/tokenizer --output /path/v41-candidate \
    --steps 1024 --width 256 --depth 8 --batch-size 4 --seq-len 256 \
    --val-batches 32 --dspark-steps 64
```

The candidate archive contains model, optimizer, metadata, draft weights, tokenizer, all four result files and logs. Archive SHA-256: `80de4261af86f7c341c5dcdb46a5c8edb6ef824546d86f25b6ad81575f2e744f`; the per-file manifest is preserved with the experiment. Weights stay outside Git and are not published as a model release.

The downloaded archive passed ZIP CRC, archive SHA-256 and all nine per-file SHA-256 checks. It was extracted and independently loaded on the local macOS CPU through the standard `load_model("base", ..., model_tag="v41-d8-w256", step=1024)` path. Repeated greedy generation was deterministic; see [local readback](../dev/benchmarks/v41-scaled-a100-2026-11-06/local-readback.json) and [artifact manifest](../dev/benchmarks/v41-scaled-a100-2026-11-06/artifact-manifest.json). This distinguishes an actual portable checkpoint from a model that only ran in the remote process.
