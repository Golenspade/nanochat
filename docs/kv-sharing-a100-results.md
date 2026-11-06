# Colab A100 KV-sharing pilot - 2026-11-06

On an isolated A100, FR and FRR improved steady training throughput by **11.4%** and **15.5%**, respectively. At batch 1 / context 4096, eager decode improved by **17.3%** and **24.6%**. The short-run validation BPB values were close; this pilot does not establish final model quality or justify changing the default from F.

## Conditions and provenance

- GPU: NVIDIA A100-SXM4-40GB, single GPU, initially idle with no GPU processes. The user's existing A100-80GB training session continued separately.
- Code: `6aed87167bd498a03735b051a55ccfdf6fd04850` on `codex/deepseek-cross-layer-kv`.
- Runtime: PyTorch `2.9.1+cu128`, CUDA 12.8, BF16, SDPA, all layers full-context (`window_pattern=L`). A100 does not exercise the FA3/Hopper branch.
- Model: depth 12, width 768, 6 query/KV heads, head dimension 128, vocabulary 49,152. Same depth and width, **different parameter counts**: 386,925,026 (F), 266,600,714 (FR), 226,492,610 (FRR). Savings include removing consumer K/V projections and moving alternating value embeddings to producers.
- Data: cached public `karpathy/climbmix-400b-shuffle`, training from `shard_00000.parquet`, validation from `shard_06542.parquet`, existing 49,152-token tokenizer, BOS best-fit packing, no Chinese stream. All six runs consumed identical pretokenized batches.
- Frozen corpus SHA-256: `9ae6fb133c2eb1aa26c407ac5e44659f75983d47b8f238d41feafff26ac21ea1`. Corpus contains token IDs and byte lengths; it is not committed to Git.
- Two seeds (42, 43), 128 steps each, batch 8 x sequence 1024: **1,048,576 training tokens per run**. Held-out validation: 16 batches / **131,072 tokens**. No checkpoint conversion, warm-start, or additional training tokens.
- Same MuonAdamW recipe: LR multiplier `sqrt(8192 / 524288) = 0.125`, unembedding/embedding/matrix/scalar base LRs `0.008/0.3/0.02/0.5`, 8-step LR warmup, final-quarter linear decay toward 0.1, Muon weight decay 0.01. This is a bounded pilot recipe, not the full speedrun schedule.
- Run order: F, FR, FRR for seed 42; FRR, FR, F for seed 43. The six training wall times including compilation totaled about 236 seconds; setup, validation and inference add overhead. The dedicated test session was released after results were downloaded and hash-verified.

## Training and validation

Training uses CUDA Inductor and the repository's fused MuonAdamW optimizer. Throughput is the mean of the two runs' median step throughputs, excluding the first 16 steps. Timing includes forward, backward, optimizer and GPU synchronization, but excludes data loading/tokenization, validation and logging. The corpus was staged on GPU before timing. Peak memory is PyTorch's allocated tensor peak during training, including optimizer and staged batches, **not** `nvidia-smi` usage or reserved memory.

| Pattern | Training token/s | Relative to F | Training peak GiB | BPB seed 42 | BPB seed 43 | Mean BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F | 125,867 | baseline | 6.29 | 1.95130 | 1.96001 | 1.95565 |
| FR | 140,236 | +11.4% | 5.35 | 1.94774 | 1.95905 | 1.95339 |
| FRR | 145,331 | +15.5% | 5.03 | 1.95163 | 1.96227 | 1.95695 |

Peak memory reduction relative to F is about **15% (FR)** / **20% (FRR)**. Mean BPB differences are -0.00226 / +0.00129, smaller than the seed-to-seed differences observed in each pattern. With only two seeds and approximately one million tokens per run, these results cannot establish quality equivalence, significance or performance after full training. This is an equal-token/depth/width comparison, not an equal-parameter comparison or an ablation isolating only the KV-cache change.

Cold/warm compile overhead is excluded from the steady table. Total per-run training times, including compile, were F 68.68/26.93 s, FR 48.85/23.70 s, FRR 45.53/22.66 s. These wall times also reflect compiler cache reuse; they are not a controlled comparison of cold compile time.

## Inference after training

Seed 42 models, batch 1, BF16, eager execution using `orig_model` under `force_eager`. Three prefill warmups, seven timed repetitions, 32 fixed-token decode steps per repetition. Reported latency includes Python and cache-position synchronization; token sampling, text tokenization, HTTP and network latency are excluded. These are **eager model measurements**, not compiled serving or FA3 benchmarks.

| Pattern | Context | Allocated KV MiB | Prefill median ms | Decode token/s | Prefill peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| F | 1,024 | 37.125 | 16.09 | 66.1 | 1.306 |
| FR | 1,024 | 18.563 | 14.17 | 78.9 | 1.049 |
| FRR | 1,024 | 12.375 | 13.20 | 83.1 | 0.964 |
| F | 4,096 | 145.125 | 19.41 | 66.2 | 2.283 |
| FR | 4,096 | 72.563 | 18.31 | 77.7 | 1.969 |
| FRR | 4,096 | 48.375 | 15.52 | 82.6 | 1.866 |

KV capacity is `context + 32` to accommodate decode; sizes are the actual CUDA tensor storages, not a projected formula. Prefill peak includes model and logits as well as cache. The raw `decode_peak_allocated_bytes` field includes the prefill performed before each decode repetition, so it should not be interpreted as a decode-only peak.

For all three trained models, full-sequence versus cached prefill/single-token/chunk output checks passed on GPU; the maximum absolute BF16 logit difference was 0.0625 (tolerance `atol=0.08`, `rtol=0.02`). The full repository suite on this A100 passed **61 tests**, with **12 FA3/Hopper skips**. Each training run completed 128 finite-loss backward/optimizer steps under CUDA Inductor.

## Artifacts and reproduction

Raw JSONs contain every training step, validation batch means, parameter counts, cache measurements, individual latency samples, code commit and corpus hash:

- [F seed 42](../dev/benchmarks/kv-sharing-a100-2026-11-06/F-seed42.json), [F seed 43](../dev/benchmarks/kv-sharing-a100-2026-11-06/F-seed43.json)
- [FR seed 42](../dev/benchmarks/kv-sharing-a100-2026-11-06/FR-seed42.json), [FR seed 43](../dev/benchmarks/kv-sharing-a100-2026-11-06/FR-seed43.json)
- [FRR seed 42](../dev/benchmarks/kv-sharing-a100-2026-11-06/FRR-seed42.json), [FRR seed 43](../dev/benchmarks/kv-sharing-a100-2026-11-06/FRR-seed43.json)

Prepare the same packing scheme from an existing cached tokenizer and train/validation shards. A different tokenizer, shard revision, packing implementation or metadata serialization can produce a different corpus hash; inspect provenance before comparing results.

```bash
NANOCHAT_BASE_DIR=/path/to/existing/cache python -m dev.prepare_kv_corpus \
    --output /tmp/nanochat_kv_corpus.pt

OMP_NUM_THREADS=4 python -m dev.benchmark_kv_sharing \
    --corpus /tmp/nanochat_kv_corpus.pt --pattern F --seed 42 \
    --output /tmp/F-seed42.json
```

Run FR and FRR with the same corpus/options, then repeat all three with `--seed 43 --skip-inference` for this pilot's layout. The default script settings match the reported 128-step budget. Each invocation should run on an otherwise idle GPU. The code saves numerical results rather than weights; the pilot models were not released as trained checkpoints.

Keep F as the default. A longer, fixed-token run with independent evaluation and multiple seeds is still needed before deciding whether the quality tradeoff is acceptable. Multi-GPU communication, FP8, H100/FA3, sliding-window combinations at scale, compiled inference, and long-context retrieval quality are outside this measurement.
