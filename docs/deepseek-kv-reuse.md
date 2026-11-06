# Experimental cross-layer KV reuse

## Report and scope

This experiment adapts the **layer dimension** of KV sharing from *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*, section 2.3.1, pages 10-11 (Figure 4). The supplied source is `DeepSeek_V41_Tech_Report.pdf`, SHA-256 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`. The PDF is not redistributed in this repository.

The report separates KV production from layer-specific queries. Its Full layers create global KV, while Reindex/Reuse layers share those entries. This PR isolates that idea in nanochat's existing dense GQA attention so it can be trained and tested on a small model.

| Component | This implementation |
| --- | --- |
| Cross-layer KV sharing | Implemented: static F/R pattern, independent queries, shared post-RoPE keys and gated values |
| Per-layer attention window | Existing window pattern remains independent of sharing |
| Separate global and local SWA branches | Not implemented; each layer has one attention branch |
| Sparse indexer / Reindex / Top-K reuse | Not implemented; all visible positions within the layer's window are attended |
| Sequence compression / MLA / FP4 KV | Not implemented; existing GQA and compute precision remain |
| CED / bounded replay | Not implemented; prefill still runs all layers |
| MoE / Single-Pass mHC / Engram / DSpark / multimodal training | Outside this experiment |

`F` and `R` here name KV producer/consumer roles, **not** complete CSA2 operating modes. The report's model-level quality, throughput, 890-byte global KV, and persistent-cache results do not apply to this implementation. In particular, the local window does not have an independent KV stream as it does in CSA2.

## Model and cache behavior

`GPTConfig.kv_share_pattern` defaults to `F`. A case-insensitive pattern must start with F and contain only F/R. It is tiled across depth without forcing the last layer to produce KV:

| Pattern at depth 6 | Logical layer to physical slot | KV slots |
| --- | --- | --- |
| `F` | `0,1,2,3,4,5` | 6 |
| `FR` | `0,0,1,1,2,2` | 3 |
| `FRR` | `0,0,0,1,1,1` | 2 |

Each producer projects K/V, applies its value embedding, rotates/normalizes K once, and passes the resulting tensors to its consumers. Every layer retains its own Q, output projection, MLP, residual weights, and attention window. Value embeddings alternate across **producers**, with the last producer always included. Consumer gradients flow into the producer's K/V and value embedding without detaching.

Inference allocates one physical K/V slot per producer. Producers append tokens once; consumers use read-only attention at the newly populated cache length. The model advances the shared cursor once after all layers. Both cache layouts (`bthd`, `bhtd`) support cloning a single prompt into a multi-sample batch, preserving smear state. The model passes an explicit cache layout so that a one-token prompt cannot be misidentified by the legacy shape heuristic. A mismatched cache mapping fails before modifying the cache. Multi-token cached chunks also carry smear state across the chunk boundary.

For custom inference code, supply the map explicitly:

```python
cache = KVCache(
    batch_size=batch_size, num_heads=config.n_kv_head,
    seq_len=capacity, head_dim=config.n_embd // config.n_head,
    num_layers=config.n_layer, device=device, dtype=dtype,
    layer_mapping=config.kv_cache_layer_map(),
)
```

`Engine.generate` and `GPT.generate` do this automatically. Existing checkpoints without `kv_share_pattern` load as F; new checkpoints serialize the field through the existing model config, including SFT/RL loading. Consumer K/V parameters do not exist, so parameter counts, optimizer groups, and the matrix part of FLOP estimates reflect the chosen architecture. Attention itself remains dense and retains its original complexity.

Use a **fresh training run and a distinct model tag** when changing patterns. An F checkpoint is not a trained FR model; there is no automatic architecture conversion. Base-training resume rejects different effective sharing maps before loading state. Default F retains the original state keys, seeded initialization, and full-sequence logits.

## Storage measurement

The allocated K/V tensor storage is:

```text
bytes = 2 * producer_count * batch * capacity * kv_heads * head_dim * element_size
```

Measured locally on 2026-11-06 using CPU-allocated BF16 tensors at depth 12, batch 1, capacity 2048, 6 KV heads and head dimension 128:

| Pattern | Physical slots | Allocated K/V bytes | MiB |
| --- | ---: | ---: | ---: |
| F | 12 | 75,497,472 | 72 |
| FR | 6 | 37,748,736 | 36 |
| FRR | 4 | 25,165,824 | 24 |

These are the actual tensor storage sizes, not process RSS, total model memory, or GPU peak-memory measurements. With a depth not divisible by the group size, use the actual producer count rather than assuming an exact 2x/3x reduction. Cache storage still grows linearly with context length.

## Validation

The local CPU suite exercises full-sequence versus cached prefill/single-token/chunk outputs, GQA/MHA, both cache layouts, unequal per-layer windows, causality, physical aliasing, cloning/reset, consumer-only gradients into producer KV, optimizer updates, default/legacy checkpoint loading, shared checkpoint round trips, and Engine versus GPT greedy generation. A separate `torch.compile(backend="aot_eager", fullgraph=True)` check compares training loss and every parameter gradient with eager execution. GPU-only FA3 read-only checks are included and skip when unavailable.

```bash
python -m pytest tests/ -v -m 'not slow'
python -m pytest tests/test_kv_sharing.py::test_compiled_training_matches_eager_gradients -v
```

The full suite completed locally with **61 passed, 12 skipped** on macOS/Python 3.10/PyTorch 2.9.1. Ordinary new behavior checks force eager CPU/SDPA execution; the compiled-training test enables Dynamo and AOTAutograd. The second command isolates that test. All skips require FA3/Hopper. These checks do not establish CUDA Inductor, multi-GPU optimizer, FP8, H100 throughput, or trained-model quality. No training datasets or weights are downloaded by the new tests.

## Controlled d12 experiment

After the normal tokenizer/data setup, run separate F and FR experiments on the same hardware, seed, dataset, token budget and total batch size:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth 12 --kv-share-pattern F --window-pattern L \
    --num-iterations 100 --total-batch-size 524288 --device-batch-size 16 \
    --model-tag d12-kv-f --run d12-kv-f --core-metric-every 999999 --sample-every -1

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth 12 --kv-share-pattern FR --window-pattern L \
    --num-iterations 100 --total-batch-size 524288 --device-batch-size 16 \
    --model-tag d12-kv-fr --run d12-kv-fr --core-metric-every 999999 --sample-every -1
```

Fixing steps and batch size matters: the automatic token horizon changes when the parameter count changes. Measure validation BPB/CORE, trained generation quality, prefill/decode latency, peak GPU memory and throughput before considering a default change. Compare inference at equal batch/context/dtype, and keep compile warmup separate. These GPU/quality experiments have not been run as part of this PR.
