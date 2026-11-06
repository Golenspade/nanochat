"""Bounded, single-GPU F/FR/FRR pilot on an identical pretokenized real corpus.

The corpus is a torch.save dictionary: vocab_size, token_bytes, source, and
splits={train: [(x,y), ...], val: [(x,y), ...]}. No data is downloaded here.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import torch

from nanochat.common import COMPUTE_DTYPE, compute_init, compute_cleanup
from nanochat.engine import KVCache
from nanochat.flash_attention import USE_FA3
from nanochat.gpt import GPT, GPTConfig


def evaluate(model, batches, token_bytes):
    """Same held-out CE/BPB denominators as nanochat.loss_eval.evaluate_bpb."""
    total_nats = total_bytes = total_tokens = ce_nats = 0
    per_batch = []
    model.eval()
    with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
        for x, y in batches:
            loss = model(x, y, loss_reduction="none").float().view_as(y)
            lengths = token_bytes[y]
            total_nats += (loss * (lengths > 0)).sum().item()
            total_bytes += lengths.sum().item()
            total_tokens += y.numel()
            ce_nats += loss.sum().item()
            per_batch.append(loss.mean().item())
    return {"ce_nats": ce_nats / total_tokens,
            "bpb": total_nats / (math.log(2) * total_bytes),
            "tokens": total_tokens, "batch_ce": per_batch}


def make_cache(config, length, batch=1):
    return KVCache(batch, config.n_kv_head, length, config.n_embd // config.n_head,
                   config.n_layer, "cuda", COMPUTE_DTYPE,
                   layer_mapping=config.kv_cache_layer_map())


def measure(fn, repeats):
    durations = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(durations), "samples_ms": durations}


def inference_benchmark(model, vocab_size, contexts, repeats, decode_tokens):
    """End-to-end eager model latency, including Python/cache synchronization."""
    model.eval()
    records = []
    with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
        # Verify trained BF16 cached outputs against the same model's full forward.
        ids = torch.randint(vocab_size, (1, 33), device="cuda")
        cache = make_cache(model.config, 64)
        expected = model(ids)
        actual = torch.cat([model(ids[:, :24], kv_cache=cache),
                            model(ids[:, 24:25], kv_cache=cache),
                            model(ids[:, 25:], kv_cache=cache)], dim=1)
        max_error = (actual.float() - expected.float()).abs().max().item()
        torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.02)
        del ids, cache, expected, actual
        for context in contexts:
            ids = torch.randint(vocab_size, (1, context), device="cuda")
            next_id = ids[:, :1].contiguous()
            cache = make_cache(model.config, context + decode_tokens)

            def prefill():
                cache.reset()
                model(ids, kv_cache=cache)

            for _ in range(3):
                prefill()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            prefill_result = measure(prefill, repeats)
            prefill_peak = torch.cuda.max_memory_allocated()
            decode_times = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(repeats):
                prefill()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(decode_tokens):
                    model(next_id, kv_cache=cache)
                torch.cuda.synchronize()
                decode_times.append((time.perf_counter() - start) * 1000 / decode_tokens)
            records.append({
                "context": context, "batch": 1,
                "cache_capacity": context + decode_tokens,
                "cache_bytes": cache.k_cache.untyped_storage().nbytes() + cache.v_cache.untyped_storage().nbytes(),
                "prefill": prefill_result,
                "prefill_peak_allocated_bytes": prefill_peak,
                "decode_median_ms_per_token": statistics.median(decode_times),
                "decode_samples_ms_per_token": decode_times,
                "decode_tokens_per_second": 1000 / statistics.median(decode_times),
                "decode_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            })
            del ids, next_id, cache
    return {"mode": "eager", "cache_parity_max_abs_logit_error": max_error, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", choices=["F", "FR", "FRR"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()
    if args.steps < 17 or args.warmup_steps < 1 or args.repeats < 1 or args.decode_tokens < 1:
        parser.error("need steps >= 17 and positive warmup/repeats/decode-tokens")
    _, _, _, world_size, device = compute_init("cuda")
    if world_size != 1:
        raise ValueError("This pilot requires a single GPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    corpus = torch.load(args.corpus, map_location="cpu", weights_only=False)
    train_cpu, val_cpu = corpus["splits"]["train"], corpus["splits"]["val"]
    if len(train_cpu) < args.steps:
        raise ValueError("Corpus has fewer batches than the fixed training budget")
    # Freeze all data before timing, avoiding network, tokenization and host I/O noise.
    train = [(x.to(device, dtype=torch.long), y.to(device, dtype=torch.long)) for x, y in train_cpu[:args.steps]]
    val = [(x.to(device, dtype=torch.long), y.to(device, dtype=torch.long)) for x, y in val_cpu]
    token_bytes = corpus["token_bytes"].to(device)
    batch, seq_len = train[0][0].shape
    model_dim = math.ceil(args.depth * 64 / 128) * 128
    config = GPTConfig(sequence_len=max(seq_len, max(args.contexts)), vocab_size=corpus["vocab_size"],
                       n_layer=args.depth, n_head=model_dim // 128, n_kv_head=model_dim // 128,
                       n_embd=model_dim, window_pattern="L", kv_share_pattern=args.pattern)
    with torch.device("meta"):
        orig_model = GPT(config)
    orig_model.to_empty(device=device)
    orig_model.init_weights()
    initial_validation = evaluate(orig_model, val[:2], token_bytes)
    model = torch.compile(orig_model, dynamic=False)
    # Identical fixed recipe across patterns. Do not scale the horizon by parameter count.
    lr_scale = math.sqrt(batch * seq_len / 524288)
    optimizer = orig_model.setup_optimizer(unembedding_lr=0.008 * lr_scale,
        embedding_lr=0.3 * lr_scale, matrix_lr=0.02 * lr_scale,
        scalar_lr=0.5 * lr_scale, weight_decay=0.01)
    orig_model.train()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    records = []
    for step, (x, y) in enumerate(train):
        multiplier = min(1.0, (step + 1) / args.warmup_steps)
        if step >= args.steps * 0.75:
            multiplier *= 0.1 + 0.9 * (args.steps - step) / (args.steps * 0.25)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - step_start
        value = loss.item()
        if not math.isfinite(value):
            raise ValueError(f"nonfinite loss at step {step}")
        records.append({"step": step + 1, "loss": value, "seconds": seconds})
        if step % 16 == 0 or step == args.steps - 1:
            print(json.dumps({"pattern": args.pattern, "seed": args.seed, **records[-1]}), flush=True)
    training_seconds = time.perf_counter() - start_time
    training_peak = torch.cuda.max_memory_allocated()
    final_validation = evaluate(orig_model, val, token_bytes)
    optimizer.zero_grad(set_to_none=True)
    del optimizer, model, loss, train, val, token_bytes
    gc.collect()
    torch.cuda.empty_cache()
    steady = [r["seconds"] for r in records[16:]]
    result = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__, "cuda": torch.version.cuda, "compute_dtype": str(COMPUTE_DTYPE),
        "attention_backend": "FA3" if USE_FA3 else "SDPA", "seed": args.seed,
        "config": asdict(config), "parameter_count": sum(p.numel() for p in orig_model.parameters()),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(), "source": corpus["source"],
        "training": {"mode": "inductor", "steps": args.steps, "batch": batch, "seq_len": seq_len,
            "tokens": args.steps * batch * seq_len, "warmup_steps": args.warmup_steps,
            "optimizer": "MuonAdamW", "lr_scale": lr_scale, "muon_weight_decay": 0.01,
            "wall_seconds_including_compile": training_seconds,
            "steady_median_step_seconds": statistics.median(steady),
            "steady_tokens_per_second": batch * seq_len / statistics.median(steady),
            "peak_allocated_bytes": training_peak, "steps_detail": records},
        "initial_validation_first_2_batches": initial_validation, "final_validation": final_validation,
    }
    if not args.skip_inference:
        result["inference"] = inference_benchmark(orig_model, corpus["vocab_size"], args.contexts, args.repeats, args.decode_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print('RESULT', args.output, json.dumps({"pattern": args.pattern, "seed": args.seed,
        "validation_bpb": final_validation["bpb"], "tokens_per_second": result["training"]["steady_tokens_per_second"]}), flush=True)
    compute_cleanup()


if __name__ == "__main__":
    main()
