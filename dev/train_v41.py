"""Bounded real-corpus training/restore/replay/FP4/DSpark acceptance for scaled V41."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from nanochat.common import compute_init, compute_cleanup
from nanochat.deepseek_v41 import DeepSeekV41, V41Config
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.v41_dspark import DSparkDrafter


def batches(corpus, split, batch, length):
    x = torch.cat([a.flatten() for a, _ in corpus["splits"][split]])
    y = torch.cat([b.flatten() for _, b in corpus["splits"][split]])
    count = x.numel() // (batch * length)
    return x[:count*batch*length].view(count, batch, length), y[:count*batch*length].view(count, batch, length)


@torch.no_grad()
def evaluate(model, data, token_bytes, count):
    model.eval()
    nats = nbytes = ntokens = 0
    for i in range(count):
        x, y = (t[i].to(model.get_device(), dtype=torch.long) for t in data)
        losses = model(x, y, loss_reduction="none").float().view_as(y)
        lengths = token_bytes[y]
        nats += (losses * (lengths > 0)).sum().item()
        nbytes += lengths.sum().item(); ntokens += y.numel()
    return dict(bpb=nats / (math.log(2) * nbytes), tokens=ntokens)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path)
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--fp4", action="store_true")
    p.add_argument("--no-engram", action="store_true")
    p.add_argument("--streams", type=int, default=2)
    p.add_argument("--dspark-steps", type=int, default=32)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = p.parse_args()
    code_commit = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if args.width % 64 or min(args.steps, args.seq_len, args.batch_size, args.val_batches) < 1:
        p.error("width must align to 64 and training sizes must be positive")
    if args.dspark_steps and args.seq_len < 32:
        p.error("DSpark acceptance needs sequence length >= 32")
    _, _, _, _, device = compute_init(args.device)
    torch.manual_seed(args.seed); torch.set_num_threads(4)
    corpus = torch.load(args.corpus, weights_only=False, map_location="cpu")
    train, val = batches(corpus, "train", args.batch_size, args.seq_len), batches(corpus, "val", args.batch_size, args.seq_len)
    if args.steps > train[0].size(0) or args.val_batches > val[0].size(0):
        raise ValueError("Fixed corpus is too small for the requested unique-token budget")
    config = V41Config(vocab_size=corpus["vocab_size"], n_embd=args.width, n_head=args.width//32,
        head_dim=32, rope_dim=8, q_rank=args.width//2, o_rank=args.width//2,
        n_layer=args.depth, sequence_len=max(2048, args.seq_len), window_size=16,
        index_dim=16, index_topk=8, candidate_block_size=4, candidate_blocks=4,
        n_experts=4, expert_dim=args.width//2, hc_mult=args.streams,
        fp4_cache=args.fp4, enable_engram=not args.no_engram)
    with torch.device("meta"):
        model = DeepSeekV41(config)
    model.to_empty(device=device); model.init_weights()
    tokenizer = None
    if args.tokenizer_dir:
        from nanochat.tokenizer import RustBPETokenizer
        tokenizer = RustBPETokenizer.from_directory(str(args.tokenizer_dir))
        model.compress_engram_tokenizer(tokenizer)
    token_bytes = corpus["token_bytes"].to(device)
    initial = evaluate(model, val, token_bytes, args.val_batches)
    optimizer = model.setup_optimizer(unembedding_lr=.0006, embedding_lr=.006, matrix_lr=.006)
    model.train(); history = []
    synchronize = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    if args.device == "cuda": torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for i in range(args.steps):
        x, y = (t[i].to(device, dtype=torch.long) for t in train)
        multiplier = min(1., (i+1)/16) * max(.1, min(1., (args.steps-i)/(args.steps*.25)))
        for group in optimizer.param_groups: group["lr"] = group["initial_lr"] * multiplier
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm): raise ValueError(f"Nonfinite training at step {i}")
        optimizer.step(); model.post_optimizer_step()
        if i % 32 == 0 or i == args.steps-1:
            history.append(dict(step=i+1, loss=loss.item(), grad_norm=norm.item()))
            print(json.dumps(history[-1]), flush=True)
    synchronize()
    training_seconds = time.monotonic()-start
    peak = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
    final = evaluate(model, val, token_bytes, args.val_batches)
    args.output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(args.output), args.steps, model.state_dict(), optimizer.state_dict(),
        dict(model_config=asdict(config), step=args.steps, val_bpb=final["bpb"],
             user_config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}))
    # Independent reconstruction from serialized metadata/weights, with exact token outputs.
    metadata = json.loads((args.output/f"meta_{args.steps:06d}.json").read_text())
    with torch.device("meta"):
        restored = DeepSeekV41(V41Config(**metadata["model_config"]))
    restored.to_empty(device=device); restored.init_weights()
    restored.load_state_dict(torch.load(args.output/f"model_{args.steps:06d}.pt", map_location=device, weights_only=True))
    restored.eval()
    ids = val[0][0, :1, :32].to(device, dtype=torch.long)
    with torch.inference_mode():
        original = model(ids); loaded = restored(ids)
        torch.testing.assert_close(loaded, original, atol=0, rtol=0)
        exact = restored.create_kv_cache(1, 128)
        incremental = torch.cat([restored(ids[:, :24], kv_cache=exact), restored(ids[:, 24:25], kv_cache=exact), restored(ids[:, 25:], kv_cache=exact)], 1)
        parity = (incremental-original).abs().max().item()
        replay = restored.create_kv_cache(1, 128, bounded_replay=True)
        approximate = restored(ids, kv_cache=replay)
        replay_error = (approximate-original[:, -approximate.size(1):]).abs().mean().item()
        generated = list(restored.generate(ids[0].tolist(), max_tokens=24, temperature=0))
        memory = dict(storage_bytes=exact.storage_bytes(), global_entries=exact.global_counts,
                      local_tokens={str(k): v.size(1) for k,v in exact.local.items()})
        prefix_cache = restored.create_kv_cache(1, 128, bounded_replay=True)
        restored(ids[:, :24], kv_cache=prefix_cache)
        old = {owner: data[:, :prefix_cache.global_counts[owner]].clone() for owner,data in prefix_cache.globals.items()}
        before_tokens = prefix_cache.layer_tokens.copy()
        prefix_cache.drop_local()
        prefix_output = restored.replay_prefix(prefix_cache, ids[:, :24], ids[:, 24:])
        for owner, value in old.items():
            torch.testing.assert_close(prefix_cache.globals[owner][:, :value.size(1)], value, atol=0, rtol=0)
        prefix_replay = dict(global_prefix_unchanged=True, final_position=prefix_cache.get_pos(),
            encoder_tokens_replayed=prefix_cache.layer_tokens[0]-before_tokens[0],
            decoder_tokens_replayed=prefix_cache.layer_tokens[config.n_layer//2]-before_tokens[config.n_layer//2],
            mean_abs_logit_error=(prefix_output-original[:, -prefix_output.size(1):]).abs().mean().item())
    draft_result = None
    if args.dspark_steps:
        draft = DSparkDrafter(restored, block_size=5, markov_rank=16).to(device); draft.init_weights()
        draft_optimizer = torch.optim.AdamW(draft.parameters(), lr=.0006)
        draft_history = []
        restored.zero_grad(set_to_none=True)
        for i in range(args.dspark_steps):
            x = train[0][i % train[0].size(0)].to(device, dtype=torch.long)
            anchor = 16 + i % (args.seq_len - 24)
            with torch.no_grad():
                teacher, hidden = restored(x, return_hidden=True)
            draft_optimizer.zero_grad(set_to_none=True)
            loss = draft.training_loss(hidden[:, :anchor], x[:, anchor], anchor,
                x[:, anchor+1:anchor+6], teacher[:, anchor:anchor+5])
            loss.backward()
            if not torch.isfinite(loss): raise ValueError("Nonfinite DSpark training")
            draft_optimizer.step()
            for block in draft.blocks: block.ffn.balance()
            draft_history.append(loss.item())
        assert all(p.grad is None for p in restored.parameters())
        torch.save(draft.state_dict(), args.output/"dspark.pt")
        draft_result = dict(steps=args.dspark_steps, losses=draft_history, backbone_gradients_none=True,
                            note="Drafter/Markov/confidence trained separately; no speculative speed claim")
    result = dict(config=asdict(config), code_commit=code_commit,
        seed=args.seed, torch=torch.__version__, gpu=torch.cuda.get_device_name() if args.device=="cuda" else "cpu",
        corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        parameters=model.num_scaling_params(), initial_validation=initial, final_validation=final,
        trained_tokens=args.steps*args.batch_size*args.seq_len, training_seconds=training_seconds,
        peak_allocated_bytes=peak, training_history=history, checkpoint_exact_reload=True,
        cache_logit_max_abs_error=parity, bounded_replay_mean_abs_error=replay_error,
        bounded_replay_decoder_tokens=replay.layer_tokens[config.n_layer//2],
        generated_ids=generated, generated_text=None if tokenizer is None else tokenizer.decode(generated),
        cache=memory, encoder_prefix_replay=prefix_replay, dspark=draft_result)
    (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print('RESULT',json.dumps({"initial_bpb":initial["bpb"],"final_bpb":final["bpb"],"parity":parity,"seconds":training_seconds}),flush=True)
    compute_cleanup()


if __name__ == "__main__": main()
