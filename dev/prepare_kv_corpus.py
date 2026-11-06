"""Freeze identical BOS-packed ClimbMix batches for the bounded KV-sharing pilot."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer, get_token_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--train-batches", type=int, default=128)
    parser.add_argument("--val-batches", type=int, default=16)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.train_batches, args.val_batches) < 1:
        parser.error("all sizes must be positive")
    torch.set_num_threads(2)
    paths = list_parquet_files()
    if len(paths) < 2:
        raise ValueError("Need separate cached training and validation parquet shards")
    tokenizer = get_tokenizer()
    payload = {"vocab_size": tokenizer.get_vocab_size(), "token_bytes": get_token_bytes(),
               "batch_size": args.batch_size, "seq_len": args.seq_len, "splits": {},
               "source": {"dataset": "karpathy/climbmix-400b-shuffle", "cn_ratio": 0,
                          "train_shard": Path(paths[0]).name, "validation_shard": Path(paths[-1]).name}}
    for split, count in [("train", args.train_batches), ("val", args.val_batches)]:
        loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.batch_size,
            args.seq_len, split, device="cpu", tokenizer_threads=2)
        batches = []
        for _ in range(count):
            x, y = next(loader)
            batches.append((x.to(torch.int32).clone(), y.to(torch.int32).clone()))
        payload["splits"][split] = batches
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"output": str(args.output), "source": payload["source"],
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
