#!/usr/bin/env python3
"""
Batch Inference & Comparative Evaluation Runner for 1,000 Questions.
Supports:
  - Running inference on Pure SFT model
  - Running inference on SFT + RL (GRPO) model
  - Generating side-by-side comparison JSON and standalone HTML report.
"""

import os
import sys
import json
import time
import argparse
import torch

# Ensure script directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nanochat.common import autodetect_device_type, compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

def run_model_inference(phase, model_tag="d12_cn", step=None, input_file="tasks/eval_1000_questions.json", output_file=None, max_new_tokens=320):
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    
    print(f"\n==================================================")
    print(f"🚀 Initializing [{phase.upper()}] model inference on {device} (tag={model_tag})...")
    print(f"==================================================")
    
    # Load model and tokenizer
    model, tokenizer, meta = load_model(phase, device, phase="eval", model_tag=model_tag, step=step)
    engine = Engine(model, tokenizer)
    
    # Load questions
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input benchmark file not found: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    print(f"Loaded {len(questions)} evaluation questions from {input_file}.")
    results = []
    start_time = time.time()
    
    for i, q in enumerate(questions):
        # Format as conversation priming for completion
        conversation = {
            "messages": [
                {"role": "user", "content": q["prompt"]},
                {"role": "assistant", "content": ""}
            ]
        }
        tokens = tokenizer.render_for_completion(conversation)
        prefix_len = len(tokens)
        
        t0 = time.time()
        # Batched generation (sample 1)
        generated_sequences, _ = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
            seed=42 + i
        )
        latency = time.time() - t0
        
        gen_tokens = generated_sequences[0][prefix_len:]
        response_text = tokenizer.decode(gen_tokens).strip()
        
        # Strip internal special tags if present
        for tag in ["<|assistant_end|>", "<|im_end|>", "<s>", "</s>", "<|endoftext|>"]:
            response_text = response_text.replace(tag, "")
            
        record = {
            "id": q["id"],
            "category": q["category"],
            "category_name": q["category_name"],
            "title": q["title"],
            "prompt": q["prompt"],
            "response": response_text.strip(),
            "tokens_generated": len(gen_tokens),
            "latency_sec": round(latency, 3)
        }
        results.append(record)
        
        if (i + 1) % 50 == 0 or (i + 1) == len(questions):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"[{phase}] Progress: {i+1}/{len(questions)} ({100*(i+1)/len(questions):.1f}%) | "
                  f"Speed: {rate:.2f} q/s | Elapsed: {elapsed:.1f}s | Avg tokens: {sum(r['tokens_generated'] for r in results)/(i+1):.1f}")
            
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(results)} outputs to {output_file}")
        
    return results

def merge_and_generate_html(sft_file, rl_file, html_out):
    print(f"\nMerging results and building comparison report...")
    with open(sft_file, "r", encoding="utf-8") as f:
        sft_data = json.load(f)
    with open(rl_file, "r", encoding="utf-8") as f:
        rl_data = json.load(f)
        
    sft_dict = {item["id"]: item for item in sft_data}
    rl_dict = {item["id"]: item for item in rl_data}
    
    comparisons = []
    for q_id, sft_item in sft_dict.items():
        rl_item = rl_dict.get(q_id, {})
        comparisons.append({
            "id": q_id,
            "category": sft_item["category"],
            "category_name": sft_item["category_name"],
            "title": sft_item["title"],
            "prompt": sft_item["prompt"],
            "sft_response": sft_item.get("response", ""),
            "sft_tokens": sft_item.get("tokens_generated", 0),
            "sft_latency": sft_item.get("latency_sec", 0.0),
            "rl_response": rl_item.get("response", ""),
            "rl_tokens": rl_item.get("tokens_generated", 0),
            "rl_latency": rl_item.get("latency_sec", 0.0),
        })
        
    # Build HTML document
    from generate_comparison_html import build_html_content
    html_str = build_html_content(comparisons)
    os.makedirs(os.path.dirname(os.path.abspath(html_out)), exist_ok=True)
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"🎉 Successfully generated HTML comparison report at: {html_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["sft", "rl", "both", "merge_html"], default="both")
    parser.add_argument("--model-tag", type=str, default="d12_cn")
    parser.add_argument("--sft-out", type=str, default="tasks/eval_1000_sft_out.json")
    parser.add_argument("--rl-out", type=str, default="tasks/eval_1000_rl_out.json")
    parser.add_argument("--html-out", type=str, default="eval_comparison_sft_vs_rl.html")
    parser.add_argument("--max-tokens", type=int, default=320)
    args = parser.parse_args()
    
    if args.mode in ["sft", "both"]:
        run_model_inference("sft", model_tag=args.model_tag, output_file=args.sft_out, max_new_tokens=args.max_tokens)
    if args.mode in ["rl", "both"]:
        run_model_inference("rl", model_tag=args.model_tag, output_file=args.rl_out, max_new_tokens=args.max_tokens)
    if args.mode in ["merge_html", "both"]:
        merge_and_generate_html(args.sft_out, args.rl_out, args.html_out)
