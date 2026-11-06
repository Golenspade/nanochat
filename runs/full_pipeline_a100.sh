#!/bin/bash
# =============================================================================
# Full Autonomous Training Pipeline on NVIDIA A100-80GB (High-RAM)
# Stages:
#   1. Dataset Download & 49,152 Chinese Tokenizer Training (~3m)
#   2. Base Pretraining (2808 steps, ~2h, checkpointing at 1000, 2000, 2808)
#   3. Chat SFT with fankex Persona + SmolTalkCN + GSM8K (~50m)
#   4. Post-train RLHF with GRPO on GSM8K reasoning (~1h)
#   5. 1,000 Questions Comprehensive Comparison Inference (SFT vs RL)
#   6. Single-file HTML Comparison Report Generation
# =============================================================================

set -euo pipefail

export NANOCHAT_BASE_DIR="/content/nanochat_cache"
export PYTHONPATH="/content/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

cd /content/nanochat
source .venv/bin/activate

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a /content/train.log
}

log "=================================================="
log "🚀 Starting NanoGPT A100 Full Pipeline (Base + SFT + RLHF + 1000 Eval)"
log "=================================================="

# Copy fankex identity dataset
cp /content/nanochat/tasks/identity_conversations_cn.jsonl "$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
cp /content/nanochat/tasks/identity_conversations_cn.jsonl "$NANOCHAT_BASE_DIR/identity_conversations_cn.jsonl"

# 1. Dataset & Tokenizer
log "=== Stage 1: Shard Datasets & Train 49,152 Tokenizer ==="
python -m nanochat.dataset -n 32
python -m scripts.tok_train --vocab-size=49152 --pattern=cn
python -m scripts.tok_eval

# 2. Base Pretraining
log "=== Stage 2: Base Pretraining (Depth 12, 124M params, 500M tokens) ==="
python -m scripts.base_train \
    --depth=12 \
    --run=d12_cn_a100 \
    --model-tag=d12_cn \
    --cn-ratio=0.5 \
    --window-pattern=L \
    --device-batch-size=32 \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=1000

log "✅ Base Pretraining completed successfully."

# 3. Chat SFT
log "=== Stage 3: Chat SFT Alignment (SmolTalkCN + fankex identity + GSM8K) ==="
cp /content/nanochat/tasks/identity_conversations_cn.jsonl "$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
python -m scripts.chat_sft \
    --run=d12_cn_a100 \
    --model-tag=d12_cn \
    --device-batch-size=16 \
    --cn-talk-epochs=1 \
    --gsm8k-cn-epochs=1 \
    --ceval-epochs=0

log "✅ Chat SFT completed successfully."

# 4. Post-train RLHF (GRPO)
log "=== Stage 4: Post-train RLHF (GRPO Algorithm on GSM8K) ==="
python -m scripts.chat_rl \
    --model-tag=d12_cn \
    --num-epochs=1 \
    --device-batch-size=8 \
    --examples-per-step=16 \
    --num-samples=8 \
    --eval-every=60 \
    --eval-examples=40 \
    --save-every=50

log "✅ Post-train RLHF (GRPO) completed successfully."

# 5. 1,000 Questions Comparative Inference
log "=== Stage 5: 1,000 Questions Comparative Inference (SFT vs RL) ==="
python /content/nanochat/dev/eval_1000_inference.py \
    --mode=both \
    --model-tag=d12_cn \
    --sft-out=/content/nanochat/tasks/eval_1000_sft_out.json \
    --rl-out=/content/nanochat/tasks/eval_1000_rl_out.json \
    --html-out=/content/nanochat/eval_comparison_sft_vs_rl.html \
    --max-tokens=280

log "✅ 1,000 Questions Inference & HTML Report completed."

# 6. Final Bundle Archiving
log "=== Stage 6: Creating Checkpoint Release Bundle ==="
tar -czf /content/nanochat_full_release.tar.gz \
    -C "$NANOCHAT_BASE_DIR" base_checkpoints/d12_cn \
    -C "$NANOCHAT_BASE_DIR" chatsft_checkpoints/d12_cn \
    -C "$NANOCHAT_BASE_DIR" chatrl_checkpoints/d12_cn \
    -C /content/nanochat eval_comparison_sft_vs_rl.html \
    -C /content/nanochat/tasks eval_1000_questions.json eval_1000_sft_out.json eval_1000_rl_out.json \
    || true

log "🎉 ALL STAGES FINISHED! Bundle created at /content/nanochat_full_release.tar.gz"
