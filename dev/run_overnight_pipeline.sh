#!/usr/bin/env bash
# =============================================================================
# NanoGPT Overnight Autonomous Pipeline (Colab A100)
# Orchestrates:
#   1. SFT Completion Monitor
#   2. Post-train RLHF (GRPO on GSM8K)
#   3. 1,000-Question Batch Inference for SFT vs RL
#   4. Standalone HTML Comparison Report Generation
#   5. Archive and Checkpoint Packaging
# =============================================================================

set -euo pipefail

export NANOCHAT_BASE_DIR="/content/nanochat_cache"
export PYTHONPATH="/content/nanochat"
cd /content/nanochat

VENV_PY="/content/nanochat/.venv/bin/python"
LOG="/content/pipeline.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "=================================================="
log "🌙 NanoGPT Overnight Autonomous Orchestrator Started"
log "=================================================="

# 1. Wait for Chat SFT to complete
log "Step 1: Monitoring Chat SFT process..."
while pgrep -f "scripts.chat_sft" > /dev/null; do
    sleep 30
done
log "✅ Chat SFT process has terminated."

# Verify SFT checkpoint
SFT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d12_cn"
if [ ! -d "$SFT_DIR" ] && [ -d "/root/.cache/nanochat/chatsft_checkpoints/d12_cn" ]; then
    mkdir -p "$SFT_DIR"
    cp -r /root/.cache/nanochat/chatsft_checkpoints/d12_cn/* "$SFT_DIR/"
fi

log "Checking SFT checkpoints in $SFT_DIR..."
ls -lh "$SFT_DIR" | tee -a "$LOG" || true

# 2. Launch Post-train RL (GRPO)
log "Step 2: Starting Post-train RL (GRPO on GSM8K reasoning)..."
$VENV_PY -m scripts.chat_rl \
    --model-tag=d12_cn \
    --num-epochs=1 \
    --device-batch-size=8 \
    --examples-per-step=16 \
    --num-samples=8 \
    --eval-every=60 \
    --eval-examples=40 \
    --save-every=60 \
    2>&1 | tee -a /content/rl.log

log "✅ Post-train RL (GRPO) completed."

# Verify RL checkpoint
RL_DIR="$NANOCHAT_BASE_DIR/chatrl_checkpoints/d12_cn"
if [ ! -d "$RL_DIR" ] && [ -d "/root/.cache/nanochat/chatrl_checkpoints/d12_cn" ]; then
    mkdir -p "$RL_DIR"
    cp -r /root/.cache/nanochat/chatrl_checkpoints/d12_cn/* "$RL_DIR/"
fi
log "Checking RL checkpoints in $RL_DIR..."
ls -lh "$RL_DIR" | tee -a "$LOG" || true

# 3. Run 1,000 Questions Comparative Inference
log "Step 3: Running 1,000-question batch inference for SFT vs RL..."
$VENV_PY /content/nanochat/dev/eval_1000_inference.py \
    --mode=both \
    --model-tag=d12_cn \
    --sft-out=/content/nanochat/tasks/eval_1000_sft_out.json \
    --rl-out=/content/nanochat/tasks/eval_1000_rl_out.json \
    --html-out=/content/nanochat/eval_comparison_sft_vs_rl.html \
    --max-tokens=280 \
    2>&1 | tee -a "$LOG"

log "✅ 1,000-question comparative inference & HTML report generated."

# 4. Package Checkpoints & Reports into Tarball
log "Step 4: Packaging all checkpoints, logs, and reports..."
ARCHIVE="/content/nanochat_release_bundle.tar.gz"
tar -czf "$ARCHIVE" \
    -C "$NANOCHAT_BASE_DIR" base_checkpoints/d12_cn \
    -C "$NANOCHAT_BASE_DIR" chatsft_checkpoints/d12_cn \
    -C "$NANOCHAT_BASE_DIR" chatrl_checkpoints/d12_cn \
    -C /content/nanochat eval_comparison_sft_vs_rl.html \
    -C /content/nanochat/tasks eval_1000_questions.json eval_1000_sft_out.json eval_1000_rl_out.json \
    || true

log "Archive size: $(ls -lh "$ARCHIVE" | awk '{print $5}')"
log "🎉 ALL OVERNIGHT TASKS COMPLETED SUCCESSFULLY!"
