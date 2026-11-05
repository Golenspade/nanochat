#!/bin/bash

# A100-oriented Chinese training pipeline.
# Example:
#   NPROC=4 DEPTH=16 RUN=d16_cn MODEL_TAG=d16_cn bash runs/chinese_a100.sh

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

NPROC="${NPROC:-1}"
DEPTH="${DEPTH:-12}"
RUN="${RUN:-d${DEPTH}_cn_a100}"
MODEL_TAG="${MODEL_TAG:-d${DEPTH}_cn}"
CN_RATIO="${CN_RATIO:-0.5}"
VOCAB_SIZE="${VOCAB_SIZE:-49152}"
WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
NUM_SHARDS="${NUM_SHARDS:-32}"
CN_TALK_EPOCHS="${CN_TALK_EPOCHS:-1}"
GSM8K_CN_EPOCHS="${GSM8K_CN_EPOCHS:-1}"
CEVAL_EPOCHS="${CEVAL_EPOCHS:-0}"
CEVAL_SPLIT="${CEVAL_SPLIT:-val}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-999999}"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

run_module() {
    module="$1"
    shift
    if [ "$NPROC" = "1" ]; then
        python -m "$module" "$@"
    else
        torchrun --standalone --nproc_per_node="$NPROC" -m "$module" -- "$@"
    fi
}

python -m nanochat.report reset

# English shards are local parquet files; SkyPile Chinese data streams on demand.
python -m nanochat.dataset -n "$NUM_SHARDS"
python -m scripts.tok_train --vocab-size="$VOCAB_SIZE" --pattern=cn
python -m scripts.tok_eval

run_module scripts.base_train \
    --depth="$DEPTH" \
    --run="$RUN" \
    --model-tag="$MODEL_TAG" \
    --cn-ratio="$CN_RATIO" \
    --window-pattern="$WINDOW_PATTERN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --core-metric-every="$CORE_METRIC_EVERY" \
    --sample-every=-1 \
    --save-every=-1

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

run_module scripts.chat_sft \
    --run="$RUN" \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --cn-talk-epochs="$CN_TALK_EPOCHS" \
    --gsm8k-cn-epochs="$GSM8K_CN_EPOCHS" \
    --ceval-epochs="$CEVAL_EPOCHS"

run_module scripts.chat_eval -i sft -g "$MODEL_TAG"
run_module scripts.eval_cn -i sft -g "$MODEL_TAG" -a 'CEval|GSM8K-CN' --ceval-split "$CEVAL_SPLIT"

python -m nanochat.report generate
