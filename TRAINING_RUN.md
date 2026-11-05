# nanochat A100 中文训练运行指南

本指南对应当前 `chinese-support` 分支的真实 CLI。目标硬件是 1x/4x A100 80G：使用 BF16，禁用 FP8，预训练强制 `--window-pattern L`，避免 SDPA fallback 下的 sliding-window 低利用率。

## 环境准备

```bash
uv sync --extra gpu --group dev
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
uv run python -c "from datasets import load_dataset; print('datasets OK')"
```

默认产物目录是 `$HOME/.cache/nanochat`，可用 `NANOCHAT_BASE_DIR=/path/to/cache` 覆盖。

## 推荐入口：A100 脚本

```bash
# 单卡 d12 默认首跑
bash runs/chinese_a100.sh

# 4 卡 d16
NPROC=4 DEPTH=16 RUN=d16_cn MODEL_TAG=d16_cn bash runs/chinese_a100.sh
```

脚本默认参数：

- `DEPTH=12`
- `CN_RATIO=0.5`
- `VOCAB_SIZE=49152`
- `WINDOW_PATTERN=L`
- `DEVICE_BATCH_SIZE=16`
- `CN_TALK_EPOCHS=1`
- `GSM8K_CN_EPOCHS=1`
- `CEVAL_EPOCHS=0`

常用覆盖：

```bash
NPROC=4 DEPTH=20 DEVICE_BATCH_SIZE=8 NUM_SHARDS=80 bash runs/chinese_a100.sh
```

## 手动分阶段命令

### 1. Tokenizer

```bash
python -m nanochat.report reset
python -m nanochat.dataset -n 32
python -m scripts.tok_train --vocab-size=49152 --pattern=cn
python -m scripts.tok_eval
```

`--pattern=cn` 会使用中文 aware split pattern，并在 tokenizer 训练数据里按 0.5 比例混入 SkyPile streaming 数据。

### 2. Base Pretraining

单卡：

```bash
OMP_NUM_THREADS=1 python -m scripts.base_train \
    --depth=12 \
    --run=d12_cn \
    --model-tag=d12_cn \
    --cn-ratio=0.5 \
    --window-pattern=L \
    --device-batch-size=16 \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
```

4 卡：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=16 \
    --run=d16_cn \
    --model-tag=d16_cn \
    --cn-ratio=0.5 \
    --window-pattern=L \
    --device-batch-size=16 \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
```

注意：A100 不要传 `--fp8`。如 OOM，先把 `--device-batch-size` 降到 `8`、`4`、`2`。

### 3. SFT

先准备 identity conversations：

```bash
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
```

单卡：

```bash
OMP_NUM_THREADS=1 python -m scripts.chat_sft \
    --run=d12_cn \
    --model-tag=d12_cn \
    --device-batch-size=16 \
    --cn-talk-epochs=1 \
    --gsm8k-cn-epochs=1 \
    --ceval-epochs=0
```

4 卡：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- \
    --run=d16_cn \
    --model-tag=d16_cn \
    --device-batch-size=16 \
    --cn-talk-epochs=1 \
    --gsm8k-cn-epochs=1 \
    --ceval-epochs=0
```

`CEval` 默认只评测，不参与 SFT。只有明确做考试专项微调时才设置 `--ceval-epochs > 0`。

### 4. Evaluation

```bash
python -m scripts.chat_eval -i sft -g d12_cn
python -m scripts.eval_cn -i sft -g d12_cn -a 'CEval|GSM8K-CN' --ceval-split val
python -m nanochat.report generate
```

4 卡评测：

```bash
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i sft -g d16_cn
torchrun --standalone --nproc_per_node=4 -m scripts.eval_cn -- -i sft -g d16_cn -a 'CEval|GSM8K-CN' --ceval-split val
```

## 监控指标

- `train/loss`：训练 loss，应整体下降。
- `val/bpb`：英文 parquet validation bits per byte，用于和原版保持可比。
- `CEval`：中文多选准确率。
- `GSM8K-CN`：中文数学准确率。
- `Chinese ChatCORE`：中文 centered metric。
- `train/mfu`：A100 上低于 H100 是预期；如果使用默认 `SSSL` 会更差，因此脚本固定 `L`。

## 建议规模

- 首次验证：`DEPTH=12`。
- 4xA100 稳定后：`DEPTH=16` 或 `DEPTH=20`。
- `DEPTH=24` 在 4xA100 上预计是多天级别，不作为默认目标。

## 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|----------|
| `datasets` 导入失败 | 依赖未安装 | `uv sync --extra gpu --group dev` |
| SkyPile streaming 慢 | 网络/HF 缓存慢 | 先跑小 `DEPTH` smoke，必要时配置 HF 镜像或本地缓存 |
| OOM | micro batch 过大 | 降低 `DEVICE_BATCH_SIZE` 或 `--device-batch-size` |
| A100 利用率低 | SDPA fallback + sliding window | 确认 `--window-pattern=L` |
| C-Eval 低 | 中文训练不足 | 增加 base tokens 或中文 SFT epochs，先不要打开 `--ceval-epochs` 过拟合验证集 |
