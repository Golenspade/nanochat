# nanochat Colab GPU 训练指引 (Chinese + Optimization)

本指引为 **Google Colab**（A100 / L4 / T4）提供了一键式端到端中文大模型训练方案。
包含完整的环境安装、中英文词表扩充、混合预训练、SFT 对话微调、双语评测及自动挂载 Google Drive 保存成果。

---

## 方式 A：单单元格一键运行（推荐）

在 Colab 新建笔记本，将运行时设置为 **GPU**（**代码执行程序 -> 更改运行时类型 -> A100 / L4 / T4**）。
新建一个代码单元格，粘贴以下全部代码并点击运行：

```python
#@title 🚀 nanochat 中文大模型全流程一键训练 (Single Cell Runner)
import os
import sys
import time
import subprocess
from pathlib import Path

# ==================== 1. 配置参数 ====================
REPO_URL = "https://github.com/Golenspade/nanochat.git"
BRANCH = "chinese-support"
REPO_DIR = "/content/nanochat"

# 是否将输出与 Checkpoint 同步到 Google Drive（防止断开后数据丢失）
USE_DRIVE = True
DRIVE_ROOT = "/content/drive/MyDrive/nanochat_run"

# 训练超参数配置
DEPTH = 12                  # 模型层数 (12 约对应 124M-150M 参数，适合快速跑通并验证)
DEVICE_BATCH_SIZE = 16      # 批大小 (T4/16G 建议 8~16, A100/40G 建议 16~32)
CN_RATIO = 0.5              # 中英文混合预训练中，中文数据的比例 (50% SkyPile-150B)
VOCAB_SIZE = 49152          # 扩充后的双语词表大小
NUM_SHARDS = 16             # 预训练下载的英文 Parquet 分片数 (快速验证设为 8~16)
CN_TALK_EPOCHS = 1          # SmolTalkCN 对话微调轮数
GSM8K_CN_EPOCHS = 1         # GSM8KCN 数学微调轮数
CEVAL_EPOCHS = 0            # C-Eval 是否参与微调 (0 为仅用于评测)
RUN_TAG = time.strftime("run_%Y%m%d_%H%M%S")

def sh(cmd, cwd=REPO_DIR):
    print(f"\n[EXEC] {cmd}\n" + "-"*50)
    subprocess.check_call(cmd, shell=True, cwd=cwd)

# ==================== 2. 检查 GPU ====================
sh("nvidia-smi", cwd="/content")
import torch
print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}")

# ==================== 3. 挂载 Google Drive ====================
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        Path(DRIVE_ROOT).mkdir(parents=True, exist_ok=True)
        print(f"✓ Google Drive 挂载成功，产出将保存至: {DRIVE_ROOT}")
    except Exception as e:
        print(f"Drive 挂载跳过/失败 ({e})，将保存在本地虚拟机。")
        USE_DRIVE = False

# ==================== 4. 拉取代码 ====================
if not Path(REPO_DIR).exists():
    sh(f"git clone -b {BRANCH} {REPO_URL} {REPO_DIR}", cwd="/content")
else:
    sh(f"git -C {REPO_DIR} checkout {BRANCH} && git -C {REPO_DIR} pull origin {BRANCH}", cwd="/content")

# ==================== 5. 安装依赖 (基于 uv) ====================
sh("curl -LsSf https://astral.sh/uv/install.sh | sh", cwd=REPO_DIR)
os.environ["PATH"] = f"{Path.home()}/.local/bin:" + os.environ["PATH"]
sh("uv venv /content/.venv", cwd=REPO_DIR)
sh("uv sync --extra gpu", cwd=REPO_DIR)

VENV_PY = "/content/.venv/bin/python"

# ==================== 6. 执行全流程训练管线 ====================
print("\n>>> [Step 1/6] 重置评测报告")
sh(f"{VENV_PY} -m nanochat.report reset")

print("\n>>> [Step 2/6] 准备英文数据分片 (中文 SkyPile 流式即时加载)")
sh(f"{VENV_PY} -m nanochat.dataset -n {NUM_SHARDS}")

print("\n>>> [Step 3/6] 训练中英双语扩展 Tokenizer (Vocab 49152)")
sh(f"{VENV_PY} -m scripts.tok_train --vocab-size={VOCAB_SIZE} --pattern=cn")
sh(f"{VENV_PY} -m scripts.tok_eval")

print("\n>>> [Step 4/6] 启动中英文混合预训练 (Base Train)")
sh(f"""{VENV_PY} -m scripts.base_train \
    --depth={DEPTH} \
    --run={RUN_TAG} \
    --model-tag=d{DEPTH}_cn \
    --cn-ratio={CN_RATIO} \
    --device-batch-size={DEVICE_BATCH_SIZE} \
    --window-pattern=L \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1""")

print("\n>>> [Step 5/6] 对话微调 (SFT Fine-Tuning: SmolTalkCN + GSM8KCN)")
sh(f"curl -L -o ~/.cache/nanochat/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
sh(f"""{VENV_PY} -m scripts.chat_sft \
    --run={RUN_TAG} \
    --model-tag=d{DEPTH}_cn \
    --device-batch-size={DEVICE_BATCH_SIZE} \
    --cn-talk-epochs={CN_TALK_EPOCHS} \
    --gsm8k-cn-epochs={GSM8K_CN_EPOCHS} \
    --ceval-epochs={CEVAL_EPOCHS}""")

print("\n>>> [Step 6/6] 双语能力全面评测 (English ChatCORE + Chinese C-Eval/GSM8K)")
sh(f"{VENV_PY} -m scripts.chat_eval -i sft -g d{DEPTH}_cn")
sh(f"{VENV_PY} -m scripts.eval_cn -i sft -g d{DEPTH}_cn -a 'CEval|GSM8K-CN' --ceval-split val")

print("\n>>> 生成最终可视化评测报告")
sh(f"{VENV_PY} -m nanochat.report generate")

# ==================== 7. 同步回 Google Drive ====================
if USE_DRIVE:
    sh(f"cp -r ~/.cache/nanochat {DRIVE_ROOT}/{RUN_TAG}")
    print(f"\n🎉 训练全流程已完成！全部模型权重、分词器和报告已持久化到 Google Drive: {DRIVE_ROOT}/{RUN_TAG}")
else:
    print(f"\n🎉 训练完成！产出保存在虚拟机 ~/.cache/nanochat")
```

---

## 方式 B：分步交互式单元格运行

若希望逐步观察损失曲线、分词压缩率与测试对话生成，可使用项目附带的已排版 Jupyter Notebook：
* 文件路径：[`nanochat/nanochat_colab.ipynb`](file:///Users/fankex/Developer/active/nanochat/nanochat_colab.ipynb)
* 打开方式：访问 [Google Colab](https://colab.research.google.com/) -> 选择 **上传** 选项卡 -> 上传该 `.ipynb` 文件即可直接分步交互执行。
