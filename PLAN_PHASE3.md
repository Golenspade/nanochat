# nanochat 中文支持计划 (Phase 3)

## 目标
让 nanochat 具备以中文为主的训练与推理能力，同时保留英文能力（中英双语）。

## 关键决策：Tokenizer

中文支持的核心瓶颈是 Tokenizer。当前 tokenizer 在纯英文数据（ClimbMix-400B）上训练，vocab=32768，中文文本会被拆成大量 byte-level token（平均 2-4 个 token/汉字），导致：
- 序列长度暴增，有效上下文缩短
- 训练效率低下
- 中文语义压缩差

**方案选择**（推荐 B）：

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| A 保持现有 | 不换 tokenizer，中文用现有 BPE 表示 | 零改动，旧 checkpoint 可用 | 中文效率极差，不适合"以中文为主" |
| B **扩展 vocab（热启动）** | 混合中英文数据扩训练 tokenizer，vocab→49152 或 65536；支持模型 embedding 扩展加载 | 保留英文能力，中文效率好，旧 checkpoint 可迁移 | 需要实现 embedding 热扩展逻辑 |
| C 全新训练 | 完全重新训练 tokenizer + 从头 pretrain | 最干净，中文最优 | 丢弃所有已有 checkpoint，耗时最长 |

**推荐方案 B**：vocab 扩展到 49152（1.5x），用中英文混合数据（ClimbMix + 中文数据）重新训练 tokenizer。模型加载时自动扩展 embedding/lm_head/value_embeds 矩阵（新 token 随机初始化），然后继续训练。

---

## 模块划分（可并行）

```
[模块A: 中文数据 + Tokenizer]     [模块B: 中文 SFT 任务]     [模块C: 中文评估]
         |                               |                         |
         v                               v                         v
    合并到 training-opt              合并到 training-opt       合并到 training-opt
         |                               |                         |
         +-------------------------------+-------------------------+
                                         v
                                统一验收测试 + d12 验证
```

### 模块 A：中文数据管道 + Tokenizer（基础模块）

**子任务 A1: 中文预训练数据源**
- **文件**: `nanochat/dataset.py`
- **目标**: 支持中英文混合预训练数据
- **实现**:
  1. 添加中文数据集下载：
     - SkyPile-150B（清华大学，CC 清洗）：`https://huggingface.co/datasets/Skywork/SkyPile-150B`
     - CCI3-HQ（高质量中文互联网）：`https://huggingface.co/datasets/BAAI/CCI3-HQ`
     - 或 Ultra-FineWeb 中文子集
  2. 在 `dataset.py` 中新增 `BASE_URL_CN` 和 `DATA_DIR_CN`，格式与现有 ClimbMix 一致（parquet shards）
  3. `list_parquet_files()` 和 `parquets_iter_batched()` 支持从多个目录读取，按配置比例混合
  4. 添加 CLI 参数到 `dataset.py` 的 `__main__`：
     ```python
     parser.add_argument("--cn-ratio", type=float, default=0.5, help="ratio of Chinese data in mix (0.0-1.0)")
     ```
  5. `tok_train.py` 的 `text_iterator()` 改为从混合数据源读取

**子任务 A2: Tokenizer 重训练（中文优化）**
- **文件**: `nanochat/tokenizer.py`, `scripts/tok_train.py`
- **目标**: 训练一个中英文双语的 BPE tokenizer，vocab_size 扩展到 49152
- **实现**:
  1. 修改 `SPLIT_PATTERN` 以更好切分中文：
     ```python
     # 在现有 pattern 基础上添加中文汉字连续匹配
     SPLIT_PATTERN_CN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{Han}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
     ```
     关键改动：`\p{Han}+` 将连续汉字作为一个整体预切分单元，避免汉字被拆散。
  2. `scripts/tok_train.py` 支持 `--vocab-size=49152`（已有），默认使用 `SPLIT_PATTERN_CN`
  3. `RustBPETokenizer.train_from_iterator()` 传入新 pattern
  4. 训练完成后验证中文压缩率：
     ```python
     test_cn = "你好世界，这是一个测试。深度学习正在改变人工智能的未来。"
     tokens = tokenizer.encode(test_cn)
     print(f"CN chars: {len(test_cn)}, tokens: {len(tokens)}, ratio: {len(tokens)/len(test_cn):.2f}")
     # 目标 ratio < 1.5（当前英文 tokenizer 约 3-4）
     ```

**子任务 A3: 模型 Vocab 扩展加载（热启动）**
- **文件**: `nanochat/checkpoint_manager.py`, `nanochat/gpt.py`
- **目标**: 旧 checkpoint（vocab=32768）能在扩展后的模型（vocab=49152）上加载，新 token 随机初始化
- **实现**:
  1. `build_model()` 中，当 `tokenizer.get_vocab_size() != model_config_kwargs["vocab_size"]` 时：
     - 不报错，而是进入"扩展模式"
     - 用新的 vocab_size 构建模型（这样 embedding/lm_head/value_embeds 都是新 shape）
     - 加载旧 checkpoint 权重到对应位置（前 32768 个 token 的 embedding）
     - 新增 token（32768~49151）用 `init_weights()` 的初始化策略重新随机初始化
  2. `GPTConfig` 中保存的 `vocab_size` 是实际 vocab size（不是 padded）
  3. 添加 `_patch_vocab_size()` 辅助函数处理 embedding/lm_head/value_embeds 的 shape 不匹配
  4. 打印日志告知用户有多少 token 是从 checkpoint 加载的，多少是随机初始化的

---

### 模块 B：中文 SFT 任务

**子任务 B1: 中文对话数据集 Task**
- **文件**: `tasks/` 目录下新建 `tasks/smoltalk_cn.py`
- **目标**: 添加中文通用对话数据
- **实现**:
  1. 调研可用的中文对话数据集：
     - `BelleGroup/train_1M_CN` 或 `BelleGroup/train_0.5M_CN`
     - `shibing624/alpaca-zh`
     - `fnlp/moss-002-sft-data`（中文）
     - `HuggingFaceH4/ultrachat_200k` 的中文子集
  2. 实现 `SmolTalkCN(Task)` 类，接口与现有 `SmolTalk` 完全一致：
     - `__init__(self, split)`
     - `num_examples()`
     - `get_example(index)` → 返回 `{"messages": [...]}` OpenAI 格式
  3. 数据格式转换：将数据集的格式转换为 nanochat 的 conversation 格式（user/assistant 交替）

**子任务 B2: 中文考试/推理 Task**
- **文件**: `tasks/` 目录下新建 `tasks/ceval.py`（或扩展 mmlu.py 支持中文子集）
- **目标**: 添加 C-Eval 或 CMMLU 作为中文评估/训练任务
- **实现**:
  1. C-Eval: `https://huggingface.co/datasets/ceval/ceval-exam`
     - 包含 52 个学科的选择题
     - 格式与 MMLU 相同（question + 4 choices + answer letter）
  2. 实现 `CEval(Task)` 类：
     - `eval_type = "categorical"`
     - `get_example()` 用 `render_mc()` 渲染多选题格式
     - `evaluate()` 对比模型输出和正确答案
  3. 支持 `split="val"`（用于训练混合）和 `split="test"`（用于评估）

**子任务 B3: SFT 混合比例调整**
- **文件**: `scripts/chat_sft.py`
- **目标**: 在 SFT 中混入中文任务
- **实现**:
  1. 添加 CLI 参数：
     ```python
     parser.add_argument("--cn-talk-epochs", type=int, default=2, help="epochs of Chinese conversation data")
     parser.add_argument("--ceval-epochs", type=int, default=2, help="epochs of C-Eval in mixture")
     ```
  2. 修改 `train_tasks` 列表，条件性加入中文任务：
     ```python
     train_tasks = [
         SmolTalk(split="train"),
         SmolTalkCN(split="train"),  # 中文对话
         # ...
         *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
         *[CEval(split="val") for _ in range(args.ceval_epochs)],  # 中文考试
         # ...
     ]
     ```

---

### 模块 C：中文评估

**子任务 C1: 中文 Benchmark 评估脚本**
- **文件**: `scripts/chat_eval.py`（或新建 `scripts/eval_cn.py`）
- **目标**: 支持 C-Eval、CMMLU 等中文 benchmark
- **实现**:
  1. 在 `tasks/ceval.py` 中实现 `evaluate()` 方法
  2. 修改 `scripts/chat_eval.py`（如果存在）或创建独立评估脚本
  3. 评估方式：
     - categorical: 用 `render_mc()` 渲染选择题，看模型输出是否匹配正确选项字母
     - generative: 看模型生成的中文回答质量（可先用人工抽检）
  4. 输出格式与现有英文评估一致（正确率百分比）

**子任务 C2: 中文推理能力测试**
- **文件**: `tasks/` 目录下新建 `tasks/gsm8k_cn.py`
- **目标**: GSM8K 的中文翻译版本（或类似的中文数学推理数据集）
- **实现**:
  1. 使用 `yahma/gsm8k-chinese` 或自己翻译 GSM8K
  2. 实现 `GSM8KCN(Task)`，接口与 `GSM8K` 一致
  3. `evaluate()` 用 exact match 或数值提取判断正确性

---

## 依赖关系

```
A1 (中文数据管道) ─┬──→ A2 (Tokenizer 重训练)
                   │        │
                   │        └──→ A3 (Vocab 扩展加载)
                   │                 │
B1 (中文对话) ◄────┘                 │
B2 (中文考试) ◄──────────────────────┘
B3 (SFT 混合) ◄── B1 + B2
C1 (中文评估) ◄── B2
C2 (中文数学) ◄── B1
```

**并行策略**：
- **Round 1**（3个 subagent 并行）：
  - Agent A: A1 + A2 + A3（数据+Tokenizer+模型加载）
  - Agent B: B1 + B2（中文对话+考试 Task 类）
  - Agent C: C1 + C2（中文评估+数学 Task 类）
- **Round 2**（1个 subagent）：
  - Agent D: B3（SFT 混合集成）+ 统一验收测试

理由：B1/B2/C1/C2 只依赖 `tasks/common.py` 的 `Task` 基类接口，不依赖 tokenizer 的具体实现。它们可以并行开发，最后在 B3 中集成。

---

## 验收标准

### 1. 数值/功能正确性
```bash
# 所有现有测试必须继续通过
uv run python -m pytest tests/ -v

# Tokenizer 中文压缩率测试
uv run python -c "
from nanochat.tokenizer import get_tokenizer
tok = get_tokenizer()
text = '你好世界，深度学习改变人工智能。'
tokens = tok.encode(text)
print(f'Chars: {len(text)}, Tokens: {len(tokens)}, Ratio: {len(tokens)/len(text):.2f}')
assert len(tokens) / len(text) < 1.8, 'Chinese tokenization too inefficient'
"
```

### 2. 模型加载兼容性
```bash
# 旧 checkpoint（vocab=32768）能在新模型（vocab=49152）上加载
uv run python -c "
from nanochat.checkpoint_manager import build_model
# 用旧 checkpoint 加载，验证不报错且新 token embedding 已初始化
"
```

### 3. Task 接口合规
```bash
# 每个新 Task 都能正确返回 conversation 格式
uv run python -c "
from tasks.smoltalk_cn import SmolTalkCN
from tasks.ceval import CEval
ds = SmolTalkCN(split='train')
print(len(ds))
ex = ds[0]
assert 'messages' in ex
assert ex['messages'][0]['role'] in ['system', 'user']
"
```

### 4. SFT 数据混合
```bash
# chat_sft.py 能正常启动并识别中文任务
uv run python -m scripts.chat_sft --help | grep -E "cn-talk|ceval"
```

---

## 风险矩阵

| 改动 | 风险等级 | 回退方式 | 最坏情况 |
|------|---------|---------|---------|
| A1 中文数据管道 | 🟢 低 | 不下载中文数据 | 无影响 |
| A2 Tokenizer 重训练 | 🟡 中 | 恢复 SPLIT_PATTERN | 中文 tokenization 差 |
| A3 Vocab 扩展加载 | 🟡 中 | 强制 vocab_size 匹配报错 | 旧 checkpoint 加载失败 |
| B1/B2 中文 Task | 🟢 低 | 从 train_tasks 移除 | 无影响 |
| B3 SFT 混合 | 🟢 低 | 调 cn 比例为 0 | 无影响 |
| C1/C2 中文评估 | 🟢 低 | 不运行中文评估 | 无影响 |

---

## 数据集参考

| 数据集 | 类型 | 大小 | HuggingFace ID | 用途 |
|--------|------|------|----------------|------|
| SkyPile-150B | Pretrain | 150B tokens | Skywork/SkyPile-150B | 中文预训练 |
| CCI3-HQ | Pretrain | ~100B | BAAI/CCI3-HQ | 中文预训练 |
| BelleGroup/train_1M_CN | SFT | 1M rows | BelleGroup/train_1M_CN | 中文对话 |
| C-Eval | Eval | 13K | ceval/ceval-exam | 中文考试 |
| CMMLU | Eval | 11K | haonan-li/cmmlu | 中文多学科 |
| GSM8K-Chinese | Eval | 8K | yahma/gsm8k-chinese | 中文数学 |

---

## 备注

- **Tokenizer 兼容性**: rustbpe 的 `pattern` 参数支持标准 regex，`\p{Han}` 需要确认 rustbpe 是否支持 Unicode property。如不支持，用 `\u{4e00}-\u{9fff}` 范围替代。
- **数据版权**: SkyPile-150B 和 CCI3-HQ 均为开源学术数据集，使用前确认最新 license。
- **混合比例**: 建议 pretrain 中英文比例 1:1 或 中文 60%:英文 40%。SFT 中英文任务数量对等。
- **评估基线**: 在 d12 规模上，C-Eval val 准确率基线约为随机猜测（~25% for 4-choice），训练后目标 >40%。
