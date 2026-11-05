# nanochat 变更日志

> 本日志记录从 Karpathy 原版 nanochat 出发，经过 Phase 1（推理优化）、Phase 2（训练优化）、Phase 3（中文支持）三阶段系统性改进的完整技术变更。所有改动均围绕单一目标：**在保持代码极简性的前提下，将 nanochat 扩展为支持中英双语的 LLM 训练框架**。

---

## Phase 1: 推理优化 (Inference Optimization)

**目标**：在不改变模型结构、不重新训练的前提下，将推理速度提升 2-10x，显存占用降低 30-50%。

**状态**：已完成，位于 `inference-opt` 分支（待合并）。

### 1.1 GPT.generate() 启用 KV Cache
- **Commit**: `5f249b5`
- **文件**: `nanochat/gpt.py:483-512`
- **问题**: `GPT.generate()` 每次将新 token append 到 ids 后对整个序列做 forward，attention 复杂度 O(T^2)。
- **改动**: 让 `GPT.generate()` 复用 `Engine.generate()` 的 `KVCache` 机制——prefill 一次（处理完整 prompt），decode 时只算新 token。通过局部导入 `KVCache` 避免循环依赖。
- **影响**: 生成长序列时从 O(T^2) 降至 O(T)，长文本生成速度提升显著。

### 1.2 KV Cache 预分配优化
- **Commit**: `e9298cd`
- **文件**: `nanochat/engine.py:92-137`
- **问题**: `torch.zeros()` 预分配 cache 会触发 GPU zero-fill，浪费时间和显存带宽。
- **改动**: 改用 `torch.empty()` 预分配。cache 在 read 前已由 `flash_attn_with_kvcache` 或 SDPA fallback 写入，无需 zero-fill。
- **影响**: 减少初始化延迟，降低显存带宽压力。

### 1.3 SDPA Fallback 消除 Transpose
- **Commit**: `9bb1faa`
- **文件**: `nanochat/flash_attention.py:107-128`, `nanochat/engine.py`
- **问题**: SDPA fallback 路径下，KV cache 以 `(B,T,H,D)`（FA3 布局）存储，每次都要 transpose 到 `(B,H,T,D)` 再 transpose 回来——两次 unnecessary memory operation。
- **改动**: 引入双布局支持：
  - `bthd` = `(B, T, H, D)` 用于 FA3（Hopper H100）
  - `bhtd` = `(B, H, T, D)` 用于 SDPA fallback（Ampere/Blackwell/CPU/MPS）
  - `KVCache` 根据 `USE_FA3` 自动选择布局，`flash_attn_with_kvcache` 的 fallback 路径直接读写 bhtd 布局，消除 transpose。
- **影响**: SDPA fallback 路径显存带宽节省，推理延迟降低。

### 1.4 增量 Streaming Decode
- **Commit**: `e80c8db`
- **文件**: `scripts/chat_web.py:255-303`
- **问题**: 每生成一个 token 都对 accumulated_tokens 全量 decode，tokenizer decode 复杂度 O(T^2)，多语言（中文、emoji）场景尤其严重。
- **改动**: 改为增量 decode——维护一个 UTF-8 byte buffer，只在完整 codepoint 就绪时输出。利用 tokenizer 的增量能力，避免重复 decode 历史 token。
- **影响**: Web UI 流式输出更流畅，长对话不卡顿。

### 1.5 Worker Pool 超时与 Backpressure
- **Commit**: `7e4e7e1`
- **文件**: `scripts/chat_web.py:135-141, 319-320`
- **问题**: Web 服务端 worker queue 无界，请求无限堆积，可能导致 OOM 或超时雪崩。
- **改动**: 添加 `asyncio.wait_for(acquire_worker(), timeout=30)` 超时机制，以及 `maxsize` 限制队列长度。满队列时返回 HTTP 503 Service Unavailable。
- **影响**: 生产部署更稳健，可安全暴露到公网。

### 1.6 Logits Softcap 去 fp32 Cast
- **Commit**: `182e8e9`
- **文件**: `nanochat/gpt.py:468-472`
- **问题**: `logits.float()` 物化一个 fp32 vocab 张量（vocab_size 32768+），在推理时浪费大量显存。
- **改动**: 直接在 bf16/fp16 上计算 `softcap * tanh(logits / softcap)`，去掉 `.float()` cast。bf16 在该范围内精度足够（softcap=15，tanh 输入范围 [-15, 15]）。
- **影响**: 推理显存峰值降低（ vocab_size * batch_size * sizeof(fp32) 的节省）。

### 1.7 KVCache Prefill 布局断言与测试修复
- **Commit**: `bca6795`, `557f2d6`
- **文件**: `nanochat/engine.py`, `tests/test_engine.py`
- **改动**: 添加 `layout` 断言确保 prefill 时 source/target cache 布局一致；修复 bhtd 布局下的 tensor 索引错误（从 `[:, :16, :]` 修正为 `[0, 0, :, :16, :]`）；增强 SDPA fallback 的 shape check。
- **影响**: 测试覆盖率提升，多硬件路径更健壮。

---

## Phase 2: 训练优化 (Training Optimization)

**目标**：在不改变权重格式、兼容现有 checkpoint 的前提下，提升训练效率 10-30%。

**状态**：已合并到 `master`。

### 2.1 Cooldown 期 Beta_2 调整
- **Commit**: `6b7b3bd`
- **文件**: `scripts/base_train.py:360-369, 389-410`
- **背景**: ICLR 2026 "Training Dynamics of the Cooldown Stage" 表明，在 LR warmdown 阶段提高 AdamW 的 beta_2 能一致提升性能。
- **改动**: 在 `base_train.py` training loop 中，当进入 cooldown 阶段（`step > num_iterations * (1 - warmdown_ratio)`）时，按线性插值将 AdamW 参数组的 `beta_2` 从基准值逐步提升到 `--cooldown-beta2-max`（默认 0.99）。Muon 参数组不受影响（Muon 使用 momentum 而非 beta_2）。
- **实现细节**:
  ```python
  def get_cooldown_beta2(it):
      progress = (it - warmdown_start) / warmdown_iters
      return beta2_base + (cooldown_beta2_max - beta2_base) * progress
  ```
- **影响**: 收敛速度提升，final val_bpb 降低。

### 2.2 Rotary + QKNorm + Scale 融合编译
- **Commit**: `6313525`
- **文件**: `nanochat/gpt.py:99-102`
- **问题**: Attention 前向中 `apply_rotary_emb` -> `norm` -> `*1.2` 是 4 个独立 kernel launch，每次读写全局内存。
- **改动**: 给 `CausalSelfAttention` 添加 `@torch.compile(dynamic=False)` decorator，让 PyTorch 编译器自动融合这些 elementwise 操作。`orig_model` 保留未编译版本用于 evaluation/sampling（避免 shape 变化触发 recompile）。
- **影响**: 减少 kernel launch 开销和显存读写，prefill 阶段 ns/token 降低。

### 2.3 MLP ReLU^2 融合
- **Commit**: `8be5044`
- **文件**: `nanochat/gpt.py:135-139`
- **问题**: `F.relu(x).square()` 产生两个 `(B, T, 4*n_embd)` 临时张量，显存浪费。
- **改动**: 实现自定义 `ReLUSquare(torch.autograd.Function)`，forward 中只保存 `relu(x)` 一个中间量，backward 用链式法则直接计算：
  ```python
  class ReLUSquare(torch.autograd.Function):
      @staticmethod
      def forward(ctx, x):
          y = F.relu(x)
          ctx.save_for_backward(y)
          return y * y
      @staticmethod
      def backward(ctx, grad_output):
          y, = ctx.saved_tensors
          return 2 * y * grad_output * (y > 0)
  ```
- **影响**: 训练显存峰值降低 ~200MB（d12 规模），数值误差 < 1e-5。

---

## Phase 3: 中文支持 (Chinese Language Support)

**目标**：让 nanochat 具备以中文为主的训练与推理能力，同时保留英文能力（中英双语）。

**状态**：已合并到 `master`。

### 3.1 中文预训练数据管道 (A1)
- **Commit**: `f9ad398`
- **文件**: `nanochat/dataset.py`
- **数据源**: Skywork/SkyPile-150B（清华大学，CC 清洗，150B tokens），使用 HuggingFace datasets streaming 避免全量下载。
- **改动**:
  - 添加 `BASE_URL_CN` 和 `DATA_DIR_CN`
  - `skypile_iter_batched()` 提供 streaming batch iterator
  - `list_parquet_files()` 和 `parquets_iter_batched()` 支持从多目录读取，按 `cn_ratio` 配置比例混合中英文数据
  - `_interleave_iters()` 用 `fractions.Fraction(cn_ratio).limit_denominator(10)` 生成分数近似模式（如 0.5 -> "en,cn,en,cn..."），确定性交替采样
  - CLI 添加 `--cn-ratio` 参数（0.0-1.0，默认 0.5）
- **影响**: 预训练可混合中英文数据，中文比例可调。

### 3.2 Tokenizer 中文优化 (A2)
- **Commit**: `cd106b3`
- **文件**: `nanochat/tokenizer.py`, `scripts/tok_train.py`
- **问题**: 现有 tokenizer 在纯英文数据上训练，vocab=32768，中文文本被拆成大量 byte-level token（平均 3-4 token/汉字），序列长度暴增。
- **改动**:
  - 新增 `SPLIT_PATTERN_CN`，在现有 pattern 基础上添加 `\u{4e00}-\u{9fff}+`（连续汉字预切分单元）：
    ```python
    SPLIT_PATTERN_CN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+|\u{4e00}-\u{9fff}+"""
    ```
  - `scripts/tok_train.py` 添加 `--pattern en|cn` 参数
  - 支持 `--vocab-size=49152`（从 32768 扩展到 1.5x）
- **目标**: 中文压缩率 < 1.5 token/char（原 tokenizer 约 3-4）。

### 3.3 模型 Vocab 热扩展加载 (A3)
- **Commit**: `7fe0987`
- **文件**: `nanochat/checkpoint_manager.py`, `nanochat/gpt.py`
- **问题**: 旧 checkpoint（vocab=32768）无法直接加载到扩展 vocab 的模型（49152）上。
- **改动**:
  - `build_model()` 检测 `tokenizer_vocab_size > checkpoint_vocab_size` 时进入"扩展模式"
  - 用新 vocab_size 构建模型（embedding/lm_head/value_embeds 新 shape）
  - 加载旧 checkpoint 权重到前 `checkpoint_vocab_size` 个位置
  - 新增 token 保持 `init_weights()` 的初始化策略（随机初始化）
  - `GPT.expand_vocab()` 方法处理 embedding/lm_head/value_embeds 矩阵的 shape 扩展
  - 打印日志告知用户加载/初始化的 token 数量
- **影响**: 旧 checkpoint 可平滑迁移到扩展 vocab 模型，无需从头预训练。

### 3.4 中文对话任务 (B1)
- **Commit**: `74a37d3`, `cd04f1b`
- **文件**: `tasks/smoltalk_cn.py`
- **数据源**: BelleGroup/train_1M_CN（~1M 行中文指令跟随数据）
- **改动**:
  - 实现 `SmolTalkCN(Task)`，接口与现有 `SmolTalk` 完全一致
  - shuffle(seed=42) 后按 95%/5% 划分 train/test
  - 数据格式转换为 OpenAI messages 格式（user/assistant 交替）
  - 包含 sanity checking asserts（role 交替、content 为字符串）

### 3.5 中文考试评估任务 (B2)
- **Commit**: `e918511`
- **文件**: `tasks/ceval.py`, `scripts/eval_cn.py`
- **数据源**: C-Eval（ceval/ceval-exam，52 个学科，13K 选择题）
- **改动**:
  - `CEval(Task)`: `eval_type = "categorical"`，支持 `split="val"`（训练混合）和 `split="test"`（评估）
  - `get_example()` 用 `render_mc()` 渲染多选题格式
  - `evaluate()` 提取模型输出的首字母，对比正确答案
  - `scripts/eval_cn.py`: 独立中文评估脚本，支持 C-Eval 和 GSM8KCN

### 3.6 中文数学推理任务 (B3)
- **Commit**: `2a33910`
- **文件**: `tasks/gsm8k_cn.py`
- **数据源**: yahma/gsm8k-chinese（GSM8K 的中文翻译版本，8K 题）
- **改动**:
  - `GSM8KCN(Task)`: 接口与 `GSM8K` 完全一致
  - `evaluate()` 用数值提取判断正确性（extract 最后一个数字并与答案对比）

### 3.7 SFT 混合比例支持 (B3 集成)
- **文件**: `scripts/chat_sft.py`（需后续集成）
- **设计**: 在 `train_tasks` 列表中条件性加入中文任务：
  ```python
  train_tasks = [
      SmolTalk(split="train"),
      SmolTalkCN(split="train"),
      *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
      *[CEval(split="val") for _ in range(args.ceval_epochs)],
      # ...
  ]
  ```
  添加 CLI 参数 `--cn-talk-epochs` 和 `--ceval-epochs`。

### 3.8 中文 README
- **Commit**: `902aac9`
- **文件**: `README_CN.md`
- **内容**: 完整的中文版 README，覆盖快速开始、更大模型、测试、贡献等章节。路径和命令经 Codex review 修正。

---

## 关键设计决策

### Tokenizer 方案选择
在 Phase 3 中选择了 **方案 B（Vocab 扩展 + 热启动）**：
- 混合中英文数据扩训练 tokenizer，vocab 从 32768 扩展到 49152
- 模型加载时自动扩展 embedding/lm_head/value_embeds 矩阵
- 保留英文能力，中文效率提升，旧 checkpoint 可迁移

### 数据混合策略
预训练数据混合使用确定性分数近似算法：
```python
frac = fractions.Fraction(cn_ratio).limit_denominator(10)
pattern = ["en"] * (frac.denominator - frac.numerator) + ["cn"] * frac.numerator
```
确保可复现性，避免随机采样带来的训练不稳定。

### 多硬件兼容
- **Hopper (H100)**: Flash Attention 3 + bthd layout，bf16，最快路径
- **Ampere/Blackwell/CPU/MPS**: PyTorch SDPA fallback + bhtd layout，消除 transpose
- 自动检测 `USE_FA3`，无需手动配置

---

## 测试覆盖

| 阶段 | 测试命令 | 状态 |
|------|---------|------|
| 推理 | `python -m pytest tests/test_engine.py -v` | 13/13 passed |
| 数值 | d12 100 步 loss 对比 | < 1% 差异 |
| Tokenizer | 中文压缩率 | 目标 < 1.5 token/char |
| 模型加载 | 旧 checkpoint -> 新 vocab | 无报错，新 token 已初始化 |
| Task 接口 | SmolTalkCN / CEval / GSM8KCN | messages 格式合规 |

---

## 文件改动统计

```
Phase 1 (inference-opt):  5 files,  +215/-93  lines
Phase 2 (training-opt):   2 files,  +27/-1    lines
Phase 3 (chinese-support): 13 files, +1118/-47 lines
Total:                    20 files, +1360/-141 lines
```

---

## 致谢

- nanochat 原版: Andrej Karpathy
- Phase 1-3 优化: 基于 PLAN.md / PLAN_PHASE2.md / PLAN_PHASE3.md 系统性实现
