# nanochat 推理优化计划 (Phase 1: Bundle 1)

## 目标
在不重新训练、不改变模型结构的前提下，将 nanochat 的推理性能提升 2-10x，显存占用降低 30-50%。

## 改动清单

### 1. GPT.generate() 启用 KV Cache
- **文件**: `nanochat/gpt.py:483-512`
- **当前**: `GPT.generate()` 每次把新 token append 到 ids 后对整个序列做 forward，attention 重算 O(T^2)
- **目标**: 复用 `Engine.generate()` 的 KVCache 机制，prefill 一次，decode 时只算新 token
- **实现**: 让 `GPT.generate()` 接受 `KVCache` 参数，走 `forward(idx, kv_cache=...)` 路径
- **风险**: 低。`forward()` 已支持 kv_cache，只是 `generate()` 没传

### 2. KV Cache 预分配优化
- **文件**: `nanochat/engine.py:92-137`
- **当前**: `torch.zeros(...)` 预分配 10x sequence_len，零填充浪费时间
- **目标**: 用 `torch.empty()` + 按需增长，省初始化时间和显存
- **实现**: `__init__` 改 `torch.empty`，`advance()` 时若超界则 `resize_` 或重新分配
- **风险**: 低。确保 cache 在 read 前已被 write 覆盖即可

### 3. SDPA fallback 消除 transpose
- **文件**: `nanochat/flash_attention.py:107-128`
- **当前**: `(B,T,H,D)` -> transpose -> `(B,H,T,D)` -> SDPA -> transpose 回来
- **目标**: 存储 KV cache 直接用 `(B,H,T,D)`，只在 FA3 路径 transpose
- **实现**: `KVCache` 存 `(B,H,T,D)`；`_sdpa_attention` 中 q 单独 transpose，k/v 直接用 cache 布局
- **风险**: 中。需确保 FA3 路径的 cache layout 不受影响

### 4. Streaming decode 改增量
- **文件**: `scripts/chat_web.py:255-303`
- **当前**: 每 token 都对 accumulated_tokens 全量 decode，O(T^2)
- **目标**: 只解码新增的字节，维护一个 byte buffer
- **实现**: 用 tokenizer 的增量 decode（如果有的话），或手动维护 UTF-8 byte buffer，只在完整字符就绪时输出
- **风险**: 低。逻辑等价，只是省重复 decode

### 5. Web 服务加 request batching
- **文件**: `scripts/chat_web.py:305-374`
- **当前**: 每个请求独占一个 worker，无 batching
- **目标**: 多个请求合并成一个 batch 喂给 model
- **实现**: 每个 worker 加 asyncio.Queue，收集 5-10ms 内的请求，pad 后一起 generate
- **风险**: 中。需要处理不同长度对话的 left-padding 和 position 偏移

### 6. Worker Pool 超时/backpressure
- **文件**: `scripts/chat_web.py:135-141, 319-320`
- **当前**: Queue 无界，请求无限堆积
- **目标**: 超时 30s，满队列返回 HTTP 503
- **实现**: `asyncio.wait_for(acquire_worker(), timeout=30)` + `maxsize` 限制
- **风险**: 低。纯服务端逻辑

### 7. logits softcap 去 fp32 cast
- **文件**: `nanochat/gpt.py:468-472`
- **当前**: `logits.float()` -> `tanh` -> 乘 softcap，物化全词表 fp32 张量
- **目标**: bf16 直接算 softcap，或只对 top-k 算
- **实现**: 去掉 `.float()`，在 bf16 上直接 `softcap * torch.tanh(logits / softcap)`
- **风险**: 低。bf16 对 tanh 在这个范围内精度足够

## 验收标准

每完成一项，跑以下测试验证：

```bash
# 1. 推理正确性
python -m pytest tests/test_engine.py -v

# 2. Web 服务启动
python -m scripts.chat_web --num-gpus 1 --source sft &
curl http://localhost:8000/health
# 期望: status=ok, ready=true

# 3. 生成一致性 (同一 seed 同一输出)
python -c "
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
model, tokenizer, _ = load_model('sft', 'cpu', phase='eval')
engine = Engine(model, tokenizer)
r1, _ = engine.generate_batch([261, 72, 101], max_tokens=5, seed=42)
r2, _ = engine.generate_batch([261, 72, 101], max_tokens=5, seed=42)
assert r1 == r2, 'Generation not deterministic!'
print('OK: deterministic')
"
```

## 迭代流程

1. Subagent 实现改动 -> 提交到 `inference-opt` 分支
2. Review: 人工检查 diff + 跑验收测试
3. Review 通过 -> 合并到 master (或保留在分支)
4. 进入 Phase 2 (训练优化 Bundle)

## 依赖关系图

```
[2 KV Cache 预分配] 依赖 [1 GPT.generate KVCache]
[3 SDPA transpose] 依赖 [2 KV Cache 预分配] (改 cache layout)
[5 Web batching] 依赖 [1 GPT.generate KVCache] (需要 batch 生成)
[4 Streaming decode] 独立
[6 Worker Pool 超时] 独立
[7 logits softcap] 独立
```

建议实现顺序: 7 -> 6 -> 1 -> 2 -> 3 -> 4 -> 5
