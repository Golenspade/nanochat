# nanochat 训练优化计划 (Phase 2: Bundle 2)

## 目标
在不改变模型结构、兼容现有 checkpoint 的前提下，将训练效率提升 10-30%，收敛速度提升 5-10%。

## 关键原则
- **不改权重格式**：老 checkpoint 能正常加载
- **不改模型结构**：forward() 输出 bitwise 一致（除 fusion 项允许微小数值差异）
- **可开关**：每项都能通过 flag/ env 回退到旧行为
- **训前验证**：改动后在 d12 上跑 100 步，loss 曲线与 baseline 对比

---

## 改动清单

### 1. Newton-Muon 替换 Polar Express Muon
- **文件**: `nanochat/optim.py:91-148` (`muon_step_fused`)，以及 `_step_muon` (lines 259-283)
- **当前**: Polar Express 正交化（5 次 Newton-Schulz 迭代）
- **目标**: Newton-Muon，添加右预条件 `W <- W - eta * msgn(G (Z Z^T)^-1)`
- **背景**: arXiv:2604.01472，在 Modded-NanoGPT speedrun 上减少 6% 迭代
- **实现**:
  1. 在 `muon_step_fused` 中，计算梯度 `G` 后，额外计算 `(Z Z^T)^-1` 的近似
  2. `Z` 是输入激活（forward 时保存的 input 到该层的激活），需要传递到 optimizer
  3. 更实际的方案：使用梯度自身的二阶矩来近似 `(Z Z^T)`，即 `G @ G.T` 的均值
  4. 或者简化版：在现有 Muon 的 `newton_schulz` 后，右乘一个基于梯度协方差的预条件矩阵
  5. **最简可行方案**：修改 `newton_schulz` 的迭代公式，在每一步同时做左/右正交化
- **验证**:
  - d12 上跑 100 步，对比 loss 曲线
  - 如果 loss 低于 baseline → 成功
  - 如果发散或 loss 更高 → 回退
- **风险**: 高。Newton-Muon 的收敛性在不同数据分布上可能有差异。必须保留旧 Muon 代码作为 fallback。
- **开关**: 通过 `MUON_VARIANT` 环境变量：`polar` (default) | `newton`

### 2. cooldown 期调 beta_2
- **文件**: `scripts/base_train.py:360-369`
- **当前**: 线性 warmdown，beta_2 固定
- **目标**: 在 LR cooldown 阶段（最后 50-65% 训练步数），将 AdamW 的 beta_2 从 0.96 逐步提升到 0.99
- **背景**: ICLR 2026 "Training Dynamics of the Cooldown Stage"，高 beta_2 在 cooldown 期一致提升性能
- **实现**:
  1. 在 `base_train.py` 的 training loop 中，每一步检查当前是否处于 cooldown 阶段
  2. cooldown 判定：`step > num_iterations * (1 - warmdown_ratio)`
  3. 在 cooldown 阶段，按线性插值调整 beta_2：
     ```python
     beta2_cooldown = beta2_base + (0.99 - beta2_base) * cooldown_frac
     for group in optimizer.param_groups:
         if group['kind'] == 'adamw':
             group['betas'] = (group['betas'][0], beta2_cooldown)
     ```
  4. 只对 AdamW 参数组调 beta_2，Muon 参数组不改（Muon 用 momentum 不是 beta_2）
- **验证**:
  - d12 上跑完整训练，对比 final val_bpb
  - 如果 val_bpb 更低 → 成功
- **风险**: 低。纯超参调整，不改代码逻辑，随时可以关掉。
- **开关**: 通过 CLI flag `--cooldown-beta2-max=0.99`（0.99 启用，默认 0.0 表示不调整）

### 3. Muon stack/unstack 优化
- **文件**: `nanochat/optim.py:259-283` (`_step_muon`)
- **当前**:
  ```python
  stacked_grads = torch.stack([p.grad for p in params])
  stacked_params = torch.stack(params)
  # ... fused kernel ...
  torch._foreach_copy_(params, list(stacked_params.unbind(0)))
  ```
  每步创建 2 个临时 stacked 张量，然后逐个参数拷贝回去。
- **目标**: 单 GPU (`MuonAdamW`) 下直接操作原始参数，消除 stack/unstack
- **实现**:
  1. 单 GPU 路径：用 `torch._foreach_*` 系列函数直接在参数列表上操作
  2. `torch._foreach_mul_`, `_foreach_add_` 等支持列表输入，不需要 stack
  3. 正交化步骤用 `torch.linalg.svd` 或 `torch.qr` 逐个参数计算（虽然慢点但省了内存）
  4. **更优方案**：在 `setup_optimizer` 时，把同 shape 的 Muon 参数分配到一个连续内存池中，用 view 访问。这样 stack/unstack 变成无操作的 view 操作。
- **验证**:
  - 内存占用对比：`torch.cuda.max_memory_allocated()` 前后对比
  - 预期省 ~200MB 临时显存（100M 参数 × 2 bytes × 2 for stack）
- **风险**: 中。内存池方案改动较大，需要确保 backward 的梯度也写到正确位置。
- **开关**: 新实现放在单独函数，运行时选择

### 4. Rotary + QKNorm + Scaling 融合
- **文件**: `nanochat/gpt.py:99-102`
- **当前**:
  ```python
  q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
  q, k = norm(q), norm(k)
  q = q * 1.2
  k = k * 1.2
  ```
  4 个独立 kernel launch，每次读写全局内存。
- **目标**: 融合成单个操作（或让 torch.compile 自动融合）
- **实现**:
  1. **方案 A（推荐）**: 给 `CausalSelfAttention` 加 `@torch.compile(dynamic=False)`，让编译器自动融合 elementwise 操作
  2. **方案 B**: 手写一个融合函数 `apply_rotary_qk_norm(q, k, cos, sin, scale)`，在 Python 层用一个函数包装，内部用 `torch.compile` 或 Triton
  3. **方案 C（最彻底）**: 用 Triton kernel 实现 rotate + rms_norm + scale 的融合
- **验证**:
  - 跑 attention benchmark（prefill 阶段），对比 ns/ token
  - 同时验证输出与逐行实现的最大误差 < 1e-4
- **风险**: 低。方案 A 几乎零风险，只是加个 decorator。
- **开关**: `torch.compile` 可开关（nanochat 已经有 `orig_model` 保留未编译版本的模式）

### 5. MLP ReLU² 融合
- **文件**: `nanochat/gpt.py:135-139`
- **当前**:
  ```python
  x = self.c_fc(x)
  x = F.relu(x).square()
  x = self.c_proj(x)
  ```
  `F.relu(x)` 产生一个 `(B, T, 4*n_embd)` 临时张量，`.square()` 又产生一个。
- **目标**: 消除中间分配，或让 torch.compile 融合
- **实现**:
  1. **方案 A**: `torch.compile` 整个 `MLP` 类，让编译器自动融合 `matmul -> relu -> square -> matmul`
  2. **方案 B**: 用 inplace 操作（需谨慎 autograd）:
     ```python
     x = F.relu(x, inplace=True)
     x = x.square_()  # 如果不需要 relu 输出做 backward，可以 inplace
     ```
     但实际上 `square_()` 会修改 tensor 的数据，而 backward 需要 relu 的输入来判断 mask... 所以 inplace 可能不安全。
  3. **方案 C（推荐）**: 自定义 autograd Function:
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
     这样只有一个中间量（y = relu(x)），square 不需要单独存。
- **验证**:
  - 内存对比：训练时 `torch.cuda.max_memory_allocated()`
  - 数值验证：与原始实现的最大误差 < 1e-5
- **风险**: 低。方案 A 和 C 都安全。
- **开关**: 通过替换 `MLP.forward()` 的实现

---

## 依赖关系图

```
[1 Newton-Muon]        独立（替换 optim.py 中的函数）
[2 cooldown beta_2]    独立（base_train.py 超参调整）
[3 Muon stack/unstack] 依赖 [1]（如果同时做，先确认 Newton-Muon 稳定）
[4 Rotary fusion]      与 [5] 可一起做（都是 torch.compile/gpt.py）
[5 MLP ReLU² fusion]   与 [4] 可一起做
```

**建议顺序**: 2 → 4+5 → 1 → 3

理由：
- 2 和 4+5 风险最低，先跑通建立信心
- 1 风险最高，等前面稳定后再做
- 3 依赖 1 的结论

---

## 验收标准

每项改动必须通过三层验证：

### 1. 数值正确性
```bash
# 跑所有现有测试
uv run python -m pytest tests/ -v
```

### 2. 训练等价性（d12 快速验证）
```bash
# baseline（改前）
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 --run="d12_baseline" --model-tag="d12_baseline" \
    --core-metric-every=999999 --sample-every=-1 --save-every=-1 --num-iterations=100

# new（改后）
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 --run="d12_new" --model-tag="d12_new" \
    --core-metric-every=999999 --sample-every=-1 --save-every=-1 --num-iterations=100

# 对比：第 100 步的 train/val loss 差异应 < 1%
```

### 3. 性能/效率验证
- **Newton-Muon**: 对比 1000 步的 wall-clock 时间，预期省 4-6%
- **cooldown beta_2**: 跑完整 d12，对比 final val_bpb
- **stack/unstack**: `torch.cuda.max_memory_allocated()` 对比，预期省 ~200MB
- **Rotary/MLP fusion**: ns/token benchmark，预期 10-25% 加速

---

## 风险矩阵

| 改动 | 风险等级 | 回退方式 | 最坏情况 |
|------|---------|---------|---------|
| 2 cooldown beta_2 | 🟢 低 | 改回固定 beta_2 | loss 略高，不影响 checkpoint |
| 4 Rotary fusion | 🟢 低 | 去掉 torch.compile | 回到旧速度，数值一致 |
| 5 MLP ReLU² fusion | 🟢 低 | 恢复原始 forward | 回到旧速度，数值一致 |
| 3 Muon stack | 🟡 中 | 恢复 stack/unstack | 训练慢/内存多，不影响正确性 |
| 1 Newton-Muon | 🔴 高 | 切回 Polar Express | 发散或不收敛，需重新训 |

---

## 迭代流程

1. Subagent 按顺序实现改动 → 提交到 `training-opt` 分支（从 `inference-opt` 或 master 切出）
2. 每完成一项：
   a. 跑 `pytest tests/ -v`
   b. 跑 d12 100 步 loss 对比
   c. Commit
3. 全部完成后：
   a. 我 review diff（deletion test + depth 评估）
   b. 跑完整 d12 验证
   c. Review 通过 → 合并，进入 Phase 3

---

## 备注

- **Newton-Muon 论文**: arXiv:2604.01472，作者 Zhanhao Du, Weijie Su
- **WSD cooldown 论文**: ICLR 2026 "Training Dynamics of the Cooldown Stage"，作者 Dremov et al.
- 如果 Newton-Muon 实现过于复杂，可以先做简化版：只在 Muon 的 Newton-Schulz 迭代中加入梯度协方差缩放，不追论文的完整公式。
