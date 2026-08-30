# P0.2 step 6: CrossEntropy 完整接入（forward + backward）

**日期**: 2026-08-30 12:06
**作者**: 苏璃珞
**关联**: ASPLOS 2027 · C3 完善优先 · 训练关键路径

## 目标

把 CrossEntropy（CE）loss 接入 C3 端到端路径——前向 + 反向。这是训练的关键节点（MLP 末端 + Softmax 的合体）。

## 改动

### 1. Forward: 自定义 c3 op + fused lowering

**`c3.include/C3/C3Ops.td`**
- 新增 `C3_CrossEntropyOp`（5 个 operand：logits, target, out, M, N）
- 输出：单元素标量（mean loss over batch）
- target 语义：one-hot / soft probability [M, N]（与现有 `CE_SIMD_kernel` line 107-108 `data_b[i*nc + j] * data_a[...]` 一致）

**`c3/src/C3/C3Dialect.cpp`**
- 手写 `CrossEntropyOp::build`（与 `SumReduceOp` 同模式，IntegerAttr 包装）

**`c3/src/C3/C3DialectLowering.cpp`** —— `CrossEntropyOpLowering`（149 行新增）
- 3 级 nested loop（`scf::ForOp` + carry）
- **数值稳定**：row max-subtraction（避免 exp 溢出）
- 算法：
  ```
  外层 for i = 0..M, carry 累加 loss:
    max_i = max_j logits[i, j]                    (中层 j-loop, carry=max)
    sum_exp_i = sum_j exp(logits[i, j] - max_i)   (中层 j-loop, carry=sum)
    inv_sum = 1 / sum_exp
    loss_i = -sum_j target[i, j] * log(exp(logits[i, j] - max_i) * inv_sum + eps)
  out[0] = total_loss / M
  ```
- 抹除原 `c3.cross_entropy` op（已 lower 到 scf+LLVM）

**`c3/src/C3/MLIRKernelGen.cpp`** —— 多节点 fused path 加 `CrossEntropyNode` dispatch
- 取 `in_ptrs[0]`（logits）和 `in_ptrs[1]`（target）
- 调 `c3::CrossEntropyOp::create(in_ptrs[0], in_ptrs[1], out_buf, M_attr, N_attr)`
- 单 op 路径暂未加（函数签名 `(a, b, out, n, M, K, N)` 只有 2 个 input ptr，CE 要 3 个）

### 2. Backward: 4 op 复合 graph

**`c3/include/C3/Graph.h`** —— `CrossEntropyNode` 加入 `NodeVariant`
- 多输入结构：`logits_desc + target_desc`
- forward: out[0] = -1/M * sum_i sum_c target_ic * log(softmax(logits)_ic)
- backward: `grad_logits = softmax(logits) - target`
- target 不需要 grad（input_index=1 返回 nullopt）

**`c3/src/C3/C3BackwardCapture.cpp`** —— `buildCrossEntropyBackwardGraph`
- 4 op 分解：Exp → SumReduce[keepdim, axis=1] → Div（重算 softmax） → Sub
- 图输入 [logits, target]，forward_inputs_indices = {0, 1}
- 下游 grad 是常数 1/M（mean reduction），公式不依赖

**`c3/include/C3/C3BackwardCapture.h`** —— `buildCrossEntropyBackwardGraph` 声明 + 文档

### 3. 分发 + 白名单

- `buildBackwardGraphForTypeAndIndex`：加 `CrossEntropyNode` 分支
  - input_index=0 → `buildCrossEntropyBackwardGraph(grad, logits, target)`
  - input_index=1 → `nullopt`（target 无需梯度）
- `supportsNodeType`：加 `CrossEntropyNode`

## 验证

```
cd build && cmake --build . --target test_c3_backward -j8
[100%] Built target test_c3_backward ✅

./test_c3_backward
✅ PASS: C3 backward 结果正确 (overall_max_diff=7.45058e-08)
```

零回归（Test 1~10 通过，max_diff 与基线 7.45e-08 一致）。

## ⚠️ 已知限制（端到端测试阻塞）

1. **Tensor::cross_entropy 暂未走 C3 路径**
   - 当前 `CtorchScheduler` 的 `op::CE` dispatch 只挂 `CrossEntropy_BASIC_kernel` / `CrossEntropy_SIMD_kernel`
   - 缺一个 C3 路径判定：形状/平台符合 → 调 C3 engine；否则 → eager
   - 这是独立小任务（dispatch 路由，不涉及 C3 dialect 改动）
2. **backward 跑通阻塞于 P0.2.1 broadcast shape-based 修复**
   - 同 Softmax backward 限制——`[M, 1] → [M, N]` 的 numel-based 广播在 M=4 N=8 等尺寸下返回错位
   - CrossEntropy backward 用到 1 次 `[M, 1]` 广播（Div 节点），需要 P0.2.1 修复
3. **axis 写死 1**：行 CE only
4. **backward 不数值稳定**：朴素 exp 没用 max-subtraction
5. **end-to-end CE test 暂未加**：`test_c3_backward.cpp` 仍只有 Test 1~10（无 CE）；等 broadcast 修了再加

## 改动量

| 文件 | 行数 |
|------|------|
| `c3/include/C3/C3Ops.td` | +24 行 |
| `c3/include/C3/Graph.h` | +12 行 |
| `c3/include/C3/C3BackwardCapture.h` | +12 行 |
| `c3/src/C3/C3Dialect.cpp` | +9 行 |
| `c3/src/C3/C3DialectLowering.cpp` | +149 行 |
| `c3/src/C3/MLIRKernelGen.cpp` | +16 行 |
| `c3/src/C3/C3BackwardCapture.cpp` | +84 行 |
| **合计** | **+306 行** |

## Commit

`c3 @ 1a4eed7`

## 下一步

1. **P0.2.1 broadcast shape-based 修复**（硬阻塞）—— 立项优先
2. **Tensor::cross_entropy → C3 路由** —— CtorchScheduler 加 C3 判定
3. **end-to-end CE test**（test_c3_backward.cpp 加 Test 11：CE forward 数值 + CE backward grad）
4. **P0.2 step 7+**：MLP 含 CE 端到端训练 benchmark（vs eager vs PyTorch）
5. **ASPLOS 2027 论文准备**：C3 dialect opset 完整性章节可以涵盖 CrossEntropy
