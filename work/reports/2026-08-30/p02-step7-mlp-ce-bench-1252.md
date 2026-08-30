# P0.2 step 7: MLP+CE 训练时延基准（暴露 multi-input fwd_input_map bug）

**日期**: 2026-08-30 12:52
**作者**: 苏璃珞
**关联**: 训练关键路径性能评估 · ASPLOS 2027

## 目标

构造端到端 MLP+CE 训练 step，量化 C3 ON vs Eager wall-clock 加速比。
这是 P0.2 算子完整性闭环后的**性能冲刺前置基准**。

## 测试配置

2 层 MLP：
- `IN=784, H=128, NC=10`（MNIST-like 输入维度）
- `B=64` batch
- forward: x@W1+b1 → ReLU → @W2+b2 → cross_entropy(target)
- backward: AutoGrad::backward
- optimizer: 朴素 SGD（每参数 `param -= lr * grad`）

`bench_mlp_ce_train [B IN H NC STEPS WARMUP]`，50 steps median 取中位数。

## 结果

| 模式 | median | p10 | p90 | 备注 |
|------|--------|-----|-----|------|
| **Eager** | **305.46 us** | 294.25 | 332.38 | SIMD AMX kernel |
| **C3 ON** | **5907.67 us** | 5725.33 | 7213.25 | 慢 19.3× ❌ |

loss 数值都对（2.37 → 2.36，Sanity check 过）—— 性能崩但**数值正确**。

## 根因分析

每 step 触发 **80 个 `unordered_map::at: key not found` compile errors**：
- cache_miss=122, cache_hit=58（MIMO 路径全 miss）
- mimo_compiles=30, mimo_hits=0（MIMO 完全没命中）
- fusion_compiles=0（fusion 没触发）

所有 C3 compile 都失败 → fallback 到 Handwritten backend → 但 C3 compile overhead 仍计入 wall-clock。

## 起源：原 supportsNodeType 注释里的预警

> 多输入节点（Add/Sub/Mul/Div/MatMul/CrossEntropy/Softmax 等）的 per-input 单节点 kernel
> 目前存在 2 个问题：① 图构造 / 输入映射 bug（unordered_map::at key not found）
>                   ② 数值正确性 bug（Mul 返回 [a,a] 而不是正确的 [b,a]）

**问题 ② 数值正确性**：P0.2 step 5/6 + P0.2.1 broadcast 修复已解决
**问题 ① 图构造 bug**：**未解决**——是性能回归的根因

## 已知 facts（已通过 Test 11/12 验证）

- Softmax/CE backward **数值 bit-identical**（max_diff=0）
- `[M, 1] → [M, N]` broadcast 正确传播
- Chain detection 放宽后，4 op 复合图能正确构造

但 `arg_ptrs.at(node_id)` 在 compile 时仍会缺 key → MLIR kernel 生成失败 → fallback。

## 改动

- 新增 `src/tests/standalone/bench_mlp_ce_train.cpp`（+194 行 / CMakeLists.txt）
- 实测后发现 C3 慢 19.3× 暴露 multi-input fwd_input_map bug
- 跳过了 `c3::shutdownAll()`（mutex lock failed 崩溃，与 benchmark 无关）

## Commit

`main @ 1436e4c`

## 下一步（multi-input fwd_input_map 修复优先级提到 P0.3）

P0.2 算子完整性 ✅ → 性能冲刺前**必须先修这个 bug**，否则 C3 永远慢于 eager。

1. **P0.2.2 multi-input fwd_input_map bug 修复**
   - 定位：`MLIRKernelGen` 或 `C3KernelRegistry::tryExecuteBackward` 里的 arg_ptrs.at 缺 key
   - 复现：bench_mlp_ce_train 跑 1 step 看堆栈
   - 修：传 fwd_input_map 索引（graph 输入 → forward_inputs 索引），不是直接 node_id
2. **修复后回归**：Test 11 (Softmax) + Test 12 (CE) + bench_mlp_ce_train
3. **预期**：C3 至少不慢于 eager，理想 ≥1.0× 加速
4. **P0.2 step 8**：DCU 端到端训练 benchmark（机时预算充足，可上节点）

## 备注：为什么 P0.2 step 7"先做 benchmark"是对的

之前在 C3 路径上做算子完整性修复（P0.2 + P0.2.1），只能保证**正确性**。
性能 bug 藏在路径里，但测试通过 → 误以为"好了"。

把 benchmark 跑出来，**19.3× 慢**的真相就摆在桌面。下一步有明确目标。
这是 P0.2.1 那种"先验证再继续"原则的具体应用：算子对了 + 跑通 benchmark 才能确认端到端 OK。
