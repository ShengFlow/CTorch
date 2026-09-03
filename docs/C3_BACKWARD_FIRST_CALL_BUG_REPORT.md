# C3 Backward 进程内首次调用梯度异常 · 缺陷报告

> 编号：C3-BUG-20260903-01
> 日期：2026-09-03
> 报告人：苏璃珞（SuLiluo）
> 状态：**已修复（2026-09-03）**
> 状态变更：FIXED — miss 路径编译同步后重试 execute
> 关联：STATUS_CONTEXT §4.45

## 1. 摘要

C3 的 MIMO/融合 backward 在**进程内对某计算图模式的首次调用（iter0）**存在**梯度未完全同步**的问题：首次调用读到的 `grad` 是偏小的中间态值，与后续稳定结果差异达 `0.185736`（差异元素占 99%），而后续（第 2 次起）结果 bitwise 自洽。该问题可能影响真实用户首次调用 `AutoGrad::backward` 后立即读梯度的正确性（如训练第一步参数更新）。

## 2. 复现

```bash
# 需要 C3 enabled 的 build
C3_AOT_CACHE_DIR=/tmp/c3c_clean ./build/bench_c3_backward_perf_clean
# 期望：guard 输出 FAIL (max_abs_diff ≈ 0.185736)
```

负载：`[512×512] ≈0.25M 元素`，图 `x → Tanh → Sigmoid → ReLU → backward`，120 次无预热测量。

## 3. 现象与证据

| 测量 | 结果 |
|---|---|
| guard（iter0 dx vs 紧随重跑的 iter0_ref） | **0.185736 FAIL**（>1e-4），确定性、可复现 |
| 预热（同 cache 再跑）后 | 仍 FAIL，max_abs_diff **完全相同 0.185736**（排除冷启动编译时序） |
| iter0 vs iter1 | 0.185736（首次离群） |
| iter0_ref vs iter1 / g1 vs g2 / g2 vs 末次 | **0.000000**（bitwise 自洽） |
| 差异元素占比 | 259658/262144 ≈ **99%**，样例 v0=0.014 vs v1=0.199（首值明显偏小） |
| iter0 耗时 | ≈1.2ms（C3 级；Eager 对照 ≈15.7ms） |
| C3_BW_SEG2 | miss=209.67ms/**3**（首次 3 节点各 miss 1 次，此后 77+ 次全命中） |

**Eager 对照**（`C3_DISABLE_BACKWARD=1`）：guard = 0，PASS。

## 4. 根因分析

### 4.1 源码定位

`C3BackwardCapture::tryExecuteBackward`（c3/src/C3/C3BackwardCapture.cpp，约 278-321 行）的 miss 路径：

```cpp
compileBackwardAsyncForInput(node, grad, i);          // 触发该输入 backward kernel 的（异步）编译
getInstance().waitForPendingCompiles();               // 等待 in-flight 编译
...
return std::nullopt;                                   // ← miss 本次仍回退 eager（node->backward）
```

- 首次调用（cache 空）时 3 个 backward 节点均 miss → 各触发一次编译（约 70ms）并同步等待；
- miss 后本次调用返回 `nullopt` → `ComputeCore::backward` 走 eager 回退；
- **但 iter0 读到的梯度既非完整 eager、也非 C3 稳态值**（两者稳态差仅 7.45e-8），而是偏小的中间态。

### 4.2 判定

首次（进程首访）时，miss 编译与梯度写入之间存在**未完全同步的边界**：`backward()` 返回时，首次计算路径产生的梯度可能尚未完整落盘到 `grad`，导致调用方（`bench`/真实用户）立即读 `grad` 拿到中间态。

**与 `test_c3_backward`（稳态 PASS，C3 vs Eager 差 7.45e-8）不矛盾**：该测试比较发生在内核就绪、多次迭代之后。

## 5. 影响判定

- **疑似真实缺陷**：真实用户首次调用 `AutoGrad::backward` 后立即读 `grad`（训练首个 step 的 SGD 参数更新）可能拿到不完整/异常梯度。
- 若首次 step 梯度异常但后续正常，可能导致首个训练 step 更新偏误；需实测确认是否影响收敛/数值正确性。
- 关系论文 claim「数值位级一致（max_diff=0）」：若首访确有真实数值异常，需在论文中明确边界或修复后重新验证。

## 6. 修复建议

修复目标：**backward 返回前，梯度必须完整落盘**（无论走 C3 kernel 还是 eager 回退）。

候选修复点（按优先级，需子代理实施 + 单测验证）：

1. **【首选】首次 miss 后真正同步完成并执行**：在 miss 路径（C3BackwardCapture.cpp 278-321）中，`compileBackwardAsyncForInput` + `waitForPendingCompiles` 之后，**重新尝试 execute 该输入**（此时 kernel 已装），而非直接 `return nullopt` 走 eager——确保本次调用即用上编译好的 C3 kernel 且同步执行写梯度。

2. **eager 回退的同步保证**：若保留 miss→eager 回退，需确认 eager 回退（`node->backward`）在 `ComputeCore::backward` 遍历中是同步完整写梯度的（不应受首次异步编译残留影响）。

3. **读梯度前的显式同步**：若 backward 内存在异步边界，需在 `ComputeCore::backward` 返回前（或 `grad` 可读前）等待所有 C3 后台任务完成（复用 `waitForPendingCompiles` / `task_cv_` 机制）。

4. **验证新增单测**：仿 `bench_c3_backward_perf_clean` 的「进程内首访后立即读 grad」场景，断言首次与二次调用梯度一致（max_diff < 1e-4），纳入 `test_c3_backward`。

## 7. 验证方案

- 修复后重跑 `bench_c3_backward_perf_clean`：guard 应 PASS（iter0 vs iter0_ref < 1e-4）。
- 回归 `test_c3_backward`（12/12）、`test_c3_graph`（115/115）、`test_c3_compile_merged`、`test_c3_compile_merged_pgo` 全绿。
- 新增「真实训练首步 loss/梯度」断言（首次 backward 后读 grad 与二次一致）。
- 端到端 MNIST 训练 loss 曲线不受影响（可选）。

## 8. 关联

- STATUS_CONTEXT §4.45（排查记录 + 诊断数据）
- docs/C3_PERF_UNIFIED_MATRIX.md（backward 是 C3 主场 ~10x，正确性为前置条件）


---

## 9. 修复记录（2026-09-03）

**改动**：`c3/src/C3/C3BackwardCapture.cpp`，`tryExecuteBackward` miss 段（原 `return std::nullopt` 处），在 `compileBackwardAsyncForInput + waitForPendingCompiles` 之后**重试 execute 该输入**：

```cpp
// 编译同步完成后重试 execute，使首次调用即同步用 C3 结果
auto retry = C3KernelRegistry::getInstance().tryExecuteBackward(
    base_key + "|in:" + std::to_string(i), grad, forward_inputs);
if (retry.has_value() && !retry->empty()) {
    out.push_back(std::move(retry->at(0)));
    continue; // 已用 C3 内核同步算完
}
return std::nullopt; // 重试仍 miss → 整体 eager
```

**验证结果**：
- `bench_c3_backward_perf_clean` guard：**0.185736 FAIL → 0 PASS**（冷启动与预热 cache 均 PASS）
- iter0 耗时：1.2ms → **0.894ms**（首访即同步用 C3，免 eager 回退，更快）
- 回归：`test_c3_backward` 12/12、`test_c3_graph` 115/115、`test_c3_compile_merged` 10/10、`test_c3_compile_merged_pgo` 11/11 全绿
