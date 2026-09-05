# C3 Fused Backward 链 grad 回填错误 · 缺陷报告

> 编号：C3-BUG-20260905-01
> 日期：2026-09-05
> 报告人：PEL25 audit（独立 Agent），苏璃珞 review
> 状态：**已缓解（disabled）** — tryExecuteFusedBackward 返回 nullopt，未 re-enable
> 关联：C3BackwardCapture::tryExecuteFusedBackward / DEBT-2
> 注意：本编号与 C3-BUG-20260903-01（进程内首访梯度未同步，已修）是**两个独立 bug**，勿混淆。

## 1. 摘要

C3 的 `tryExecuteFusedBackward`（多算子 element-wise 链反向融合）存在 grad 回填错误的确定性风险：
ReLU→Sigmoid / ReLU→ReLU 等链在异步 kernel 安装后，`pending_intercepted_` 拦截机制可能把错误的
upstream grad 回填到后续节点，导致训练数值错误。

## 2. 现状（PEL25 mitigation）

- 该函数在更早的 commit f8161c6（2026-08-30）就因 pool buffer 生命周期 bug 被 early-return 禁用。
- PEL25（2026-09-05）将已禁用函数体内 289 行 unreachable dead code 删除并补充诊断文档。
- 当前策略：始终返回 std::nullopt，所有反向融合走单节点 C3 backward + 必要 eager fallback。
- 影响：此 element-wise 链融合路径 fusion_hit=0；**MIMO 反向融合（mimo_hit=4678）走独立路径，不受影响**。

## 3. 历史实据（为什么当初禁用）

首次禁用 commit f8161c6 记录了真实 bug：
- pool buffer 生命周期：多归约图共享依赖 → 后写覆盖前写 → 越界
- Gt scalar SelectOp 参数反了（cmp 真时应返回 1，原返回 0）

## 4. 待补证据（PEL26+）

- 需要在 re-enable 状态下实测 ReLU→Sigmoid / ReLU→ReLU 链的 C3 fused grad vs eager grad，
  以独立确认 DEBT-2 根因（当前死代码已删，无法在启用态复现）。
- 对照基准：src/tests/standalone/test_fused_bw_debt2.cpp

## 5. 重新启用条件

1. 修 DEBT-2 真根因（chain_forward_inputs 构造 / installBackward shape 校验 / pending_intercepted_ 生命周期）
2. 加数值正确性回归测试（ReLU→Sigmoid / ReLU→ReLU 链，C3 fused vs eager grad 差 < 1e-5）
3. 灰度启用（C3_FUSED_BW=1），先 1 epoch 验证 loss 与 eager 一致

