# P0.2 CrossEntropy 端到端 PASS

**日期**: 2026-08-30 12:42
**作者**: 苏璃珞
**关联**: P0.2 step 6 闭环 + P0.2.1 broadcast 修复合并验证 · ASPLOS 2027

## 收尾

Test 12 (CrossEntropy end-to-end, M=4 N=6 故意非平凡尺寸) 一次性 PASS：

```
[Test 12] CrossEntropy end-to-end (eager forward + C3 backward)
  Eager forward loss: 1.8852
  Eager grad_logits (ref):
    row 0 (class=0): -0.866038 0.145604 0.158257 0.172011 0.186959 0.203207
    row 1 (class=1):  0.133962 -0.854396 0.158257 0.172011 0.186959 0.203207
    row 2 (class=2):  0.133962  0.145604 -0.841743 0.172011 0.186959 0.203207
    row 3 (class=3):  0.133962  0.145604  0.158257 -0.827989 0.186959 0.203207
  ...
  Test 12 CrossEntropy max_diff=0  ✅

✅ PASS: C3 backward 结果正确 (overall_max_diff=5.96046e-08)
```

整体 PASS（Test 1~12 + MLP），max_diff 上限 5.96e-08 来自 Test 7 MatMul 精度。

## 这意味着什么

`grad_logits = softmax(logits) - target` 这条**训练关键路径**的 C3 backward 现在：

1. **数值 bit-identical**（max_diff=0）—— 跟 eager CrossEntropyNode.backward 完全一致
2. **走通完整链路**：Tensor::cross_entropy → eager forward → CrossEntropyNode 注册 → AutoGrad::backward 触发 → C3 recordBackwardNode 识别 → buildCrossEntropyBackwardGraph 构造 4 op graph → MLIR JIT 编译 → 执行
3. **包含 P0.2.1 broadcast 修复**：[M, 1] → [M, N] 广播在 CE backward 的 Div 节点上跑通

## 改动量

- 加 1 个 test (Test 12, +64 行)
- 0 个 production 改动（Test 12 复用 step 5/6 + P0.2.1 已实装的所有代码）

## P0.2 全部 step 状态

| Step | 状态 | 验证 |
|------|------|------|
| 1. c3.softmax op（C3Ops.td） | ✅ | — |
| 2. SoftmaxNode 加入 Graph.h | ✅ | — |
| 3. SoftmaxOpLowering（→ linalg.softmax） | ✅ | — |
| 4. MLIRKernelGen forward dispatch | ✅ | Step 4 fix IntegerAttr bug |
| 5. Softmax backward graph + dispatch + keepdim | ✅ | Test 11 max_diff=0 |
| 6. CrossEntropy（op + node + forward + backward） | ✅ | Test 12 max_diff=0 |
| P0.2.1 broadcast shape-based 修复 | ✅ | 解锁 step 5/6 端到端 |

## Commit

`main @ 08e7b89`

## 下一步（基于现在 P0.2 全闭环的状态）

- **P0.2 step 7**：MLP+CE 端到端训练 benchmark（C3 vs eager vs PyTorch-CPU）
  - 现在 4-6-10 MLP backward 正确，下一步加 CE loss + 训练 step 数比对
- **Tensor::cross_entropy forward 走 C3**：现在 forward 还走 eager SIMD，C3 路径可加（C3_CrossEntropyOp 已在 step 6 实装）
- **axis=0 支持**：Softmax/CE 当前 axis=1 hardcode
- **ASPLOS 2027 论文**：C3 dialect opset 完整性 + shape-based broadcast 章节可写
- **P1.x 性能冲刺**（如果时间允许）：MatMul epilogue vector lowering + 区域融合达标 1.0+×

## 阶段总结

今天 session 完成的全部：
- P0.1 backward 覆盖率统计
- P0.3 supportsNodeType 多输入节点
- P0.4 stub 完整化诊断
- P0.5 compile 失败原因统计
- P0.6B async timing 修复（6.25% → 81%）
- P1.4 JITCache key 完整化
- P0.2 step 1-6：Softmax + CrossEntropy 完整接入
- P0.2.1 shape-based broadcast 修复
- 战略调整：CGO 2027 → **ASPLOS 2027**（CCF A）
- Test 11 (Softmax) + Test 12 (CrossEntropy) 端到端 PASS

C3 路径在训练关键算子上**数值完全等价 eager**，下一步可以做性能冲刺。
