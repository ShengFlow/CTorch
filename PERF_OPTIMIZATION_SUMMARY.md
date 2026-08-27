# CTorch MNIST 性能优化小结

> 本文档沉淀 2026-08-26 ~ 2026-08-27 对 C3（JIT 自动优化）在 MNIST MLP 上端到端性能的归因与优化成果。
> 详细逐条记录见 `STATUS_CONTEXT.md` 4.30 ~ 4.40。

## 1. 一句话结论

C3（区域融合 + 反向 MIMO 融合）把 MNIST 训练 epoch 从 **Eager 9577ms 压到 ~192ms**，比自家 Eager 快 **~49×**，距 PyTorch eager（CPU，5 线程，~160ms）只差 **~1.2×**。Foward 融合 / 反向 MIMO / 调度层 / 自动微分编排 / 梯度写回五大块均已逐项归因到底，全套 C3 优化已逼近这套硬件 + 框架的物理账本尽头。

## 2. 性能演进时间线

| 阶段 | epoch | 关键动作 |
|------|-------|---------|
| 基线 Eager（无 C3） | 9577ms | —— |
| C3 早期（含死转置） | 425ms | 区域融合启用，但命中死转置 |
| 死转置折叠修复 | 285ms | 消除 ~116ms 死转置拷贝 |
| 门控 + 调度裁剪 | ~257ms | 位掩码 / `isRegionCandidateOp` |
| 调度 `start` 削半 | ~240ms | `Tensor::shallow()` + backward 短路 |
| P0 现代 C++ | ~198ms | copy/assign 不再深拷 grad（`initAutogradSelf` DRY） |
| P1 低风险批 | ~192ms | `getUpStreamNodes` 返 const ref、`scoped_lock`、`static_cast` |
| **当前稳态** | **~188–193ms** | 本轮 GradAccumulator 快速路径（零回归） |

## 3. 当前每-epoch 字节级账目（192.8ms 口径）

```
Foward region（融合 kernel + prewalk） ≈ 46ms
    └ 融合 kernel 真实计算（end）      ≈ 34ms   ← 该花的钱
    └ prewalk 启动（start）             ≈ 11ms   ← 必需机制
Backward ≈ 94.5ms
    └ MIMO 反向融合（含 keybuild/dispatch/execute）≈ 68ms
        └ cblas 必算                    ≈ 46ms
        └ epilogue/setup/keybuild       ≈ 22ms   ← 判不强推
    └ eager node.backward（nbwd）      ≈ 26ms
        └ GradAccumulator 梯度写回      ≈ 15ms   ← 本轮已优化
        └ CrossEntropy                  ≈ 1.9ms
        └ Layer3 Add(bias)              ≈ 0.65ms
    └ 图清理（clearRecursive）         ≈ 1ms
Loss（CrossEntropy 前向）≈ 1.5ms
Optimizer（SGD）           ≈ 6.8ms
```

## 4. 五大块归因结论（决定性）

### 4.1 GEMM 计算 —— 已是硬件极限，手动并行是负优化
- 探针证伪：对 MNIST 全部 6 个真实形状，`cblas_sgemm`（Accelerate）本身就多核高效；手动 M 行分块 P=2/4/8 全部更慢（P1 最优），整 batch 加速 **1.00×**。
- ⇒ 「GEMM 多线程化」对当前小型线性层**不做**。

### 4.2 调度层 `tryRegionDispatch`(rd/rm) —— 已到极限
- `backward` 短路后 `early=0.02µs`、`tail=0.1µs` 免费；唯一可削的 `start=11.6µs` 大头是**一次性 placeholder/LazyMaterializer 创建**（非查找，`findRegionByFirstOp` 只遍历 2~3 个 region）；`end=36.4µs` 是融合 kernel **真实计算**。
- memo 缓存最多省 1~2ms/ep，收益过低，不做。

### 4.3 反向 MIMO —— 已近极限
- MIMO exec ≈ 68ms/ep，其中调度（keybuild + dispatch）< 1.5ms，主要靠 `mimo_keybuild≈0.6ms`、`bw_dispatch≈0.7ms`。
- cblas 46ms 为必算（与裸调等速，探针铁证）；剩 epilogue ~10ms（`[4.28/4.40]` 判不强推）。

### 4.4 自动微分编排层 —— 已证明免费
- `GradBucket` 线性查找 + 锁 + 就绪队列（pop/get/add/dec/push）实测 **全部 ≈0ms**。
- 修正早期"~38ms 编排黑洞"的误判：真身是 eager 小算子（见 4.5）。

### 4.5 eager 梯度写回（nbwd）—— 机制必要，最后一块已优化
- 按 node 类型分桶：**GradAccumulator ≈15ms/ep**（每参数每 batch 一次写回 `.grad`），CrossEntropy 1.9ms/ep、Add 0.65ms/ep。
- 已落地：单梯度快速路径 + `grad_ptr()` 探测（避免 `grad()` 整 Tensor 拷贝），acc 97.16% 零回归，稳态 epoch 持平。

## 5. 方法学：确定性探针工具链（默认关，零开销）

- `C3_RD_SEG`：`tryRegionDispatch` 五段计时（early/start/mid/end/tail）。
- `C3_BW_SEG` + `[BW-NBWD]`：`ComputeCore::backward` 八段（pop/get/nbwd/mimo/dec/add/push/clear）+ 按 node 类型分桶。
- `C3_CBLAS_PROBE`：cblas_sgemm 强符号拦截，按 (M,N,transA,transB) 分桶钉死每个 GEMM 真实耗时。
- macOS Release 无 `perf` 符号 → 一律用确定性探针而非采样。

## 6. 与 PyTorch 的差距与展望

- **现状**：C3 ≈192ms vs PyTorch eager ≈160ms，**1.2×**（三种实现 acc 均 ≈97.18%，可比）。PyTorch CPU 用 **5 线程**（OpenMP）逐元素 op 也并行，我们没有这层。
- **已封顶**：五大块归因完毕，单 op 层面没有大块浪费可再啃。
- **进一步的思路**（若要走）：大 batch 并行 / 跨 batch 流水线 / thread-level 逐元素并行（对正 PyTorch 的 5 线程）/ 更大真实网络（在更大的 GEMM 上，分块并行与数值调优才有发挥空间）。对小线性层抠单 op 已进入收益递减区。

## 附：本阶段相关提交

- `c229054`：tryRegionDispatch prewalk 优化 + `Tensor::shallow()`
- `a501909`：P0/P1 现代 C++ + Node 移动构造 `_dependencies` 修复
- `c353e81`：GradAccumulator 快速路径 + backward 归因探针（C3_BW_SEG / BW-NBWD）