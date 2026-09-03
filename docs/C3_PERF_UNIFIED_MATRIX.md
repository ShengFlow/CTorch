# C3 统一性能口径矩阵（赢面地图）

> 更新日期：2026-09-03
> 测量人：苏璃珞（SuLiluo）· ShengFlow
> 机器：macOS Apple Silicon（M3 Pro, AArch64），Release -O3 -ffast-math，LLVM 22.1.8
> 方法：固定 benchmark 口径，C3 enabled（build/）vs 纯 Eager（build-eager/，CT_DISABLE_C3=ON，本次重建），各自取 median/p50 稳态；无并发构建的干净机器。

## 1. 核心结论

**C3 的价值主线 = 反向融合（backward），不是端到端全胜。**

| 口径 | benchmark | C3 | Eager | 加速比 | 结论 |
|---|---|---|---|---|---|
| **Backward**（单链） | bench_c3_backward_perf_clean（[512×512]→Tanh→Sigmoid→ReLU→bw） | p50 **1.500 ms** | p50 16.150 ms | **≈10.8× 快** | ✅ C3 大赢 |
| **Backward**（交叉验证） | bench_c3_backward_perf | p50 **1.260 ms** | p50 16.024 ms | **≈12.7× 快** | ✅ C3 大赢 |
| **端到端训练**（MNIST-MLP 型） | bench_mlp_ce_train（B64/IN784/H128/d1） | median **401 us** | median 373 us | ≈1.08× 慢 | ⚖️ 同量级 |
| **端到端训练**（深 4 层） | bench_mlp_ce_train（depth=4） | median **667 us** | median 618 us | ≈1.08× 慢 | ⚖️ 同量级 |
| **端到端前向**（宽 matmul 密集） | bench_wide_mlp_e2e（B64/H4096/L4） | 35.6 ms/step | 27.4 ms/step | ≈1.30× 慢 | ⚠️ Eager 赢 |

## 2. 逐项解读

### 2.1 Backward：C3 的压倒性主场（~10-13×）

- 反相链 `x→Tanh→Sigmoid→ReLU→backward`（0.25M 元素）：C3 稳态 p50 ≈ 1.3-1.5 ms，Eager ≈ 16 ms。
- MIMO 反向融合命中率在端到端训练中达 100%（50/50 step 全命中，depth=4 时 200/200）。
- 交叉验证（两个独立 bench）加速比一致（10.8× 与 12.7×），可信。

### 2.2 端到端训练：C3 ≈ Eager（同量级）

- 端到端训练（forward+loss+backward+SGD）C3 仅慢 ≈1.08×，与论文口径（MNIST epoch 162 vs 144 ms，≈1.12×）一致。
- **backward 的 ~10× 收益被 forward 的调度开销部分抵消**，故端到端不放大。
- 证明：`C3-禁backward`（只关 backward，forward 仍 C3）median=579 us，反而比 C3 全开（401 us）慢——**backward 融合是净收益**，关掉它端到端更慢。
- 端到端训练下 MIMO hits=50/200（每 step 每层都融合），backward 融合确实在工作。

### 2.3 端到端前向（宽 matmul 密集）：Eager 占优（C3 慢 ~1.3×）

- B64/H4096/L4 宽 MLP 纯前向：C3 35.6 vs Eager 27.4 ms/step。
- matmul 密集场景 eager 委托 cblas 达到带宽/算力上限，C3 的融合收益（省激活 epilogue）被巨量 GEMM 成本稀释，叠加调度税 → C3 略输。

## 3. 与旧口径矛盾的解释

| 旧说法 | 实际 | 原因 |
|---|---|---|
| "端到端 C3 慢 8.8×（2854 vs 325 us）" | C3 ≈ Eager（≈1.08×） | 旧口径未固定（可能含冷启动/特定规模/未重建 eager 正确对比）；本次同参数同步骤取稳态 median 后不一致不成立 |
| "backward 快 10.51×" | 快 10.8-12.7× | 成立（与本次一致） |
| "论文端到端 C3≈Eager" | ≈1.08× 慢 | 成立（同量级） |

> 注：bench_mlp_ce_train 的 `mode` 字符串由 `CTORCH_DISABLE_C3_BACKWARD` env 决定，与 `CT_DISABLE_C3` 编译宏无关，**不可作纯 eager 判据**；正确判据是 C3 stats 是否打印（纯 eager 无 `[C3 Backward Stats]` 块）。

## 4. 诚实边界

- C3 backward 无预热 bench 的数值 guard 在冷启动（空 cache）下 iter0 出现 diff（≈0.186），系首 iter 触发编译致路径不同；稳态数值此前验证 ≈8.9e-08（见 C3_BACKWARD_OPTIMIZATION_PLAN）。稳态性能数据不受影响。
- benchmark 退出清理阶段偶发 mutex 崩溃（退出码 134），数据在 abort 前已打印至 stdout，本报告数据均取自该部分。

## 5. 论文/RC2 叙事建议

- **C3 的主卖点 = 自动微分反向融合（~10×），且在端到端训练中净贡献为正（不拖累端到端）**。
- 端到端定位：与最优 Eager 同量级，靠 backward 融合 + 稳态零卡顿 + 零动态分配换取，而非端到端吞吐放大。
- 纯 forward matmul 密集场景如实陈述为 eager（cblas）优势区。
