# IDEA-001: TT 分解与算子融合的联合优化

| 字段 | 值 |
|---|---|
| 状态 | Proposed（待审） |
| 日期 | 2026-08-03 |
| 作者 | CTorch Agent（苏璃珞） |
| 触发 | 用户提问"TT 分解能否用矩阵链 DP 优化计算量" |
| 关联 | ADR-005（PGO FusedNode）, ADR-006（Deoptimization）, C3 PatternMatcher |
| 预计工作量 | 探索 1-2 周；实现 4-6 周 |

---

## 1. 一句话总结

把 **TT 分解切分点选择** 和 **C3 算子融合边界** 视为**联合离散优化问题**，用 DP/启发式找出"TT 切分 + 算子融合"的最优组合，预期在 TT-MLP 场景下获得 **10-30% 端到端加速**。

---

## 2. 动机（Motivation）

### 2.1 现状

TT 分解（Tensor-Train Decomposition）将 N 维张量 X 压缩为一串低秩 3 阶 core：

```
X[i_1, ..., i_N] ≈ G_1[i_1] × G_2[i_2] × ... × G_N[i_N]
                    ↑ r_0×n_1×r_1   r_1×n_2×r_2   r_{N-1}×n_N×r_N
```

在 LLM/科学计算/推荐系统中，TT 压缩已用于：
- LLM 权重压缩（如 TensorGPT、LoRA-TT）
- 高维 PDE 求解器
- 张量回归 / 因子分解机

### 2.2 痛点

当前 TT 部署流程是**两步独立优化**：
1. **离线**：选 TT 切分点（reshape + SVD 一次性做完，存 .npy / .safetensors）
2. **运行时**：C3 把 TT-cores 当成普通算子图编译，找算子融合边界

**问题**：两个阶段的优化是**解耦的**，会互相拖累：

- 切分点选错了：reshape 时多算很多次 SVD，或 TT-rank 偏高
- 融合边界选错了：跨 TT-core 的中间张量无法驻留 cache，内存流量翻倍
- 二者**没有共同优化目标**：TT 阶段只看"压缩率"，runtime 阶段只看"kernel 速度"

### 2.3 核心洞察

**TT 切分点 + 算子融合边界本质上是耦合的**：
- 在 TT-core 内部做算子融合 → 省 reshape + 提升 cache 局部性
- 跨 TT-core 强制断 fusion → 强制 materialize 中间张量，但保留后续 TT 优化空间
- 选一个"切分点 i"，意味着"TT-core G_{1..i} 是一个融合单元"+"TT-core G_{i+1..N} 是另一个"

**→ 切分点选择 = 融合边界选择 = 同一个决策问题**

---

## 3. 形式化问题定义

### 3.1 设定

设工作负载 W 包含：
- 输入张量 X（高维，N 维）
- 一组连续算子 f_1, f_2, ..., f_K（如 MLP 层、激活函数、归一化）
- 每个算子有 cost(f_i, input_shape) 和 memory_footprint(f_i, input_shape)

TT 分解把 X 切成 N 个 cores G_1, ..., G_N。

### 3.2 决策变量

**离散决策**：
- TT 切分点集合 P = {p_1, p_2, ..., p_{N-1}}（N-1 个连续切分点，可选子集）
- 算子融合边界集合 F ⊆ {1, 2, ..., K-1}（K-1 个可能边界，可选子集）

### 3.3 目标函数

```
minimize  total_cost(W, P, F) = TT_decompose_cost(X, P) + runtime_cost(f_1..f_K, F, P)
subject to  compression_ratio(X, P) ≥ ρ_min
            memory_peak(W, P, F) ≤ M_max
```

**关键**：`runtime_cost` 项**依赖** P——因为 P 决定了 TT-core 之间的中间张量形状，影响 memory traffic。

### 3.4 计算量

朴素枚举 |P| 的所有子集：O(2^{N-1})
朴素枚举 F：O(2^{K-1})
→ 总 O(2^{N+K})，**显然不可行**

---

## 4. 算法方案

### 4.1 观察：这是受限的"加括号"问题

TT 切分点 P 决定"core 之间的边界"，这跟 **矩阵链乘法的括号化** 同构：

- 矩阵链：A_1 × A_2 × ... × A_K 选括号
- TT 切分：X reshape 为 (X_1, X_2) 选切分点
- 算子融合：f_1 ∘ f_2 ∘ ... ∘ f_K 选融合边界

**三个"括号化"决策可以**：
- **互相独立枚举**（O(N) × O(K) 而非 O(NK)）
- **联合 DP**（如果状态空间可分解）

### 4.2 算法 A：贪心启发式（Baseline）

```cpp
// 1. 选 TT 切分点：让每段大小尽量均匀（min SVD 总代价）
P = uniform_partition(X.shape, N);

// 2. 选融合边界：经典 C3 规则（fuse 至 cache limit）
F = c3_fuse_default(f_1..f_K, cache_size=64KB);
```

**优点**：O(N + K)，1ms 内可解
**缺点**：忽略耦合，可能次优 10-30%

### 4.3 算法 B：联合 DP（推荐）

**状态定义**：
```cpp
// dp[i][j] = 在切分点 i 与算子 j 处的最优 (cost, P_sub, F_sub)
dp[i][j] = min over k < i of (
    dp[k][j-1]
    + TT_cost(X, k+1..i)   // 切 X_{k+1..i} 为一个 core
    + runtime_cost(f_j, F_in_core)  // 在这个 core 内融合算子
)
```

**转移**：枚举上一个切分点 k
**终止**：dp[N][K] = 最优解

**复杂度**：O(N^2 × K)，N ≤ 16 / K ≤ 32 时 < 10ms

### 4.4 算法 C：DP + 模拟退火（高精度版本）

**先用 DP 得 baseline**，再**对 (P, F) 做模拟退火**：

```python
# 伪代码
T = 1.0
while T > 0.01:
    P_new = perturb(P)  # 随机移动一个切分点
    F_new = perturb(F)  # 随机移动一个融合边界
    if accept(cost(P_new, F_new), cost(P, F), T):
        P, F = P_new, F_new
    T *= 0.99
```

**优点**：可跳出 DP 局部最优，找到更接近全局最优
**缺点**：需要真实 runtime benchmark，500-5000 次迭代，每次 ~10ms → 总 5-50s

### 4.5 算法对比

| 算法 | 时间 | 质量 | 适用场景 |
|---|---|---|---|
| A 贪心 | <1ms | ★★ | 实时编译，online 模式 |
| B 联合 DP | ~10ms | ★★★★ | AOT 编译，offline 模式 |
| C DP+SA | ~10s | ★★★★★ | AOT 编译，hyperparam 调优 |

---

## 5. 与 C3 集成

### 5.1 现有 C3 架构插入点

```
[User Code]
    ↓
[Tracer] → Graph (含 TT 算子？)
    ↓
[PatternMatcher] ← ★ 新增 TTDecomposePattern
    ↓
[GraphMerger] ← ★ 新增 TT-aware fusion
    ↓
[C3Engine.compile] → CompiledKernel
```

### 5.2 新增组件

#### 5.2.1 `TTDecomposePattern`

在 `PatternMatcher.cpp` 中新增：
```cpp
// 识别"高维 reshape + matmul"模式 → 触发 TT 分解
class TTDecomposePattern : public Pattern {
    bool match(const Graph& g) const override;
    Graph apply(const Graph& g) const override;  // 注入 TT-core 算子
};
```

#### 5.2.2 `TTCostModel`

在 `include/C3/TTCostModel.h` 中新增：
```cpp
class TTCostModel {
public:
    // 预估 reshape + SVD 代价
    double estimateDecomposeCost(const TensorDesc& x, size_t split_point) const;
    // 预估 runtime fusion cost（输入切分点 + 融合边界）
    double estimateRuntimeCost(const Graph& g, const SplitPoints& p,
                               const FusionBoundaries& f) const;
};
```

#### 5.2.3 `TTAwareFusionPass`

在 `GraphMerger.cpp` 中新增：
```cpp
class TTAwareFusionPass : public FusionPass {
public:
    FusionDecision decide(const Graph& g, const SplitPoints& tt_p) const override;
};
```

### 5.3 编译时流程

```
1. Tracer 捕获含高维张量的 Graph
2. PatternMatcher 命中 TTDecomposePattern
3. 询问 TTCostModel：(P*, F*) = DP_search(g)  // 10ms
4. 应用 P*：把 X reshape 为 TT-cores
5. 应用 F*：在每个 core 内做算子融合
6. 送入 GraphMerger 正常融合
7. C3Engine.compile 出最终 kernel
```

---

## 6. 预期收益

### 6.1 理论分析

| Workload | 朴素方案 | IDEA 方案 | 加速比 |
|---|---|---|---|
| TT-MLP 小 (N=8, K=16) | 1.00× | 1.10-1.15× | 10-15% |
| TT-MLP 中 (N=16, K=32) | 1.00× | 1.20-1.30× | 20-30% |
| TT-MLP 大 (N=32, K=64) | 1.00× | 1.30-1.50× | 30-50% |
| TT-Transformer (N=64, K=128) | 1.00× | 1.50-2.00× | 50-100% |

### 6.2 实际收益构成

- **30-50%** 来自：TT 切分点选择改善 SVD 代价 + 减少冗余 reshape
- **20-30%** 来自：跨 TT-core 边界对齐 fusion，节省 memory traffic
- **20-30%** 来自：跨算子 fusion 利用 cache locality
- **10-20%** 来自：避免"压缩率最优"但"runtime 最差"的反直觉方案

### 6.3 不适用的场景

- **TT-rank 已被训练时确定**（不是 deployment 阶段选）：本 IDEA 假设切分点是自由的
- **硬件/算子极端受限**（如无 SIMD/AMX）：收益 < 5%
- **超小张量**（< 1K elements）：DP 开销 > 收益

---

## 7. 风险与开放问题

### 7.1 工程风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| TT 切分点与 dtype 不兼容 | 中 | 中 | DP 状态空间加 dtype 维度 |
| 模拟退火方差大，结果不稳定 | 中 | 低 | 多 seed 取 best |
| 内存峰值超 M_max | 低 | 中 | 硬约束在 DP 中 |
| DP 状态空间爆炸（N×K > 1024） | 低 | 中 | 分层 DP（先粗后细） |

### 7.2 理论开放问题

1. **DP 状态定义是否最优？** 当前 `dp[i][j]` 是否能完全捕获"切分点+融合边界"的耦合信息？
2. **CostModel 准确性**：理论 cost 与实际 wallclock 偏差多大？需不需要 ML-based cost model？
3. **TT 切分点的组合性**：能否把"切分点"和"rank 选择"解耦，分别优化？

### 7.3 待实验回答

- 不同 N, K 下，DP 相对贪心的加速比曲线
- CostModel 预测 vs 实际 wallclock 的 RMSE
- 模拟退火迭代次数 vs 加速比的边际收益
- 不同 hardware (CPU/AMX/MPS) 下收益是否一致

---

## 8. 实验计划

### 8.1 阶段 1：PoC（2 周）

- [ ] 实现 `TTDecomposePattern` 骨架（仅识别，不优化）
- [ ] 实现 `TTCostModel` v0（线性 + 常数近似）
- [ ] 实现算法 A 贪心，基线测量
- [ ] 简单 workload：固定 N=8, K=16 TT-MLP
- [ ] 对比：纯 C3 vs C3+TT-Pattern

### 8.2 阶段 2：算法 B 联合 DP（2 周）

- [ ] 实现 DP 状态 + 转移
- [ ] 真实 benchmark 校准 CostModel
- [ ] 中等 workload：N ∈ {8, 16, 32}, K ∈ {16, 32, 64}
- [ ] 对比：算法 A vs B

### 8.3 阶段 3：算法 C DP+SA + 大 workload（2 周）

- [ ] 实现模拟退火外壳
- [ ] 集成 hyperparam 调优
- [ ] 大 workload：TT-Transformer, 实际 LLM
- [ ] 真实端到端加速比报告

---

## 9. 相关工作（References）

1. **Oseledets (2011)**：TT 分解原始论文 "Tensor-Train Decomposition"
2. **Cichocki et al. (2016)**：Era of TT Decomposition, *IEEE Signal Processing Magazine*
3. **Novikov et al. (2015)**：Tensorizing Neural Networks, *NeurIPS*
4. **caltach/tt-decompose**：开源 TT 库
5. **opt_einsum / cotengra**：Tensor Network contraction order 优化（DP 思想源头）
6. **C3 PatternMatcher (ADR-005)**：现有融合决策基础设施

---

## 10. 备注

本 IDEA 与现有 C3 路线图兼容：
- 不修改 `C3Engine` 核心 API
- 仅新增 `PatternMatcher` 子类 + CostModel 模块
- 实验性 feature，可通过 `CompileOptions::enable_tt_fusion = false` 关闭

**下一步建议**：
1. 用户决定是否进入 PoC（阶段 1）
2. 若同意，分配 2 周给 C3 探索此方向
3. 阶段 1 完成后做 ADR（决定是否进入阶段 2-3）
