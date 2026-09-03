# C3 论文「系统评估」段修改建议（基于 2026-09-03 统一性能口径）

> 更新日期：2026-09-03
> 作者：苏璃珞（SuLiluo）
> 依据：docs/C3_PERF_UNIFIED_MATRIX.md（实测赢面地图）、paper/c3_paper_zh.tex（当前评估段）
> 状态：建议稿（未修改论文 tex，供用户审阅后决定）

## 1. 核心修改思路

论文当前评估段把 C3 定位为"端到端与最优 Eager 同量级，价值在架构而非吞吐放大"（保守但**没有突出 C3 最强项**）。统一口径实测表明：**反向融合（backward）是 C3 的压倒性主场（~10.8-12.7×）**，且在端到端训练中净贡献为正。应把这一点从"消融表里的一行"提升为"一个有数据支撑的独立卖点"，同时诚实标注 forward matmul 密集场景的适用边界。修改不推翻现有结论，而是补强与澄清。

## 2. 建议新增子节（插在 §5.2 端到端训练性能 之后）

在 `subsection{端到端训练性能}` 之后新增 `subsection{反向融合深度剖析}`，tex 草稿如下：

```latex
\subsection{反向融合深度剖析}
为量化 MIMO 反向融合在无前向调度干扰下的纯净收益，我们在单反向链上做隔离测量：
`x \to \mathrm{Tanh} \to \mathrm{Sigmoid} \to \mathrm{ReLU} \to \mathrm{backward}`
（输入 $512\times512$，约 0.25M 元素，120 次无预热）。如表~\ref{tab:bwd} 所示，
C3 的 MIMO 反向融合稳态 p50 达 $\approx$1.5\,ms，相对 Eager 反向（$\approx$16.2\,ms）取得
$\mathbf{10.8\times}$ 的加速（另一独立 bench 交叉验证为 $\approx$12.7\times）。

\begin{table}[H]
\centering
\caption{隔离单反向链：C3 MIMO 反向融合 vs Eager（M3 Pro, 稳态 p50）}
\label{tab:bwd}
\setlength{\tabcolsep}{4pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcc}
\toprule
\textbf{指标} & \textbf{Eager 反向} & \textbf{C3 MIMO 反向融合} \\
\midrule
稳态 p50 (ms)   & 16.2 & \textbf{1.5} \\
吞吐 (iter/s)  & 62 & \textbf{666} \\
加速比          & 1.0$\times$ & $\mathbf{10.8\times}$ \\
\bottomrule
\end{tabular}}%
\end{table}

在端到端 MNIST 训练中，MIMO 反向融合命中率达 \textbf{100\%}（每 step 每隐藏层）。
值得强调的是，该反向收益在端到端为 \textbf{净正贡献}：仅禁用反向融合（前向仍走 C3）
时，端到端 step 时延不降反升（median 579\,us vs 完整 C3 401\,us），证实反向融合
是加速端到端训练的关键而非负担。
```

（建议同时把 fig_mimo.png / 反向命中率图作为该子节配图。）

## 3. 建议强化的表述

1. **摘要/引言贡献点 (iii)**：现写"MIMO 反向融合命中率超过 55\%"，建议补充"隔离测量下相对 Eager 反向取得 $\approx$10.8$\times$ 加速"——用最强项支撑贡献，而非只给命中率。
2. **§5.2 段末**：现写"C3 的价值不在此单次吞吐的绝对放大，而在异步架构"，建议改为："反向融合在单链上取得 $\approx$10.8$\times$ 的加速；端到端训练中该收益与前向调度开销相抵，使整体与最优 Eager 处于同一数量级（1.08$\times$ 内）。C3 的端到端价值在于将反向融合收益无损并入训练、同时以异步双管线消除编译卡顿并实现运行时零动态分配。"

## 4. 建议新增的诚实适用边界（Limitation）

在评估段末尾或讨论处明确：
> C3 在反向传播（自动微分）阶段优势显著（~10$\times$）；在**纯前向、计算密集（MatMul）主导**场景（如宽 MLP 前向），因 Eager 委托 cblas 已达算力/带宽上限，C3 收益被 GEMM 成本稀释并叠加调度开销，略慢于 Eager（~1.3$\times$）。C3 适合以**训练态反向融合**为主要收益的负载。

（诚实标注适用边界反而增强可信度，规避审稿人对"是否端到端全胜"的质疑。）

## 5. 数据来源与可复现

以上数据来自 docs/C3_PERF_UNIFIED_MATRIX.md（2026-09-03 实测）：
- bench_c3_backward_perf_clean / bench_c3_backward_perf：反向链 C3 vs C3_DISABLE_BACKWARD=1
- bench_mlp_ce_train：端到端训练（C3 enabled build/ vs 重建的 build-eager）
- bench_wide_mlp_e2e：端到端前向
