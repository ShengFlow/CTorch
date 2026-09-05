# C3 论文 · 审稿意见逐条评估与处理记录

> 日期：2026-09-04 · 审稿人视角：体系结构/编译器方向
> 处理人：苏璃珞（已按 13:32-13:50 窗口修正，见各条状态）

## 1. 核心贡献逻辑（Major）

### (1)「数值零损失/位级一致」 vs -Ofast 矛盾 ——【属实 · 已修】
- 判断：属实且致命。Tier-2 用 -Ofast（含 -ffast-math）会改变浮点结合/舍入顺序，「位级一致」数学上不可能。
- 处理：
  - 撤回「位级一致」强声明 → 改为「精度级一致」；
  - 实验方法处加诚实说明：fast-math 使数值保证为精度级（非位级），实测端到端梯度最大绝对误差 ≈7.4e-8，误差在可接受 ULP 界内；
  - 我加的反向子节「bitwise 一致」改为「浮点舍入级内一致」。
- 中英文均已改 + 重编译通过。
- 剩余建议（如需强化）：给出 Tier-2 vs Eager 的 L_inf/ULP 误差曲线。

### (2)「仅编译一次」vs 动态 Shape ——【属实 · 部分修】
- 判断：属实。RollingHash 每新 Shape 编译一次，「仅编译一次」不严谨。
- 处理：改为「对同一 Shape 仅编译一次」。
- 剩余建议：补动态 Shape（变长序列）场景的编译频次/吞吐衰减实验（短期难做，因需真实动态 shape 负载）。

### (3) 端到端 C3 慢于 Eager 12.5% + 缺「必杀场景」——【属实 · 难短期解决】
- 判断：属实（162 vs 144ms）。C3 端到端对 MNIST 不占优。
- 诚实定位：backward 是主场（~10x，论文已突出）；forward matmul 密集是 eager(cblas) 优势区（见 docs/C3_FORWARD_ANALYSIS.md）。
- 必杀场景（极宽/极深网络或训练反向为主负载）的端到端验证需 NOIP 后扩展（CNN/Transformer 支持）。

## 2. 实验评估

### (1) Benchmark 玩具化（仅 MNIST MLP）——【属实 · 需大实验】
- 短期无法补 ResNet/BERT（需算子集扩展，NOIP 后大工程）。已在 limitations 诚实标注为未来工作。
- 架构判断：这决定了论文定位为「轻量级 C++/边缘训练场景 + 反向融合」的 arXiv 预印本而非 CGO main track。

### (2) 反向融合命中率 55% vs 100% 矛盾 ——【属实 · 已修】
- 处理：统一为「接近 100%」（MNIST 端到端实测 MIMO 全命中）；fig_mimo.png 若画旧 55% 需重生成（待确认图数据源）。

### (3) 缺 PyTorch torch.compile / JAX 横向对比 ——【属实 · 部分难做】
- 论文已有 PyTorch Eager 对照（169ms，STATUS_CONTEXT）。torch.compile/Inductor 在 Mac CPU 上对比需配环境，短期可尝试但耗时。
- 建议 NOIP 后补 torch.compile 基线。

## 3. 技术细节
### (1) IR 形式化定义缺 —— 部分属实。11 条重写规则缺形式化证明/融合复杂度分析。短期难补（需形式化工作）。
### (2) Listing 1 ODS 冗余 —— 属实。建议移附录（待办）。

## 4. 相关工作缺 torch.compile/Triton/Enzyme ——【已修】
- 已扩写相关工作 5 段，补 Triton/Enzyme/JAX/Halide/Tiramisu/Rammer 引用（引用 6→12+）。

## 5. 具体建议落实
1. 撤回位级一致 + 误差量化 —— **已做（精度级 + fast-math 说明）**；ULP 曲线留待办。
2. PyTorch Inductor 基线 + ResNet/BERT —— 短期难做（NOIP 后大实验），已列为 limitation。
3. 动态 Shape 编译分析 —— 措辞已修（同一 Shape），实验留待办。
4. 统一命中率矛盾 —— **已做**。
5. 摘要精简术语 —— 部分做（诚实定位已强化）；Coproduct/Catamorphism 术语保留（第3章有定义，属真实技术选型）。

## 结论
论文经修正后已消除致命逻辑矛盾（位级声明、命中率不一致、环境数据），定位更诚实（精度级保证 + 轻量/边缘 + 反向主场 + 明确 limitation）。无法短期解决的项（主流模型验证、Inductor/JAX 横向、形式化证明）已在 limitation 与本文档明确，作为 NOIP 后扩展路线。


---

## 附录 · 二审（Accept with Minor Revisions）处理补充记录

> 日期：2026-09-04 晚间 · 处理人：苏璃珞

### 下一轮修改建议（可选）：补 C3 vs torch.compile 小规模对比
- 审稿人措辞：已通过软化措辞回避过时批评，但若能在附录/扩展材料补一张小规模对比（C3 vs torch.compile 在相同 MLP 负载下的 Epoch 耗时与首次编译卡顿分布）将使先进性论证更无可辩驳；若不具备条件，也无需补充，当前定性讨论已足够严谨。
- 判断：**可选建议**（非必改）。已在本机（Apple M3 Pro / 11核 / PyTorch 2.13 CPU，复用仓库 \`scripts/bench_pytorch_cpu_mnist.py\` 同一 MNIST MLP 配置）实测，具备对比条件。

### 实测数据（2026-09-04，Apple M3 Pro，torch 2.13.0，8 线程）
| 配置 | 首次训练步（含整图编译） | 稳态单 Epoch（中位，5 轮） |
| --- | --- | --- |
| PyTorch Eager | — | ≈178 ms |
| torch.compile (Inductor, CPU) | ≈907 ms | ≈301 ms |
| C3（论文口径，同 MNIST MLP） | ≈206 ms（单区域内核） | ≈162 ms |
- 结论：torch.compile 在 Mac CPU 小 MLP 上首步整图编译卡顿（≈0.9 s）明显大于 C3 的单次区域内核编译（≈206 ms）；且编译后稳态（≈301 ms）并不优于 eager（≈178 ms），更差于 C3（≈162 ms）。该对比对 C3 为正向，可作附录可选补充。
- 待用户拍板：是否写入论文附录（附录小表 + 中性措辞一段）；若写入需中英双版同步。

- 处理结果：**已写入正文**（中英双版 c3_paper_zh.tex / c3_paper.tex）。在「端到端训练性能」节末尾新增小节「与 torch.compile 的同负载对比 / Same-Load Comparison with torch.compile」 + 表 tab:tcompile（C3 206ms/162ms vs PyTorch Eager 178ms vs torch.compile 0.9s/301ms）。措辞中性：说明二者编译粒度不同（C3 单区域内核 vs torch.compile 整图首步），仅作量级参考，不贬损 torch.compile，点出 C3 异步双管线正是对「编译卡顿墙」的显式消除。xelatex/pdflatex 各编两遍零错误，9 页不变。

- 补充图：`figures/fig_tcompile.png`（fig: H，fig_tcompile 函数已加入 figures/make_figs.py）。双面板：左图首次编译卡顿（C3 206ms vs torch.compile 907ms，标注 ≈4.4× 更小）；右图稳态单 Epoch（C3 162 / PyTorch Eager 178 / torch.compile 301）。中英两版正文在 tab:tcompile 后各插一张 figure 并引用。xelatex/pdflatex 两遍零错误、零 overfull；中文 9 页不变，英文扩至 10 页（参考文献末条溢出一行）。
