# C3 论文「相关工作」扩写素材

> 更新日期：2026-09-04
> 用途：补齐 paper/c3_paper_zh.tex 相关工作章节（现仅一段、引用 6 次，为论文最短板）
> 说明：以下均为真实存在的经典论文/系统，bib 信息建议在引前复核 arXiv/会议卷号。

## 一、扩写框架（按 5 类组织相关工作）

### 1. 静态图编译器（整图优化）
- **XLA**（Google）："XLA: Optimizing Compiler for Machine Learning" (arXiv:1705.10363, 2017)；另 Sabne, *XLA: Compiling Machine Learning for Peak Performance* (2020)。
  - 呼应：整图 HLO 层融合/平铺；冷启动高、同步编译。→ C3 的异步双管线消除冷启动。
- **TVM**（Chen et al., OSDI'18）：自动化端到端优化栈 + Halide 式调度搜索（含 Ansor AutoTVM）。
  - 呼应：调度搜索 vs C3 热路径自动编译；占用前台线程同步 vs C3 异步。
- **TensorFlow/GraphDef**（Abadi et al., OSDI'16）。

### 2. DSL 与代码生成
- **Halide**（Ragan-Kelley et al., PLDI'13）：计算与调度分离。
- **Triton**（Tillet et al., MAPL'19）：tile 级语言 + 编译器，PyTorch Inductor 内核目标。
- **Tiramisu**（Baghdadi et al., CGO'19）：多面体 + 调度 DSL。
- **MLIR**（Lattner et al., CGO'21）：多级 IR 基础设施（本论文已引 lattner2021mlir）。

### 3. 反向模式自动微分与编译
- **Enzyme**（Moses et al., PLDI'21）：LLVM IR 层源码级 AD，GPU kernel 自动微分（SC'21）。
  - 呼应：AD 与编译器融合——C3 在 MLIR/计算图层做反向融合，Enzyme 在 LLVM IR 层做 AD，可对比粒度与覆盖。
- **JAX**（Bradbury et al., 2018）：jaxpr 追踪 + jit。
  - 呼应：jaxpr 静态追踪 vs C3 运行时热路径捕获。

### 4. 动态/异步图执行
- **Rammer**（微软，OSDI'20）：基于算子依赖 DAG 的 GPU 并行调度，提高设备利用率。
- **TorchScript / torch.compile (TorchInductor)**：PyTorch 动态图编译路线。
  - 呼应：PyTorch 编译卡顿墙（冷启动）vs C3 稳态零卡顿。

### 5. 算子融合与内核工程
- **NVIDIA cuDNN/cuBLAS / 自研 GEMM**（底层，可简述）。
- **TVM Ansor / FlexTensor**（调度/分块自动搜索）。

## 二、建议扩写的「相关工作」段落骨架

> 静态图编译器（XLA/TVM）在整图层面优化，但冷启动代价高、占用前台线程同步等待；
> DSL 类（Halide/Triton/Tiramisu）把计算与调度分离，让领域专家手工或搜索调度，却较少处理训练态反向传播；
> 反向模式 AD 编译器（Enzyme）在 LLVM IR 层求导，粒度粗、依赖整程序分析；
> 动态图框架（PyTorch torch.compile/JAX jit）以追踪/图捕获实现编译，但存在编译卡顿墙或静态化牺牲灵活性；
> C3 的差异在于：以**异步非阻塞双管线**从机制上消除冷启动、以**余积 IR + MIMO 反向融合**覆盖静态编译器较少处理的训练反向传播，并以诚实规模扫描刻画融合边界。

## 三、参考文献（建议加入 references.bib，引前复核）

1. @misc{sabne2020xla, XLA: Compiling Machine Learning for Peak Performance, 2020}  [已引 sabne2020xla]
2. @inproceedings{chen2018tvm, OSDI 2018}  [已引 chen2018tvm]
3. @inproceedings{tillet2019triton, MAPL 2019}
4. @inproceedings{ragankelly2013halide, PLDI 2013}
5. @inproceedings{baghdadi2019tiramisu, CGO 2019}
6. @inproceedings{lattner2021mlir, CGO 2021}  [已引 lattner2021mlir]
7. @inproceedings{moses2021enzyme, PLDI 2021 "Reverse-mode automatic differentiation and optimization of GPU kernels via Enzyme"}
8. @software{bradbury2018jax}
9. @inproceedings{chen2020rammer, OSDI 2020}
10. @inproceedings{abadi2016tensorflow, OSDI 2016}

> 注：可先补 3/4/7/8/9（Triton/Halide/Enzyme/JAX/Rammer），把引用从 6 提到 11+，相关工作扩到 ~1.5-2 段，审稿观感即显著改善。
