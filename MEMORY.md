# 🧠 CTorch AutoDiff & C3 Compiler Memory

## 🌟 核心架构与定位
- **CTorch**：轻量级、高度优化的 C++ 深度学习张量与自动求导（AutoDiff）框架。
- **C3 编译器**：CTorch 的核心 JIT 编译器后端，使用 MLIR/LLVM 进行动态 kernel 编译，支持算子融合与并行化。
- **feature-DCU 分支**：当前核心工作分支，目标是修复编译器 P0 级严重 Bug，支持 MLIR Backward 路径，实现自动的区域融合（Region Fusion），并为最终迁移到 DCU 异构计算打下基础。

---

## 🚀 重大历史技术突破与经验

### 1. MPS 后端正确性与性能双重突围 (2026-08)
- **发现与修复**：
  - 解决了多节点在 MPS 上异步执行时的时序同步问题（通过在 Critical Path 插入 `MPS_flush_wait(true)` 强刷缓存）。
  - 将 `Storage` 拷贝语义由链式重构为引用计数浅拷贝，从而解决了高频训练中的内存爆炸与写写冲突。
  - 修正了 `GradAccumulator` 在累加过程中的浅拷贝覆盖行为，彻底对齐了 CPU/MPS 梯度 L2 误差（误差降至 `< 1e-5`，完全可比）。
  - **准确率**：MNIST 15 epoch 训练准确率从 **9.87%（不收敛）** 直线飙升至 **99.31%（收敛且超越历史基线）**。
- **性能剖析瓶颈**：
  - 尽管优化了 MPS kernel 融合与 command buffer 批处理，由于 MPS 反向传播中有大量高频的同步锁（`waitUntilCompleted`）与显存分配（`allocate`），在 batch=128 下，**CPU (AMX+SIMD) 仍然比 MPS 快约 32.8 倍**。
  - 性能金科玉律：小模型/小 batch 走 CPU，大模型/大 batch 走 MPS。

### 2. 编译优化方案 (CPU 最佳实践)
- **Thin-LTO 开启**：在 `CMakeLists.txt` 中开启 `-flto=thin` 可将 CPU 训练性能提升 **58.7%**。
- **PGO 冲突现象**：实测表明，在 Thin-LTO 已经开启的前提下，使用 PGO（Profile-Guided Optimization）反而会导致性能变差（慢 44%）。这是由于 PGO 的热路径统计干扰了 LTO 的内联/代码布局决策。**因此，目前的最佳配置为 Release + 仅 LTO**。

---

## 🛠️ C3 编译器 MLIR 后端核心优化

### 1. MLIR 级别的“显式/强力”向量化 (Explicit Vectorization)
- **设计原理**：直接在 MLIR 的 `linalg.generic` 级别，通过重写 Pattern 将算子转换为 Vector Dialect 表达（如 `vector.transfer_read`、`vector.transfer_write` 和 `vector.add` 等），显式声明向量宽度（如 `<8xf32>`），消除对 LLVM 自动猜想向量化（LoopVectorize）的依赖，最终降解为 SIMD 指令。
- **物理收益**：在前端直接显式生成 SIMD 表达，面对复杂步长、含取模（bmod）的周期性广播等 LLVM 往往“猜不出/不敢向量化”的场景，能够强制实现 100% 向量化，提升非常稳定。此外对生成的 `c3_kernel` 参数添加 `llvm.noalias` 消除别名疑虑。

### 2. Linalg Tiling 与缓存分块优化 (Loop Tiling)
- **设计原理**：利用 `mlir::linalg::createLinalgTilingPass(options)` 对循环进行切片分块（如指定 `Tiling Sizes = [64, 64]`），替换原本对大张量使用的一维连续拉平单层大循环。
- **物理收益**：将高维迭代空间切分为 `64x64` 的小块，使得计算时的数据块能够完美塞入 L1 Cache 中。通过极高的数据复用与 Cache 命中率，大幅减少访问高延迟 DDR 内存的次数，针对超大张量能获得数倍的速度提升。

### 3. 多核多线程并行化 (Parallelization)
- **设计原理**：在 Lowering 阶段阻止 parallel 属性退化为串行 `scf.for`。
- **CPU 多核**：采用 `mlir::createConvertSCFToOpenMPPass()`，将 parallel 映射为 `omp.parallel`，结合 OpenMP 运行时利用服务器的数十个 CPU 核心进行多线程并行计算。
- **GPU 异构**：利用 `mlir::createLinalgStrategyTileAndFusePass()` + `mlir::createConvertGPUToSPIRVPass()`（或 GPU-to-CUDA），将 Linalg 算子分块并分发给 GPU 的 Grid 和 Block，在 GPU CUDA Core 上并发运行。

### 4. 静态形状特化 (Static Shape Specialization & Constant Folding)
- **设计原理**：对于神经网络中批大小或维度固定的模型（如批大小固定的模型或 ViT 形状），在 MLIR 模块构建时直接传入静态维度（如 `RankedTensorType::get({1024}, f32)`），取代动态形状（`ShapedType::kDynamic`）。
- **物理收益**：
  - 彻底消灭 `tensor.dim` / `memref.dim` 动态探测指令。
  - 下层 LLVM 优化器会进行极度激进的常量折叠，使循环上界变为立即数。
  - LLVM 可以进行完美的、无尾部的 Loop Unrolling（循环展开），消除循环步长跳转开销。

### 5. MLIR 官方的高级 Buffer 提纯 Passes
- **堆转栈分配 (`createPromoteBuffersToStackPass`)**：对于复杂融合中产生的微小临时 MemRef 空间（例如大小 <= 64 字节的中间标量），自动把本该执行 malloc（在堆上分配）的操作转换为 `llvm.alloca`（在 CPU 栈帧上分配），直接免去了向操作系统申请堆内存的开销。
- **Buffer 提升 (`createBufferHoistingPass`)**：自动把嵌套循环内部的临时 Buffer 分配“提升（Hoist）”到循环体外部，防止在百万次的高频迭代中重复创建、销毁分配，分配性能提升立竿见影。

### 6. 算子 JIT 编译零堆内存分配（M2 阶段优化 2026-08-13）
- **设计原理**：多节点融合 JIT 内部之前通过 `malloc`/`free` 动态在堆上为中间 pool buffer 申请空间，这在高频训练中会引入大量的 OS 上下文和内存竞争延迟。
- **核心重构**：
  - **JIT IR 端**：升级 `c3_kernel` 函数签名，追加第 7 个参数 `scratchpad_ptr`（`float*` 指针）。内部完全删除 `malloc` / `free` 外部函数声明和调用，采用 `GEP` 偏移物理划分的方式直接从 `scratchpad_ptr` 切片获得中间 buffer。
  - **JIT 执行端**：在 `C3Engine.cpp` 中通过 `thread_local std::vector<float>` 在 Host 侧提供和托管暂存空间，确保 100% 线程安全，并且在 JIT Hot-Path 运行期间实现了**极致的零动态堆内存分配（Zero dynamic allocations）**。

### 3. 区域融合 (Region Fusion) 输入对齐机制
- **技术缺陷**：以往训练期虽然有 Backward MLIR 编译，但由于 `tryRegionDispatch` 中外部输入提取逻辑错位，区域融合无法命中。
- **重构机制**：
  - 恢复 `tryRegionDispatch` 极速、原生的外部输入收集逻辑。
  - 在测试中，由于前序 MatMul/Add 算子可能已经 JIT 单算子化并强行命中缓存从而“截胡”了 `recordCall` 记账，因此在 EXP 各个重置边界必须调用 `C3Engine::getInstance().clearCache()` 确保“环境隔离”。
  - 修复后，**`test_region_fusion_auto` 的 12/12 单元测试项全部以 100% 正确率斩获 `✅` 满分通过**，并且没有任何 input 数量不匹配（`need 3 inputs, got 2`）的警告！

---

---

## 📐 JIT 3.0 (C3 Dialect & Linalg 声明式大一统) 成果与对标 XLA 的第三阶段展望 (2026-08-15)

### 1. 第一阶段与第二阶段及第三阶段任务 1 的收官战果
* **高层 Dialect 与 ODS 统一落地（100% 算子收编）**：`c3.matmul`、`c3.transpose`、`c3.sum_reduce` 以及所有 **10 种逐元素算子**（`add`、`sub`、`mul`、`div`、`neg`、`relu`、`sigmoid`、`tanh`、`exp`、`log`）已全面在 TableGen ODS 中定义完成，前端 `buildMLIRModule` (单/多节点) 构建全面转换为纯净的 `c3` Dialect 拓扑，旧的标量/逐元素内联生成被彻底废除！
* **统一的 Lowering 设计与降级转换**：在 `MLIRKernelGen.cpp` 中定义并注册了完整的 10 种逐元素 Lowering 模式（`AddOpLowering` 等），优雅将 `c3` Dialect 降解为 `scf` 循环、数学函数与向量化指令，前端计算构建与优化机制实现了完美的物理/硬件解耦！
* **图级代数化化简（Canonicalization）**：完成并修复了 `Add(x, x) -> Mul(x, 2.0)` 重建，并追加了 `Sub(x,0)->x`、`Div(x,1)->x`、`Sub(0,x)->Neg(x)`、`Mul(x,-1)->Neg(x)` 四大新重写规则，单元测试 13/13 全绿通过！
* **Linalg 声明式逐元素大一统**：
  * **单节点**：`LinalgElementwiseGen` 组件 100% 接管单节点逐元素算子，支持标量广播与 1D 向量周期广播，打通了 **JITCache 2.0 (LLVM Bitcode 磁盘持久化缓存)**。
  * **多节点**：`LinalgFusedGen` 组件接管多输出纯逐元素图的编译融合（如 `Mul->Add->Div->Sub`），成功通过多维向量化对 `Sigmoid`、`Tanh`、`Exp`、`Log`、`Div` 5 大复杂算子进行寄存器级并行（Neon/AVX-512）。
  * **测试战果**：100+ 单元测试 100% 成功通过！在 MNIST MLP 端到端训练实验上，C3 相比 Eager 实现了 **5.93× 极致加速且精度零损失（loss/acc 完全对齐）**。前向区域融合也已完全激活（`fused_hit=4676`），且 JITCache 2.0 在训练期间完美运行。同时，系统于 2026-08-16 实现了首创的并发双管线 JIT (Tier 1 & Tier 2) 与注册表级别自适应优化级抢占热更新，彻底抹平了 JIT 编译延迟与运行时极致性能的矛盾。

### 2. 对标 XLA：第三阶段——统一变换管道（Unified Transform Pipeline）演进方向
* **统一 C3 Dialect 接入**：将目前用手写标量循环生成的 `Add`/`Sub`/`ReLU`/`Sigmoid` 等基础逐元素算子正式“收编”入 `C3Ops.td` 成为高层 `c3` 算子，实现前端图输入时 100% 的 `C3 Dialect` 表达。
* **统一降低到 Linalg（C3-to-Linalg Lowering）**：编写通用的 lowering 转换 Pass，将所有高层 `c3` 算子统一降低为标准的 `linalg.matmul` 和带有对应 `AffineMap` 的 `linalg.generic`，精简编译器后端代码。
* **Transform Dialect 联动**：将 `AutoTuner` 的分块参数（Tile Sizes）作为参数传入 `LinalgTilePass` 和 `LinalgVectorizationPass`，在 Linalg 层次通过 Transform Dialect 声明式地应用分块与向量化。
* **全局 One-Shot Bufferization**：在 Linalg 层次上执行张量生命周期与活性分析，最后通过 `createOneShotBufferizePass()` 在 SSA 阶就地复用 memref 物理缓存，消除所有的临时分配，达到极致的物理性能红利。

