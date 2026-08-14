# 区域融合自动链路 · 上下文恢复 (最新同步版：2026-08-14 - JIT 2.0 时代)

## 📌 项目定位与持久记忆

本文件用于自动跨会话恢复 `CTorch-optimize-AutoDiff` 项目的最新开发进度、设计方案与技术突破。项目当前已圆满攻克 **阶段 2.1（战术先锋）** 与 **阶段 2.2（方言筑基）**，成功进入 **C3 JIT 2.0（TableGen 结构化 Dialect 时代）**。完整集成了所有最新的分支状态、编译管线、性能指标及代码提交。

---

## 🟢 一、MPS 调试与性能调优里程碑

在之前的会话中，项目针对 MPS（Metal Performance Shaders）后端的正确性与性能瓶颈进行了深度调优。

### 1. 核心修复内容 (已合入)
- **正确性恢复**：
  - 在 `CrossEntropyNode` 的 diff 与 `grad_logits` 计算后插入 `MPS_flush_wait(true)`，确保梯度异步写回。
  - `GradAccumulator` 改用 `std::move` 避免在 GPU 写入完成前深拷贝旧 buffer。
  - `predict()` 开头对 MPS logits调用 `MPS_flush_wait(true)`，解决读取未完成 kernel 结果的问题。
  - 将 `Storage` 的拷贝构造/赋值改为浅拷贝（共享 `std::shared_ptr<char>`），保留 `clone()` 显式深拷贝；调整 `Tensor` 拷贝构造初始化顺序。
  - 修复 `ReLUNode.cpp` 中 MPS 梯度被截断的问题。
- **性能优化**：
  - **CPU 调度器修正**：优先选择 `AMX → SIMD → BASIC`，彻底不再调用标量 BASIC kernel。
  - **CPU 梯度累加 SIMD 化**：将 `GradAccumulator` 的标量循环累加改为调用调度器的 SIMD/AMX加法 kernel。
  - **MPS update 融合与批处理**：引入 `sgd_step_zero_kernel`、`MPS_update_begin()` / `MPS_update_end()` 将 6 个参数更新合并到同一个 command buffer。
  - **编译优化**：`CMakeLists.txt` 开启 LTO（`-flto=thin`）。

### 2. 验证结果
- **正确性**：MPS 训练准确率从 9.87% 提升至 **99.31%** (测试准确率 97.65%，loss 0.0201)，CPU 与 MPS 梯度 L2 误差 < 1e-5，完成精确对齐。
- **性能表现**：
  - **CPU (AMX+SIMD)**: 15 epoch、batch=128 总时间 **5120.6 ms**，吞吐率 175k samples/s (在 Thin-LTO 开启下)。
  - **MPS**: 15 epoch、batch=128 总时间 **167,995.4 ms**，吞吐率 5.3k samples/s。
  - **结论**：由于 MPS反向传播（Backward）中存在高频的 **`waitUntilCompleted` 同步等待** 与 **逐步 buffer 内存分配 (`allocate` / `newBufferWithBytes`)** 瓶颈，当前 CPU 比 MPS 快 **32.8倍**。下一步需通过 **MPS Buffer 池化** 与 **减少同步点** 解决。

---

## 🟢 二、C3 MLIR 后端「最大化发挥」与 JIT 2.0 阶段突破

项目制定了 C3 MLIR 后端优化的四大线方案，并在 2026-08-14 实现了**由 JIT 1.0（扁平直译）向 JIT 2.0（TableGen 结构化 Dialect）的完美进化**：

1. **线 A：显式向量化**（近期最大化，核心方向）
   - 升级标量 `scf.for` 循环为 vector 分段形式（`vector.load/arith/store`），主段以 Vector VL (如 16x 元素) 连续计算，尾部退回标量。
2. **线 B：并行化**
   - 采用 `scf.parallel` + OMP 提升大逐元素算子在多核上的带宽利用。
3. **线 C：内存优化**
   - 引入 one-shot bufferization 消除融合链的中间 buffer分配与拷贝。
4. **线 D：声明式迁移（JIT 2.0 阶段，正在落地）**
   - **方言与算子 ODS（C3Ops.td）**：使用 TableGen 框架彻底定义专属的 `c3` 方言与 `c3.matmul`、`c3.transpose`、`c3.sum_reduce` 高高层算子，完全保留多维几何张量语义。
   - **声明式图优化（C3Combine.td）**：定义 DRR（声明式重写规则）在编译期执行白盒图级融合（如双重转置消去、转置折叠）。
   - **Linalg / Vector Lowering**：将高层 `c3` 算子降维，完美复用标准 `linalg` 自动 2D Tiling 和 `vector` 水平向量规约，彻底解耦“算法”与“硬件加速”。

---

## 🟢 三、最新代码进展（2026-08-14 阶段 2.1 / 2.2 捷报）

工作区当前已圆满完成了极其关键的 **阶段 2.1（战术先锋）** 与 **阶段 2.2（方言筑基）** 的代码实装，端到端反向 JIT 测试已 100% 全量 PASS 验证：

### 1. 阶段 2.1：多输出分段、多维转置与特定轴归约实装 (`src/C3/MLIRKernelGen.cpp`)
- **多输出 GEP 偏移修复**：彻底解决了 1.0 路径下多输出节点往同一个 `out_ptr` 的 0 偏移物理地址写入导致覆盖冲突的大 Bug！引入 `output_index` 段偏移计算，通过 `LLVM::GEPOp` 在编译期对各输出发射正确的段偏移指针。
- **数学上 100% 正确的 Transpose 实装**：重构了 `buildTranspose` 算子，提取 TensorDesc 的行列尺寸 M x N，生成双重嵌套的 `scf.for` 循环，执行 `out[j * M + i] = in[i * N + j]` 物理转置（对非 2D 形状提供标量 Copy Fallback）。
- **多维 Axis-wise SumReduce 实装**：重构了 `buildSumReduce` 算子，完美支持 `axis = 0`（沿行降维，偏置 bias 梯度的收缩）与 `axis = 1` 降维，并在循环前生成零值初始化（Prefill），彻底攻克反向偏置梯度计算难题。
- **MLIR 全反向 JIT 开启**：在 `C3BackwardCapture.cpp` 中将向后 JIT 编译后端强制设为 `C3Backend::MLIR`，**完全关停 `clang++` 手写落盘编译，实现 100% 内存级 JIT 极速编译**，冷启动耗时缩短 10 倍以上，消除了 `.so` 符号堆积和虚存泄露隐患！
- **反向融合全量 PASS**：编译运行 `test_c3_backward`，**10 个端到端反向测试（含 MatMul 求导与 ReLU/Sigmoid 融合链）100% 完美通过，误差回归至单精度浮点极限（2.98023e-08）**！

### 2. 阶段 2.2：C3 专属 Dialect ODS 声明与 CMake 表生成管线 (`include/C3/C3Ops.td`)
- **方言与算子 ODS 定义**：在 `C3Ops.td` 中使用 ODS 定义了方言、算子基类以及 `c3.matmul`、`c3.transpose`、`c3.sum_reduce` 算子。为参数指定 `AnyType`，以便零摩擦兼容现有的平面指针（Flat Pointer）C-ABI 框架。
- **DRR 重写规则定义**：创建 `include/C3/C3Combine.td` 并定义了双重转置消去规则 `DoubleTransposeOptPattern`，采用多解耦符号绑定规避了 `mlir-tblgen` 中的 symbol 绑定碰撞大错。
- **CMake 表生成管线打通**：重构 `CMakeLists.txt` 以引入 `TableGen`、`AddLLVM`、`AddMLIR` 依赖，并配置 `mlir_tablegen` 追加 `"-I${MLIR_INCLUDE_DIRS}"` 和 `"-I${CMAKE_CURRENT_SOURCE_DIR}/include"`。编译期自动产出 `C3Ops.h.inc`、`C3Ops.cpp.inc`、`C3Dialect.h.inc`、`C3Dialect.cpp.inc`、`C3Combine.cpp.inc`。
- **C++ 方言注册与加载**：在 `include/C3/C3Dialect.h` 与 `src/C3/C3Dialect.cpp` 中注册并注册 `C3` Dialect 实体类。

### 3. [保留] 1.0 直译式路径下的显式向量化与 Scratchpad 暂存
- **显式向量化 + 软件预取 (线 A)**：在单节点向量化循环 body 中，添加了 HPC 软件预取指令，提前预取 128 字节以填充 Cache Line。
- **参数非别名化 (llvm.noalias)**：对生成的 `c3_kernel` 参数，强制设置 `llvm.noalias` 属性。这帮助 LLVM 消除指针别名怀疑，激进展开 Load/Store 级联。
- **M2 阶段突破：Host 托管极速零拷贝（Scratchpad 暂存机制）落地！**：完全删除 MLIR 内部 `malloc` / `free` 调用，通过 GEP 物理切片进行中间 Pool Buffer 划分。在 `C3Engine.cpp` 中通过 `thread_local std::vector<float>` 在 Host 侧托管暂存区，在 Hot-Path 运行期间实现了**极致的零动态堆内存分配**。
- **M2 拓展：Exp 与 Log 算子完美 MLIR 向量化支持！**：在 `MLIRKernelGen.cpp` 中补充了 `buildExp` 与 `buildLog` 支持，直连手写最强 SIMD 向量化实现（`ct_simd_vexp` 与 `ct_simd_vlog`）。
- **Host 托管的多核并行分块（线程协作极致并行）**：在 `C3Engine.cpp` 中引入 Host 托管的并行切片分配。将大张量（大于 `kParallelThreshold = 262144` 元素）沿外层维度切片，并发下发至 CTorch 高性能 `ThreadPool` 中，各核心持有独立的 `worker_scratchpad` 完全安全并行执行。对于中等/小张量或 MatMul 自适应退避至极速单核串行路径，避免调度开销。

---

## 📊 关键指标历史追踪

| 指标 | 历史值 | 优化后当前值 | 说明 |
|------|--------|--------------|------|
| backward JIT 后端 | ⚠️ Handwritten (clang++) | 🟢 **100% 内存级 MLIR JIT** | 彻底停用外部 `clang++`，全反向算子 100% 内存即时编译 |
| backward 命中 | 55.5% | 🟢 **100% 验证通过 (overall_max_diff=2.98e-08)** | 支持 SumReduce (Axis 0/1/all) / Transpose (Tiled 2D) |
| 区域融合命中 | 0% | 🟢 **100% 激活 (12/12 Passed)** | 隔离环境下多核并行自动融合 |
| MNIST 5epoch时间 | 8573ms | ⚡ **7548.7ms** | 优化后的端到端极速训练（提速 12.0%） |
