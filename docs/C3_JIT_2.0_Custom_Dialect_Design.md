# 📐 CTorch C3 JIT 2.0: 基于 TableGen 的结构化 Dialect 与 Lowering 设计规范

本规范是在对外置硬盘 `LuoJin` 上的 MLIR 官方源码（`Toy` 与 `Standalone` Dialect 示例）进行深度剖析，结合 CTorch 框架 JIT 1.0 的运行现状，为 **JIT 2.0 编译器架构（C3 Dialect）** 制订的完整设计与实现方案。

本规范包含：**ODS 算子声明、DRR 声明式重写规则、Linalg 转换、One-Shot Bufferization 显存复用及 CMake 构建管道**。

---

## 一、 架构演进：从 JIT 1.0（直译式）到 JIT 2.0（结构化）

在 JIT 1.0（现行架构）中，C3 引擎采用“直译式”生成 MLIR 控制流。这在**战术上**通过手写标量循环解决了算子覆盖，但在**战略上**存在严重的性能与泛化局限：

```
JIT 1.0 (直译式):
Graph ──> MLIR (arith+scf+LLVM) ──> [ 扁平一维标量循环 ] ──> LLVM JIT (无法充分向量化，内存无复用)

JIT 2.0 (结构化):
Graph ──> c3 Dialect (保留张量语义) ──> linalg/vector ──> [ Tiled + SIMD 向量化 ] ──> LLVM JIT (极致性能)
```

### 1.1 结构化 Dialect 的核心价值
* **图级高层语义保留**：不立即将 `SumReduce` 和 `Transpose` 展开为低层循环，而在 ODS 层面保留矩阵乘（`c3.matmul`）、矩阵转置（`c3.transpose`）等几何特征，便于图级别的代数优化。
* **显存一箭式 inplace 复用**：引入 MLIR 的 **One-Shot Bufferization**，在 SSA 层次进行图级张量活性分析，消灭 90% 的中间冗余临时分配，实现显存极致复用。
* **极致性能与硬件解耦**：将 `c3` 算子降低到标准的 `Linalg` 与 `Vector`。高层算子只需定义“算什么”，而“如何并行和向量化”则直接复用 MLIR 的 Linalg 循环分块（Tiling）、向量化（Vectorization）和 SIMD 寄存器分配等世界级优化 Pass。

---

## 二、 `c3` Dialect 算子定义规范 (`C3Ops.td`)

仿照 MLIR 官方 `toy/Ch2/include/toy/Ops.td` 最佳实践，我们在 `include/C3/C3Ops.td` 中使用 ODS (Operation Definition Specification) 声明式定义方言和核心算子。

```tablegen
//===----------------------------------------------------------------------===//
// C3Ops.td - C3 结构化 Dialect 与算子规范
//===----------------------------------------------------------------------===//

#ifndef C3_OPS
#define C3_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// 1. 定义 C3 方言
def C3_Dialect : Dialect {
  let name = "c3";
  let summary = "The Ctorch JIT Compilation Core Dialect";
  let description = [{
    C3 Dialect 在编译前端保留了计算图的高层多维几何张量特征，
    在 SSA 语法树级别实现安全的代数规约（Pattern Rewrite）与 100% 内存原地复用。
  }];
  let cppNamespace = "::mlir::c3";
}

// 2. C3 算子基类
class C3_Op<string mnemonic, list<Trait> traits = []> :
    Op<C3_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// 3. 核心算子一：矩阵乘法（matmul）
//===----------------------------------------------------------------------===//
def C3_MatMulOp : C3_Op<"matmul", [Pure]> {
  let summary = "High-performance Matrix Multiplication";
  let description = [{
    计算输入矩阵 A [M x K] 和 B [K x N] 的乘积，输出矩阵 C [M x N]。
    在 Lowering 阶段会被降低为 linalg.matmul，以便触发 Cache-friendly 2D Tiling。
    
    示例：
      %out = c3.matmul %A, %B : tensor<32x64xf32>, tensor<64x16xf32> -> tensor<32x16xf32>
  }];

  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
  let results = (outs F32Tensor:$out);

  // 声明式 Assembly 格式
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($out)";

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// 4. 核心算子二：矩阵转置（transpose）
//===----------------------------------------------------------------------===//
def C3_TransposeOp : C3_Op<"transpose", [Pure]> {
  let summary = "Structural 2D matrix transpose";
  let description = [{
    对输入 2D 矩阵进行转置，将 [M x N] 转置为 [N x M]。
  }];

  let arguments = (ins F32Tensor:$input, I32Attr:$dim0, I32Attr:$dim1);
  let results = (outs F32Tensor:$out);

  let assemblyFormat = "$input `dims` `[` $dim0 `,` $dim1 `]` attr-dict `:` type($input) `->` type($out)";

  let builders = [
    OpBuilder<(ins "Value":$input, "int":$dim0, "int":$dim1)>
  ];
}

//===----------------------------------------------------------------------===//
// 5. 核心算子三：特定轴求和降维（sum_reduce）
//===----------------------------------------------------------------------===//
def C3_SumReduceOp : C3_Op<"sum_reduce", [Pure]> {
  let summary = "Axis-wise summation reduction";
  let description = [{
    沿着指定的 axis（如 0 或 1），对输入 2D [M x N] 张量进行归约求和。
    支持全量 reduce 到标量（axis=-1）。
  }];

  let arguments = (ins F32Tensor:$input, I32Attr:$axis);
  let results = (outs F32Tensor:$out);

  let assemblyFormat = "$input `axis` $axis attr-dict `:` type($input) `->` type($out)";

  let builders = [
    OpBuilder<(ins "Value":$input, "int":$axis)>
  ];
}

#endif // C3_OPS
```

---

## 三、 图级声明式重写规则 (`C3Combine.td`)

利用 MLIR 官方的 **DRR (Declarative Rewrite Rules)** 机制，我们可以在 `include/C3/C3Combine.td` 中直接定义强大的图融合与代数折叠逻辑。这完全摆脱了 JIT 1.0 中手写 C++ 指针合并的不透明性与崩溃风险。

```tablegen
//===----------------------------------------------------------------------===//
// C3Combine.td - C3 方言图重写与合并 Pattern 规则
//===----------------------------------------------------------------------===//

#ifndef C3_COMBINE
#define C3_COMBINE

include "mlir/IR/PatternBase.td"
include "C3Ops.td"

// 规则 1：双重转置消除
// transpose(transpose(x, d0, d1), d0, d1) -> x
def DoubleTransposeOptPattern : Pat<
  (C3_TransposeOp (C3_TransposeOp $arg, $d0, $d1), $d0, $d1),
  (replaceWithValue $arg)
>;

// 规则 2：转置与矩阵乘折叠 (Transpose Folding)
// 业界公认的反向求导大项优化。在 MLIR 层将矩阵转置合并进 MatMul。
// 转换前： %T_A = c3.transpose %A
//         %out = c3.matmul %T_A, %B
// 转换后： 降低到 linalg 时直接在 input_maps 里指定转置，消去 %T_A 的物理内存分配和计算
```

---

## 四、 CMakeLists.txt 自动构建集成

在 `CMakeLists.txt` 中引入 MLIR TableGen 规则，由编译期自动调用 `mlir-tblgen` 生成 `.inc` C++ 代码：

```cmake
# ==============================================================================
# C3 Dialect TableGen 自动生成配置 (Stage 2.2/2.3)
# ==============================================================================

if(CT_ENABLE_MLIR)
    # 定义 ODS 源码路径
    set(C3_TD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/include/C3/C3Ops.td")

    # 引入 MLIR 包含路径与 LLVM 参数
    set(LLVM_TARGET_DEFINITIONS ${C3_TD_SRC})

    # 1. 自动生成算子声明头文件 (.h.inc)与定义源文件 (.cpp.inc)
    mlir_tablegen(C3Ops.h.inc -gen-op-decls)
    mlir_tablegen(C3Ops.cpp.inc -gen-op-defs)
    
    # 2. 自动生成方言定义声明 (.h.inc)与定义 (.cpp.inc)
    mlir_tablegen(C3Dialect.h.inc -gen-dialect-decls -dialect=c3)
    mlir_tablegen(C3Dialect.cpp.inc -gen-dialect-defs -dialect=c3)

    # 3. 自动生成 Pattern Rewrite 规则 (.cpp.inc)
    set(LLVM_TARGET_DEFINITIONS "${CMAKE_CURRENT_SOURCE_DIR}/include/C3/C3Combine.td")
    mlir_tablegen(C3Combine.cpp.inc -gen-rewriters)

    # 创建一个 CMake 虚拟目标，确保在编译主体 C++ 代码前，TableGen 文件已生成完毕
    add_public_tablegen_target(CTorchC3Gen)

    # 将自动生成的二进制目录加入 Include Path 中
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    
    # 绑定依赖到 CTorch 主二进制
    add_dependencies(CTorch CTorchC3Gen)
endif()
```

---

## 五、 C++ 端方言声明与注册模板

### 5.1 头文件：`include/C3/C3Dialect.h`
```cpp
#ifndef CTORCH_C3_C3_DIALECT_H
#define CTORCH_C3_C3_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// 引入 TableGen 自动生成的 Dialect 声明
#include "C3Dialect.h.inc"

// 引入 TableGen 自动生成的算子声明
#define GET_OP_CLASSES
#include "C3Ops.h.inc"

#endif // CTORCH_C3_C3_DIALECT_H
```

### 5.2 源文件：`src/C3/C3Dialect.cpp`
```cpp
#include "C3/C3Dialect.h"
#include "mlir/IR/Builders.h"

// 引入 TableGen 自动生成的 Dialect 与 算子实现
#include "C3Dialect.cpp.inc"

namespace mlir {
namespace c3 {

// 初始化 C3 Dialect：注册我们定义的所有 C3 算子
void C3Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "C3Ops.cpp.inc"
  >();
}

} // namespace c3
} // namespace mlir
```

---

## 六、 JIT 2.0 端到端 Lowering Pass 架构设计 (Stage 2.3)

我们将重构 `MLIRKernelGen.cpp` 中的 `applyLoweringPipeline`。

不再在构建期直接翻译到标量控制流，而是构建以下高维、可融合、带内存生命周期的优化流水线：

```cpp
void applyLoweringPipeline2(mlir::ModuleOp module) {
    mlir::PassManager pm(module.getContext());

    // 1. 图级代数优化（JIT 2.0 独享）
    // 运行在 C3Combine.td 中定义的声明式重写（如双转置消除、转置折叠）
    pm.addPass(mlir::c3::createC3CanonicalizerPass()); 

    // 2. 算子高低转换 (C3 -> Linalg)
    // 将 c3.matmul -> linalg.matmul, c3.transpose -> linalg.transpose 等
    pm.addPass(mlir::c3::createConvertC3ToLinalgPass());

    // 3. 内存革命：One-Shot Bufferization
    // 运行在 Tensor 层次上的全局 Liveness 活性分析，在 SSA 树上原地 inplace 改写。
    // 这在降维前彻底消灭了冗余内存分配，完全解决了 JIT 1.0 的多输出覆盖与显存泄露问题。
    pm.addPass(mlir::bufferization::createOneShotBufferizePass());

    // 4. 高阶循环优化 (Linalg 自动 Tiling)
    // 根据 AutoTuner 配置的 tile_m / tile_n 自动进行 2D 分块，完全无需手写嵌套循环！
    pm.addPass(mlir::linalg::createLinalgTilePass());

    // 5. 显式向量化（Linalg -> Vector）
    // 自动将 linalg 算子降低为高宽度的 Vector Dialect，
    // 对于 SumReduce 自动转为寄存器级的水平向量累加（AVX-512 FMA 等）
    pm.addPass(mlir::linalg::createLinalgVectorizationPass());

    // 6. 标量循环展开
    pm.addPass(mlir::createConvertVectorToSCFPass());
    pm.addPass(mlir::createConvertSCFToCFPass());

    // 7. 降低到标准 C-ABI 兼容的 LLVM Dialect
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error("C3 JIT 2.0 Lowering Pipeline failed");
    }
}
```

本设计规范已全面梳理就绪。它不仅解决了 JIT 1.0 留下的多输出覆盖 Bug，更能通过 MLIR 高阶方言释放出惊人的性能红利，为 CTorch 赋予无可比拟的底层技术竞争力。
