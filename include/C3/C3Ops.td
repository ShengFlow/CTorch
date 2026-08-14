//===- C3Ops.td - C3 dialect operation definitions ---------*- tablegen -*-===//
//
// Defines the operations of the C3 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef C3_OPS
#define C3_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// C3 Dialect Definition
//===----------------------------------------------------------------------===//

def C3_Dialect : Dialect {
  let name = "c3";
  let summary = "The Ctorch JIT Compilation Core Dialect";
  let description = [{
    C3 Dialect 在编译前端保留了计算图的高层多维几何张量特征，
    在 SSA 语法树级别实现安全的代数规约（Pattern Rewrite）与 100% 内存原地复用。
  }];
  let cppNamespace = "::mlir::c3";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base C3 Operation Definition
//===----------------------------------------------------------------------===//

class C3_Op<string mnemonic, list<Trait> traits = []> :
    Op<C3_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// C3 Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

def C3_MatMulOp : C3_Op<"matmul", [Pure]> {
  let summary = "High-performance Matrix Multiplication";
  let description = [{
    计算输入矩阵 A [M x K] 和 B [K x N] 的乘积，输出矩阵 C [M x N]。
    在 Lowering 阶段会被降低为 linalg.matmul 或优化的 Tiled scf.for，以便触发 Cache-friendly 2D Tiling。
    
    示例：
      %out = c3.matmul %A, %B {M = 32, K = 64, N = 16} : !llvm.ptr, !llvm.ptr -> !llvm.ptr
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs, I64Attr:$M, I64Attr:$K, I64Attr:$N);
  let results = (outs AnyType:$out);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($out)";

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "int64_t":$M, "int64_t":$K, "int64_t":$N)>
  ];
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

def C3_TransposeOp : C3_Op<"transpose", [Pure]> {
  let summary = "Structural 2D matrix transpose";
  let description = [{
    对输入 2D 矩阵进行转置，将 [M x N] 转置为 [N x M]。
  }];

  let arguments = (ins AnyType:$input, I64Attr:$M, I64Attr:$N, I32Attr:$dim0, I32Attr:$dim1);
  let results = (outs AnyType:$out);

  let assemblyFormat = "$input `dims` `[` $M `,` $N `]` `[` $dim0 `,` $dim1 `]` attr-dict `:` type($input) `->` type($out)";

  let builders = [
    OpBuilder<(ins "Value":$input, "int64_t":$M, "int64_t":$N, "int":$dim0, "int":$dim1)>
  ];
}

//===----------------------------------------------------------------------===//
// SumReduceOp
//===----------------------------------------------------------------------===//

def C3_SumReduceOp : C3_Op<"sum_reduce", [Pure]> {
  let summary = "Axis-wise summation reduction";
  let description = [{
    沿着指定的 axis（如 0 或 1），对输入 2D [M x N] 张量进行归约求和。
    支持全量 reduce 到标量（axis=-1）。
  }];

  let arguments = (ins AnyType:$input, I64Attr:$M, I64Attr:$N, I32Attr:$axis);
  let results = (outs AnyType:$out);

  let assemblyFormat = "$input `dims` `[` $M `,` $N `]` `axis` $axis attr-dict `:` type($input) `->` type($out)";

  let builders = [
    OpBuilder<(ins "Value":$input, "int64_t":$M, "int64_t":$N, "int":$axis)>
  ];
}

#endif // C3_OPS
