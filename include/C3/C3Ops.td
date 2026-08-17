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

def C3_MatMulOp : C3_Op<"matmul", [MemoryEffects<[MemWrite]>]> {
  let summary = "High-performance Matrix Multiplication";
  let description = [{
    计算输入矩阵 A [M x K] 和 B [K x N] 的乘积，输出矩阵 C [M x N]。
    采用 out-as-operand 语义：结果写入预分配 buffer（$out），无 SSA result。
    
    在 Lowering 阶段按 total_ops 选择策略：
      - total_ops < 256        → 小矩阵内联循环（无 tiling）
      - < kTiledMatMulThreshold → 中矩阵 2D tiled scf.for（Cache-friendly）
      - 否则                    → 委托 cblas_sgemm（BLAS 最优实现）
    
    可选 epilogue 融合：$bias 加法 + 激活函数（act: 0=None, 1=ReLU, 2=Sigmoid, 3=Tanh）。
    可选 transpose folding：transA/transB 携带上游 Transpose 折叠信息（111=NoTrans, 112=Trans）。
    M/K/N 为 0 时表示维度在运行时确定（走 cblas 分支）。
    
    示例：
      c3.matmul %A, %B, %C, %bias {M = 32, K = 64, N = 16, transA = 111, transB = 111, act = 1} : !llvm.ptr, !llvm.ptr, !llvm.ptr
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$out,
    Optional<AnyType>:$bias,
    I64Attr:$M,
    I64Attr:$K,
    I64Attr:$N,
    I32Attr:$transA,
    I32Attr:$transB,
    I32Attr:$act,
    I64Attr:$tileM,
    I64Attr:$tileN,
    I64Attr:$biasNumel
  );
  let results = (outs);

  let assemblyFormat = "$lhs `,` $rhs `,` $out (`,` $bias^)? attr-dict `:` type($lhs) `,` type($rhs) `,` type($out) `,` type($bias)";

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$out, "Value":$bias,
                  "int64_t":$M, "int64_t":$K, "int64_t":$N,
                  "int":$transA, "int":$transB, "int":$act,
                  "int64_t":$tileM, "int64_t":$tileN, "int64_t":$biasNumel)>
  ];
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

def C3_TransposeOp : C3_Op<"transpose", [MemoryEffects<[MemWrite]>]> {
  let summary = "Structural 2D matrix transpose";
  let description = [{
    对输入 2D 矩阵进行转置，将 [M x N] 转置为 [N x M]。
  }];

  let arguments = (ins AnyType:$input, AnyType:$out, I64Attr:$M, I64Attr:$N, I32Attr:$dim0, I32Attr:$dim1);
  let results = (outs);

  let assemblyFormat = "$input `,` $out `dims` `[` $M `,` $N `]` `[` $dim0 `,` $dim1 `]` attr-dict `:` type($input) `,` type($out)";

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$out, "int64_t":$M, "int64_t":$N, "int":$dim0, "int":$dim1)>
  ];
}

//===----------------------------------------------------------------------===//
// SumReduceOp
//===----------------------------------------------------------------------===//

def C3_SumReduceOp : C3_Op<"sum_reduce", [MemoryEffects<[MemWrite]>]> {
  let summary = "Axis-wise summation reduction";
  let description = [{
    沿着指定的 axis（如 0 或 1），对输入 2D [M x N] 张量进行归约求和。
    支持全量 reduce 到标量（axis=-1）。
  }];

  let arguments = (ins AnyType:$input, AnyType:$out, I64Attr:$M, I64Attr:$N, I32Attr:$axis);
  let results = (outs);

  let assemblyFormat = "$input `,` $out `dims` `[` $M `,` $N `]` `axis` $axis attr-dict `:` type($input) `,` type($out)";

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$out, "int64_t":$M, "int64_t":$N, "int":$axis)>
  ];
}

//===----------------------------------------------------------------------===//
// Elementwise Operations Base Classes
//===----------------------------------------------------------------------===//

class C3_BinaryElementwiseOp<string mnemonic> :
    C3_Op<mnemonic, [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$out,
    AnyType:$numel,
    I64Attr:$bmod
  );
  let results = (outs);
  let assemblyFormat = "$lhs `,` $rhs `,` $out `,` $numel attr-dict `:` type($lhs) `,` type($rhs) `,` type($out) `,` type($numel)";
}

class C3_UnaryElementwiseOp<string mnemonic> :
    C3_Op<mnemonic, [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins
    AnyType:$input,
    AnyType:$out,
    AnyType:$numel
  );
  let results = (outs);
  let assemblyFormat = "$input `,` $out `,` $numel attr-dict `:` type($input) `,` type($out) `,` type($numel)";
}

//===----------------------------------------------------------------------===//
// Tensor-based Elementwise Operations (Stage 3, Task 1/2)
//===----------------------------------------------------------------------===//

class C3_BinaryTensorOp<string mnemonic> :
    C3_Op<mnemonic, [Pure]> {
  let arguments = (ins
    F32Tensor:$lhs,
    F32Tensor:$rhs,
    F32Tensor:$dest,
    I64Attr:$bmod
  );
  let results = (outs F32Tensor:$out);
  let assemblyFormat = "$lhs `,` $rhs `,` $dest attr-dict `:` type($lhs) `,` type($rhs) `,` type($dest) `->` type($out)";
}

class C3_UnaryTensorOp<string mnemonic> :
    C3_Op<mnemonic, [Pure]> {
  let arguments = (ins
    F32Tensor:$input,
    F32Tensor:$dest
  );
  let results = (outs F32Tensor:$out);
  let assemblyFormat = "$input `,` $dest attr-dict `:` type($input) `,` type($dest) `->` type($out)";
}

def C3_AddTensorOp : C3_BinaryTensorOp<"add_tensor">;
def C3_SubTensorOp : C3_BinaryTensorOp<"sub_tensor">;
def C3_MulTensorOp : C3_BinaryTensorOp<"mul_tensor">;
def C3_DivTensorOp : C3_BinaryTensorOp<"div_tensor">;
def C3_GtTensorOp  : C3_BinaryTensorOp<"gt_tensor">;

def C3_NegTensorOp : C3_UnaryTensorOp<"neg_tensor">;
def C3_ReLUTensorOp : C3_UnaryTensorOp<"relu_tensor">;
def C3_SigmoidTensorOp : C3_UnaryTensorOp<"sigmoid_tensor">;
def C3_TanhTensorOp : C3_UnaryTensorOp<"tanh_tensor">;
def C3_ExpTensorOp : C3_UnaryTensorOp<"exp_tensor">;
def C3_LogTensorOp : C3_UnaryTensorOp<"log_tensor">;

def C3_ConstTensorOp : C3_Op<"const_tensor", [Pure]> {
  let arguments = (ins
    F32Attr:$value,
    F32Tensor:$dest
  );
  let results = (outs F32Tensor:$out);
  let assemblyFormat = "$value `,` $dest attr-dict `:` type($dest) `->` type($out)";
}

//===----------------------------------------------------------------------===//
// Binary Elementwise Operations
//===----------------------------------------------------------------------===//

def C3_AddOp : C3_BinaryElementwiseOp<"add"> {
  let summary = "Elementwise addition operation";
}

def C3_SubOp : C3_BinaryElementwiseOp<"sub"> {
  let summary = "Elementwise subtraction operation";
}

def C3_MulOp : C3_BinaryElementwiseOp<"mul"> {
  let summary = "Elementwise multiplication operation";
}

def C3_DivOp : C3_BinaryElementwiseOp<"div"> {
  let summary = "Elementwise division operation";
}

//===----------------------------------------------------------------------===//
// Unary Elementwise Operations
//===----------------------------------------------------------------------===//

def C3_NegOp : C3_UnaryElementwiseOp<"neg"> {
  let summary = "Elementwise negation operation";
}

def C3_ReLUOp : C3_UnaryElementwiseOp<"relu"> {
  let summary = "Elementwise ReLU activation operation";
}

def C3_SigmoidOp : C3_UnaryElementwiseOp<"sigmoid"> {
  let summary = "Elementwise Sigmoid activation operation";
}

def C3_TanhOp : C3_UnaryElementwiseOp<"tanh"> {
  let summary = "Elementwise Tanh activation operation";
}

def C3_ExpOp : C3_UnaryElementwiseOp<"exp"> {
  let summary = "Elementwise exponential operation";
}

def C3_LogOp : C3_UnaryElementwiseOp<"log"> {
  let summary = "Elementwise natural logarithm operation";
}

//===----------------------------------------------------------------------===//
// MatMul / Transpose / SumReduce Tensor Operations (JIT 4.0 Unified Pipeline)
//===----------------------------------------------------------------------===//

def C3_MatMulTensorOp : C3_Op<"matmul_tensor", [Pure]> {
  let summary = "High-performance Tensor Matrix Multiplication";
  let arguments = (ins
    F32Tensor:$lhs,
    F32Tensor:$rhs,
    F32Tensor:$dest,
    I64Attr:$M,
    I64Attr:$K,
    I64Attr:$N,
    I32Attr:$transA,
    I32Attr:$transB,
    I32Attr:$act,
    I64Attr:$tileM,
    I64Attr:$tileN,
    I64Attr:$biasNumel
  );
  let results = (outs F32Tensor:$out);
  let assemblyFormat = "$lhs `,` $rhs `,` $dest attr-dict `:` type($lhs) `,` type($rhs) `,` type($dest) `->` type($out)";
  
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$dest,
                  "int64_t":$M, "int64_t":$K, "int64_t":$N,
                  "int":$transA, "int":$transB, "int":$act,
                  "int64_t":$tileM, "int64_t":$tileN, "int64_t":$biasNumel)>
  ];
}

def C3_TransposeTensorOp : C3_Op<"transpose_tensor", [Pure]> {
  let summary = "High-performance Tensor Transpose";
  let arguments = (ins
    F32Tensor:$input,
    F32Tensor:$dest,
    I64Attr:$M,
    I64Attr:$N,
    I32Attr:$dim0,
    I32Attr:$dim1
  );
  let results = (outs F32Tensor:$out);
  let assemblyFormat = "$input `,` $dest attr-dict `:` type($input) `,` type($dest) `->` type($out)";
  
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$dest, "int64_t":$M, "int64_t":$N, "int":$dim0, "int":$dim1)>
  ];
}

def C3_SumReduceTensorOp : C3_Op<"sum_reduce_tensor", [Pure]> {
  let summary = "High-performance Tensor Sum Reduction";
  let arguments = (ins
    F32Tensor:$input,
    F32Tensor:$dest,
    I64Attr:$M,
    I64Attr:$N,
    I32Attr:$axis
  );
  let results = (outs F32Tensor:$out);
  let assemblyFormat = "$input `,` $dest attr-dict `:` type($input) `,` type($dest) `->` type($out)";
  
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$dest, "int64_t":$M, "int64_t":$N, "int":$axis)>
  ];
}

#endif // C3_OPS
