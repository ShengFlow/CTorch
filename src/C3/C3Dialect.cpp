#include "C3/C3Dialect.h"
#include "mlir/IR/Builders.h"

// 引入 TableGen 自动生成的 Dialect 与算子实现
#include "C3Dialect.cpp.inc"

#define GET_OP_CLASSES
#include "C3Ops.cpp.inc"

namespace mlir {
namespace c3 {

// 初始化 C3 Dialect：注册我们定义的所有 C3 算子
void C3Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "C3Ops.cpp.inc"
  >();
}

// ==================== C3 算子自定义 builder 实现 ====================
// ODS 中声明的自定义 builders（参数为 int64_t/int）在 TableGen 里只生成声明、
// 不生成定义（算子无 result type 可推断时无法自动生成），需在此手动实现。
// 语义与自动生成的 uint64_t/uint32_t 版本完全一致。

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                        ::mlir::Value input, ::mlir::Value out,
                        int64_t M, int64_t N, int dim0, int dim1) {
  odsState.addOperands(input);
  odsState.addOperands(out);
  auto i64 = odsBuilder.getI64Type();
  auto i32 = odsBuilder.getI32Type();
  odsState.getOrAddProperties<TransposeOp::Properties>().M = odsBuilder.getIntegerAttr(i64, M);
  odsState.getOrAddProperties<TransposeOp::Properties>().N = odsBuilder.getIntegerAttr(i64, N);
  odsState.getOrAddProperties<TransposeOp::Properties>().dim0 = odsBuilder.getIntegerAttr(i32, dim0);
  odsState.getOrAddProperties<TransposeOp::Properties>().dim1 = odsBuilder.getIntegerAttr(i32, dim1);
}

void SumReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                        ::mlir::Value input, ::mlir::Value out,
                        int64_t M, int64_t N, int axis) {
  odsState.addOperands(input);
  odsState.addOperands(out);
  auto i64 = odsBuilder.getI64Type();
  auto i32 = odsBuilder.getI32Type();
  odsState.getOrAddProperties<SumReduceOp::Properties>().M = odsBuilder.getIntegerAttr(i64, M);
  odsState.getOrAddProperties<SumReduceOp::Properties>().N = odsBuilder.getIntegerAttr(i64, N);
  odsState.getOrAddProperties<SumReduceOp::Properties>().axis = odsBuilder.getIntegerAttr(i32, axis);
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                     ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value out,
                     ::mlir::Value bias,
                     int64_t M, int64_t K, int64_t N,
                     int transA, int transB, int act,
                     int64_t tileM, int64_t tileN, int64_t biasNumel) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(out);
  if (bias) odsState.addOperands(bias);
  auto i64 = odsBuilder.getI64Type();
  auto i32 = odsBuilder.getI32Type();
  odsState.getOrAddProperties<MatMulOp::Properties>().M = odsBuilder.getIntegerAttr(i64, M);
  odsState.getOrAddProperties<MatMulOp::Properties>().K = odsBuilder.getIntegerAttr(i64, K);
  odsState.getOrAddProperties<MatMulOp::Properties>().N = odsBuilder.getIntegerAttr(i64, N);
  odsState.getOrAddProperties<MatMulOp::Properties>().transA = odsBuilder.getIntegerAttr(i32, transA);
  odsState.getOrAddProperties<MatMulOp::Properties>().transB = odsBuilder.getIntegerAttr(i32, transB);
  odsState.getOrAddProperties<MatMulOp::Properties>().act = odsBuilder.getIntegerAttr(i32, act);
  odsState.getOrAddProperties<MatMulOp::Properties>().tileM = odsBuilder.getIntegerAttr(i64, tileM);
  odsState.getOrAddProperties<MatMulOp::Properties>().tileN = odsBuilder.getIntegerAttr(i64, tileN);
  odsState.getOrAddProperties<MatMulOp::Properties>().biasNumel = odsBuilder.getIntegerAttr(i64, biasNumel);
}

void MatMulTensorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                           ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value dest,
                           int64_t M, int64_t K, int64_t N,
                           int transA, int transB, int act,
                           int64_t tileM, int64_t tileN, int64_t biasNumel) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(dest);
  odsState.addTypes(dest.getType());
  auto i64 = odsBuilder.getI64Type();
  auto i32 = odsBuilder.getI32Type();
  odsState.getOrAddProperties<MatMulTensorOp::Properties>().M = odsBuilder.getIntegerAttr(i64, M);
  odsState.getOrAddProperties<MatMulTensorOp::Properties>().K = odsBuilder.getIntegerAttr(i64, K);
  odsState.getOrAddProperties<MatMulTensorOp::Properties>().N = odsBuilder.getIntegerAttr(i64, N);
  odsState.getOrAddProperties<MatMulTensorOp::Properties>().transA = odsBuilder.getIntegerAttr(i32, transA);
  odsState.getOrAddProperties<MatMulTensorOp::Properties>().transB = odsBuilder.getIntegerAttr(i32, transB);
  odsState.getOrAddProperties<MatMulTensorOp::Properties>().act = odsBuilder.getIntegerAttr(i32, act);
  odsState.getOrAddProperties<MatMulTensorOp::Properties>().tileM = odsBuilder.getIntegerAttr(i64, tileM);
  odsState.getOrAddProperties<MatMulTensorOp::Properties>().tileN = odsBuilder.getIntegerAttr(i64, tileN);
  odsState.getOrAddProperties<MatMulTensorOp::Properties>().biasNumel = odsBuilder.getIntegerAttr(i64, biasNumel);
}

void TransposeTensorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                              ::mlir::Value input, ::mlir::Value dest,
                              int64_t M, int64_t N, int dim0, int dim1) {
  odsState.addOperands(input);
  odsState.addOperands(dest);
  odsState.addTypes(dest.getType());
  auto i64 = odsBuilder.getI64Type();
  auto i32 = odsBuilder.getI32Type();
  odsState.getOrAddProperties<TransposeTensorOp::Properties>().M = odsBuilder.getIntegerAttr(i64, M);
  odsState.getOrAddProperties<TransposeTensorOp::Properties>().N = odsBuilder.getIntegerAttr(i64, N);
  odsState.getOrAddProperties<TransposeTensorOp::Properties>().dim0 = odsBuilder.getIntegerAttr(i32, dim0);
  odsState.getOrAddProperties<TransposeTensorOp::Properties>().dim1 = odsBuilder.getIntegerAttr(i32, dim1);
}

void SumReduceTensorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                              ::mlir::Value input, ::mlir::Value dest,
                              int64_t M, int64_t N, int axis) {
  odsState.addOperands(input);
  odsState.addOperands(dest);
  odsState.addTypes(dest.getType());
  auto i64 = odsBuilder.getI64Type();
  auto i32 = odsBuilder.getI32Type();
  odsState.getOrAddProperties<SumReduceTensorOp::Properties>().M = odsBuilder.getIntegerAttr(i64, M);
  odsState.getOrAddProperties<SumReduceTensorOp::Properties>().N = odsBuilder.getIntegerAttr(i64, N);
  odsState.getOrAddProperties<SumReduceTensorOp::Properties>().axis = odsBuilder.getIntegerAttr(i32, axis);
}

// ==================== C3 方言类型解析/打印 ====================
// 声明了 useDefaultTypePrinterParser，但 C3 方言当前没有任何自定义类型
// （算子直接使用 LLVM 指针/标量类型），因此 TableGen 只生成声明不生成定义，
// 这里提供最小实现：解析一律失败、打印为空（正常流程不会走到）。

::mlir::Type C3Dialect::parseType(::mlir::DialectAsmParser &parser) const {
  parser.emitError(parser.getCurrentLocation(),
                   "c3 dialect does not define any custom types");
  return ::mlir::Type();
}

void C3Dialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter &os) const {
  (void)type;
  (void)os;
}

} // namespace c3
} // namespace mlir
