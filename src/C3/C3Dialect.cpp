#include "C3/C3Dialect.h"
#include "mlir/IR/Builders.h"

// 引入 TableGen 自动生成的 Dialect 与算子实现
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
