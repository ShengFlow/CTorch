#ifndef CTORCH_C3_C3_DIALECT_H
#define CTORCH_C3_C3_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/DialectImplementation.h"

// 引入 TableGen 自动生成的 Dialect 声明
#include "C3Dialect.h.inc"

// 引入 TableGen 自动生成的算子声明
#define GET_OP_CLASSES
#include "C3Ops.h.inc"

#endif // CTORCH_C3_C3_DIALECT_H
