//===- C3Combine.td - Pattern Match Optimizations for C3 ----*- tablegen -*-===//
//
// Defines pattern match optimizations for the C3 dialect using Declarative
// Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef C3_COMBINE
#define C3_COMBINE

include "mlir/IR/PatternBase.td"
include "C3/C3Ops.td"

//===----------------------------------------------------------------------===//
// C3 Pattern-Match and Rewrite Rules
//===----------------------------------------------------------------------===//

// 规则 1: 双重转置消去
// transpose(transpose(x, M1, N1, d0, d1), M2, N2, d2, d3) -> x
def DoubleTransposeOptPattern : Pat<
  (C3_TransposeOp (C3_TransposeOp $arg, $M1, $N1, $d0, $d1), $M2, $N2, $d2, $d3),
  (replaceWithValue $arg)
>;

#endif // C3_COMBINE
