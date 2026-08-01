/**
 * @file MLIRKernelGen.cpp
 * @brief C3 JIT MLIR kernel 生成器（Phase 1: MLIR/LLVM 后端）
 * @details 将 Graph 编译为 MLIR module，通过标准 lowering pipeline 降至 LLVM IR，
 *          再经 ExecutionEngine JIT 编译为原生函数指针。
 *
 *          Pipeline:
 *            Graph → MLIR (arith+scf+func+LLVM) → Canonicalize → SCF→CF →
 *            Arith→LLVM → CF→LLVM → MemRef→LLVM → Func→LLVM → JIT
 *
 *          函数签名: c3_kernel(f32* a, f32* b, f32* out, i64 n, i64 M, i64 K, i64 N)
 *          使用 LLVM 指针类型构建，确保 C ABI 兼容。
 * @date 2026/8/1
 */

#include "MLIRKernelGen.h"

#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <variant>

#include "C3/TuningState.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Target/TargetMachine.h"

namespace ct {
namespace c3 {

// ======================= 辅助函数 =======================

static mlir::Value indexToI64(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value idx) {
    return builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), idx);
}

static mlir::Value i64ToIndex(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value val) {
    return builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), val);
}

// ======================= Kernel 构建（LLVM 指针版本） =======================

template <typename ArithOp>
static void buildElementwiseBinary(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value a, mlir::Value b, mlir::Value out,
                                   mlir::Value n) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();
    mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value n_idx = i64ToIndex(builder, loc, n);

    auto loop = builder.create<mlir::scf::ForOp>(loc, c0, n_idx, c1);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value idx = loop.getInductionVar();
    mlir::Value idx_i64 = indexToI64(builder, loc, idx);

    mlir::Value a_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, a, mlir::ValueRange{idx_i64});
    mlir::Value b_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, b, mlir::ValueRange{idx_i64});
    mlir::Value out_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});

    mlir::Value av = builder.create<mlir::LLVM::LoadOp>(loc, f32, a_ptr);
    mlir::Value bv = builder.create<mlir::LLVM::LoadOp>(loc, f32, b_ptr);
    mlir::Value rv = builder.create<ArithOp>(loc, av, bv);
    builder.create<mlir::LLVM::StoreOp>(loc, rv, out_ptr);

    builder.setInsertionPointAfter(loop);
}

/// Div 专用：含除零检查，零除时存储 NaN 并继续
static void buildDiv(mlir::OpBuilder& builder, mlir::Location loc,
                     mlir::Value a, mlir::Value b, mlir::Value out,
                     mlir::Value n) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();
    mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value n_idx = i64ToIndex(builder, loc, n);

    auto loop = builder.create<mlir::scf::ForOp>(loc, c0, n_idx, c1);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value idx = loop.getInductionVar();
    mlir::Value idx_i64 = indexToI64(builder, loc, idx);

    mlir::Value a_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, a, mlir::ValueRange{idx_i64});
    mlir::Value b_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, b, mlir::ValueRange{idx_i64});
    mlir::Value out_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});

    mlir::Value av = builder.create<mlir::LLVM::LoadOp>(loc, f32, a_ptr);
    mlir::Value bv = builder.create<mlir::LLVM::LoadOp>(loc, f32, b_ptr);

    // 除零检查：若 bv == 0.0f，存储 NaN；否则正常除法
    mlir::Value zero = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(0.0f));
    mlir::Value is_zero = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, bv, zero);

    auto if_op = builder.create<mlir::scf::IfOp>(loc, f32, is_zero, true);
    builder.setInsertionPointToStart(&if_op.getThenRegion().front());
    mlir::Value nan_val = mlir::arith::ConstantFloatOp::create(
        builder, loc, f32, llvm::APFloat::getNaN(llvm::APFloat::IEEEsingle()));
    builder.create<mlir::scf::YieldOp>(loc, nan_val);

    builder.setInsertionPointToStart(&if_op.getElseRegion().front());
    mlir::Value div_result = builder.create<mlir::arith::DivFOp>(loc, av, bv);
    builder.create<mlir::scf::YieldOp>(loc, div_result);

    builder.setInsertionPointAfter(if_op);
    builder.create<mlir::LLVM::StoreOp>(loc, if_op.getResult(0), out_ptr);

    builder.setInsertionPointAfter(loop);
}

static void buildMatMul(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::Value a, mlir::Value b, mlir::Value out,
                        mlir::Value M, mlir::Value K, mlir::Value N) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();
    auto i64_type = builder.getI64Type();
    mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value M_idx = i64ToIndex(builder, loc, M);
    mlir::Value K_idx = i64ToIndex(builder, loc, K);
    mlir::Value N_idx = i64ToIndex(builder, loc, N);

    // 分块大小：从 TuningState 获取（支持自动调优），默认 64
    auto tp = TuningState::instance().get();
    const int64_t TILE_M = tp.tile_m;
    const int64_t TILE_N = tp.tile_n;
    const int64_t TILE_K = tp.tile_k;
    mlir::Value tile_m = builder.create<mlir::arith::ConstantIndexOp>(loc, TILE_M);
    mlir::Value tile_n = builder.create<mlir::arith::ConstantIndexOp>(loc, TILE_N);
    mlir::Value tile_k = builder.create<mlir::arith::ConstantIndexOp>(loc, TILE_K);

    // 辅助函数：计算 tile 上界 min(val + tile, limit)
    auto tileBound = [&](mlir::Value val_i64, int64_t tile, mlir::Value limit_i64) -> mlir::Value {
        mlir::Value tile_val = builder.create<mlir::arith::ConstantIntOp>(loc, tile, 64);
        mlir::Value sum = builder.create<mlir::arith::AddIOp>(loc, val_i64, tile_val);
        mlir::Value clamped = builder.create<mlir::arith::MinSIOp>(loc, sum, limit_i64);
        return i64ToIndex(builder, loc, clamped);
    };

    // 预初始化 C 矩阵为 0
    {
        auto loop_init_i = builder.create<mlir::scf::ForOp>(loc, c0, M_idx, c1);
        builder.setInsertionPointToStart(loop_init_i.getBody());
        mlir::Value ii = loop_init_i.getInductionVar();
        mlir::Value ii64 = indexToI64(builder, loc, ii);
        auto loop_init_j = builder.create<mlir::scf::ForOp>(loc, c0, N_idx, c1);
        builder.setInsertionPointToStart(loop_init_j.getBody());
        mlir::Value jj = loop_init_j.getInductionVar();
        mlir::Value jj64 = indexToI64(builder, loc, jj);
        mlir::Value iN = builder.create<mlir::arith::MulIOp>(loc, ii64, N);
        mlir::Value ij = builder.create<mlir::arith::AddIOp>(loc, iN, jj64);
        mlir::Value out_init_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{ij});
        mlir::Value zero = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(0.0f));
        builder.create<mlir::LLVM::StoreOp>(loc, zero, out_init_ptr);
        builder.setInsertionPointAfter(loop_init_j);
        builder.setInsertionPointAfter(loop_init_i);
    }

    // i0 tile (M 维度) → j0 tile (N 维度) → k0 tile (K 维度) → i → j → k
    auto loop_i0 = builder.create<mlir::scf::ForOp>(loc, c0, M_idx, tile_m);
    builder.setInsertionPointToStart(loop_i0.getBody());
    mlir::Value i0 = loop_i0.getInductionVar();
    mlir::Value i0_i64 = indexToI64(builder, loc, i0);
    mlir::Value i_end = tileBound(i0_i64, TILE_M, M);

    auto loop_j0 = builder.create<mlir::scf::ForOp>(loc, c0, N_idx, tile_n);
    builder.setInsertionPointToStart(loop_j0.getBody());
    mlir::Value j0 = loop_j0.getInductionVar();
    mlir::Value j0_i64 = indexToI64(builder, loc, j0);
    mlir::Value j_end = tileBound(j0_i64, TILE_N, N);

    auto loop_k0 = builder.create<mlir::scf::ForOp>(loc, c0, K_idx, tile_k);
    builder.setInsertionPointToStart(loop_k0.getBody());
    mlir::Value k0 = loop_k0.getInductionVar();
    mlir::Value k0_i64 = indexToI64(builder, loc, k0);
    mlir::Value k_end = tileBound(k0_i64, TILE_K, K);

    auto loop_i = builder.create<mlir::scf::ForOp>(loc, i0, i_end, c1);
    builder.setInsertionPointToStart(loop_i.getBody());
    mlir::Value i = loop_i.getInductionVar();
    mlir::Value i64 = indexToI64(builder, loc, i);

    auto loop_j = builder.create<mlir::scf::ForOp>(loc, j0, j_end, c1);
    builder.setInsertionPointToStart(loop_j.getBody());
    mlir::Value j = loop_j.getInductionVar();
    mlir::Value j64 = indexToI64(builder, loc, j);

    // 加载 C[i][j] 作为累加器初值
    mlir::Value iN = builder.create<mlir::arith::MulIOp>(loc, i64, N);
    mlir::Value ij = builder.create<mlir::arith::AddIOp>(loc, iN, j64);
    mlir::Value out_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{ij});
    mlir::Value acc_init = builder.create<mlir::LLVM::LoadOp>(loc, f32, out_ptr);

    auto loop_k = builder.create<mlir::scf::ForOp>(loc, k0, k_end, c1, mlir::ValueRange{acc_init});
    builder.setInsertionPointToStart(loop_k.getBody());
    mlir::Value k = loop_k.getInductionVar();
    mlir::Value k64 = indexToI64(builder, loc, k);
    mlir::Value acc = loop_k.getRegionIterArgs()[0];

    // A[i * K + k]
    mlir::Value iK = builder.create<mlir::arith::MulIOp>(loc, i64, K);
    mlir::Value ik = builder.create<mlir::arith::AddIOp>(loc, iK, k64);
    mlir::Value a_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, a, mlir::ValueRange{ik});
    mlir::Value aik = builder.create<mlir::LLVM::LoadOp>(loc, f32, a_ptr);

    // B[k * N + j]
    mlir::Value kN = builder.create<mlir::arith::MulIOp>(loc, k64, N);
    mlir::Value kj = builder.create<mlir::arith::AddIOp>(loc, kN, j64);
    mlir::Value b_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, b, mlir::ValueRange{kj});
    mlir::Value bkj = builder.create<mlir::LLVM::LoadOp>(loc, f32, b_ptr);

    // acc += A[i][k] * B[k][j]
    mlir::Value prod = builder.create<mlir::arith::MulFOp>(loc, aik, bkj);
    mlir::Value new_acc = builder.create<mlir::arith::AddFOp>(loc, acc, prod);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{new_acc});

    builder.setInsertionPointAfter(loop_k);
    builder.create<mlir::LLVM::StoreOp>(loc, loop_k.getResult(0), out_ptr);

    builder.setInsertionPointAfter(loop_j);
    builder.setInsertionPointAfter(loop_i);
    builder.setInsertionPointAfter(loop_k0);
    builder.setInsertionPointAfter(loop_j0);
    builder.setInsertionPointAfter(loop_i0);
}

static void buildNegate(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::Value in, mlir::Value out, mlir::Value n) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();
    mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value n_idx = i64ToIndex(builder, loc, n);

    auto loop = builder.create<mlir::scf::ForOp>(loc, c0, n_idx, c1);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value idx = loop.getInductionVar();
    mlir::Value idx_i64 = indexToI64(builder, loc, idx);

    mlir::Value in_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, in, mlir::ValueRange{idx_i64});
    mlir::Value out_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});

    mlir::Value iv = builder.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
    mlir::Value rv = builder.create<mlir::arith::NegFOp>(loc, iv);
    builder.create<mlir::LLVM::StoreOp>(loc, rv, out_ptr);

    builder.setInsertionPointAfter(loop);
}

static void buildReLU(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value in, mlir::Value out, mlir::Value n) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();
    mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value n_idx = i64ToIndex(builder, loc, n);

    auto loop = builder.create<mlir::scf::ForOp>(loc, c0, n_idx, c1);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value idx = loop.getInductionVar();
    mlir::Value idx_i64 = indexToI64(builder, loc, idx);

    mlir::Value in_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, in, mlir::ValueRange{idx_i64});
    mlir::Value out_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});

    mlir::Value iv = builder.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
    mlir::Value zero = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(0.0f));
    mlir::Value rv = builder.create<mlir::arith::MaxNumFOp>(loc, iv, zero);
    builder.create<mlir::LLVM::StoreOp>(loc, rv, out_ptr);

    builder.setInsertionPointAfter(loop);
}

// ======================= 融合 Kernel 构建 =======================

static void buildFused(mlir::OpBuilder& builder, mlir::Location loc,
                       mlir::Value inputs, mlir::Value out, mlir::Value n,
                       const std::vector<NodeVariant>& ops,
                       const std::vector<std::vector<size_t>>& op_inputs,
                       const std::vector<size_t>& arg_node_ids) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();
    auto i64_type = builder.getI64Type();
    mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value n_idx = i64ToIndex(builder, loc, n);

    // 构建 node_id → arg_index 的映射
    std::unordered_map<size_t, size_t> node_to_arg;
    for (size_t i = 0; i < arg_node_ids.size(); ++i) {
        node_to_arg[arg_node_ids[i]] = i;
    }

    // === 循环外：预加载所有外部输入指针（循环不变量提升） ===
    // 收集所有被引用的外部输入 node_id
    std::set<size_t> referenced_nodes;
    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        const auto& inputs_for_op = op_inputs[op_idx];
        for (size_t in_id : inputs_for_op) {
            if (op_idx > 0 && in_id == op_inputs[op_idx][0]) continue;
            referenced_nodes.insert(in_id);
        }
    }

    // 为每个引用的外部输入预加载指针：ptr = inputs[arg_idx]
    std::unordered_map<size_t, mlir::Value> preloaded_ptrs;
    for (size_t node_id : referenced_nodes) {
        size_t arg_idx = node_to_arg.at(node_id);
        mlir::Value kc = builder.create<mlir::arith::ConstantIndexOp>(loc, arg_idx);
        mlir::Value kc_i64 = indexToI64(builder, loc, kc);
        mlir::Value ptr_addr = builder.create<mlir::LLVM::GEPOp>(
            loc, ptr_type, ptr_type, inputs, mlir::ValueRange{kc_i64});
        mlir::Value ptr = builder.create<mlir::LLVM::LoadOp>(loc, ptr_type, ptr_addr);
        preloaded_ptrs[node_id] = ptr;
    }

    // === 循环体 ===
    auto loop = builder.create<mlir::scf::ForOp>(loc, c0, n_idx, c1);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value idx = loop.getInductionVar();
    mlir::Value idx_i64 = indexToI64(builder, loc, idx);

    /// 辅助函数：从预加载的指针加载元素值（仅 GEP + Load，无指针间接访问）
    auto loadExternal = [&](size_t node_id) -> mlir::Value {
        mlir::Value ptr = preloaded_ptrs.at(node_id);
        mlir::Value elem_addr = builder.create<mlir::LLVM::GEPOp>(
            loc, ptr_type, f32, ptr, mlir::ValueRange{idx_i64});
        return builder.create<mlir::LLVM::LoadOp>(loc, f32, elem_addr);
    };

    mlir::Value prev_val;

    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        const NodeVariant& op = ops[op_idx];
        const auto& inputs_for_op = op_inputs[op_idx];
        bool is_last = (op_idx == ops.size() - 1);

        // 获取外部输入节点 ID（排除 chain 内部）
        std::vector<size_t> ext_inputs;
        for (size_t in_id : inputs_for_op) {
            if (op_idx > 0 && in_id == inputs_for_op[0]) continue;
            ext_inputs.push_back(in_id);
        }

        mlir::Value result;
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            mlir::Value lhs, rhs;

            if constexpr (std::is_same_v<T, NegNode>) {
                lhs = (op_idx > 0) ? prev_val : loadExternal(ext_inputs[0]);
                result = builder.create<mlir::arith::NegFOp>(loc, lhs);
            } else if constexpr (std::is_same_v<T, ReLUNode>) {
                lhs = (op_idx > 0) ? prev_val : loadExternal(ext_inputs[0]);
                mlir::Value zero = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(0.0f));
                result = builder.create<mlir::arith::MaxNumFOp>(loc, lhs, zero);
            } else if constexpr (std::is_same_v<T, AddNode>) {
                if (op_idx > 0) {
                    lhs = prev_val;
                    rhs = loadExternal(ext_inputs[0]);
                } else {
                    lhs = loadExternal(ext_inputs[0]);
                    rhs = loadExternal(ext_inputs[1]);
                }
                result = builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
            } else if constexpr (std::is_same_v<T, SubNode>) {
                if (op_idx > 0) {
                    lhs = prev_val;
                    rhs = loadExternal(ext_inputs[0]);
                } else {
                    lhs = loadExternal(ext_inputs[0]);
                    rhs = loadExternal(ext_inputs[1]);
                }
                result = builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
            } else if constexpr (std::is_same_v<T, MulNode>) {
                if (op_idx > 0) {
                    lhs = prev_val;
                    rhs = loadExternal(ext_inputs[0]);
                } else {
                    lhs = loadExternal(ext_inputs[0]);
                    rhs = loadExternal(ext_inputs[1]);
                }
                result = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
            } else if constexpr (std::is_same_v<T, DivNode>) {
                if (op_idx > 0) {
                    lhs = prev_val;
                    rhs = loadExternal(ext_inputs[0]);
                } else {
                    lhs = loadExternal(ext_inputs[0]);
                    rhs = loadExternal(ext_inputs[1]);
                }
                // 除零检查：零除时产出 NaN
                mlir::Value zero_c = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(0.0f));
                mlir::Value is_zero = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, rhs, zero_c);
                auto div_if = builder.create<mlir::scf::IfOp>(loc, f32, is_zero, true);
                builder.setInsertionPointToStart(&div_if.getThenRegion().front());
                mlir::Value nan_v = mlir::arith::ConstantFloatOp::create(
                    builder, loc, f32, llvm::APFloat::getNaN(llvm::APFloat::IEEEsingle()));
                builder.create<mlir::scf::YieldOp>(loc, nan_v);
                builder.setInsertionPointToStart(&div_if.getElseRegion().front());
                mlir::Value div_r = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
                builder.create<mlir::scf::YieldOp>(loc, div_r);
                builder.setInsertionPointAfter(div_if);
                result = div_if.getResult(0);
            }
        }, op);

        if (is_last) {
            mlir::Value out_ptr = builder.create<mlir::LLVM::GEPOp>(
                loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});
            builder.create<mlir::LLVM::StoreOp>(loc, result, out_ptr);
        } else {
            prev_val = result;
        }
    }

    builder.setInsertionPointAfter(loop);
}

// ======================= 多节点 MLIR 构建 =======================

/// 判断节点是否为计算节点（非输入、非 Const）
static bool isComputeNodeMLIR(const Node& node, const std::vector<size_t>& input_ids) {
    for (size_t in_id : input_ids) {
        if (node.id == in_id) return false;
    }
    if (node.inputs.empty()) return false;
    if (std::holds_alternative<ConstNode>(node.op)) return false;
    return true;
}

/// 为多节点图构建 MLIR 模块
static mlir::OwningOpRef<mlir::ModuleOp> buildMultiNodeMLIR(
    mlir::MLIRContext& context, const Graph& graph)
{
    auto loc = mlir::UnknownLoc::get(&context);
    mlir::OpBuilder builder(&context);

    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto ptr_type = mlir::LLVM::LLVMPointerType::get(&context);
    auto f32 = builder.getF32Type();
    auto i64_type = builder.getI64Type();

    const auto& nodes = graph.nodes();
    const auto& inputs = graph.inputs();
    const auto& outputs = graph.outputs();

    // 步骤 1: 收集计算节点（拓扑顺序）
    std::vector<const Node*> compute_nodes;
    for (const auto& node : nodes) {
        if (isComputeNodeMLIR(node, inputs)) {
            compute_nodes.push_back(&node);
        }
    }

    // 步骤 2: 构建 node_id → 外部输入索引 的映射
    std::unordered_map<size_t, size_t> external_input_map;
    for (size_t i = 0; i < inputs.size(); ++i) {
        external_input_map[inputs[i]] = i;
    }

    // 步骤 3: 分配缓冲区索引
    std::unordered_map<size_t, size_t> node_to_buffer;
    size_t num_intermediates = 0;
    for (size_t i = 0; i < compute_nodes.size(); ++i) {
        size_t node_id = compute_nodes[i]->id;
        bool is_output = false;
        for (size_t out_id : outputs) {
            if (node_id == out_id) { is_output = true; break; }
        }
        if (i == compute_nodes.size() - 1) is_output = true;
        if (!is_output) {
            node_to_buffer[node_id] = num_intermediates++;
        } else {
            node_to_buffer[node_id] = SIZE_MAX;
        }
    }

    // 步骤 4: 创建函数 (MultiNodeKernelFunc 签名)
    auto func_type = builder.getFunctionType(
        {ptr_type, ptr_type, i64_type, i64_type, i64_type, i64_type}, {});
    auto func = builder.create<mlir::func::FuncOp>(loc, "c3_kernel", func_type);
    func.setPrivate();
    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    mlir::Value inputs_ptr = entry->getArgument(0);  // const float* const*
    mlir::Value out_ptr = entry->getArgument(1);       // float*
    mlir::Value n_val = entry->getArgument(2);          // n
    mlir::Value M_val = entry->getArgument(3);          // M
    mlir::Value K_val = entry->getArgument(4);          // K
    mlir::Value N_val = entry->getArgument(5);          // N

    // 步骤 4.5: 声明 malloc/free 外部函数（在 module 级别）
    auto malloc_type = mlir::LLVM::LLVMFunctionType::get(ptr_type, {i64_type}, false);
    auto free_type = mlir::LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(&context), {ptr_type}, false);
    mlir::OpBuilder::InsertPoint saved_ip = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(module.getBody());
    auto malloc_func = builder.create<mlir::LLVM::LLVMFuncOp>(loc, "malloc", malloc_type);
    malloc_func.setVisibility(mlir::SymbolTable::Visibility::Private);
    auto free_func = builder.create<mlir::LLVM::LLVMFuncOp>(loc, "free", free_type);
    free_func.setVisibility(mlir::SymbolTable::Visibility::Private);
    builder.restoreInsertionPoint(saved_ip);

    // 步骤 5: 分配中间缓冲区 (malloc, 避免 alloca 栈溢出)
    std::vector<mlir::Value> tmp_buffers;
    mlir::Value c4 = builder.create<mlir::arith::ConstantIntOp>(loc, 4, 64);
    mlir::Value size_bytes = builder.create<mlir::arith::MulIOp>(loc, n_val, c4);
    for (size_t i = 0; i < num_intermediates; ++i) {
        auto call = builder.create<mlir::LLVM::CallOp>(loc, malloc_func, mlir::ValueRange{size_bytes});
        tmp_buffers.push_back(call.getResult());
    }

    // 辅助函数：获取节点输入的 MLIR Value
    auto getInputPtr = [&](size_t in_node_id) -> mlir::Value {
        auto ext_it = external_input_map.find(in_node_id);
        if (ext_it != external_input_map.end()) {
            // 外部输入：GEP into inputs array
            mlir::Value idx_val = builder.create<mlir::arith::ConstantIntOp>(loc, (int64_t)ext_it->second, 64);
            mlir::Value slot = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, ptr_type, inputs_ptr, mlir::ValueRange{idx_val});
            return builder.create<mlir::LLVM::LoadOp>(loc, ptr_type, slot);
        }
        auto buf_it = node_to_buffer.find(in_node_id);
        if (buf_it != node_to_buffer.end()) {
            if (buf_it->second == SIZE_MAX) return out_ptr;
            return tmp_buffers[buf_it->second];
        }
        return out_ptr; // fallback
    };

    // 步骤 6: 生成每个计算节点的 MLIR 代码
    for (size_t ci = 0; ci < compute_nodes.size(); ++ci) {
        const Node* node = compute_nodes[ci];
        bool is_last = (ci == compute_nodes.size() - 1);
        mlir::Value out_buf = is_last ? out_ptr : tmp_buffers[node_to_buffer.at(node->id)];
        const NodeVariant& op = node->op;

        if (std::holds_alternative<FusedNode>(op)) continue;

        // 收集输入指针
        std::vector<mlir::Value> in_ptrs;
        for (size_t in_id : node->inputs) {
            in_ptrs.push_back(getInputPtr(in_id));
        }

        if (std::holds_alternative<MatMulNode>(op)) {
            buildMatMul(builder, loc, in_ptrs[0], in_ptrs[1], out_buf, M_val, K_val, N_val);
        } else if (std::holds_alternative<AddNode>(op)) {
            buildElementwiseBinary<mlir::arith::AddFOp>(builder, loc, in_ptrs[0], in_ptrs[1], out_buf, n_val);
        } else if (std::holds_alternative<SubNode>(op)) {
            buildElementwiseBinary<mlir::arith::SubFOp>(builder, loc, in_ptrs[0], in_ptrs[1], out_buf, n_val);
        } else if (std::holds_alternative<MulNode>(op)) {
            buildElementwiseBinary<mlir::arith::MulFOp>(builder, loc, in_ptrs[0], in_ptrs[1], out_buf, n_val);
        } else if (std::holds_alternative<DivNode>(op)) {
            buildDiv(builder, loc, in_ptrs[0], in_ptrs[1], out_buf, n_val);
        } else if (std::holds_alternative<NegNode>(op)) {
            buildNegate(builder, loc, in_ptrs[0], out_buf, n_val);
        } else if (std::holds_alternative<ReLUNode>(op)) {
            buildReLU(builder, loc, in_ptrs[0], out_buf, n_val);
        }
    }

    // 步骤 7: 释放中间缓冲区
    for (auto& tmp : tmp_buffers) {
        builder.create<mlir::LLVM::CallOp>(loc, free_func, mlir::ValueRange{tmp});
    }

    builder.create<mlir::func::ReturnOp>(loc);
    return module;
}

static size_t countComputeNodesMLIR(const Graph& graph) {
    size_t count = 0;
    for (const auto& node : graph.nodes()) {
        if (isComputeNodeMLIR(node, graph.inputs())) count++;
    }
    return count;
}

// ======================= 模块构建 =======================

static mlir::OwningOpRef<mlir::ModuleOp> buildMLIRModule(
    mlir::MLIRContext& context, const Graph& graph)
{
    // 多节点图：使用多节点 MLIR kernel
    if (countComputeNodesMLIR(graph) > 1) {
        auto module = buildMultiNodeMLIR(context, graph);
        if (mlir::failed(mlir::verify(*module))) {
            module->emitError();
            module->dump();
            throw std::runtime_error("MLIRKernelGen: multi-node module verification failed");
        }
        return module;
    }
    auto loc = mlir::UnknownLoc::get(&context);
    mlir::OpBuilder builder(&context);

    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto ptr_type = mlir::LLVM::LLVMPointerType::get(&context);
    auto i64_type = builder.getI64Type();

    // 先找到计算节点，确定使用哪种函数签名
    const auto& nodes = graph.nodes();
    const Node* compute_node = nullptr;
    for (const auto& node : nodes) {
        bool is_input = false;
        for (size_t in_id : graph.inputs()) {
            if (node.id == in_id) { is_input = true; break; }
        }
        if (!is_input && !node.inputs.empty()) { compute_node = &node; break; }
    }
    if (!compute_node) throw std::runtime_error("MLIRKernelGen: no compute node");

    const NodeVariant& op = compute_node->op;

    // 融合节点：使用 FusedKernelFunc 签名 (ptr, ptr, i64) → void
    if (std::holds_alternative<FusedNode>(op)) {
        const auto& fnode = std::get<FusedNode>(op);
        auto func_type = builder.getFunctionType(
            {ptr_type, ptr_type, i64_type}, {});  // inputs(ptr*), out(ptr), n
        auto func = builder.create<mlir::func::FuncOp>(loc, "c3_kernel", func_type);
        func.setPrivate();
        auto* entry = func.addEntryBlock();
        builder.setInsertionPointToStart(entry);
        mlir::Value inputs = entry->getArgument(0);
        mlir::Value out_val = entry->getArgument(1);
        mlir::Value n_val = entry->getArgument(2);
        buildFused(builder, loc, inputs, out_val, n_val, fnode.ops, fnode.op_inputs, fnode.arg_node_ids);
        builder.create<mlir::func::ReturnOp>(loc);
    } else {
        // 普通算子：使用 C3KernelFunc 签名 (ptr, ptr, ptr, i64, i64, i64, i64) → void
        auto func_type = builder.getFunctionType(
            {ptr_type, ptr_type, ptr_type, i64_type, i64_type, i64_type, i64_type}, {});
        auto func = builder.create<mlir::func::FuncOp>(loc, "c3_kernel", func_type);
        func.setPrivate();

        auto* entry = func.addEntryBlock();
        builder.setInsertionPointToStart(entry);

        mlir::Value a = entry->getArgument(0);
        mlir::Value b = entry->getArgument(1);
        mlir::Value out = entry->getArgument(2);
        mlir::Value n = entry->getArgument(3);
        mlir::Value M = entry->getArgument(4);
        mlir::Value K = entry->getArgument(5);
        mlir::Value N = entry->getArgument(6);

        if (std::holds_alternative<AddNode>(op))
            buildElementwiseBinary<mlir::arith::AddFOp>(builder, loc, a, b, out, n);
        else if (std::holds_alternative<SubNode>(op))
            buildElementwiseBinary<mlir::arith::SubFOp>(builder, loc, a, b, out, n);
        else if (std::holds_alternative<MulNode>(op))
            buildElementwiseBinary<mlir::arith::MulFOp>(builder, loc, a, b, out, n);
        else if (std::holds_alternative<DivNode>(op))
            buildDiv(builder, loc, a, b, out, n);
        else if (std::holds_alternative<MatMulNode>(op))
            buildMatMul(builder, loc, a, b, out, M, K, N);
        else if (std::holds_alternative<NegNode>(op))
            buildNegate(builder, loc, a, out, n);
        else if (std::holds_alternative<ReLUNode>(op))
            buildReLU(builder, loc, a, out, n);
        else
            throw std::runtime_error("MLIRKernelGen: unsupported op " + std::to_string(op.index()));

        builder.create<mlir::func::ReturnOp>(loc);
    }

    if (mlir::failed(mlir::verify(module))) {
        module.emitError();
        module->dump();
        throw std::runtime_error("MLIRKernelGen: module verification failed");
    }

    return module;
}

// ======================= Lowering Pipeline =======================

static void applyLoweringPipeline(mlir::ModuleOp module) {
    mlir::PassManager pm(module.getContext());
    // 1. 高级优化：常量折叠 + 公共子表达式消除
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    // 2. 循环不变量外提
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    // 3. lowering: SCF → CF → LLVM
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(pm.run(module))) {
        module.emitError();
        throw std::runtime_error("MLIRKernelGen: lowering pipeline failed");
    }
}

// ======================= 主入口 =======================

GeneratedKernel generateFromGraphMLIR(const Graph& graph) {
    static std::once_flag llvm_init_flag;
    std::call_once(llvm_init_flag, []() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    });

    // 每次调用创建独立 MLIRContext（MLIRContext 非线程安全，异步编译需 per-task 隔离）
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    auto context = std::make_shared<mlir::MLIRContext>(registry);
    context->loadDialect<mlir::arith::ArithDialect>();
    context->loadDialect<mlir::scf::SCFDialect>();
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::memref::MemRefDialect>();
    context->loadDialect<mlir::LLVM::LLVMDialect>();
    mlir::registerBuiltinDialectTranslation(*context);
    mlir::registerLLVMDialectTranslation(*context);

    auto module = buildMLIRModule(*context, graph);
    applyLoweringPipeline(*module);

    // 创建 TargetMachine 以启用 LLVM 自动向量化（NEON/SIMD）
    auto tm = std::shared_ptr<llvm::TargetMachine>(
        llvm::EngineBuilder()
            .setEngineKind(llvm::EngineKind::JIT)
            .selectTarget());

    mlir::ExecutionEngineOptions engineOpts;
    if (tm) {
        engineOpts.transformer = mlir::makeOptimizingTransformer(/*optLevel=*/3, /*sizeLevel=*/0,
                                                                  tm.get());
    } else {
        engineOpts.transformer = {};
    }
    engineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOpts);
    if (!maybeEngine)
        throw std::runtime_error("MLIRKernelGen: failed to create ExecutionEngine");

    auto expectedPtr = maybeEngine->get()->lookup("c3_kernel");
    if (!expectedPtr)
        throw std::runtime_error("MLIRKernelGen: failed to lookup c3_kernel");

    GeneratedKernel result;

    // 检查是否为多节点 / 融合 kernel
    const auto& nodes = graph.nodes();
    const Node* compute_node = nullptr;
    for (const auto& node : nodes) {
        bool is_input = false;
        for (size_t in_id : graph.inputs()) {
            if (node.id == in_id) { is_input = true; break; }
        }
        if (!is_input && !node.inputs.empty()) { compute_node = &node; break; }
    }

    // 多节点图
    if (countComputeNodesMLIR(graph) > 1) {
        result.is_multi_node = true;
        result.num_inputs = graph.inputCount();
        result.multi_func = reinterpret_cast<MultiNodeKernelFunc>(*expectedPtr);
        for (const auto& node : nodes) {
            if (isComputeNodeMLIR(node, graph.inputs())) {
                if (std::holds_alternative<MatMulNode>(node.op)) {
                    const auto& mm = std::get<MatMulNode>(node.op);
                    if (mm.lhs_desc.shape.size() == 2 && mm.rhs_desc.shape.size() == 2) {
                        result.M = mm.lhs_desc.shape[0];
                        result.K = mm.lhs_desc.shape[1];
                        result.N = mm.rhs_desc.shape[1];
                    }
                }
                result.elem_n = std::max(result.elem_n, node.out_desc.numel);
            }
        }
    } else if (compute_node && std::holds_alternative<FusedNode>(compute_node->op)) {
        const auto& fnode = std::get<FusedNode>(compute_node->op);
        result.is_fused = true;
        result.num_inputs = fnode.arg_descs.size();
        result.fused_func = reinterpret_cast<FusedKernelFunc>(*expectedPtr);
    } else {
        result.func = reinterpret_cast<C3KernelFunc>(*expectedPtr);
    }

    // engine/tm/context 持有所有权，deleter 确保三者生命周期覆盖 kernel 执行期
    auto engine = std::shared_ptr<mlir::ExecutionEngine>(maybeEngine->release());
    result.handle = nullptr;
    result.deleter = [tm, context, engine]() {};

    if (!result.is_fused && !result.is_multi_node) {
        for (const auto& node : nodes) {
            if (std::holds_alternative<MatMulNode>(node.op)) {
                result.is_matmul = true;
                const auto& mm = std::get<MatMulNode>(node.op);
                const auto& lhs = mm.lhs_desc.shape;
                const auto& rhs = mm.rhs_desc.shape;
                if (lhs.size() == 2 && rhs.size() == 2) {
                    result.M = lhs[0]; result.K = lhs[1]; result.N = rhs[1];
                }
                break;
            }
        }
    }

    return result;
}

} // namespace c3
} // namespace ct
