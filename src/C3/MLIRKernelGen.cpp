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

    // 分块大小：适配 M1/M2 L1D cache (128KB)
    // 64×64×3×4 = 48KB < 128KB，三块矩阵同时驻留 L1D
    const int64_t TILE_M = 64;
    const int64_t TILE_N = 64;
    const int64_t TILE_K = 64;
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
                result = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
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

// ======================= 模块构建 =======================

static mlir::OwningOpRef<mlir::ModuleOp> buildMLIRModule(
    mlir::MLIRContext& context, const Graph& graph)
{
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
            buildElementwiseBinary<mlir::arith::DivFOp>(builder, loc, a, b, out, n);
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
    engineOpts.transformer = mlir::makeOptimizingTransformer(/*optLevel=*/3, /*sizeLevel=*/0,
                                                              tm.get());
    engineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOpts);
    if (!maybeEngine)
        throw std::runtime_error("MLIRKernelGen: failed to create ExecutionEngine");

    auto expectedPtr = maybeEngine->get()->lookup("c3_kernel");
    if (!expectedPtr)
        throw std::runtime_error("MLIRKernelGen: failed to lookup c3_kernel");

    GeneratedKernel result;

    // 检查是否为融合 kernel
    const auto& nodes = graph.nodes();
    const Node* compute_node = nullptr;
    for (const auto& node : nodes) {
        bool is_input = false;
        for (size_t in_id : graph.inputs()) {
            if (node.id == in_id) { is_input = true; break; }
        }
        if (!is_input && !node.inputs.empty()) { compute_node = &node; break; }
    }

    if (compute_node && std::holds_alternative<FusedNode>(compute_node->op)) {
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

    if (!result.is_fused) {
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
