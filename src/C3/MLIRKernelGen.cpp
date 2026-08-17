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
#include "C3/C3Dialect.h"
#include "C3/LinalgElementwiseGen.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// 导入 TableGen 生成的 C3Combine 模式重写实现 (已解耦移入 C3DialectLowering.cpp)

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <variant>

#include "C3/TuningState.h"
#include "C3/JITCache.h"
#include <mlir/Target/LLVMIR/Export.h>

// ======================= Profile timestamps (region fusion 探针) =======================
// 由 test_region_fusion.cpp 通过 extern "C" 引用。JIT kernel 内部应调用
// c3_profile_mark(idx) 写入 g_profile_ts[idx]，调度器据此分析 region fusion
// 内核耗时分布。当前 region fusion 关闭（C3 编译期宏 CT_C3_DISABLE_REGION_FUSION=ON），
// kernel 不会调用 c3_profile_mark，所以这里只提供符号定义让链接通过。
extern "C" {
    uint64_t g_profile_ts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    void c3_profile_mark(int /*idx*/) { /* no-op: region fusion disabled */ }
}

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/Affine/Transforms/Passes.h>
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

std::mutex c3_global_mlir_mutex;

enum class MatMulActivation { None, ReLU, Sigmoid, Tanh };

namespace {
    struct TileCache {
        int64_t tile_m = 0;
        int64_t tile_n = 0;
        int64_t tile_k = 0;
        bool fetched = false;
    };

    inline TileCache& currentTileCache() {
        thread_local TileCache cache;
        if (!cache.fetched) {
            auto t = ct::c3::TuningState::instance().get();
            cache.tile_m = static_cast<int64_t>(t.tile_m);
            cache.tile_n = static_cast<int64_t>(t.tile_n);
            cache.tile_k = static_cast<int64_t>(t.tile_k);
            cache.fetched = true;
        }
        return cache;
    }

    constexpr int64_t kDefaultTileM = 32;
    constexpr int64_t kDefaultTileN = 32;
}

// ======================= 辅助函数 =======================

static mlir::Value indexToI64(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value idx) {
    return builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), idx);
}

static mlir::Value i64ToIndex(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value val) {
    return builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), val);
}

// ======================= 循环/展开辅助函数（Phase C: Shape 特化循环展开） =======================

/// 生成循环或展开的循环体
/// 当 known_numel 是已知的小常量（<=16）时，生成完全展开的代码（无循环开销），
/// 否则生成标准的 scf.for 循环。
/// 对于小张量，循环展开消除了循环控制指令（i++、分支比较、跳转），
/// 使 LLVM 能进一步优化指令流水线和寄存器分配。
static void buildLoop(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value n, int64_t known_numel,
                      const std::function<void(mlir::OpBuilder&, mlir::Location, mlir::Value)>& body_fn) {
    if (known_numel > 0 && known_numel <= 16) {
        // 展开版本：生成 known_numel 个独立的操作序列
        for (int64_t i = 0; i < known_numel; ++i) {
            mlir::Value idx = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
            mlir::Value idx_i64 = indexToI64(builder, loc, idx);
            body_fn(builder, loc, idx_i64);
        }
    } else {
        // 循环版本
        mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
        mlir::Value n_idx = i64ToIndex(builder, loc, n);
        auto loop = builder.create<mlir::scf::ForOp>(loc, c0, n_idx, c1);
        builder.setInsertionPointToStart(loop.getBody());
        mlir::Value idx = loop.getInductionVar();
        mlir::Value idx_i64 = indexToI64(builder, loc, idx);
        body_fn(builder, loc, idx_i64);
        builder.setInsertionPointAfter(loop);
    }
}

/// 在 module 中声明或查找已有的 expf 外部函数
static mlir::LLVM::LLVMFuncOp getOrDeclareExpf(mlir::OpBuilder& builder, mlir::Location loc) {
    auto* ctx = builder.getContext();
    auto module_op = builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
    if (!module_op)
        throw std::runtime_error("getOrDeclareExpf: not inside a module");
    auto existing = module_op.lookupSymbol<mlir::LLVM::LLVMFuncOp>("expf");
    if (existing) return existing;

    auto f32 = mlir::Float32Type::get(ctx);
    auto expf_type = mlir::LLVM::LLVMFunctionType::get(f32, {f32}, false);

    auto saved_ip = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(module_op.getBody());
    auto func = builder.create<mlir::LLVM::LLVMFuncOp>(loc, "expf", expf_type);
    func.setVisibility(mlir::SymbolTable::Visibility::Private);
    builder.restoreInsertionPoint(saved_ip);
    return func;
}

static void buildGt(mlir::OpBuilder& builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs, mlir::Value out, mlir::Value n) {
    // out = (lhs > rhs) ? 1.0f : 0.0f
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();
    
    // 常量: 0, 1
    auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto n_idx = i64ToIndex(builder, loc, n);
    auto zero_f = builder.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
    auto one_f = builder.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(1.0f));

    // [HPC 优化] rhs 是大小为 1 的标量常量（如 0.0f），在循环外仅加载一次，避免循环内重复加载和越界访问
    auto c0_i64 = indexToI64(builder, loc, c0);
    auto rhs_ptr_base = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, rhs, mlir::ValueRange{c0_i64});
    auto rhs_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, rhs_ptr_base);
    
    // 循环: for i in 0..n-1: out[i] = (lhs[i] > rhs) ? 1.0f : 0.0f
    auto loop = builder.create<mlir::scf::ForOp>(loc, c0, n_idx, c1);
    builder.setInsertionPointToStart(loop.getBody());
    
    auto idx = loop.getInductionVar();
    auto idx_i64 = indexToI64(builder, loc, idx);
    auto lhs_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, lhs, mlir::ValueRange{idx_i64});
    auto out_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});
    
    auto lhs_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, lhs_ptr);
    
    // 比较: lhs > rhs
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs_val, rhs_val);
    
    // select: cmp ? 1.0f : 0.0f
    auto result = builder.create<mlir::arith::SelectOp>(loc, f32, cmp, one_f, zero_f);
    builder.create<mlir::LLVM::StoreOp>(loc, result, out_ptr);
    
    // 【关键】恢复插入点到循环之后
    builder.setInsertionPointAfter(loop);
}

// ======================= Fused Kernel 构建 (JIT 3.0 FusedNode 内联实现) =======================
// 注意：逐元素算子(Add/Mul/ReLU/Sigmoid/...)已全面迁移至 C3DialectLowering.cpp 的 JIT 3.0 统一管线。
// 此处仅保留 FusedNode 的标量/向量化循环展开实现，供多节点图直接内联使用。

static void buildFused(mlir::OpBuilder& builder, mlir::Location loc,
                       mlir::Value inputs, mlir::Value out, mlir::Value n,
                       const std::vector<NodeVariant>& ops,
                       const std::vector<std::vector<size_t>>& op_inputs,
                       const std::vector<size_t>& op_node_ids,
                       const std::vector<size_t>& arg_node_ids,
                       int64_t known_numel = 0) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();

    // 构建 node_id → arg_index 的映射
    std::unordered_map<size_t, size_t> node_to_arg;
    for (size_t i = 0; i < arg_node_ids.size(); ++i) {
        node_to_arg[arg_node_ids[i]] = i;
    }

    // 循环外：预加载所有外部输入指针（循环不变量提升）
    std::set<size_t> referenced_nodes;
    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        const auto& inputs_for_op = op_inputs[op_idx];
        for (size_t in_id : inputs_for_op) {
            if (node_to_arg.find(in_id) != node_to_arg.end()) {
                referenced_nodes.insert(in_id);
            }
        }
    }

    // 为每个引用的外部输入预加载指针
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
    buildLoop(builder, loc, n, known_numel,
        [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx_i64) {
            /// 从预加载的指针加载元素值
            auto loadExternal = [&](size_t node_id) -> mlir::Value {
                mlir::Value ptr = preloaded_ptrs.at(node_id);
                mlir::Value elem_addr = b.create<mlir::LLVM::GEPOp>(
                    loc, ptr_type, f32, ptr, mlir::ValueRange{idx_i64});
                return b.create<mlir::LLVM::LoadOp>(loc, f32, elem_addr);
            };

            std::unordered_map<size_t, mlir::Value> val_map;
            auto getValue = [&](size_t node_id) -> mlir::Value {
                auto it = val_map.find(node_id);
                if (it != val_map.end()) {
                    return it->second;
                }
                if (node_to_arg.find(node_id) != node_to_arg.end()) {
                    mlir::Value val = loadExternal(node_id);
                    val_map[node_id] = val;
                    return val;
                }
                throw std::runtime_error("buildFused: node_id " + std::to_string(node_id) + " not found in val_map or arg_node_ids");
            };

            for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
                const NodeVariant& op = ops[op_idx];
                const auto& inputs_for_op = op_inputs[op_idx];
                bool is_last = (op_idx == ops.size() - 1);

                mlir::Value result;
                std::visit([&](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    mlir::Value lhs, rhs;

                    if constexpr (std::is_same_v<T, NegNode>) {
                        lhs = getValue(inputs_for_op[0]);
                        result = b.create<mlir::arith::NegFOp>(loc, lhs);
                    } else if constexpr (std::is_same_v<T, ReLUNode>) {
                        lhs = getValue(inputs_for_op[0]);
                        mlir::Value zero = mlir::arith::ConstantFloatOp::create(b, loc, f32, llvm::APFloat(0.0f));
                        result = b.create<mlir::arith::MaxNumFOp>(loc, lhs, zero);
                    } else if constexpr (std::is_same_v<T, AddNode>) {
                        lhs = getValue(inputs_for_op[0]);
                        rhs = getValue(inputs_for_op[1]);
                        result = b.create<mlir::arith::AddFOp>(loc, lhs, rhs);
                    } else if constexpr (std::is_same_v<T, SubNode>) {
                        lhs = getValue(inputs_for_op[0]);
                        rhs = getValue(inputs_for_op[1]);
                        result = b.create<mlir::arith::SubFOp>(loc, lhs, rhs);
                    } else if constexpr (std::is_same_v<T, MulNode>) {
                        lhs = getValue(inputs_for_op[0]);
                        rhs = getValue(inputs_for_op[1]);
                        result = b.create<mlir::arith::MulFOp>(loc, lhs, rhs);
                    } else if constexpr (std::is_same_v<T, DivNode>) {
                        lhs = getValue(inputs_for_op[0]);
                        rhs = getValue(inputs_for_op[1]);
                        mlir::Value zero_c = mlir::arith::ConstantFloatOp::create(b, loc, f32, llvm::APFloat(0.0f));
                        mlir::Value is_zero = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, rhs, zero_c);
                        auto div_if = b.create<mlir::scf::IfOp>(loc, f32, is_zero, true);
                        b.setInsertionPointToStart(&div_if.getThenRegion().front());
                        mlir::Value nan_v = mlir::arith::ConstantFloatOp::create(
                            b, loc, f32, llvm::APFloat::getNaN(llvm::APFloat::IEEEsingle()));
                        b.create<mlir::scf::YieldOp>(loc, nan_v);
                        b.setInsertionPointToStart(&div_if.getElseRegion().front());
                        mlir::Value div_r = b.create<mlir::arith::DivFOp>(loc, lhs, rhs);
                        b.create<mlir::scf::YieldOp>(loc, div_r);
                        b.setInsertionPointAfter(div_if);
                        result = div_if.getResult(0);
                    } else if constexpr (std::is_same_v<T, SigmoidNode>) {
                        lhs = getValue(inputs_for_op[0]);
                        mlir::Value neg_x = b.create<mlir::arith::NegFOp>(loc, lhs);
                        auto expf_func = getOrDeclareExpf(b, loc);
                        mlir::Value exp_x = b.create<mlir::LLVM::CallOp>(loc, expf_func, mlir::ValueRange{neg_x}).getResult();
                        mlir::Value one = mlir::arith::ConstantFloatOp::create(b, loc, f32, llvm::APFloat(1.0f));
                        mlir::Value denom = b.create<mlir::arith::AddFOp>(loc, one, exp_x);
                        result = b.create<mlir::arith::DivFOp>(loc, one, denom);
                    } else if constexpr (std::is_same_v<T, TanhNode>) {
                        lhs = getValue(inputs_for_op[0]);
                        auto expf_func = getOrDeclareExpf(b, loc);
                        mlir::Value exp_x = b.create<mlir::LLVM::CallOp>(loc, expf_func, mlir::ValueRange{lhs}).getResult();
                        mlir::Value neg_x = b.create<mlir::arith::NegFOp>(loc, lhs);
                        mlir::Value exp_neg_x = b.create<mlir::LLVM::CallOp>(loc, expf_func, mlir::ValueRange{neg_x}).getResult();
                        mlir::Value num = b.create<mlir::arith::SubFOp>(loc, exp_x, exp_neg_x);
                        mlir::Value denom = b.create<mlir::arith::AddFOp>(loc, exp_x, exp_neg_x);
                        result = b.create<mlir::arith::DivFOp>(loc, num, denom);
                    }
                }, op);

                val_map[op_node_ids[op_idx]] = result;

                if (is_last) {
                    mlir::Value out_ptr = b.create<mlir::LLVM::GEPOp>(
                        loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});
                    b.create<mlir::LLVM::StoreOp>(loc, result, out_ptr);
                }
            }
        });
}

// ======================= 多节点融合 Kernel 构建 =======================

static bool isFusedChainVectorizable(const std::vector<NodeVariant>& ops,
                                     const std::vector<std::vector<size_t>>& op_inputs,
                                     const std::unordered_map<size_t, int64_t>& arg_numels,
                                     size_t n) {
    if (std::getenv("C3_MLIR_NO_VECTORIZE") != nullptr) return false;
    for (const auto& inputs : op_inputs) {
        for (size_t in_id : inputs) {
            auto it = arg_numels.find(in_id);
            if (it != arg_numels.end() && it->second > 0 && it->second < (int64_t)n) {
                return false;
            }
        }
    }
    for (const auto& op : ops) {
        bool op_ok = std::visit([](auto&& arg) -> bool {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, AddNode> ||
                          std::is_same_v<T, SubNode> ||
                          std::is_same_v<T, MulNode> ||
                          std::is_same_v<T, NegNode> ||
                          std::is_same_v<T, ReLUNode> ||
                          std::is_same_v<T, GtNode> ||
                          std::is_same_v<T, SigmoidNode> ||
                          std::is_same_v<T, TanhNode> ||
                          std::is_same_v<T, ExpNode> ||
                          std::is_same_v<T, LogNode> ||
                          std::is_same_v<T, DivNode>) {
                return true;
            }
            return false;
        }, op);
        if (!op_ok) return false;
    }
    return true;
}

static void buildFusedMultiNodeVectorized(mlir::OpBuilder& builder, mlir::Location loc,
                                          mlir::Value out, mlir::Value n,
                                          const std::vector<NodeVariant>& ops,
                                          const std::vector<std::vector<size_t>>& op_inputs,
                                          const std::vector<size_t>& arg_node_ids,
                                          const std::unordered_map<size_t, mlir::Value>& arg_ptrs,
                                          const std::unordered_map<size_t, int64_t>& arg_numels = {}) {
    constexpr int64_t VL = 8;
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();
    auto vec_ty = mlir::VectorType::get({VL}, f32);

    std::set<size_t> referenced_nodes;
    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        const auto& inputs_for_op = op_inputs[op_idx];
        for (size_t in_id : inputs_for_op) {
            if (op_idx > 0 && in_id == op_inputs[op_idx][0]) continue;
            referenced_nodes.insert(in_id);
        }
    }

    std::unordered_map<size_t, mlir::Value> preloaded_ptrs;
    for (size_t node_id : referenced_nodes) {
        auto it = arg_ptrs.find(node_id);
        if (it != arg_ptrs.end()) {
            preloaded_ptrs[node_id] = it->second;
        }
    }

    mlir::Value c0_i = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1_i = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value VL_i = builder.create<mlir::arith::ConstantIndexOp>(loc, VL);

    mlir::Value n_idx = i64ToIndex(builder, loc, n);
    mlir::Value rem = builder.create<mlir::arith::RemUIOp>(loc, n_idx, VL_i);
    mlir::Value n_vec = builder.create<mlir::arith::SubIOp>(loc, n_idx, rem);

    mlir::Value zero_vec = builder.create<mlir::arith::ConstantOp>(
        loc, mlir::DenseElementsAttr::get(
            vec_ty, llvm::ArrayRef<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    mlir::Value one_vec = builder.create<mlir::arith::ConstantOp>(
        loc, mlir::DenseElementsAttr::get(
            vec_ty, llvm::ArrayRef<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));

    auto vloop = builder.create<mlir::scf::ForOp>(loc, c0_i, n_vec, VL_i);
    builder.setInsertionPointToStart(vloop.getBody());
    mlir::Value base = indexToI64(builder, loc, vloop.getInductionVar());

    auto loadExternalVector = [&](size_t node_id) -> mlir::Value {
        mlir::Value ptr = preloaded_ptrs.at(node_id);
        mlir::Value addr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, ptr, mlir::ValueRange{base});
        return builder.create<mlir::LLVM::LoadOp>(loc, vec_ty, addr);
    };

    mlir::Value prev_val_v;

    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        const NodeVariant& op = ops[op_idx];
        const auto& inputs_for_op = op_inputs[op_idx];
        bool is_last = (op_idx == ops.size() - 1);

        std::vector<size_t> ext_inputs;
        for (size_t in_id : inputs_for_op) {
            if (op_idx > 0 && in_id == inputs_for_op[0]) continue;
            ext_inputs.push_back(in_id);
        }

        mlir::Value result_v;
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            mlir::Value lhs, rhs;

            if constexpr (std::is_same_v<T, NegNode>) {
                lhs = (op_idx > 0) ? prev_val_v : loadExternalVector(ext_inputs[0]);
                result_v = builder.create<mlir::arith::NegFOp>(loc, lhs);
            } else if constexpr (std::is_same_v<T, ReLUNode>) {
                lhs = (op_idx > 0) ? prev_val_v : loadExternalVector(ext_inputs[0]);
                result_v = builder.create<mlir::arith::MaxNumFOp>(loc, lhs, zero_vec);
            } else if constexpr (std::is_same_v<T, AddNode>) {
                if (op_idx > 0) { lhs = prev_val_v; rhs = loadExternalVector(ext_inputs[0]); }
                else { lhs = loadExternalVector(ext_inputs[0]); rhs = loadExternalVector(ext_inputs[1]); }
                result_v = builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
            } else if constexpr (std::is_same_v<T, SubNode>) {
                if (op_idx > 0) { lhs = prev_val_v; rhs = loadExternalVector(ext_inputs[0]); }
                else { lhs = loadExternalVector(ext_inputs[0]); rhs = loadExternalVector(ext_inputs[1]); }
                result_v = builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
            } else if constexpr (std::is_same_v<T, MulNode>) {
                if (op_idx > 0) { lhs = prev_val_v; rhs = loadExternalVector(ext_inputs[0]); }
                else { lhs = loadExternalVector(ext_inputs[0]); rhs = loadExternalVector(ext_inputs[1]); }
                result_v = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
            } else if constexpr (std::is_same_v<T, GtNode>) {
                if (op_idx > 0) { lhs = prev_val_v; rhs = loadExternalVector(ext_inputs[0]); }
                else { lhs = loadExternalVector(ext_inputs[0]); rhs = loadExternalVector(ext_inputs[1]); }
                mlir::Value cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
                result_v = builder.create<mlir::arith::SelectOp>(loc, cmp, one_vec, zero_vec);
            } else if constexpr (std::is_same_v<T, SigmoidNode>) {
                lhs = (op_idx > 0) ? prev_val_v : loadExternalVector(ext_inputs[0]);
                mlir::Value neg_x = builder.create<mlir::arith::NegFOp>(loc, lhs);
                mlir::Value exp_x = builder.create<mlir::math::ExpOp>(loc, neg_x);
                mlir::Value denom = builder.create<mlir::arith::AddFOp>(loc, one_vec, exp_x);
                result_v = builder.create<mlir::arith::DivFOp>(loc, one_vec, denom);
            } else if constexpr (std::is_same_v<T, TanhNode>) {
                lhs = (op_idx > 0) ? prev_val_v : loadExternalVector(ext_inputs[0]);
                result_v = builder.create<mlir::math::TanhOp>(loc, lhs);
            } else if constexpr (std::is_same_v<T, ExpNode>) {
                lhs = (op_idx > 0) ? prev_val_v : loadExternalVector(ext_inputs[0]);
                result_v = builder.create<mlir::math::ExpOp>(loc, lhs);
            } else if constexpr (std::is_same_v<T, LogNode>) {
                lhs = (op_idx > 0) ? prev_val_v : loadExternalVector(ext_inputs[0]);
                result_v = builder.create<mlir::math::LogOp>(loc, lhs);
            } else if constexpr (std::is_same_v<T, DivNode>) {
                if (op_idx > 0) { lhs = prev_val_v; rhs = loadExternalVector(ext_inputs[0]); }
                else { lhs = loadExternalVector(ext_inputs[0]); rhs = loadExternalVector(ext_inputs[1]); }
                result_v = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
            }
        }, op);

        if (is_last) {
            mlir::Value out_addr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{base});
            builder.create<mlir::LLVM::StoreOp>(loc, result_v, out_addr);
        } else {
            prev_val_v = result_v;
        }
    }
    builder.setInsertionPointAfter(vloop);

    auto tloop = builder.create<mlir::scf::ForOp>(loc, n_vec, n_idx, c1_i);
    builder.setInsertionPointToStart(tloop.getBody());
    mlir::Value idx = indexToI64(builder, loc, tloop.getInductionVar());

    auto loadExternalScalar = [&](size_t node_id) -> mlir::Value {
        mlir::Value ptr = preloaded_ptrs.at(node_id);
        mlir::Value addr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, ptr, mlir::ValueRange{idx});
        return builder.create<mlir::LLVM::LoadOp>(loc, f32, addr);
    };

    mlir::Value prev_val_s;

    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        const NodeVariant& op = ops[op_idx];
        const auto& inputs_for_op = op_inputs[op_idx];
        bool is_last = (op_idx == ops.size() - 1);

        std::vector<size_t> ext_inputs;
        for (size_t in_id : inputs_for_op) {
            if (op_idx > 0 && in_id == inputs_for_op[0]) continue;
            ext_inputs.push_back(in_id);
        }

        mlir::Value result_s;
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            mlir::Value lhs, rhs;

            if constexpr (std::is_same_v<T, NegNode>) {
                lhs = (op_idx > 0) ? prev_val_s : loadExternalScalar(ext_inputs[0]);
                result_s = builder.create<mlir::arith::NegFOp>(loc, lhs);
            } else if constexpr (std::is_same_v<T, ReLUNode>) {
                lhs = (op_idx > 0) ? prev_val_s : loadExternalScalar(ext_inputs[0]);
                mlir::Value zero = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(0.0f));
                result_s = builder.create<mlir::arith::MaxNumFOp>(loc, lhs, zero);
            } else if constexpr (std::is_same_v<T, AddNode>) {
                if (op_idx > 0) { lhs = prev_val_s; rhs = loadExternalScalar(ext_inputs[0]); }
                else { lhs = loadExternalScalar(ext_inputs[0]); rhs = loadExternalScalar(ext_inputs[1]); }
                result_s = builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
            } else if constexpr (std::is_same_v<T, SubNode>) {
                if (op_idx > 0) { lhs = prev_val_s; rhs = loadExternalScalar(ext_inputs[0]); }
                else { lhs = loadExternalScalar(ext_inputs[0]); rhs = loadExternalScalar(ext_inputs[1]); }
                result_s = builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
            } else if constexpr (std::is_same_v<T, MulNode>) {
                if (op_idx > 0) { lhs = prev_val_s; rhs = loadExternalScalar(ext_inputs[0]); }
                else { lhs = loadExternalScalar(ext_inputs[0]); rhs = loadExternalScalar(ext_inputs[1]); }
                result_s = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
            } else if constexpr (std::is_same_v<T, GtNode>) {
                if (op_idx > 0) { lhs = prev_val_s; rhs = loadExternalScalar(ext_inputs[0]); }
                else { lhs = loadExternalScalar(ext_inputs[0]); rhs = loadExternalScalar(ext_inputs[1]); }
                mlir::Value cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
                mlir::Value zero_f = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(0.0f));
                mlir::Value one_f = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(1.0f));
                result_s = builder.create<mlir::arith::SelectOp>(loc, cmp, zero_f, one_f);
            } else if constexpr (std::is_same_v<T, SigmoidNode>) {
                lhs = (op_idx > 0) ? prev_val_s : loadExternalScalar(ext_inputs[0]);
                mlir::Value neg_x = builder.create<mlir::arith::NegFOp>(loc, lhs);
                mlir::Value exp_x = builder.create<mlir::math::ExpOp>(loc, neg_x);
                mlir::Value one = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(1.0f));
                mlir::Value denom = builder.create<mlir::arith::AddFOp>(loc, one, exp_x);
                result_s = builder.create<mlir::arith::DivFOp>(loc, one, denom);
            } else if constexpr (std::is_same_v<T, TanhNode>) {
                lhs = (op_idx > 0) ? prev_val_s : loadExternalScalar(ext_inputs[0]);
                result_s = builder.create<mlir::math::TanhOp>(loc, lhs);
            } else if constexpr (std::is_same_v<T, ExpNode>) {
                lhs = (op_idx > 0) ? prev_val_s : loadExternalScalar(ext_inputs[0]);
                result_s = builder.create<mlir::math::ExpOp>(loc, lhs);
            } else if constexpr (std::is_same_v<T, LogNode>) {
                lhs = (op_idx > 0) ? prev_val_s : loadExternalScalar(ext_inputs[0]);
                result_s = builder.create<mlir::math::LogOp>(loc, lhs);
            } else if constexpr (std::is_same_v<T, DivNode>) {
                if (op_idx > 0) { lhs = prev_val_s; rhs = loadExternalScalar(ext_inputs[0]); }
                else { lhs = loadExternalScalar(ext_inputs[0]); rhs = loadExternalScalar(ext_inputs[1]); }
                result_s = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
            }
        }, op);

        if (is_last) {
            mlir::Value out_addr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx});
            builder.create<mlir::LLVM::StoreOp>(loc, result_s, out_addr);
        } else {
            prev_val_s = result_s;
        }
    }
    builder.setInsertionPointAfter(tloop);
}

/// 在多节点 MLIR kernel 中生成融合节点的循环代码
/// @details 每个融合节点包含一个 element-wise 操作链，在单次循环中按顺序执行。
///          输入指针已由调用方解析（可能是外部输入或中间缓冲区），直接使用。
static void buildFusedMultiNode(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value out, mlir::Value n,
                                const std::vector<NodeVariant>& ops,
                                const std::vector<std::vector<size_t>>& op_inputs,
                                const std::vector<size_t>& arg_node_ids,
                                const std::unordered_map<size_t, mlir::Value>& arg_ptrs,
                                const std::unordered_map<size_t, int64_t>& arg_numels = {},
                                int64_t known_numel = 0) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();

    // 构建 node_id → arg_index 的映射
    std::unordered_map<size_t, size_t> node_to_arg;
    for (size_t i = 0; i < arg_node_ids.size(); ++i) {
        node_to_arg[arg_node_ids[i]] = i;
    }

    // 循环外：预加载所有被引用的外部输入指针
    std::set<size_t> referenced_nodes;
    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        const auto& inputs_for_op = op_inputs[op_idx];
        for (size_t in_id : inputs_for_op) {
            if (op_idx > 0 && in_id == op_inputs[op_idx][0]) continue;
            referenced_nodes.insert(in_id);
        }
    }

    // 预加载指针到局部变量（循环不变量提升）
    std::unordered_map<size_t, mlir::Value> preloaded_ptrs;
    for (size_t node_id : referenced_nodes) {
        auto it = arg_ptrs.find(node_id);
        if (it != arg_ptrs.end()) {
            preloaded_ptrs[node_id] = it->second;
        }
    }

    // 检查是否需要广播：如果有任何输入 numel < 已知 numel，则不能展开
    bool needs_broadcast = false;
    if (known_numel > 0) {
        for (size_t node_id : referenced_nodes) {
            auto it = arg_numels.find(node_id);
            if (it != arg_numels.end() && it->second > 0 && it->second < known_numel) {
                needs_broadcast = true;
                break;
            }
        }
    }
    int64_t effective_known = needs_broadcast ? 0 : known_numel;

    // === 循环体 ===
    buildLoop(builder, loc, n, effective_known,
        [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx_i64) {
            /// 从预加载的指针加载元素值（支持广播）
            auto loadExternal = [&](size_t node_id) -> mlir::Value {
                mlir::Value ptr = preloaded_ptrs.at(node_id);
                mlir::Value load_idx = idx_i64;
                auto it = arg_numels.find(node_id);
                if (it != arg_numels.end() && it->second > 0) {
                    mlir::Value node_numel = b.create<mlir::arith::ConstantIntOp>(
                        loc, it->second, 64);
                    mlir::Value need_broadcast = b.create<mlir::arith::CmpIOp>(
                        loc, mlir::arith::CmpIPredicate::ult, node_numel, n);
                    auto if_broadcast = b.create<mlir::scf::IfOp>(loc, b.getI64Type(), need_broadcast, true);
                    b.setInsertionPointToStart(&if_broadcast.getThenRegion().front());
                    mlir::Value mod_idx = b.create<mlir::arith::RemUIOp>(loc, idx_i64, node_numel);
                    b.create<mlir::scf::YieldOp>(loc, mod_idx);
                    b.setInsertionPointToStart(&if_broadcast.getElseRegion().front());
                    b.create<mlir::scf::YieldOp>(loc, idx_i64);
                    b.setInsertionPointAfter(if_broadcast);
                    load_idx = if_broadcast.getResult(0);
                }
                mlir::Value elem_addr = b.create<mlir::LLVM::GEPOp>(
                    loc, ptr_type, f32, ptr, mlir::ValueRange{load_idx});
                return b.create<mlir::LLVM::LoadOp>(loc, f32, elem_addr);
            };

            mlir::Value prev_val;

            for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
                const NodeVariant& op = ops[op_idx];
                const auto& inputs_for_op = op_inputs[op_idx];
                bool is_last = (op_idx == ops.size() - 1);

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
                        result = b.create<mlir::arith::NegFOp>(loc, lhs);
                    } else if constexpr (std::is_same_v<T, ReLUNode>) {
                        lhs = (op_idx > 0) ? prev_val : loadExternal(ext_inputs[0]);
                        mlir::Value zero = mlir::arith::ConstantFloatOp::create(b, loc, f32, llvm::APFloat(0.0f));
                        result = b.create<mlir::arith::MaxNumFOp>(loc, lhs, zero);
                    } else if constexpr (std::is_same_v<T, SigmoidNode>) {
                        lhs = (op_idx > 0) ? prev_val : loadExternal(ext_inputs[0]);
                        mlir::Value neg_x = b.create<mlir::arith::NegFOp>(loc, lhs);
                        auto expf_func = getOrDeclareExpf(b, loc);
                        mlir::Value exp_x = b.create<mlir::LLVM::CallOp>(loc, expf_func, mlir::ValueRange{neg_x}).getResult();
                        mlir::Value one = mlir::arith::ConstantFloatOp::create(b, loc, f32, llvm::APFloat(1.0f));
                        mlir::Value denom = b.create<mlir::arith::AddFOp>(loc, one, exp_x);
                        result = b.create<mlir::arith::DivFOp>(loc, one, denom);
                    } else if constexpr (std::is_same_v<T, TanhNode>) {
                        lhs = (op_idx > 0) ? prev_val : loadExternal(ext_inputs[0]);
                        auto expf_func = getOrDeclareExpf(b, loc);
                        mlir::Value exp_x = b.create<mlir::LLVM::CallOp>(loc, expf_func, mlir::ValueRange{lhs}).getResult();
                        mlir::Value neg_x = b.create<mlir::arith::NegFOp>(loc, lhs);
                        mlir::Value exp_neg_x = b.create<mlir::LLVM::CallOp>(loc, expf_func, mlir::ValueRange{neg_x}).getResult();
                        mlir::Value num = b.create<mlir::arith::SubFOp>(loc, exp_x, exp_neg_x);
                        mlir::Value denom = b.create<mlir::arith::AddFOp>(loc, exp_x, exp_neg_x);
                        result = b.create<mlir::arith::DivFOp>(loc, num, denom);
                    } else if constexpr (std::is_same_v<T, AddNode>) {
                        if (op_idx > 0) { lhs = prev_val; rhs = loadExternal(ext_inputs[0]); }
                        else { lhs = loadExternal(ext_inputs[0]); rhs = loadExternal(ext_inputs[1]); }
                        result = b.create<mlir::arith::AddFOp>(loc, lhs, rhs);
                    } else if constexpr (std::is_same_v<T, SubNode>) {
                        if (op_idx > 0) { lhs = prev_val; rhs = loadExternal(ext_inputs[0]); }
                        else { lhs = loadExternal(ext_inputs[0]); rhs = loadExternal(ext_inputs[1]); }
                        result = b.create<mlir::arith::SubFOp>(loc, lhs, rhs);
                    } else if constexpr (std::is_same_v<T, MulNode>) {
                        if (op_idx > 0) { lhs = prev_val; rhs = loadExternal(ext_inputs[0]); }
                        else { lhs = loadExternal(ext_inputs[0]); rhs = loadExternal(ext_inputs[1]); }
                        result = b.create<mlir::arith::MulFOp>(loc, lhs, rhs);
                    } else if constexpr (std::is_same_v<T, DivNode>) {
                        if (op_idx > 0) { lhs = prev_val; rhs = loadExternal(ext_inputs[0]); }
                        else { lhs = loadExternal(ext_inputs[0]); rhs = loadExternal(ext_inputs[1]); }
                        mlir::Value zero_c = mlir::arith::ConstantFloatOp::create(b, loc, f32, llvm::APFloat(0.0f));
                        mlir::Value is_zero = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, rhs, zero_c);
                        auto div_if = b.create<mlir::scf::IfOp>(loc, f32, is_zero, true);
                        b.setInsertionPointToStart(&div_if.getThenRegion().front());
                        mlir::Value nan_val = mlir::arith::ConstantFloatOp::create(
                            b, loc, f32, llvm::APFloat::getNaN(llvm::APFloat::IEEEsingle()));
                        b.create<mlir::scf::YieldOp>(loc, nan_val);
                        b.setInsertionPointToStart(&div_if.getElseRegion().front());
                        mlir::Value div_result = b.create<mlir::arith::DivFOp>(loc, lhs, rhs);
                        b.create<mlir::scf::YieldOp>(loc, div_result);
                        b.setInsertionPointAfter(div_if);
                        result = div_if.getResult(0);
                    }
                    (void)result;
                }, op);

                if (is_last) {
                    mlir::Value out_addr = b.create<mlir::LLVM::GEPOp>(
                        loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});
                    b.create<mlir::LLVM::StoreOp>(loc, result, out_addr);
                } else {
                    prev_val = result;
                }
            }
        });
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

    // 步骤 3: 分配缓冲区索引，记录每个缓冲区的 numel
    // 支持多输出（反向融合图）：输出节点写入 output 平面 buffer 的对应段（output_index * elem_n）
    std::unordered_map<size_t, size_t> node_to_buffer;
    std::unordered_map<size_t, size_t> output_index; // 输出节点 id → 平面 buffer 段索引（按 graph.outputs() 顺序）
    std::vector<size_t> buffer_numels; // buffer index → numel
    size_t num_intermediates = 0;

    // 为图中的所有常量节点分配专属的（不参与复用的）缓冲区索引
    std::unordered_map<size_t, size_t> const_to_pool_idx;
    size_t num_constants = 0;
    for (const auto& node : nodes) {
        if (std::holds_alternative<ConstNode>(node.op)) {
            const_to_pool_idx[node.id] = num_constants++;
        }
    }

    auto assign_output_mlir = [&](size_t node_id) {
        if (output_index.count(node_id)) return; // 去重
        const size_t seg = output_index.size();
        output_index[node_id] = seg;
        node_to_buffer[node_id] = SIZE_MAX;
    };
    for (size_t i = 0; i < compute_nodes.size(); ++i) {
        size_t node_id = compute_nodes[i]->id;
        bool is_output = false;
        for (size_t out_id : outputs) {
            if (node_id == out_id) { is_output = true; break; }
        }
        if (is_output) {
            assign_output_mlir(node_id);
        } else {
            node_to_buffer[node_id] = num_intermediates++;
            buffer_numels.push_back(compute_nodes[i]->out_desc.numel);
        }
    }
    // 安全网：确保最后一个计算节点总是输出
    assign_output_mlir(compute_nodes.back()->id);

    // 确定 elem_n（最大输出元素数，作为多输出平面偏移的步长）
    size_t elem_n = 0;
    for (const auto* node : compute_nodes) {
        elem_n = std::max(elem_n, node->out_desc.numel);
    }

    // 步骤 3a: Buffer 原地复用分析
    // 当 FusedNode 紧跟在 MatMul（或其他单输出计算节点）之后，
    // 且输出 shape 相同时，FusedNode 可以原地复用前驱节点的 buffer。
    // 这消除了中间 buffer 的分配和内存流量。
    // node_buffer_reuse[node_id] = 被复用的前驱 buffer 索引
    std::unordered_map<size_t, size_t> node_buffer_reuse;
    std::unordered_set<size_t> reused_source_nodes; // 被复用的前驱节点 ID
    for (size_t ci = 0; ci + 1 < compute_nodes.size(); ++ci) {
        const Node* cur = compute_nodes[ci];
        const Node* next = compute_nodes[ci + 1];
        size_t cur_id = cur->id;
        size_t next_id = next->id;

        // 当前节点必须有 buffer（不是最后节点）且不是 FusedNode
        auto cur_buf_it = node_to_buffer.find(cur_id);
        if (cur_buf_it == node_to_buffer.end()) continue;
        if (cur_buf_it->second == SIZE_MAX) continue;
        if (std::holds_alternative<FusedNode>(cur->op)) continue;

        // 下一个节点必须是 FusedNode
        if (!std::holds_alternative<FusedNode>(next->op)) continue;

        // 检查 FusedNode 是否消费了当前节点
        bool consumes_cur = false;
        for (size_t in_id : next->inputs) {
            if (in_id == cur_id) { consumes_cur = true; break; }
        }
        if (!consumes_cur) continue;

        // 检查输出 shape 是否匹配（numel 相同即可原地复用）
        if (next->out_desc.numel != cur->out_desc.numel) continue;

        // 复用 buffer！
        node_buffer_reuse[next_id] = cur_buf_it->second;
        reused_source_nodes.insert(cur_id);
    }

    // 步骤 4: 创建函数 (MultiNodeKernelFunc 签名)
    auto func_type = builder.getFunctionType(
        {ptr_type, ptr_type, i64_type, i64_type, i64_type, i64_type, ptr_type}, {});
    auto func = builder.create<mlir::func::FuncOp>(loc, "c3_kernel", func_type);
    func.setArgAttr(0, "llvm.noalias", builder.getUnitAttr());
    func.setArgAttr(1, "llvm.noalias", builder.getUnitAttr());
    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    mlir::Value inputs_ptr = entry->getArgument(0);  // const float* const*
    mlir::Value out_ptr = entry->getArgument(1);       // float*
    mlir::Value n_val = entry->getArgument(2);          // n
    mlir::Value M_val = entry->getArgument(3);          // M
    mlir::Value K_val = entry->getArgument(4);          // K
    mlir::Value N_val = entry->getArgument(5);          // N
    mlir::Value scratchpad_ptr = entry->getArgument(6); // float* (scratchpad)

    // 步骤 5: 分配 Buffer Pool（跨层复用）
    // 在串行图（如 MLP）中，中间 buffer 不会被同时使用，
    // 因此可以用一个 pool 替代 N 个独立 malloc，减少分配/释放开销。
    // 为安全起见，分配 2 个 pool buffer（同时读/写场景）。
    size_t max_numel = 0;
    for (size_t i = 0; i < num_intermediates; ++i) {
        if (buffer_numels[i] > max_numel) max_numel = buffer_numels[i];
    }
    size_t pool_buf_count = (num_intermediates > 0)
        ? std::min(num_intermediates, (size_t)2)
        : 0;
    std::vector<mlir::Value> tmp_buffers;
    if (pool_buf_count > 0) {
        for (size_t pi = 0; pi < pool_buf_count; ++pi) {
            int64_t offset = (int64_t)(pi * max_numel);
            mlir::Value offset_val = builder.create<mlir::arith::ConstantIntOp>(loc, offset, 64);
            mlir::Value buf = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, scratchpad_ptr, mlir::ValueRange{offset_val});
            tmp_buffers.push_back(buf);
        }
    }

    // 在 pool buffers 后面追加常量的专属 buffer (不参与逻辑 buffer 的分配和复用)
    size_t const_buf_start_idx = tmp_buffers.size();
    for (size_t ci = 0; ci < num_constants; ++ci) {
        // 每个常量只占用 1 个 float 空间，偏移量接着 pool buffers 后面
        int64_t offset = (int64_t)(pool_buf_count * max_numel + ci);
        mlir::Value offset_val = builder.create<mlir::arith::ConstantIntOp>(loc, offset, 64);
        mlir::Value buf = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, scratchpad_ptr, mlir::ValueRange{offset_val});
        tmp_buffers.push_back(buf);
    }

    // 逻辑 buffer → pool buffer 的映射（交替分配，确保串行图的正确性）
    // logical_buf_idx → pool_buf_idx (0 或 1)
    std::vector<size_t> logical_to_pool(num_intermediates, SIZE_MAX);
    for (size_t i = 0; i < num_intermediates; ++i) {
        logical_to_pool[i] = i % std::max(pool_buf_count, (size_t)1);
    }

    // 写入所有常量节点的初始值
    for (const auto& node : nodes) {
        auto const_it = const_to_pool_idx.find(node.id);
        if (const_it != const_to_pool_idx.end()) {
            const auto& const_op = std::get<ConstNode>(node.op);
            float value = const_op.value;
            mlir::Value buf = tmp_buffers[const_buf_start_idx + const_it->second];
            
            mlir::Value c0_i64 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 64);
            mlir::Value addr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, buf, mlir::ValueRange{c0_i64});
            mlir::Value val = builder.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(value));
            builder.create<mlir::LLVM::StoreOp>(loc, val, addr);
        }
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
        auto const_it = const_to_pool_idx.find(in_node_id);
        if (const_it != const_to_pool_idx.end()) {
            return tmp_buffers[const_buf_start_idx + const_it->second];
        }
        auto buf_it = node_to_buffer.find(in_node_id);
        if (buf_it != node_to_buffer.end()) {
            if (buf_it->second == SIZE_MAX) {
                auto oi = output_index.find(in_node_id);
                size_t seg = (oi != output_index.end()) ? oi->second : 0;
                if (seg == 0) return out_ptr;
                mlir::Value offset_val = builder.create<mlir::arith::ConstantIntOp>(loc, (int64_t)(seg * elem_n), 64);
                return builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out_ptr, mlir::ValueRange{offset_val});
            }
            size_t pool_idx = logical_to_pool[buf_it->second];
            if (pool_idx < tmp_buffers.size()) return tmp_buffers[pool_idx];
            return out_ptr; // fallback
        }
        return out_ptr; // fallback
    };

    // 步骤 6: 生成每个计算节点的 MLIR 代码
    for (size_t ci = 0; ci < compute_nodes.size(); ++ci) {
        const Node* node = compute_nodes[ci];
        // 确定输出 buffer：输出节点 → 写入 output 平面 buffer 对应段（output_index * elem_n）；
        //   否则写中间 tmp buffer（优先原地复用）。
        mlir::Value out_buf;
        auto oci = output_index.find(node->id);
        if (oci != output_index.end()) {
            size_t seg = oci->second;
            if (seg == 0) {
                out_buf = out_ptr;
            } else {
                mlir::Value offset_val = builder.create<mlir::arith::ConstantIntOp>(loc, (int64_t)(seg * elem_n), 64);
                out_buf = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out_ptr, mlir::ValueRange{offset_val});
            }
        } else {
            auto reuse_it = node_buffer_reuse.find(node->id);
            if (reuse_it != node_buffer_reuse.end()) {
                // 原地复用前驱节点的 buffer（pool buffer 索引）
                size_t pool_idx = logical_to_pool[reuse_it->second];
                out_buf = (pool_idx < tmp_buffers.size()) ? tmp_buffers[pool_idx] : out_ptr;
            } else {
                size_t logical_idx = node_to_buffer.at(node->id);
                size_t pool_idx = logical_to_pool[logical_idx];
                out_buf = (pool_idx < tmp_buffers.size()) ? tmp_buffers[pool_idx] : out_ptr;
            }
        }
        const NodeVariant& op = node->op;

        if (std::holds_alternative<FusedNode>(op)) {
            // 节点自身输出 numel
            int64_t node_numel = (int64_t)node->out_desc.numel;
            // [Fix 2026-08-15] 并行切片下 n_val < node_numel：循环上界必须取
            //   min(node_numel, n)，否则每个线程写满全尺寸导致越界（同 Handwritten 侧修复）
            auto node_numel_c = builder.create<mlir::arith::ConstantIntOp>(loc, node_numel, 64);
            auto fn_node_n = builder.create<mlir::arith::MinSIOp>(loc, node_numel_c, n_val);
            const auto& fnode = std::get<FusedNode>(op);
            // 为每个 arg_node_id 获取输入指针（外部输入或中间缓冲区）
            std::unordered_map<size_t, mlir::Value> fused_arg_ptrs;
            std::unordered_map<size_t, int64_t> fused_arg_numels;
            for (size_t aidx = 0; aidx < fnode.arg_node_ids.size(); ++aidx) {
                size_t nid = fnode.arg_node_ids[aidx];
                fused_arg_ptrs[nid] = getInputPtr(nid);
                // 查找该节点的输出 numel（从 graph 的 nodes 中获取）
                for (const auto& gn : nodes) {
                    if (gn.id == nid) {
                        fused_arg_numels[nid] = (int64_t)gn.out_desc.numel;
                        break;
                    }
                }
            }
            // 判断是否可以使用多核向量化
            if (isFusedChainVectorizable(fnode.ops, fnode.op_inputs, fused_arg_numels, (size_t)node_numel)) {
                buildFusedMultiNodeVectorized(builder, loc, out_buf, fn_node_n, fnode.ops, fnode.op_inputs,
                                              fnode.arg_node_ids, fused_arg_ptrs, fused_arg_numels);
            } else {
                buildFusedMultiNode(builder, loc, out_buf, fn_node_n, fnode.ops, fnode.op_inputs,
                                    fnode.arg_node_ids, fused_arg_ptrs, fused_arg_numels);
            }
            continue;
        }

        // 收集输入指针
        std::vector<mlir::Value> in_ptrs;
        for (size_t in_id : node->inputs) {
            in_ptrs.push_back(getInputPtr(in_id));
        }

        // 计算广播取模（用于 element-wise binary ops）
        auto getBroadcastMod = [&](const Node* n_ptr) -> int64_t {
            if (n_ptr->inputs.size() < 2) return 0;
            const auto& lhs_desc = graph.node(n_ptr->inputs[0]).out_desc;
            const auto& rhs_desc = graph.node(n_ptr->inputs[1]).out_desc;
            auto lhs = lhs_desc.shape;
            auto rhs = rhs_desc.shape;
            if (lhs.empty() || rhs.empty() || lhs == rhs) return 0;
            size_t lhs_numel = 1, rhs_numel = 1;
            for (size_t d : lhs) lhs_numel *= d;
            for (size_t d : rhs) rhs_numel *= d;
            if (rhs_numel < lhs_numel && rhs_numel == 1) return 1; // RHS scalar broadcast
            if (lhs_numel < rhs_numel && lhs_numel == 1) return -1; // LHS scalar broadcast
            if (rhs.size() == 1 && !lhs.empty() && lhs.back() == rhs[0]) {
                return (int64_t)rhs[0]; // 1D vector broadcast to last dim
            }
            if (lhs.size() == 1 && !rhs.empty() && rhs.back() == lhs[0]) {
                return -(int64_t)lhs[0]; // 1D vector broadcast to last dim (LHS)
            }
            return 0; // unsupported broadcast pattern
        };

        // 节点自身输出 numel（用于 element-wise 循环计数）
        int64_t node_numel = (int64_t)node->out_desc.numel;
        // [Fix 2026-08-15] 并行切片下 n_val < node_numel：循环上界取
        //   min(node_numel, n)（同 FusedNode 与 Handwritten 侧修复）
        auto node_numel_c = builder.create<mlir::arith::ConstantIntOp>(loc, node_numel, 64);
        auto node_n = builder.create<mlir::arith::MinSIOp>(loc, node_numel_c, n_val);

        if (std::holds_alternative<MatMulNode>(op)) {
            // 每个 MatMul 使用自己的 M, K, N 维度
            const auto& mm = std::get<MatMulNode>(op);
            int64_t matM = (int64_t)mm.lhs_desc.shape[0];
            int64_t matK = (int64_t)mm.lhs_desc.shape[1];
            int64_t matN = (int64_t)mm.rhs_desc.shape[1];
            auto mm_M = builder.create<mlir::arith::ConstantIntOp>(loc, matM, 64);
            auto mm_K = builder.create<mlir::arith::ConstantIntOp>(loc, matK, 64);
            auto mm_N = builder.create<mlir::arith::ConstantIntOp>(loc, matN, 64);

            // === Transpose Folding 转置折叠优化 (M2 阶段 2026-08-14) ===
            int transA = 111; // 111 = CblasNoTrans, 112 = CblasTrans
            int transB = 111;
            size_t in_id_a = node->inputs[0];
            size_t in_id_b = node->inputs[1];
            mlir::Value matmul_a_ptr = getInputPtr(in_id_a);
            mlir::Value matmul_b_ptr = getInputPtr(in_id_b);

            for (const auto& gn : nodes) {
                if (gn.id == in_id_a && std::holds_alternative<TransposeNode>(gn.op)) {
                    transA = 112;
                    matmul_a_ptr = getInputPtr(gn.inputs[0]);
                    break;
                }
            }
            for (const auto& gn : nodes) {
                if (gn.id == in_id_b && std::holds_alternative<TransposeNode>(gn.op)) {
                    transB = 112;
                    matmul_b_ptr = getInputPtr(gn.inputs[0]);
                    break;
                }
            }

            // === MatMul Epilogue Fusion 检测 ===
            // 检测 MatMul→Add(bias)→Activation 模式并融合为一个 kernel
            mlir::Value fused_bias_ptr = nullptr;
            MatMulActivation fused_act = MatMulActivation::None;
            int64_t fused_skip = 0; // 跳过的后续节点数
            size_t fused_bias_numel = 0; // DEBT-NEW-5: bias 张量元素数 (0=无 bias)

            // 检查下一个节点是否为 Add（偏置加法）
            if (ci + 1 < compute_nodes.size()) {
                const Node* next_node = compute_nodes[ci + 1];
                if (std::holds_alternative<AddNode>(next_node->op) &&
                    next_node->inputs.size() == 2 &&
                    next_node->inputs[0] == node->id) {
                    // 匹配：MatMul 的输出作为 Add 的第一个输入
                    size_t bias_node_id = next_node->inputs[1];
                    fused_bias_ptr = getInputPtr(bias_node_id);
                    fused_skip = 1;
                    // DEBT-NEW-5: 记录 bias 形状 numel, 供 epilogue 选择行/列广播索引
                    fused_bias_numel = 1;
                    for (size_t d : std::get<AddNode>(next_node->op).rhs_desc.shape)
                        fused_bias_numel *= d;

                    // 检查再下一个节点是否为激活函数
                    if (ci + 2 < compute_nodes.size()) {
                        const Node* act_node = compute_nodes[ci + 2];
                        if (act_node->inputs.size() == 1 &&
                            act_node->inputs[0] == next_node->id) {
                            if (std::holds_alternative<ReLUNode>(act_node->op)) {
                                fused_act = MatMulActivation::ReLU;
                                fused_skip = 2;
                            } else if (std::holds_alternative<SigmoidNode>(act_node->op)) {
                                fused_act = MatMulActivation::Sigmoid;
                                fused_skip = 2;
                            } else if (std::holds_alternative<TanhNode>(act_node->op)) {
                                fused_act = MatMulActivation::Tanh;
                                fused_skip = 2;
                            }
                        }
                    }

                    // 当融合发生时，计算正确的输出 buffer（使用最后一个融合节点的 buffer）
                    if (fused_skip > 0) {
                        size_t last_fused_ci = ci + fused_skip;
                        bool last_fused_is_last = (last_fused_ci == compute_nodes.size() - 1);
                        if (last_fused_is_last) {
                            out_buf = out_ptr;
                        } else {
                            const Node* last_fused = compute_nodes[last_fused_ci];
                            auto reuse_it = node_buffer_reuse.find(last_fused->id);
                            if (reuse_it != node_buffer_reuse.end()) {
                                size_t pool_idx = logical_to_pool[reuse_it->second];
                                out_buf = (pool_idx < tmp_buffers.size()) ? tmp_buffers[pool_idx] : out_ptr;
                            } else {
                                size_t logical_idx = node_to_buffer.at(last_fused->id);
                                size_t pool_idx = logical_to_pool[logical_idx];
                                out_buf = (pool_idx < tmp_buffers.size()) ? tmp_buffers[pool_idx] : out_ptr;
                            }
                        }
                    }
                }
            }

            // === 创建 c3.matmul op（JIT 3.0 三 op 收口 2026-08-15）===
            // 策略选择（small_inline / tiled / cblas）下沉到 MatMulOpLowering，
            // 这里只负责打包 M/K/N + transpose folding + epilogue 融合信息。
            // M1 1.2 (2026-08-09): 读 AutoTuner 调优结果替换写死的 kDefaultTileM/N
            // 调优未跑 (tuned=false) 时 thread_local cache 持有 {0,0,0}, 落到默认 32/32
            auto& tile = currentTileCache();
            int64_t tile_m = (tile.tile_m > 0) ? tile.tile_m : kDefaultTileM;
            int64_t tile_n = (tile.tile_n > 0) ? tile.tile_n : kDefaultTileN;
            builder.create<mlir::c3::MatMulOp>(loc, matmul_a_ptr, matmul_b_ptr, out_buf,
                                               fused_bias_ptr,
                                               matM, matK, matN,
                                               transA, transB, (int)fused_act,
                                               tile_m, tile_n, (int64_t)fused_bias_numel);

            // 跳过被融合的后续节点
            ci += fused_skip;
        } else if (std::holds_alternative<AddNode>(op)) {
            int64_t bmod = getBroadcastMod(node);
            builder.create<mlir::c3::AddOp>(loc, in_ptrs[0], in_ptrs[1], out_buf, node_n, bmod);
        } else if (std::holds_alternative<SubNode>(op)) {
            int64_t bmod = getBroadcastMod(node);
            builder.create<mlir::c3::SubOp>(loc, in_ptrs[0], in_ptrs[1], out_buf, node_n, bmod);
        } else if (std::holds_alternative<MulNode>(op)) {
            int64_t bmod = getBroadcastMod(node);
            builder.create<mlir::c3::MulOp>(loc, in_ptrs[0], in_ptrs[1], out_buf, node_n, bmod);
        } else if (std::holds_alternative<DivNode>(op)) {
            int64_t bmod = getBroadcastMod(node);
            builder.create<mlir::c3::DivOp>(loc, in_ptrs[0], in_ptrs[1], out_buf, node_n, bmod);
        } else if (std::holds_alternative<NegNode>(op)) {
            builder.create<mlir::c3::NegOp>(loc, in_ptrs[0], out_buf, node_n);
        } else if (std::holds_alternative<ReLUNode>(op)) {
            builder.create<mlir::c3::ReLUOp>(loc, in_ptrs[0], out_buf, node_n);
        } else if (std::holds_alternative<SigmoidNode>(op)) {
            builder.create<mlir::c3::SigmoidOp>(loc, in_ptrs[0], out_buf, node_n);
        } else if (std::holds_alternative<TanhNode>(op)) {
            builder.create<mlir::c3::TanhOp>(loc, in_ptrs[0], out_buf, node_n);
        } else if (std::holds_alternative<ExpNode>(op)) {
            builder.create<mlir::c3::ExpOp>(loc, in_ptrs[0], out_buf, node_n);
        } else if (std::holds_alternative<LogNode>(op)) {
            builder.create<mlir::c3::LogOp>(loc, in_ptrs[0], out_buf, node_n);
        } else if (std::holds_alternative<SumReduceNode>(op)) {
            const auto& sr = std::get<SumReduceNode>(op);
            int64_t M = sr.in_desc.shape.size() > 0 ? sr.in_desc.shape[0] : 1;
            int64_t N = sr.in_desc.shape.size() > 1 ? sr.in_desc.shape[1] : 1;
            builder.create<mlir::c3::SumReduceOp>(loc, in_ptrs[0], out_buf, M, N, sr.axis);
        } else if (std::holds_alternative<TransposeNode>(op)) {
            const auto& tr = std::get<TransposeNode>(op);
            int64_t M = tr.in_desc.shape.size() > 0 ? tr.in_desc.shape[0] : 1;
            int64_t N = tr.in_desc.shape.size() > 1 ? tr.in_desc.shape[1] : 1;
            builder.create<mlir::c3::TransposeOp>(loc, in_ptrs[0], out_buf, M, N, tr.dim0, tr.dim1);
        } else if (std::holds_alternative<GtNode>(op)) {
            buildGt(builder, loc, in_ptrs[0], in_ptrs[1], out_buf, node_n);
        } else {
            // 支持的 op 列表 (M1 路线图 9/15 + M2 完成):
            //   MatMul/Add/Sub/Mul/Div/Neg/ReLU/Sigmoid/Tanh/SumReduce/Transpose/Gt/Exp/Log
            // 未支持 (M2 未完成): Const
            const std::string op_name = std::visit(
                [](const auto& n) -> std::string { return typeid(n).name(); },
                op);
            throw std::runtime_error(
                "MLIRKernelGen: unsupported op in multi-node graph: " + op_name +
                " (per MLIR backend 完整化路线图, M2 范畴 v0.5.3+ 实装)");
        }
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

// ======================= linalg.generic 逐元素路线（JIT 3.0 声明式大一统 接入） =======================

/// 尝试用 linalg.generic 声明式路径编译单节点逐元素 kernel，
/// 替换主库手写 if-else 标量 IR 分支（buildElementwiseBinary/buildReLU/buildSigmoid/...）。
/// 命中条件：
///   1. 恰好 1 个计算节点（非融合、非多节点）；
///   2. 算子属于 {ReLU, Sigmoid, Tanh, Exp, Log, Add, Sub, Mul}；
///   3. 二元算子支持 同尺寸 / 标量（rhs size=1）/ 1D vector 周期广播
///      （2D+ lhs 最后一维 == rhs[0]，linalg 1D 视角下映射为 `d0 -> d0 mod k`）；
///   4. env C3_LINALG_EW != "0"（默认开启，逃生开关回退手写）。
/// 命中时填充 out.func_any（执行器捕获 shared_ptr<LinalgElementwiseKernel> 保证生命周期）。
static bool tryBuildLinalgElementwise(const Graph& graph, GeneratedKernel& out, int opt_level) {
    if (opt_level >= 2) return false; // 强行对 opt_level >= 2 开启 MLIR 级真实向量化（走标准多节点向量化管线），不走标量 linalg 降解
    static const bool disabled = [] {
        const char* v = std::getenv("C3_LINALG_EW");
        return v != nullptr && std::string(v) == "0";
    }();
    if (disabled) return false;

    if (countComputeNodesMLIR(graph) != 1) return false;

    const Node* compute_node = nullptr;
    for (const auto& node : graph.nodes()) {
        if (isComputeNodeMLIR(node, graph.inputs())) { compute_node = &node; break; }
    }
    if (!compute_node || compute_node->out_desc.numel == 0) return false;

    const NodeVariant& op = compute_node->op;
    ElementwiseOp eop = ElementwiseOp::ReLU;
    int rhs_mod = RhsNoBroadcast;  // 二元第二输入广播模数（0 同尺寸 / 1 标量 / k>1 周期）
    bool eligible = true;

    // 计算二元第二输入广播模数（语义对齐手写路径 buildElementwiseBinary 的 rhs_broadcast_mod）：
    //   lhs==rhs → 0（同尺寸）；rhs_numel==1 → 1（标量）；rhs 1D 且 lhs 最后一维==rhs[0] → k（周期）；
    //   其余（含任一 shape 为空、numel==0）→ -1（不支持，回退手写）。
    auto computeRhsMod = [&](const NodeVariant& v) -> int {
        auto getRhsShape = [](const NodeVariant& nv) -> std::vector<size_t> {
            if (std::holds_alternative<AddNode>(nv)) return std::get<AddNode>(nv).rhs_desc.shape;
            if (std::holds_alternative<SubNode>(nv)) return std::get<SubNode>(nv).rhs_desc.shape;
            if (std::holds_alternative<MulNode>(nv)) return std::get<MulNode>(nv).rhs_desc.shape;
            return {};
        };
        auto getLhsShape = [](const NodeVariant& nv) -> std::vector<size_t> {
            if (std::holds_alternative<AddNode>(nv)) return std::get<AddNode>(nv).lhs_desc.shape;
            if (std::holds_alternative<SubNode>(nv)) return std::get<SubNode>(nv).lhs_desc.shape;
            if (std::holds_alternative<MulNode>(nv)) return std::get<MulNode>(nv).lhs_desc.shape;
            return {};
        };
        const auto lhs = getLhsShape(v);
        const auto rhs = getRhsShape(v);
        if (lhs.empty() || rhs.empty()) return -1;
        if (lhs == rhs) return RhsNoBroadcast;      // 同尺寸
        size_t rhs_numel = 1;
        for (size_t d : rhs) rhs_numel *= d;
        if (rhs_numel == 1) return RhsScalarBroadcast;  // 标量广播（size=1）
        if (rhs.size() == 1 && lhs.back() == rhs[0])
            return static_cast<int>(rhs[0]);        // 1D vector 周期广播（k>1，mod k）
        return -1;                                  // 其余多维/不支持广播
    };

    if (std::holds_alternative<ReLUNode>(op)) {
        eop = ElementwiseOp::ReLU;
    } else if (std::holds_alternative<SigmoidNode>(op)) {
        eop = ElementwiseOp::Sigmoid;
    } else if (std::holds_alternative<TanhNode>(op)) {
        eop = ElementwiseOp::Tanh;
    } else if (std::holds_alternative<ExpNode>(op)) {
        eop = ElementwiseOp::Exp;
    } else if (std::holds_alternative<LogNode>(op)) {
        eop = ElementwiseOp::Log;
    } else if (std::holds_alternative<AddNode>(op)) {
        eop = ElementwiseOp::Add;
        rhs_mod = computeRhsMod(op);
        if (rhs_mod < 0) eligible = false;
    } else if (std::holds_alternative<SubNode>(op)) {
        eop = ElementwiseOp::Sub;
        rhs_mod = computeRhsMod(op);
        if (rhs_mod < 0) eligible = false;
    } else if (std::holds_alternative<MulNode>(op)) {
        eop = ElementwiseOp::Mul;
        rhs_mod = computeRhsMod(op);
        if (rhs_mod < 0) eligible = false;
    } else {
        eligible = false;
    }
    if (!eligible) return false;

    // 编译 linalg kernel（走共享缓存，同一 (op,opt,rhs_mod) 只 JIT 一次）；失败静默回退手写
    std::shared_ptr<LinalgElementwiseKernel> kernel;
    try {
        kernel = getCachedLinalgKernel(eop, 3, rhs_mod);
    } catch (const std::exception& e) {
        fprintf(stderr, "C3 linalg: %s compile failed (%s), fallback to handwritten\n",
                elementwiseOpName(eop), e.what());
        return false;
    }

    const size_t num_inputs = elementwiseOpNumInputs(eop);
    out.func = nullptr;
    out.func_any = [kernel, num_inputs](const float* a, const float* b, float* out_ptr,
                                        size_t n, size_t, size_t, size_t) {
        // 一元取 in_ptrs[0]=a；二元取 [a, b]（num_inputs 控制 execute 读取个数；
        // 标量广播时 execute 对 b 按 size=1 处理，只读 b[0]；
        // 周期广播时 execute 对 b 按 size=k 处理，沿主维度周期复用）
        const float* in_ptrs[2] = {a, b};
        kernel->execute(in_ptrs, out_ptr, n);
    };
    out.is_matmul = false;
    out.num_inputs = num_inputs;
    out.elem_n = compute_node->out_desc.numel;

    // [2026-08-15] 路由命中诊断：env C3_LINALG_EW_TRACE=1 时打印，默认静默
    static const bool trace = [] {
        const char* v = std::getenv("C3_LINALG_EW_TRACE");
        return v != nullptr && std::string(v) == "1";
    }();
    if (trace) {
        fprintf(stderr, "C3 linalg: routed %s (num_inputs=%zu, n=%zu) via linalg.generic\n",
                elementwiseOpName(eop), num_inputs, compute_node->out_desc.numel);
    }
    return true;
}

// ======================= 模块构建 =======================

// [Dev] v0.5.2 DCU 接入 refactor (2026-08-10):
// buildMLIRModule 从 file-static 改成公开 API, 跟 MLIRToLLVMIR.cpp 的
// mlirToLLVMIRFromGraph 复用同一份 build / lower 逻辑
mlir::OwningOpRef<mlir::ModuleOp> buildMLIRModule(
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
        func.setArgAttr(0, "llvm.noalias", builder.getUnitAttr());
        func.setArgAttr(1, "llvm.noalias", builder.getUnitAttr());
        auto* entry = func.addEntryBlock();
        builder.setInsertionPointToStart(entry);
        mlir::Value inputs = entry->getArgument(0);
        mlir::Value out_val = entry->getArgument(1);
        mlir::Value n_val = entry->getArgument(2);
        buildFused(builder, loc, inputs, out_val, n_val, fnode.ops, fnode.op_inputs, fnode.op_node_ids, fnode.arg_node_ids);
        builder.create<mlir::func::ReturnOp>(loc);
    } else {
        // 普通算子：使用 C3KernelFunc 签名 (ptr, ptr, ptr, i64, i64, i64, i64) → void
        auto func_type = builder.getFunctionType(
            {ptr_type, ptr_type, ptr_type, i64_type, i64_type, i64_type, i64_type}, {});
        auto func = builder.create<mlir::func::FuncOp>(loc, "c3_kernel", func_type);
        func.setArgAttr(0, "llvm.noalias", builder.getUnitAttr());
        func.setArgAttr(1, "llvm.noalias", builder.getUnitAttr());
        func.setArgAttr(2, "llvm.noalias", builder.getUnitAttr());

        auto* entry = func.addEntryBlock();
        builder.setInsertionPointToStart(entry);

        mlir::Value a = entry->getArgument(0);
        mlir::Value b = entry->getArgument(1);
        mlir::Value out = entry->getArgument(2);
        mlir::Value n = entry->getArgument(3);
        mlir::Value M = entry->getArgument(4);
        mlir::Value K = entry->getArgument(5);
        mlir::Value N = entry->getArgument(6);

        if (std::holds_alternative<AddNode>(op)) {
            const auto& add = std::get<AddNode>(op);
            int64_t broadcast_mod = 0;
            if (add.rhs_desc.numel < add.lhs_desc.numel && add.rhs_desc.numel > 0) {
                broadcast_mod = (int64_t)add.rhs_desc.numel;
            } else if (add.lhs_desc.numel < add.rhs_desc.numel && add.lhs_desc.numel > 0) {
                broadcast_mod = -(int64_t)add.lhs_desc.numel;
            }
            builder.create<mlir::c3::AddOp>(loc, a, b, out, n, broadcast_mod);
        }
        else if (std::holds_alternative<SubNode>(op)) {
            const auto& sub = std::get<SubNode>(op);
            int64_t broadcast_mod = 0;
            if (sub.rhs_desc.numel < sub.lhs_desc.numel && sub.rhs_desc.numel > 0) {
                broadcast_mod = (int64_t)sub.rhs_desc.numel;
            } else if (sub.lhs_desc.numel < sub.rhs_desc.numel && sub.lhs_desc.numel > 0) {
                broadcast_mod = -(int64_t)sub.lhs_desc.numel;
            }
            builder.create<mlir::c3::SubOp>(loc, a, b, out, n, broadcast_mod);
        }
        else if (std::holds_alternative<MulNode>(op)) {
            const auto& mul = std::get<MulNode>(op);
            int64_t broadcast_mod = 0;
            if (mul.rhs_desc.numel < mul.lhs_desc.numel && mul.rhs_desc.numel > 0) {
                broadcast_mod = (int64_t)mul.rhs_desc.numel;
            } else if (mul.lhs_desc.numel < mul.rhs_desc.numel && mul.lhs_desc.numel > 0) {
                broadcast_mod = -(int64_t)mul.lhs_desc.numel;
            }
            builder.create<mlir::c3::MulOp>(loc, a, b, out, n, broadcast_mod);
        }
        else if (std::holds_alternative<DivNode>(op)) {
            const auto& div = std::get<DivNode>(op);
            int64_t broadcast_mod = 0;
            if (div.rhs_desc.numel < div.lhs_desc.numel && div.rhs_desc.numel > 0) {
                broadcast_mod = (int64_t)div.rhs_desc.numel;
            } else if (div.lhs_desc.numel < div.rhs_desc.numel && div.lhs_desc.numel > 0) {
                broadcast_mod = -(int64_t)div.lhs_desc.numel;
            }
            builder.create<mlir::c3::DivOp>(loc, a, b, out, n, broadcast_mod);
        }
        else if (std::holds_alternative<MatMulNode>(op)) {
            // JIT 3.0 三 op 收口：单节点 MatMul 也走 c3.matmul op，
            // 维度从 MatMulNode 的 desc 取编译期常量，策略选择由 MatMulOpLowering 完成。
            const auto& mm = std::get<MatMulNode>(op);
            int64_t matM = mm.lhs_desc.shape.size() > 0 ? (int64_t)mm.lhs_desc.shape[0] : 0;
            int64_t matK = mm.lhs_desc.shape.size() > 1 ? (int64_t)mm.lhs_desc.shape[1] : 0;
            int64_t matN = mm.rhs_desc.shape.size() > 1 ? (int64_t)mm.rhs_desc.shape[1] : 0;
            builder.create<mlir::c3::MatMulOp>(loc, a, b, out, nullptr,
                                               matM, matK, matN,
                                               111, 111, (int)MatMulActivation::None,
                                               /*tileM=*/0, /*tileN=*/0, /*biasNumel=*/0);
        }
        else if (std::holds_alternative<NegNode>(op)) {
            builder.create<mlir::c3::NegOp>(loc, a, out, n);
        }
        else if (std::holds_alternative<ReLUNode>(op)) {
            builder.create<mlir::c3::ReLUOp>(loc, a, out, n);
        }
        else if (std::holds_alternative<SigmoidNode>(op)) {
            builder.create<mlir::c3::SigmoidOp>(loc, a, out, n);
        }
        else if (std::holds_alternative<TanhNode>(op)) {
            builder.create<mlir::c3::TanhOp>(loc, a, out, n);
        }
        else if (std::holds_alternative<ExpNode>(op)) {
            builder.create<mlir::c3::ExpOp>(loc, a, out, n);
        }
        else if (std::holds_alternative<LogNode>(op)) {
            builder.create<mlir::c3::LogOp>(loc, a, out, n);
        }
        else if (std::holds_alternative<SumReduceNode>(op)) {
            const auto& sr = std::get<SumReduceNode>(op);
            int64_t M = sr.in_desc.shape.size() > 0 ? sr.in_desc.shape[0] : 1;
            int64_t N = sr.in_desc.shape.size() > 1 ? sr.in_desc.shape[1] : 1;
            builder.create<mlir::c3::SumReduceOp>(loc, a, out, M, N, sr.axis);
        }
        else if (std::holds_alternative<TransposeNode>(op)) {
            const auto& tr = std::get<TransposeNode>(op);
            int64_t M = tr.in_desc.shape.size() > 0 ? tr.in_desc.shape[0] : 1;
            int64_t N = tr.in_desc.shape.size() > 1 ? tr.in_desc.shape[1] : 1;
            builder.create<mlir::c3::TransposeOp>(loc, a, out, M, N, tr.dim0, tr.dim1);
        }
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

// [CGO 2027 重构]: C3ToLLVM 降低模式、DRR 图合并规则及 applyLoweringPipeline 已完全解耦移入 C3DialectLowering.cpp。
// MLIRKernelGen.cpp 仅保留单节点与多节点方言图构建入口。
GeneratedKernel generateFromGraphMLIR(const Graph& graph, int opt_level) {
    // [2026-08-15] linalg.generic 声明式逐元素路线（JIT 3.0 声明式大一统 接入）：
    // 单节点逐元素算子（无广播）直接走 LinalgElementwiseKernel（自带 ExecutionEngine，
    // func_any 捕获 shared_ptr 保证生命周期），跳过下方手写 if-else 标量 IR 构建。
    // 未命中（多节点/融合/MatMul/广播/逃生开关 C3_LINALG_EW=0）时回退原 MLIR 构建。
    {
        GeneratedKernel linalg_gen;
        if (tryBuildLinalgElementwise(graph, linalg_gen, opt_level)) {
            return linalg_gen;
        }
    }

    static std::once_flag llvm_init_flag;
    std::call_once(llvm_init_flag, []() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    });

    // 每次编译创建独立的 MLIRContext，通过 DialectRegistry 集中管理所有 dialect。
    // LLVMDialect 必须放入 DialectRegistry 以正确初始化类型系统（LLVMPointerType 等）。
    // 参考 JIT-MLIR-Debug-Experience.md §1: Dialect 注册必须通过 DialectRegistry 集中管理。
    mlir::DialectRegistry reg;
    reg.insert<mlir::arith::ArithDialect>();
    reg.insert<mlir::math::MathDialect>();
    reg.insert<mlir::scf::SCFDialect>();
    reg.insert<mlir::func::FuncDialect>();
    reg.insert<mlir::memref::MemRefDialect>();
    reg.insert<mlir::LLVM::LLVMDialect>();
    reg.insert<mlir::c3::C3Dialect>();
    mlir::registerBuiltinDialectTranslation(reg);
    mlir::registerLLVMDialectTranslation(reg);

    std::shared_ptr<mlir::MLIRContext> context;
    {
        std::lock_guard<std::mutex> lock(ct::c3::c3_global_mlir_mutex);
        context = std::make_shared<mlir::MLIRContext>(reg);
        context->getOrLoadDialect<mlir::arith::ArithDialect>();
        context->getOrLoadDialect<mlir::math::MathDialect>();
        context->getOrLoadDialect<mlir::scf::SCFDialect>();
        context->getOrLoadDialect<mlir::func::FuncDialect>();
        context->getOrLoadDialect<mlir::memref::MemRefDialect>();
        context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        context->getOrLoadDialect<mlir::c3::C3Dialect>();
    }

    auto module = buildMLIRModule(*context, graph);
    applyLoweringPipeline(*module, opt_level);

    // [TEMP-DBG] 环境变量 C3_MLIR_DUMP=1 时打印 lowering 后的 module
    if (std::getenv("C3_MLIR_DUMP")) {
        llvm::errs() << "==== C3 lowered module ====\n";
        module->dump();
        llvm::errs() << "==== end ====\n";
    }

    // 创建 TargetMachine 以启用 LLVM 自动向量化（NEON/SIMD）
    // [Fix 2026-08-12] 添加 opt level pour correspondre au Pass Manager
    auto tm = std::shared_ptr<llvm::TargetMachine>(
        llvm::EngineBuilder()
            .setEngineKind(llvm::EngineKind::JIT)
            .setOptLevel(static_cast<llvm::CodeGenOptLevel>(opt_level))
            .selectTarget());

    // 【dispatch 优化 2026-08-12】重新启用 LLVM 优化 transformer。
    // 此前 DEBT-NEW-5 实验置空（隔离 MLIR vs LLVM 数值差异来源），导致 MLIR 生成的
    // 标量逐元素循环无 LoopVectorize/SLP 等 pass 自动向量化，实测比原生 SIMD kernel 慢 ~3.6x，
    // 是 dispatch 中单 kernel 执行的主要开销来源。LLVM pass 语义保持，安全性由
    // test_c3_backward / test_c3_mnist_train 数值回归保障。
    // 生命周期: ① transformer 仅在 ExecutionEngine::create() 的 JIT 编译期调用，此时 tm 仍在
    // 作用域，tm.get() 不悬垂；② engineOpts.transformer 是 llvm::function_ref（非拥有引用），
    // 必须由命名局部量 opt_transformer 持有 std::function 存活到 create() 之后。
    const bool mlir_noopt = std::getenv("C3_MLIR_NOOPT") != nullptr;
    std::function<llvm::Error(llvm::Module *)> opt_transformer =
        (!mlir_noopt && tm)
        ? mlir::makeOptimizingTransformer(static_cast<unsigned>(opt_level), 0, tm.get())
        : std::function<llvm::Error(llvm::Module *)>();
    mlir::ExecutionEngineOptions engineOpts;
    if (opt_transformer) {
        engineOpts.transformer = opt_transformer;
    }
    // 当 opt_level >= 3 时使用 Aggressive（Ofast），否则匹配 opt_level
    engineOpts.jitCodeGenOptLevel = (opt_level >= 3)
        ? llvm::CodeGenOptLevel::Aggressive
        : (opt_level == 2) ? llvm::CodeGenOptLevel::Default
        : (opt_level == 1) ? llvm::CodeGenOptLevel::Less
        : llvm::CodeGenOptLevel::None;

    // [Dev] v0.5.2 (4) JITCache 1.0 store-only (2026-08-09):
    // 在 ExecutionEngine::create 之前,翻译 MLIR module → LLVM module → 写 bitcode 落盘
    // 1.0 实装: store 完整 (写 .bc + .meta), lookup 走 disk check 但不实际反序列化
    // read path (loadBitcode → ExecutionEngine) 留 v0.5.2 follow-up (需要 ExecutionEngine 重建 hook)
    // 用户测试注意 (per 洛锦 2026-08-09):
    //   - 性能测试前必须 JITCache::evict() (避免命中作弊)
    //   - MLIR backend 改动后必须 evict() (旧 .bc 跟新 MLIR IR 不兼容)
    //   - 正确性测试允许 warm cache (cache deterministic)
    if (JITCache::isEnabled()) {
        try {
            std::string jit_key = JITCache::makeKey(graph.toString(), opt_level);
            std::string bc_path = JITCache::getInstance().lookup(jit_key);
            if (bc_path.empty()) {
                // miss: 翻译 + 写 bitcode
                llvm::LLVMContext bc_ctx;
                auto llvm_module = mlir::translateModuleToLLVMIR(*module, bc_ctx);
                if (llvm_module) {
                    JITCache::getInstance().store(jit_key, *llvm_module);
                }
            } else {
                // 命中 (有 .bc 文件),但 1.0 不实际反序列化,直接走正常 ExecutionEngine
                JITCache::getInstance().recordHit();
            }
        } catch (...) {
            // 静默失败,不影响正常 ExecutionEngine 编译
        }
    }

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
        
        std::vector<const Node*> compute_nodes;
        for (const auto& node : nodes) {
            if (isComputeNodeMLIR(node, graph.inputs())) {
                compute_nodes.push_back(&node);
            }
        }
        size_t max_numel = 0;
        size_t num_intermediates = 0;
        for (size_t i = 0; i < compute_nodes.size(); ++i) {
            size_t node_id = compute_nodes[i]->id;
            bool is_output = false;
            for (size_t out_id : graph.outputs()) {
                if (node_id == out_id) { is_output = true; break; }
            }
            if (i == compute_nodes.size() - 1) is_output = true;
            if (!is_output) {
                num_intermediates++;
                if (compute_nodes[i]->out_desc.numel > max_numel) max_numel = compute_nodes[i]->out_desc.numel;
            }
        }
        size_t num_constants = 0;
        for (const auto& node : nodes) {
            if (std::holds_alternative<ConstNode>(node.op)) {
                num_constants++;
            }
        }
        size_t pool_buf_count = (num_intermediates > 0) ? std::min(num_intermediates, (size_t)2) : 0;
        result.scratch_size = max_numel * pool_buf_count + num_constants;

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

    // engine/tm/context/module 持有所有权，deleter 确保四者生命周期覆盖 kernel 执行期
    // 注意：context 和 module 必须捕获，否则函数返回后 MLIRContext 和 ModuleOp 被销毁，
    // 导致 ExecutionEngine 内部引用悬空，触发 StorageUniquerImpl 段错误。
    // OwningOpRef 是 move-only 类型，需要用 shared_ptr 包装以便 std::function 拷贝。
    auto engine = std::shared_ptr<mlir::ExecutionEngine>(maybeEngine->release());
    auto module_holder = std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>(std::move(module));
    result.handle = nullptr;
    result.deleter = [tm, engine, context, module_holder]() {};

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
