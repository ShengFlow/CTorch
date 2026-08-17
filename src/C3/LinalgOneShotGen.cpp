/**
 * @file LinalgOneShotGen.cpp
 * @generation JIT-3.0 One-Shot 统一融合后端实现
 * @brief C3 JIT 3.0: 统一 C3-to-Linalg Lowering + Linalg Fusion + One-Shot Bufferization 极致优化管线
 * @date 2026/08/15
 */

#include "C3/LinalgOneShotGen.h"
#include "C3/C3Dialect.h"
#include "C3/JITCache.h"
#include "MLIRKernelGen.h"
#include "Ctools.h"
#include "CtorchError.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Math/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h>
#include <mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h>
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>

#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/IR/Function.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <stdexcept>
#include <iostream>

namespace ct {
namespace c3 {

// ======================= MemRef 描述符 C-ABI =======================

struct MemRefDesc {
    float* alloc;
    float* aligned;
    int64_t offset;
    int64_t sizes[1];
    int64_t strides[1];
};

static void appendMemRefDescArgs(const MemRefDesc& desc, void** args, int& idx) {
    args[idx++] = const_cast<float**>(&desc.alloc);
    args[idx++] = const_cast<float**>(&desc.aligned);
    args[idx++] = const_cast<int64_t*>(&desc.offset);
    args[idx++] = const_cast<int64_t*>(&desc.sizes[0]);
    args[idx++] = const_cast<int64_t*>(&desc.strides[0]);
}

// ======================= 调试与优雅 TRACE 辅助 (一切以用户为中心) =======================

static bool isVerboseDebugEnabled() {
    const char* debug_env = std::getenv("C3_LINALG_DEBUG");
    const char* dump_env = std::getenv("C3_MLIR_DUMP");
    bool debug_v = debug_env != nullptr && (std::string(debug_env) == "1" || std::string(debug_env) == "verbose");
    bool dump_v = dump_env != nullptr && std::string(dump_env) == "verbose";
    return debug_v || dump_v;
}

static bool isDebugEnabled() {
    const char* debug_env = std::getenv("C3_LINALG_DEBUG");
    const char* dump_env = std::getenv("C3_MLIR_DUMP");
    return debug_env != nullptr || dump_env != nullptr;
}

/// 是否启用 Fast-Math（多项式逼近 + 后端不安全浮点）。
/// 受 C3_FAST_MATH=1 控制；会改变 exp/log/tanh 的数值（相对误差 ~1e-6），
/// 通常不影响分类精度（如 MNIST 97.18% 保持不变），但精度敏感场景需自测。
static bool isFastMathEnabled() {
    const char* v = std::getenv("C3_FAST_MATH");
    return v != nullptr && std::string(v) == "1";
}

/// 是否启用 OpenMP 真多核并行（scf.parallel → omp.wsloop → libomp）。
/// 受 C3_LINALG_OMP=1 控制；仅当管线生成 scf.parallel 时才有意义
/// （即 tile_size > 0 的路径，见 Phase 6）。
static bool isOpenMPEnabled() {
    const char* v = std::getenv("C3_LINALG_OMP");
    return v != nullptr && std::string(v) == "1";
}

static void dumpPhaseIR(mlir::ModuleOp module, const std::string& phase_id, const std::string& phase_name) {
    std::string prefix = std::string(ESC_START) + COLOR_DEBUG + "[C3-JIT 3.0 TRACE]" + ESC_END;
    std::string separator = "================================================================================";
    llvm::errs() << "\n" << prefix << " " << separator << "\n";
    llvm::errs() << prefix << "  编译阶段: " << phase_id << " - " << phase_name << "\n";
    llvm::errs() << prefix << " " << separator << "\n";
    module.dump();
    llvm::errs() << prefix << " " << separator << "\n\n";
}

// ======================= C3-to-Linalg Lowering Patterns =======================

namespace {
using namespace mlir;

template <typename SrcOp, typename ArithOp>
struct BinaryTensorOpLowering : public OpRewritePattern<SrcOp> {
    using OpRewritePattern<SrcOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(SrcOp op, PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        Value dest = op.getDest();

        auto ctx = rewriter.getContext();
        auto tensorTy = mlir::cast<RankedTensorType>(dest.getType());
        auto lhsTy = mlir::cast<RankedTensorType>(lhs.getType());
        auto rhsTy = mlir::cast<RankedTensorType>(rhs.getType());

        auto lhs_shape = lhsTy.getShape();
        auto rhs_shape = rhsTy.getShape();
        auto out_shape = tensorTy.getShape();

        size_t rank = out_shape.size();
        if (rank == 0) rank = 1;

        std::vector<mlir::AffineExpr> lhs_exprs, rhs_exprs, out_exprs;
        lhs_exprs.reserve(rank);
        rhs_exprs.reserve(rank);
        out_exprs.reserve(rank);

        size_t lhs_offset = rank - lhs_shape.size();
        size_t rhs_offset = rank - rhs_shape.size();

        for (size_t i = 0; i < rank; ++i) {
            mlir::AffineExpr d = rewriter.getAffineDimExpr(i);
            out_exprs.push_back(d);

            // Map LHS dimension i
            if (i < lhs_offset || lhs_shape[i - lhs_offset] == 1) {
                lhs_exprs.push_back(mlir::getAffineConstantExpr(0, ctx));
            } else {
                lhs_exprs.push_back(d);
            }

            // Map RHS dimension i
            if (i < rhs_offset || rhs_shape[i - rhs_offset] == 1) {
                rhs_exprs.push_back(mlir::getAffineConstantExpr(0, ctx));
            } else {
                int64_t r_sz = rhs_shape[i - rhs_offset];
                int64_t l_sz = (i >= lhs_offset) ? lhs_shape[i - lhs_offset] : 1;
                if (l_sz > r_sz && l_sz % r_sz == 0) {
                    rhs_exprs.push_back(d % mlir::getAffineConstantExpr(r_sz, ctx));
                } else {
                    rhs_exprs.push_back(d);
                }
            }
        }

        std::vector<AffineMap> indexingMaps = {
            AffineMap::get(rank, 0, lhs_exprs, ctx),
            AffineMap::get(rank, 0, rhs_exprs, ctx),
            AffineMap::get(rank, 0, out_exprs, ctx)
        };

        std::vector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{tensorTy},
            ValueRange{lhs, rhs},
            ValueRange{dest},
            indexingMaps,
            iterTypes,
            [&](OpBuilder& b, Location regionLoc, ValueRange args) {
                Value res = b.create<ArithOp>(regionLoc, args[0], args[1]);
                b.create<linalg::YieldOp>(regionLoc, ValueRange{res});
            });

        rewriter.replaceOp(op, genericOp.getResults());
        return success();
    }
};

using AddTensorOpLowering = BinaryTensorOpLowering<mlir::c3::AddTensorOp, arith::AddFOp>;
using SubTensorOpLowering = BinaryTensorOpLowering<mlir::c3::SubTensorOp, arith::SubFOp>;
using MulTensorOpLowering = BinaryTensorOpLowering<mlir::c3::MulTensorOp, arith::MulFOp>;
using DivTensorOpLowering = BinaryTensorOpLowering<mlir::c3::DivTensorOp, arith::DivFOp>;

template <typename SrcOp, typename TargetOp>
struct UnaryTensorOpLowering : public OpRewritePattern<SrcOp> {
    using OpRewritePattern<SrcOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(SrcOp op, PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        Value input = op.getInput();
        Value dest = op.getDest();

        auto ctx = rewriter.getContext();
        auto tensorTy = mlir::cast<RankedTensorType>(dest.getType());

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        auto identityMap = AffineMap::get(1, 0, {d0}, ctx);
        std::vector<AffineMap> indexingMaps = {identityMap, identityMap};

        std::vector<utils::IteratorType> iterTypes{utils::IteratorType::parallel};

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{tensorTy},
            ValueRange{input},
            ValueRange{dest},
            indexingMaps,
            iterTypes,
            [&](OpBuilder& b, Location regionLoc, ValueRange args) {
                Value res = b.create<TargetOp>(regionLoc, args[0]);
                b.create<linalg::YieldOp>(regionLoc, ValueRange{res});
            });

        rewriter.replaceOp(op, genericOp.getResults());
        return success();
    }
};

using TanhTensorOpLowering = UnaryTensorOpLowering<mlir::c3::TanhTensorOp, math::TanhOp>;
using ExpTensorOpLowering = UnaryTensorOpLowering<mlir::c3::ExpTensorOp, math::ExpOp>;
using LogTensorOpLowering = UnaryTensorOpLowering<mlir::c3::LogTensorOp, math::LogOp>;
using NegTensorOpLowering = UnaryTensorOpLowering<mlir::c3::NegTensorOp, arith::NegFOp>;

struct ReLUTensorOpLowering : public OpRewritePattern<mlir::c3::ReLUTensorOp> {
    using OpRewritePattern<mlir::c3::ReLUTensorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::c3::ReLUTensorOp op, PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        Value input = op.getInput();
        Value dest = op.getDest();

        auto ctx = rewriter.getContext();
        auto f32 = rewriter.getF32Type();
        auto tensorTy = mlir::cast<RankedTensorType>(dest.getType());

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        auto identityMap = AffineMap::get(1, 0, {d0}, ctx);
        std::vector<AffineMap> indexingMaps = {identityMap, identityMap};

        std::vector<utils::IteratorType> iterTypes{utils::IteratorType::parallel};

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{tensorTy},
            ValueRange{input},
            ValueRange{dest},
            indexingMaps,
            iterTypes,
            [&](OpBuilder& b, Location regionLoc, ValueRange args) {
                Value zero = b.create<arith::ConstantFloatOp>(regionLoc, f32, llvm::APFloat(0.0f));
                Value res = b.create<arith::MaxNumFOp>(regionLoc, args[0], zero);
                b.create<linalg::YieldOp>(regionLoc, ValueRange{res});
            });

        rewriter.replaceOp(op, genericOp.getResults());
        return success();
    }
};

struct SigmoidTensorOpLowering : public OpRewritePattern<mlir::c3::SigmoidTensorOp> {
    using OpRewritePattern<mlir::c3::SigmoidTensorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::c3::SigmoidTensorOp op, PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        Value input = op.getInput();
        Value dest = op.getDest();

        auto ctx = rewriter.getContext();
        auto f32 = rewriter.getF32Type();
        auto tensorTy = mlir::cast<RankedTensorType>(dest.getType());

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        auto identityMap = AffineMap::get(1, 0, {d0}, ctx);
        std::vector<AffineMap> indexingMaps = {identityMap, identityMap};

        std::vector<utils::IteratorType> iterTypes{utils::IteratorType::parallel};

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{tensorTy},
            ValueRange{input},
            ValueRange{dest},
            indexingMaps,
            iterTypes,
            [&](OpBuilder& b, Location regionLoc, ValueRange args) {
                Value x = args[0];
                Value neg_x = b.create<arith::NegFOp>(regionLoc, x);
                Value exp_neg_x = b.create<math::ExpOp>(regionLoc, neg_x);
                Value one = b.create<arith::ConstantFloatOp>(regionLoc, f32, llvm::APFloat(1.0f));
                Value denom = b.create<arith::AddFOp>(regionLoc, one, exp_neg_x);
                Value res = b.create<arith::DivFOp>(regionLoc, one, denom);
                b.create<linalg::YieldOp>(regionLoc, ValueRange{res});
            });

        rewriter.replaceOp(op, genericOp.getResults());
        return success();
    }
};

struct GtTensorOpLowering : public OpRewritePattern<mlir::c3::GtTensorOp> {
    using OpRewritePattern<mlir::c3::GtTensorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::c3::GtTensorOp op, PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        Value dest = op.getDest();
        int64_t bmod = op.getBmod();

        auto ctx = rewriter.getContext();
        auto f32 = rewriter.getF32Type();
        auto tensorTy = mlir::cast<RankedTensorType>(dest.getType());

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        auto identityMap = AffineMap::get(1, 0, {d0}, ctx);
        auto zeroMap = AffineMap::get(1, 0, {getAffineConstantExpr(0, ctx)}, ctx);
        AffineMap modMap;
        if (bmod > 1) {
            modMap = AffineMap::get(1, 0, {d0 % getAffineConstantExpr(bmod, ctx)}, ctx);
        }

        std::vector<AffineMap> indexingMaps;
        if (bmod > 0) {
            indexingMaps = {identityMap, bmod == 1 ? zeroMap : modMap, identityMap};
        } else {
            indexingMaps = {identityMap, identityMap, identityMap};
        }

        std::vector<utils::IteratorType> iterTypes{utils::IteratorType::parallel};

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{tensorTy},
            ValueRange{lhs, rhs},
            ValueRange{dest},
            indexingMaps,
            iterTypes,
            [&](OpBuilder& b, Location regionLoc, ValueRange args) {
                Value cmp = b.create<arith::CmpFOp>(regionLoc, arith::CmpFPredicate::OGT, args[0], args[1]);
                Value zero = b.create<arith::ConstantFloatOp>(regionLoc, f32, llvm::APFloat(0.0f));
                Value one = b.create<arith::ConstantFloatOp>(regionLoc, f32, llvm::APFloat(1.0f));
                Value res = b.create<arith::SelectOp>(regionLoc, cmp, one, zero);
                b.create<linalg::YieldOp>(regionLoc, ValueRange{res});
            });

        rewriter.replaceOp(op, genericOp.getResults());
        return success();
    }
};

struct ConstTensorOpLowering : public OpRewritePattern<mlir::c3::ConstTensorOp> {
    using OpRewritePattern<mlir::c3::ConstTensorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::c3::ConstTensorOp op, PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        Value dest = op.getDest();
        float value = op.getValue().convertToFloat();

        auto ctx = rewriter.getContext();
        auto f32 = rewriter.getF32Type();
        auto tensorTy = mlir::cast<RankedTensorType>(dest.getType());

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        auto identityMap = AffineMap::get(1, 0, {d0}, ctx);
        std::vector<AffineMap> indexingMaps = {identityMap};

        std::vector<utils::IteratorType> iterTypes{utils::IteratorType::parallel};

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{tensorTy},
            ValueRange{},
            ValueRange{dest},
            indexingMaps,
            iterTypes,
            [&](OpBuilder& b, Location regionLoc, ValueRange /*args*/) {
                Value res = b.create<arith::ConstantFloatOp>(regionLoc, f32, llvm::APFloat(value));
                b.create<linalg::YieldOp>(regionLoc, ValueRange{res});
            });

        rewriter.replaceOp(op, genericOp.getResults());
        return success();
    }
};

struct MatMulTensorOpLowering : public OpRewritePattern<mlir::c3::MatMulTensorOp> {
    using OpRewritePattern<mlir::c3::MatMulTensorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::c3::MatMulTensorOp op, PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        Value dest = op.getDest();

        auto ctx = rewriter.getContext();
        auto tensorTy = mlir::cast<RankedTensorType>(dest.getType());

        auto matmulOp = rewriter.create<linalg::MatmulOp>(
            loc,
            TypeRange{tensorTy},
            ValueRange{lhs, rhs},
            ValueRange{dest}
        );

        rewriter.replaceOp(op, matmulOp.getResults());
        return success();
    }
};

struct TransposeTensorOpLowering : public OpRewritePattern<mlir::c3::TransposeTensorOp> {
    using OpRewritePattern<mlir::c3::TransposeTensorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::c3::TransposeTensorOp op, PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        Value input = op.getInput();
        Value dest = op.getDest();

        auto ctx = rewriter.getContext();
        auto tensorTy = mlir::cast<RankedTensorType>(dest.getType());

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        AffineExpr d1 = rewriter.getAffineDimExpr(1);
        auto map_in = AffineMap::get(2, 0, {d0, d1}, ctx);
        auto map_out = AffineMap::get(2, 0, {d1, d0}, ctx);

        std::vector<AffineMap> indexingMaps = {map_in, map_out};
        std::vector<utils::IteratorType> iterTypes = {utils::IteratorType::parallel, utils::IteratorType::parallel};

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{tensorTy},
            ValueRange{input},
            ValueRange{dest},
            indexingMaps,
            iterTypes,
            [&](OpBuilder& b, Location regionLoc, ValueRange args) {
                b.create<linalg::YieldOp>(regionLoc, ValueRange{args[0]});
            });

        rewriter.replaceOp(op, genericOp.getResults());
        return success();
    }
};

struct SumReduceTensorOpLowering : public OpRewritePattern<mlir::c3::SumReduceTensorOp> {
    using OpRewritePattern<mlir::c3::SumReduceTensorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::c3::SumReduceTensorOp op, PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        Value input = op.getInput();
        Value dest = op.getDest();
        int axis = op.getAxis();

        auto ctx = rewriter.getContext();
        auto tensorTy = mlir::cast<RankedTensorType>(dest.getType());

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        AffineExpr d1 = rewriter.getAffineDimExpr(1);
        auto map_in = AffineMap::get(2, 0, {d0, d1}, ctx);

        AffineMap map_out;
        std::vector<utils::IteratorType> iterTypes;

        if (axis == 0) {
            map_out = AffineMap::get(2, 0, {d1}, ctx);
            iterTypes = {utils::IteratorType::reduction, utils::IteratorType::parallel};
        } else {
            map_out = AffineMap::get(2, 0, {d0}, ctx);
            iterTypes = {utils::IteratorType::parallel, utils::IteratorType::reduction};
        }

        std::vector<AffineMap> indexingMaps = {map_in, map_out};

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{tensorTy},
            ValueRange{input},
            ValueRange{dest},
            indexingMaps,
            iterTypes,
            [&](OpBuilder& b, Location regionLoc, ValueRange args) {
                Value res = b.create<arith::AddFOp>(regionLoc, args[1], args[0]);
                b.create<linalg::YieldOp>(regionLoc, ValueRange{res});
            });

          rewriter.replaceOp(op, genericOp.getResults());
          return success();
      }
  };

} // namespace

// ======================= MLIR Module Builder with Tensors =======================

static mlir::OwningOpRef<mlir::ModuleOp> buildTensorMLIRModule(mlir::MLIRContext& context, const Graph& graph,
                                                               size_t num_inputs, size_t num_outputs) {
    auto loc = mlir::UnknownLoc::get(&context);
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto f32 = builder.getF32Type();

    // 全静态形状特化通道：默认开启 (C3_LINALG_STATIC=0 可关闭)
    const char* static_env = std::getenv("C3_LINALG_STATIC");
    bool use_static_shape = (static_env == nullptr || std::string(static_env) != "0");

    const auto& graph_inputs = graph.inputs();
    const auto& graph_outputs = graph.outputs();

    // 辅助函数：根据算子输出获取其 1D 扁平张量类型 (静态/动态)
    auto getFlatTensorTypeForNode = [&](size_t node_id) {
        if (!use_static_shape) {
            return mlir::RankedTensorType::get({mlir::ShapedType::kDynamic}, f32);
        }
        int64_t node_numel = static_cast<int64_t>(graph.node(node_id).out_desc.numel);
        return mlir::RankedTensorType::get({node_numel}, f32);
    };

    // 辅助函数：根据算子输出获取其真实多维张量类型 (静态/动态)
    auto getMultiDimTensorTypeForNode = [&](size_t node_id) {
        const auto& shape = graph.node(node_id).out_desc.shape;
        if (!use_static_shape || shape.empty()) {
            return mlir::RankedTensorType::get({mlir::ShapedType::kDynamic}, f32);
        }
        std::vector<int64_t> static_shape;
        for (size_t s : shape) {
            static_shape.push_back(static_cast<int64_t>(s));
        }
        return mlir::RankedTensorType::get(static_shape, f32);
    };

    int64_t base_numel = 1;
    if (!graph_outputs.empty()) {
        base_numel = static_cast<int64_t>(graph.node(graph_outputs[0]).out_desc.numel);
    }

    // 函数签名: N 个输入 tensors + M 个输出 tensors -> M 个输出 tensors
    std::vector<mlir::Type> arg_types;
    arg_types.reserve(num_inputs + num_outputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        arg_types.push_back(getFlatTensorTypeForNode(graph_inputs[i]));
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        arg_types.push_back(getFlatTensorTypeForNode(graph_outputs[i]));
    }

    std::vector<mlir::Type> ret_types;
    ret_types.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        ret_types.push_back(getFlatTensorTypeForNode(graph_outputs[i]));
    }

    auto funcType = builder.getFunctionType(arg_types, ret_types);
    auto func = builder.create<mlir::func::FuncOp>(loc, "c3_kernel", funcType);

    // 声明输出 arguments 是可写的
    for (size_t i = 0; i < num_outputs; ++i) {
        func.setArgAttr(num_inputs + i, "bufferization.writable", builder.getBoolAttr(true));
    }

    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // 内部临时变量映射：node_id -> Value (tensor)
    std::unordered_map<size_t, mlir::Value> val_map;

    for (size_t i = 0; i < num_inputs; ++i) {
        size_t input_id = graph_inputs[i];
        mlir::Value flat_val = entry->getArgument(static_cast<unsigned>(i));
        const auto& shape = graph.node(input_id).out_desc.shape;
        if (shape.size() > 1) {
            std::vector<int64_t> static_shape;
            for (size_t s : shape) static_shape.push_back(static_cast<int64_t>(s));
            auto multiTy = mlir::RankedTensorType::get(static_shape, f32);
            mlir::ReassociationIndices indices;
            for (size_t d = 0; d < shape.size(); ++d) indices.push_back(static_cast<int64_t>(d));
            mlir::SmallVector<mlir::ReassociationIndices> reassociation;
            reassociation.push_back(indices);
            val_map[input_id] = builder.create<mlir::tensor::ExpandShapeOp>(loc, multiTy, flat_val, reassociation);
        } else {
            val_map[input_id] = flat_val;
        }
    }

    auto createEmptyTensorOfShape = [&](const std::vector<size_t>& shape) -> mlir::Value {
        if (!use_static_shape || shape.empty()) {
            mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            mlir::Value d0_dim = builder.create<mlir::tensor::DimOp>(loc, entry->getArgument(0), c0);
            return builder.create<mlir::tensor::EmptyOp>(loc, llvm::ArrayRef<int64_t>{mlir::ShapedType::kDynamic}, f32, mlir::ValueRange{d0_dim});
        }
        std::vector<int64_t> static_shape;
        for (size_t s : shape) {
            static_shape.push_back(static_cast<int64_t>(s));
        }
        return builder.create<mlir::tensor::EmptyOp>(loc, static_shape, f32);
    };

    // 占位空张量
    mlir::Value empty_tensor;
    if (use_static_shape) {
        empty_tensor = builder.create<mlir::tensor::EmptyOp>(loc, llvm::ArrayRef<int64_t>{base_numel}, f32);
    } else {
        mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        mlir::Value d0_dim = builder.create<mlir::tensor::DimOp>(loc, entry->getArgument(0), c0);
        empty_tensor = builder.create<mlir::tensor::EmptyOp>(loc, llvm::ArrayRef<int64_t>{mlir::ShapedType::kDynamic}, f32, mlir::ValueRange{d0_dim});
    }

    // 拓扑排序计算节点
    std::vector<const Node*> compute_nodes;
    for (const auto& node : graph.nodes()) {
        if (std::find(graph_inputs.begin(), graph_inputs.end(), node.id) != graph_inputs.end()) {
            continue;
        }
        compute_nodes.push_back(&node);
    }

    for (const auto* node : compute_nodes) {
        // 查找此算子的输出是否属于图输出
        auto it = std::find(graph_outputs.begin(), graph_outputs.end(), node->id);
        mlir::Value dest;
        if (it != graph_outputs.end()) {
            size_t out_idx = std::distance(graph_outputs.begin(), it);
            mlir::Value flat_dest = entry->getArgument(static_cast<unsigned>(num_inputs + out_idx));
            const auto& shape = node->out_desc.shape;
            if (shape.size() > 1) {
                std::vector<int64_t> static_shape;
                for (size_t s : shape) static_shape.push_back(static_cast<int64_t>(s));
                auto multiTy = mlir::RankedTensorType::get(static_shape, f32);
                mlir::ReassociationIndices indices;
                for (size_t d = 0; d < shape.size(); ++d) indices.push_back(static_cast<int64_t>(d));
                mlir::SmallVector<mlir::ReassociationIndices> reassociation;
                reassociation.push_back(indices);
                dest = builder.create<mlir::tensor::ExpandShapeOp>(loc, multiTy, flat_dest, reassociation);
            } else {
                dest = flat_dest;
            }
        } else {
            dest = createEmptyTensorOfShape(node->out_desc.shape);
        }

        mlir::Value result;
        auto tensorType = getMultiDimTensorTypeForNode(node->id);

        std::visit([&](auto&& op_node) {
            using T = std::decay_t<decltype(op_node)>;

            if constexpr (std::is_same_v<T, AddNode>) {
                mlir::Value lhs = val_map.at(node->inputs[0]);
                mlir::Value rhs = val_map.at(node->inputs[1]);
                result = builder.create<mlir::c3::AddTensorOp>(loc, tensorType, lhs, rhs, dest, 0);
            }
            else if constexpr (std::is_same_v<T, SubNode>) {
                mlir::Value lhs = val_map.at(node->inputs[0]);
                mlir::Value rhs = val_map.at(node->inputs[1]);
                result = builder.create<mlir::c3::SubTensorOp>(loc, tensorType, lhs, rhs, dest, 0);
            }
            else if constexpr (std::is_same_v<T, MulNode>) {
                mlir::Value lhs = val_map.at(node->inputs[0]);
                mlir::Value rhs = val_map.at(node->inputs[1]);
                result = builder.create<mlir::c3::MulTensorOp>(loc, tensorType, lhs, rhs, dest, 0);
            }
            else if constexpr (std::is_same_v<T, DivNode>) {
                mlir::Value lhs = val_map.at(node->inputs[0]);
                mlir::Value rhs = val_map.at(node->inputs[1]);
                result = builder.create<mlir::c3::DivTensorOp>(loc, tensorType, lhs, rhs, dest, 0);
            }
            else if constexpr (std::is_same_v<T, NegNode>) {
                mlir::Value input = val_map.at(node->inputs[0]);
                result = builder.create<mlir::c3::NegTensorOp>(loc, tensorType, input, dest);
            }
            else if constexpr (std::is_same_v<T, ReLUNode>) {
                mlir::Value input = val_map.at(node->inputs[0]);
                result = builder.create<mlir::c3::ReLUTensorOp>(loc, tensorType, input, dest);
            }
            else if constexpr (std::is_same_v<T, SigmoidNode>) {
                mlir::Value input = val_map.at(node->inputs[0]);
                result = builder.create<mlir::c3::SigmoidTensorOp>(loc, tensorType, input, dest);
            }
            else if constexpr (std::is_same_v<T, TanhNode>) {
                mlir::Value input = val_map.at(node->inputs[0]);
                result = builder.create<mlir::c3::TanhTensorOp>(loc, tensorType, input, dest);
            }
            else if constexpr (std::is_same_v<T, ExpNode>) {
                mlir::Value input = val_map.at(node->inputs[0]);
                result = builder.create<mlir::c3::ExpTensorOp>(loc, tensorType, input, dest);
            }
            else if constexpr (std::is_same_v<T, LogNode>) {
                mlir::Value input = val_map.at(node->inputs[0]);
                result = builder.create<mlir::c3::LogTensorOp>(loc, tensorType, input, dest);
            }
            else if constexpr (std::is_same_v<T, GtNode>) {
                mlir::Value lhs = val_map.at(node->inputs[0]);
                mlir::Value rhs = val_map.at(node->inputs[1]);
                result = builder.create<mlir::c3::GtTensorOp>(loc, tensorType, lhs, rhs, dest, 0);
            }
            else if constexpr (std::is_same_v<T, ConstNode>) {
                auto attr = builder.getFloatAttr(f32, op_node.value);
                result = builder.create<mlir::c3::ConstTensorOp>(loc, tensorType, attr, dest);
            }
            else if constexpr (std::is_same_v<T, MatMulNode>) {
                mlir::Value lhs = val_map.at(node->inputs[0]);
                mlir::Value rhs = val_map.at(node->inputs[1]);
                int64_t matM = op_node.lhs_desc.shape.size() > 0 ? (int64_t)op_node.lhs_desc.shape[0] : 0;
                int64_t matK = op_node.lhs_desc.shape.size() > 1 ? (int64_t)op_node.lhs_desc.shape[1] : 0;
                int64_t matN = op_node.rhs_desc.shape.size() > 1 ? (int64_t)op_node.rhs_desc.shape[1] : 0;
                result = builder.create<mlir::c3::MatMulTensorOp>(
                    loc, tensorType, lhs, rhs, dest,
                    matM, matK, matN,
                    111, 111, 0, 0, 0, 0
                );
            }
            else if constexpr (std::is_same_v<T, TransposeNode>) {
                mlir::Value input = val_map.at(node->inputs[0]);
                int64_t M = op_node.in_desc.shape.size() > 0 ? op_node.in_desc.shape[0] : 1;
                int64_t N = op_node.in_desc.shape.size() > 1 ? op_node.in_desc.shape[1] : 1;
                result = builder.create<mlir::c3::TransposeTensorOp>(
                    loc, tensorType, input, dest,
                    M, N, op_node.dim0, op_node.dim1
                );
            }
            else if constexpr (std::is_same_v<T, SumReduceNode>) {
                mlir::Value input = val_map.at(node->inputs[0]);
                int64_t M = op_node.in_desc.shape.size() > 0 ? op_node.in_desc.shape[0] : 1;
                int64_t N = op_node.in_desc.shape.size() > 1 ? op_node.in_desc.shape[1] : 1;
                result = builder.create<mlir::c3::SumReduceTensorOp>(
                    loc, tensorType, input, dest,
                    M, N, op_node.axis
                );
            }
            else if constexpr (std::is_same_v<T, FusedNode>) {
                // 内置 FusedNode 展开：将其内部所有 ops 作为独立的 c3 Dialect Tensor 算子内联发射
                const auto& fnode = op_node;
                std::unordered_map<size_t, mlir::Value> f_map;
                // 建立 FusedNode 输入映射
                for (size_t aidx = 0; aidx < fnode.arg_node_ids.size(); ++aidx) {
                    f_map[fnode.arg_node_ids[aidx]] = val_map.at(node->inputs[aidx]);
                }

                mlir::Value f_prev_val;
                for (size_t f_idx = 0; f_idx < fnode.ops.size(); ++f_idx) {
                    const auto& f_op = fnode.ops[f_idx];
                    const auto& f_inputs = fnode.op_inputs[f_idx];

                    std::vector<size_t> ext_inputs;
                    for (size_t in_id : f_inputs) {
                        if (f_idx > 0 && in_id == f_inputs[0]) continue;
                        ext_inputs.push_back(in_id);
                    }

                    auto loadExt = [&](size_t id) -> mlir::Value {
                        return f_map.at(id);
                    };

                    mlir::Value f_res;
                    mlir::Value f_dest = (f_idx == fnode.ops.size() - 1) ? dest : empty_tensor;

                    std::visit([&](auto&& f_inner_node) {
                        using FT = std::decay_t<decltype(f_inner_node)>;
                        if constexpr (std::is_same_v<FT, ConstNode>) {
                            auto attr = builder.getFloatAttr(f32, f_inner_node.value);
                            f_res = builder.create<mlir::c3::ConstTensorOp>(loc, tensorType, attr, f_dest);
                        }
                        else if constexpr (std::is_same_v<FT, NegNode>) {
                            mlir::Value f_in = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                            f_res = builder.create<mlir::c3::NegTensorOp>(loc, tensorType, f_in, f_dest);
                        }
                        else if constexpr (std::is_same_v<FT, ReLUNode>) {
                            mlir::Value f_in = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                            f_res = builder.create<mlir::c3::ReLUTensorOp>(loc, tensorType, f_in, f_dest);
                        }
                        else if constexpr (std::is_same_v<FT, SigmoidNode>) {
                            mlir::Value f_in = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                            f_res = builder.create<mlir::c3::SigmoidTensorOp>(loc, tensorType, f_in, f_dest);
                        }
                        else if constexpr (std::is_same_v<FT, TanhNode>) {
                            mlir::Value f_in = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                            f_res = builder.create<mlir::c3::TanhTensorOp>(loc, tensorType, f_in, f_dest);
                        }
                        else if constexpr (std::is_same_v<FT, ExpNode>) {
                            mlir::Value f_in = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                            f_res = builder.create<mlir::c3::ExpTensorOp>(loc, tensorType, f_in, f_dest);
                        }
                        else if constexpr (std::is_same_v<FT, LogNode>) {
                            mlir::Value f_in = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                            f_res = builder.create<mlir::c3::LogTensorOp>(loc, tensorType, f_in, f_dest);
                        }
                        else if constexpr (std::is_same_v<FT, AddNode>) {
                            mlir::Value f_lhs = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                            mlir::Value f_rhs = loadExt(ext_inputs[f_idx > 0 ? 0 : 1]);
                            f_res = builder.create<mlir::c3::AddTensorOp>(loc, tensorType, f_lhs, f_rhs, f_dest, 0);
                        }
                        else if constexpr (std::is_same_v<FT, SubNode>) {
                            mlir::Value f_lhs = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                            mlir::Value f_rhs = loadExt(ext_inputs[f_idx > 0 ? 0 : 1]);
                            f_res = builder.create<mlir::c3::SubTensorOp>(loc, tensorType, f_lhs, f_rhs, f_dest, 0);
                        }
                        else if constexpr (std::is_same_v<FT, MulNode>) {
                            mlir::Value f_lhs = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                            mlir::Value f_rhs = loadExt(ext_inputs[f_idx > 0 ? 0 : 1]);
                            f_res = builder.create<mlir::c3::MulTensorOp>(loc, tensorType, f_lhs, f_rhs, f_dest, 0);
                        }
                        else if constexpr (std::is_same_v<FT, DivNode>) {
                            mlir::Value f_lhs = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                            mlir::Value f_rhs = loadExt(ext_inputs[f_idx > 0 ? 0 : 1]);
                            f_res = builder.create<mlir::c3::DivTensorOp>(loc, tensorType, f_lhs, f_rhs, f_dest, 0);
                        }
                        else if constexpr (std::is_same_v<FT, GtNode>) {
                            mlir::Value f_lhs = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                            mlir::Value f_rhs = loadExt(ext_inputs[f_idx > 0 ? 0 : 1]);
                            f_res = builder.create<mlir::c3::GtTensorOp>(loc, tensorType, f_lhs, f_rhs, f_dest, 0);
                        }
                    }, f_op);

                    f_prev_val = f_res;
                    f_map[fnode.op_node_ids[f_idx]] = f_res;
                    if (f_idx == fnode.ops.size() - 1) {
                        result = f_res;
                    }
                }
            }
        }, node->op);

        val_map[node->id] = result;
    }

    std::vector<mlir::Value> returns;
    for (size_t out_id : graph_outputs) {
        mlir::Value multi_val = val_map.at(out_id);
        const auto& shape = graph.node(out_id).out_desc.shape;
        if (shape.size() > 1) {
            int64_t node_numel = static_cast<int64_t>(graph.node(out_id).out_desc.numel);
            auto flatTy = mlir::RankedTensorType::get({node_numel}, f32);
            mlir::ReassociationIndices indices;
            for (size_t d = 0; d < shape.size(); ++d) indices.push_back(static_cast<int64_t>(d));
            mlir::SmallVector<mlir::ReassociationIndices> reassociation;
            reassociation.push_back(indices);
            mlir::Value flat_val = builder.create<mlir::tensor::CollapseShapeOp>(loc, flatTy, multi_val, reassociation);
            returns.push_back(flat_val);
        } else {
            returns.push_back(multi_val);
        }
    }

    builder.create<mlir::func::ReturnOp>(loc, returns);
    return module;
}

// ======================= 對标 XLA 编译转换管线 =======================

static void applyUnifiedTransformPipeline(mlir::ModuleOp module, size_t num_inputs, size_t num_outputs, size_t base_numel) {
    bool is_verbose = isVerboseDebugEnabled();
    if (is_verbose) {
        dumpPhaseIR(module, "Phase 1.0", "原始 Tensor-based C3 Dialect IR");
    }

    // 1. C3-to-Linalg Lowering
    {
        mlir::RewritePatternSet patterns(module.getContext());
        patterns.add<AddTensorOpLowering, SubTensorOpLowering, MulTensorOpLowering, DivTensorOpLowering,
                     NegTensorOpLowering, ReLUTensorOpLowering, SigmoidTensorOpLowering, TanhTensorOpLowering,
                     ExpTensorOpLowering, LogTensorOpLowering, GtTensorOpLowering, ConstTensorOpLowering,
                     MatMulTensorOpLowering, TransposeTensorOpLowering, SumReduceTensorOpLowering>(module.getContext());
        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
            throw std::runtime_error("C3 to Linalg lowering failed");
        }
        if (is_verbose) {
            dumpPhaseIR(module, "Phase 2.0", "C3-to-Linalg Lowering 后的 Tensor IR");
        }
    }

    // 1.2 [Fast-Math] 多项式逼近 + 代数化简（受 C3_FAST_MATH=1 控制）
    //    使用多项式近似替换 math::TanhOp / math::ExpOp / math::LogOp 等超越函数，
    //    避免浮点函数调用开销（~10-30 cycle/op → ~2-4 cycle/op），
    //    相对误差 ~1e-6，通常不影响分类精度（MNIST 97.18% 可保持）。
    if (isFastMathEnabled()) {
        mlir::RewritePatternSet fast_math_patterns(module.getContext());
        mlir::populateMathAlgebraicSimplificationPatterns(fast_math_patterns);
        mlir::populateMathPolynomialApproximationPatterns(fast_math_patterns);
        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module, std::move(fast_math_patterns)))) {
            throw std::runtime_error("Fast-Math polynomial approximation patterns failed");
        }
        if (is_verbose) {
            dumpPhaseIR(module, "Phase 2.1", "Fast-Math 多项式逼近后的 Tensor IR");
        }
    }

    // 1.5 Linalg Elementwise 自动图融合与折叠
    {
        mlir::PassManager pm(module.getContext());
        pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
        pm.addPass(mlir::createLinalgFoldIntoElementwisePass());
        pm.addPass(mlir::createLinalgFoldUnitExtentDimsPass());
        pm.addPass(mlir::createLinalgInlineScalarOperandsPass());
        pm.addPass(mlir::createLinalgSpecializeGenericOpsPass());
        pm.addPass(mlir::createInlinerPass());
        pm.addPass(mlir::createSCCPPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        if (mlir::failed(pm.run(module))) {
            throw std::runtime_error("Linalg fusion and folding optimizations failed");
        }
        if (is_verbose) {
            dumpPhaseIR(module, "Phase 3.0", "Linalg Elementwise 自动算子融合后的 Tensor IR");
        }
    }

    // 2. [极致优化] 擦除 Linalg Fusion 产生的临时 tensor.empty 并绑定到 writable 输出参数上 (DPS 目的绑定)
    {
        auto func = module.lookupSymbol<mlir::func::FuncOp>("c3_kernel");
        if (func) {
            func.walk([&](mlir::func::ReturnOp returnOp) {
                for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
                    mlir::Value retVal = returnOp.getOperand(i);
                    mlir::Value out_arg = func.getArgument(static_cast<unsigned>(num_inputs + i));
                    // 泛化处理所有 Destination-Style op（linalg.generic 与 linalg.mul 等命名算子），
                    // 避免 Linalg 融合把 generic 折叠成命名 op 后 DPS 重绑定失效（输出全 0 回归）
                    // LLVM 22 的 DPS 接口使用 getDpsInits/getTiedOpOperand 系列 API
                    if (auto dstOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(retVal.getDefiningOp())) {
                        if (auto retOpResult = mlir::dyn_cast<mlir::OpResult>(retVal)) {
                            if (mlir::OpOperand* initOperand = dstOp.getTiedOpOperand(retOpResult)) {
                                mlir::Value outVal = initOperand->get();
                                if (auto emptyOp = outVal.getDefiningOp<mlir::tensor::EmptyOp>()) {
                                    initOperand->set(out_arg);
                                    if (emptyOp->use_empty()) {
                                        emptyOp->erase();
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
        if (is_verbose) {
            dumpPhaseIR(module, "Phase 4.0", "DPS 目的地址重绑定后的 Tensor IR");
        }
    }

    // 3. One-Shot Bufferization (Tensor-to-MemRef 转换)
    {
        mlir::PassManager pm(module.getContext());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        mlir::bufferization::OneShotBufferizePassOptions options;
        options.bufferizeFunctionBoundaries = true;
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
        if (mlir::failed(pm.run(module))) {
            throw std::runtime_error("One-Shot Bufferization failed");
        }
        if (is_verbose) {
            dumpPhaseIR(module, "Phase 5.0", "One-Shot Bufferization (Tensor-to-MemRef) 转换后的 MemRef IR");
        }
    }

    // 3.5 [极致堆转栈与Buffer提升] Buffer Hoisting, Loop Hoisting, Stack Promotion & Liveness Optimization
    {
        mlir::PassManager pm(module.getContext());
        auto& func_pm = pm.nest<mlir::func::FuncOp>();
        func_pm.addPass(mlir::bufferization::createBufferHoistingPass());
        func_pm.addPass(mlir::bufferization::createBufferLoopHoistingPass());
        func_pm.addPass(mlir::bufferization::createPromoteBuffersToStackPass());
        func_pm.addPass(mlir::bufferization::createOptimizeAllocationLivenessPass());
        func_pm.addPass(mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
        func_pm.addPass(mlir::bufferization::createBufferDeallocationSimplificationPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        if (mlir::failed(pm.run(module))) {
            throw std::runtime_error("Buffer optimization passes (Hoisting / Stack Promotion) failed");
        }
        if (is_verbose) {
            dumpPhaseIR(module, "Phase 5.5", "应用 PromoteBuffersToStack 与 BufferHoisting 等优化后的 MemRef IR");
        }
    }

    // 4. [修复] 擦除返回值，将函数签名在 memref 层级重写为 void()，避免 JIT Struct ABI 平台崩溃
    {
        auto func = module.lookupSymbol<mlir::func::FuncOp>("c3_kernel");
        if (func) {
            mlir::OpBuilder builder(module.getContext());
            auto newFuncType = builder.getFunctionType(func.getArgumentTypes(), {});
            func.setType(newFuncType);
            func.walk([](mlir::func::ReturnOp returnOp) {
                mlir::OpBuilder b(returnOp);
                b.create<mlir::func::ReturnOp>(returnOp.getLoc());
                returnOp.erase();
            });
        }
        if (is_verbose) {
            dumpPhaseIR(module, "Phase 6.0", "擦除返回值重写为 void() 后的 MemRef IR");
        }
    }

    // 5. 为所有 memref 参数注入 llvm.noalias 属性，确保硬件 SIMD 100% 自动向量化
    {
        auto func = module.lookupSymbol<mlir::func::FuncOp>("c3_kernel");
        if (func) {
            mlir::OpBuilder builder(module.getContext());
            for (unsigned i = 0; i < func.getNumArguments(); ++i) {
                func.setArgAttr(i, "llvm.noalias", builder.getUnitAttr());
            }
        }
        if (is_verbose) {
            dumpPhaseIR(module, "Phase 7.0", "注入 llvm.noalias 属性后的 MemRef IR");
        }
    }

    // 6. Linalg to Loops & SCF/Arith/Math to LLVM
    {
        mlir::PassManager pm(module.getContext());
        const char* tile_size_env = std::getenv("C3_LINALG_TILE_SIZE");
        int64_t tile_size = tile_size_env ? std::stoll(tile_size_env) : 0;

        bool use_openmp = isOpenMPEnabled();

        // 自动高维分块展开：若未显式指定（且为 0），大张量（>= 65536 元素）默认使用 2048 进行 1D 缓存分块，适应 L1 Cache Line
        // OpenMP 模式下不做显式 tiling，直接生成 scf.parallel 交给 libomp 调度分块
        if (tile_size == 0 && base_numel >= 65536 && !use_openmp) {
            tile_size = 2048;
        }

        if (tile_size > 0) {
            pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
            pm.addPass(mlir::createParallelLoopTilingPass({tile_size}));
        } else {
            pm.addPass(mlir::createConvertLinalgToLoopsPass());
        }

        // [Pass 全家桶 2026-08-16] 低风险语义保持优化（均为纯语义保持 pass，可安全开启）
        pm.addPass(mlir::createForLoopSpecializationPass());   // scf.for 特化（常量步长 / 常数边界）
        if (use_openmp) {
            pm.addPass(mlir::createParallelLoopFusionPass());  // 相邻并行循环融合：仅在 OpenMP 多线程并行循环模式下有意义，串行时为 no-op
        }
        pm.addPass(mlir::createControlFlowSinkPass());         // 控制流下沉（将区域不变的 op 下沉到分支）
        pm.addPass(mlir::createRemoveDeadValuesPass());        // 前向死值消除（配合 store 消除，辅助向量化）
        pm.addPass(mlir::createLoopInvariantCodeMotionPass()); // 运行 LICM (循环不变代码外提)

        // [OpenMP 真多核 2026-08-16] C3_LINALG_OMP=1 时走 scf.parallel → omp → LLVM，
        // 生成 __kmpc_* 运行时调用，libomp 做真实多线程分块（不再退化到串行 scf.for）。
        if (use_openmp) {
            pm.addPass(mlir::createConvertSCFToOpenMPPass());   // scf.parallel → omp.parallel + omp.wsloop
            pm.addPass(mlir::createConvertOpenMPToLLVMPass());  // omp → LLVM（__kmpc_* 调用）
        }

        pm.addPass(mlir::createSCFToControlFlowPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertMathToLLVMPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        if (mlir::failed(pm.run(module))) {
            throw std::runtime_error("MLIROneShotGen: lowering pipeline failed");
        }
        if (isDebugEnabled()) {
            dumpPhaseIR(module, "Phase 8.0 (最终)", "最终 Lowering 后的 LLVM Dialect IR 模块");
        }
    }
}

static std::unique_ptr<mlir::ExecutionEngine> createEngine(
    mlir::ModuleOp module, int opt_level, const std::string& cache_graph,
    std::function<std::unique_ptr<llvm::Module>(mlir::Operation*, llvm::LLVMContext&)>& builder_slot) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // [OpenMP 真多核 2026-08-16] 提前将 OpenMP 运行时加载进进程，
    // 使 ORC JIT 解析 __kmpc_* 符号（omp.parallel 展开为 libomp 运行时调用）。
    // 失败不致命：只会导致该 kernel 走串行路径（或 JIT 链接报未定义符号）。
    if (isOpenMPEnabled()) {
        const char* env_lib = std::getenv("C3_OMP_LIBRARY");
        llvm::sys::DynamicLibrary::LoadLibraryPermanently(
            env_lib ? env_lib : "libomp.dylib", /*errMsg=*/nullptr);
        llvm::sys::DynamicLibrary::LoadLibraryPermanently(
            env_lib ? env_lib : "libomp.so", /*errMsg=*/nullptr);
    }

    auto tm = std::shared_ptr<llvm::TargetMachine>(
        llvm::EngineBuilder()
            .setEngineKind(llvm::EngineKind::JIT)
            .setOptLevel(static_cast<llvm::CodeGenOptLevel>(opt_level))
            .selectTarget());

    std::function<llvm::Error(llvm::Module*)> opt_transformer =
        tm ? mlir::makeOptimizingTransformer(static_cast<unsigned>(opt_level), 0, tm.get())
           : std::function<llvm::Error(llvm::Module*)>();

    mlir::ExecutionEngineOptions engineOpts;
    if (opt_transformer) {
        engineOpts.transformer = opt_transformer;
    }
    engineOpts.jitCodeGenOptLevel = (opt_level >= 3) ? llvm::CodeGenOptLevel::Aggressive : llvm::CodeGenOptLevel::Default;

    // [持久化 AOT 缓存 2.0]
    if (JITCache::isEnabled()) {
        try {
            // Fast-Math 编译产物（多项式逼近 + 后端 fast-math）与普通产物不兼容，
            // 在缓存 key 上附加标记，避免相互污染缓存。
            std::string cache_graph_key = isFastMathEnabled() ? (cache_graph + ":fastmath") : cache_graph;
            std::string jit_key = JITCache::makeKey(cache_graph_key, opt_level);
            std::string bc_path = JITCache::getInstance().lookup(jit_key);

            if (!bc_path.empty()) {
                if (isDebugEnabled()) {
                    CtorchError::trace(ErrorPlatform::kAutoDiff, "AOT 缓存命中! 正在载入 AOT 预编译 bitcode 路径: " + bc_path);
                }
                builder_slot = [bc_path](mlir::Operation*, llvm::LLVMContext& ctx) {
                    return JITCache::getInstance().loadBitcode(bc_path, ctx);
                };
            } else {
                if (isDebugEnabled()) {
                    CtorchError::trace(ErrorPlatform::kAutoDiff, "AOT 缓存未命中! 正在实时 JIT 编译、优化并将 bitcode 暂存至 Key: " + jit_key);
                }
                builder_slot = [module, jit_key](mlir::Operation*, llvm::LLVMContext& ctx) {
                    auto llvm_module = mlir::translateModuleToLLVMIR(module, ctx);
                    if (llvm_module) {
                        JITCache::getInstance().store(jit_key, *llvm_module);
                    }
                    return llvm_module;
                };
            }

            // [Fast-Math] 后端级不安全浮点：在 LLVM 模块上打 unsafe-fp-math / fp-contract 函数属性，
            // 让 SelectionDAG 把 exp/log 替换为 exp2+多项式、并允许 FMA 融合（对应 -ffast-math）。
            // 仅 C3_FAST_MATH=1 时生效；命中缓存路径（loadBitcode）同样被包裹。
            if (isFastMathEnabled()) {
                auto inner = builder_slot;
                builder_slot = [inner](mlir::Operation* op, llvm::LLVMContext& ctx) {
                    auto mod = inner(op, ctx);
                    if (mod) {
                        for (auto& fn : mod->functions()) {
                            fn.addFnAttr("unsafe-fp-math", "true");
                            fn.addFnAttr("fp-contract", "fast");
                        }
                    }
                    return mod;
                };
            }

            engineOpts.llvmModuleBuilder = builder_slot;
        } catch (...) {}
    }

    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOpts);
    if (!maybeEngine) {
        throw std::runtime_error("LinalgOneShotGen: failed to create ExecutionEngine");
    }
    return std::move(*maybeEngine);
}

// ======================= LinalgOneShotKernel Impl =======================

struct LinalgOneShotKernel::Impl {
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> heldModule;
    std::function<std::unique_ptr<llvm::Module>(mlir::Operation*, llvm::LLVMContext&)> aotBuilder;
    std::unique_ptr<mlir::ExecutionEngine> engine;

    Impl() {
        mlir::DialectRegistry registry;
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::math::MathDialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::memref::MemRefDialect>();
        registry.insert<mlir::tensor::TensorDialect>();
        registry.insert<mlir::linalg::LinalgDialect>();
        registry.insert<mlir::c3::C3Dialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
        if (isOpenMPEnabled()) {
            registry.insert<mlir::omp::OpenMPDialect>();
        }

        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
        if (isOpenMPEnabled()) {
            mlir::registerOpenMPDialectTranslation(registry);
        }

        mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();
    }
};

LinalgOneShotKernel::LinalgOneShotKernel(const Graph& graph, int opt_level) {
    std::unique_ptr<Impl> temp_impl;
    {
        std::lock_guard<std::mutex> lock(ct::c3::c3_global_mlir_mutex);
        temp_impl = std::make_unique<Impl>();
    }

    impl_ = std::move(temp_impl);
    num_inputs_ = graph.inputs().size();
    num_outputs_ = graph.outputs().size();

    impl_->heldModule = buildTensorMLIRModule(impl_->context, graph, num_inputs_, num_outputs_);
    mlir::ModuleOp module = *impl_->heldModule;

    size_t base_numel = 1;
    if (graph.outputCount() > 0) {
        base_numel = graph.node(graph.outputs()[0]).out_desc.numel;
    }
    applyUnifiedTransformPipeline(module, num_inputs_, num_outputs_, base_numel);

    std::string cache_graph = std::string("linalg_oneshot_") + graph.toString()
                              + "_ol" + std::to_string(opt_level);
    if (isOpenMPEnabled()) {
        cache_graph += ":omp";  // OpenMP 编译产物与串行产物不兼容，隔离缓存 key
    }
    impl_->engine = createEngine(module, opt_level, cache_graph, impl_->aotBuilder);
    if (!impl_->engine->lookup("c3_kernel")) {
        throw std::runtime_error("LinalgOneShotGen: lookup c3_kernel failed");
    }
}

LinalgOneShotKernel::~LinalgOneShotKernel() = default;
LinalgOneShotKernel::LinalgOneShotKernel(LinalgOneShotKernel&&) noexcept = default;
LinalgOneShotKernel& LinalgOneShotKernel::operator=(LinalgOneShotKernel&&) noexcept = default;

void LinalgOneShotKernel::execute(const float* const* in_ptrs, float* const* out_ptrs, size_t n) const {
    const size_t num_memrefs = num_inputs_ + num_outputs_;

    std::vector<MemRefDesc> descs(num_memrefs);
    std::vector<void*> args(num_memrefs * 5);
    int arg_idx = 0;

    for (size_t i = 0; i < num_inputs_; ++i) {
        descs[i] = MemRefDesc{const_cast<float*>(in_ptrs[i]), const_cast<float*>(in_ptrs[i]),
                              0, {static_cast<int64_t>(n)}, {1}};
        appendMemRefDescArgs(descs[i], args.data(), arg_idx);
    }
    for (size_t i = 0; i < num_outputs_; ++i) {
        size_t desc_idx = num_inputs_ + i;
        descs[desc_idx] = MemRefDesc{out_ptrs[i], out_ptrs[i], 0, {static_cast<int64_t>(n)}, {1}};
        appendMemRefDescArgs(descs[desc_idx], args.data(), arg_idx);
    }

    auto err = impl_->engine->invokePacked("c3_kernel", args);
    if (err) {
        throw std::runtime_error("LinalgOneShotGen: invokePacked failed: " + llvm::toString(std::move(err)));
    }
}

// ======================= 共享融合 kernel 缓存工厂 =======================

std::shared_ptr<LinalgOneShotKernel> getCachedLinalgOneShotKernel(
    const Graph& graph, const std::string& graph_key, int opt_level) {
    static const bool cache_disabled = [] {
        const char* v = std::getenv("C3_LINALG_CACHE");
        return v != nullptr && std::string(v) == "0";
    }();
    if (cache_disabled) {
        return std::make_shared<LinalgOneShotKernel>(graph, opt_level);
    }

    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<LinalgOneShotKernel>> cache;

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(graph_key);
    if (it != cache.end()) {
        if (auto sp = it->second.lock()) {
            return sp;
        }
        cache.erase(it);
    }

    auto kernel = std::make_shared<LinalgOneShotKernel>(graph, opt_level);
    cache[graph_key] = kernel;
    return kernel;
}

} // namespace c3
} // namespace ct
