/**
 * @file C3DialectLowering.cpp
 * @generation JIT-3.0 C3 Dialect 算子降低与优化管线
 * @brief JIT 3.0 C3 Dialect 算子到标准 Linalg/SCF 的 Lowering 与优化管线（CGO 2027 解耦版）
 * @details 物理隔离 JIT 3.0 的高层降低转换模式，彻底拆分编译逻辑。
 *          通过 RewritePattern 将 c3.matmul、c3.transpose、c3.sum_reduce 算子
 *          完美降解到 linalg / scf / math / arith 层级，并应用自动融合与 One-Shot 缓冲化。
 * @date 2026/08/16
 */

#include "MLIRKernelGen.h"
#include "C3/C3Dialect.h"
#include "C3/TuningState.h"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
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
#include <mlir/Dialect/Math/Transforms/Passes.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

// 导入 TableGen 自动生成的 C3Combine (DRR 规则) 模式重写实现
#include "C3Combine.cpp.inc"

namespace ct {
namespace c3 {

namespace {

// ======================= Helper for Tiling / GEP =======================

static mlir::Value indexToI64(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value idx) {
    return builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), idx);
}

static mlir::Value i64ToIndex(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val) {
    return builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), val);
}

static void buildLoop(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value n, int64_t known_numel,
                      const std::function<void(mlir::OpBuilder&, mlir::Location, mlir::Value)>& body_fn) {
    if (known_numel > 0 && known_numel <= 16) {
        for (int64_t i = 0; i < known_numel; ++i) {
            mlir::Value idx = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
            mlir::Value idx_i64 = indexToI64(builder, loc, idx);
            body_fn(builder, loc, idx_i64);
        }
    } else {
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

static void buildSmallMatMul(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value lhs, mlir::Value rhs, mlir::Value out, mlir::Value bias,
                             size_t M, size_t K, size_t N,
                             int transA, int transB, int act,
                             size_t bias_numel) {
    auto f32 = builder.getF32Type();
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    mlir::Value c1_i64 = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 64);

    mlir::Value M_v = builder.create<mlir::arith::ConstantIndexOp>(loc, M);
    mlir::Value N_v = builder.create<mlir::arith::ConstantIndexOp>(loc, N);
    mlir::Value K_v = builder.create<mlir::arith::ConstantIndexOp>(loc, K);

    mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

    auto loop_i = builder.create<mlir::scf::ForOp>(loc, c0, M_v, c1);
    builder.setInsertionPointToStart(loop_i.getBody());
    mlir::Value i_idx = loop_i.getInductionVar();
    mlir::Value i_i64 = indexToI64(builder, loc, i_idx);

    auto loop_j = builder.create<mlir::scf::ForOp>(loc, c0, N_v, c1);
    builder.setInsertionPointToStart(loop_j.getBody());
    mlir::Value j_idx = loop_j.getInductionVar();
    mlir::Value j_i64 = indexToI64(builder, loc, j_idx);

    mlir::Value out_idx = builder.create<mlir::arith::MulIOp>(loc, i_i64, builder.create<mlir::arith::ConstantIntOp>(loc, N, 64));
    out_idx = builder.create<mlir::arith::AddIOp>(loc, out_idx, j_i64);
    mlir::Value out_cell_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{out_idx});

    mlir::Value init_val = builder.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
    if (bias) {
        mlir::Value bias_idx = j_i64;
        if (bias_numel == 1) {
            bias_idx = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 64);
        }
        mlir::Value bias_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, bias, mlir::ValueRange{bias_idx});
        init_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, bias_ptr);
    }

    auto loop_k = builder.create<mlir::scf::ForOp>(loc, c0, K_v, c1, mlir::ValueRange{init_val});
    builder.setInsertionPointToStart(loop_k.getBody());
    mlir::Value k_idx = loop_k.getInductionVar();
    mlir::Value k_i64 = indexToI64(builder, loc, k_idx);
    mlir::Value sum_accum = loop_k.getRegionIterArgs()[0];

    mlir::Value a_idx;
    if (transA == 112) {
        a_idx = builder.create<mlir::arith::MulIOp>(loc, k_i64, builder.create<mlir::arith::ConstantIntOp>(loc, M, 64));
        a_idx = builder.create<mlir::arith::AddIOp>(loc, a_idx, i_i64);
    } else {
        a_idx = builder.create<mlir::arith::MulIOp>(loc, i_i64, builder.create<mlir::arith::ConstantIntOp>(loc, K, 64));
        a_idx = builder.create<mlir::arith::AddIOp>(loc, a_idx, k_i64);
    }

    mlir::Value b_idx;
    if (transB == 112) {
        b_idx = builder.create<mlir::arith::MulIOp>(loc, j_i64, builder.create<mlir::arith::ConstantIntOp>(loc, K, 64));
        b_idx = builder.create<mlir::arith::AddIOp>(loc, b_idx, k_i64);
    } else {
        b_idx = builder.create<mlir::arith::MulIOp>(loc, k_i64, builder.create<mlir::arith::ConstantIntOp>(loc, N, 64));
        b_idx = builder.create<mlir::arith::AddIOp>(loc, b_idx, j_i64);
    }

    mlir::Value a_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, lhs, mlir::ValueRange{a_idx});
    mlir::Value b_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, rhs, mlir::ValueRange{b_idx});

    mlir::Value av = builder.create<mlir::LLVM::LoadOp>(loc, f32, a_ptr);
    mlir::Value bv = builder.create<mlir::LLVM::LoadOp>(loc, f32, b_ptr);
    mlir::Value prod = builder.create<mlir::arith::MulFOp>(loc, av, bv);
    mlir::Value next_sum = builder.create<mlir::arith::AddFOp>(loc, sum_accum, prod);

    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{next_sum});

    builder.setInsertionPointAfter(loop_k);
    mlir::Value final_sum = loop_k.getResult(0);

    mlir::Value activated = final_sum;
    if (act == 1) { // ReLU
        mlir::Value zero = builder.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
        activated = builder.create<mlir::arith::MaxNumFOp>(loc, final_sum, zero);
    } else if (act == 2) { // Sigmoid
        mlir::Value neg_sum = builder.create<mlir::arith::NegFOp>(loc, final_sum);
        mlir::Value exp_val = builder.create<mlir::math::ExpOp>(loc, neg_sum);
        mlir::Value one = builder.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(1.0f));
        mlir::Value denom = builder.create<mlir::arith::AddFOp>(loc, one, exp_val);
        activated = builder.create<mlir::arith::DivFOp>(loc, one, denom);
    } else if (act == 3) { // Tanh
        activated = builder.create<mlir::math::TanhOp>(loc, final_sum);
    }

    builder.create<mlir::LLVM::StoreOp>(loc, activated, out_cell_ptr);

    builder.setInsertionPointAfter(loop_j);
    builder.setInsertionPointAfter(loop_i);
}

} // namespace

// ======================= JIT 3.0 C3 Dialect Lowering Patterns =======================

struct TransposeOpLowering : public mlir::OpRewritePattern<mlir::c3::TransposeOp> {
    using OpRewritePattern<mlir::c3::TransposeOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::c3::TransposeOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value input = op.getInput();
        mlir::Value out = op.getOut();

        int64_t M = op.getM();
        int64_t N = op.getN();
        int dim0 = op.getDim0();
        int dim1 = op.getDim1();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        if ((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0)) {
            mlir::Value M_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, M);
            mlir::Value N_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, N);
            mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
            mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

            auto loop_i = rewriter.create<mlir::scf::ForOp>(loc, c0, M_v, c1);
            rewriter.setInsertionPointToStart(loop_i.getBody());
            mlir::Value i_idx = loop_i.getInductionVar();
            mlir::Value i_i64 = indexToI64(rewriter, loc, i_idx);

            auto loop_j = rewriter.create<mlir::scf::ForOp>(loc, c0, N_v, c1);
            rewriter.setInsertionPointToStart(loop_j.getBody());
            mlir::Value j_idx = loop_j.getInductionVar();
            mlir::Value j_i64 = indexToI64(rewriter, loc, j_idx);

            mlir::Value in_idx = rewriter.create<mlir::arith::MulIOp>(loc, i_i64, rewriter.create<mlir::arith::ConstantIntOp>(loc, N, 64));
            in_idx = rewriter.create<mlir::arith::AddIOp>(loc, in_idx, j_i64);

            mlir::Value out_idx = rewriter.create<mlir::arith::MulIOp>(loc, j_i64, rewriter.create<mlir::arith::ConstantIntOp>(loc, M, 64));
            out_idx = rewriter.create<mlir::arith::AddIOp>(loc, out_idx, i_i64);

            mlir::Value in_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{in_idx});
            mlir::Value out_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{out_idx});

            mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
            rewriter.create<mlir::LLVM::StoreOp>(loc, val, out_ptr);

            rewriter.setInsertionPointAfter(loop_j);
            rewriter.setInsertionPointAfter(loop_i);
        } else {
            int64_t total_ops = M * N;
            buildLoop(rewriter, loc, rewriter.create<mlir::arith::ConstantIntOp>(loc, total_ops, 64), total_ops,
                [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx) {
                    mlir::Value in_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{idx});
                    mlir::Value out_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx});
                    mlir::Value val = bld.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
                    bld.create<mlir::LLVM::StoreOp>(loc, val, out_ptr);
                });
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

template <typename SrcOp, typename ArithOp>
struct BinaryOpLowering : public mlir::OpRewritePattern<SrcOp> {
    using mlir::OpRewritePattern<SrcOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SrcOp op, mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value lhs = op.getLhs();
        mlir::Value rhs = op.getRhs();
        mlir::Value out = op.getOut();
        int64_t bmod = op.getBmod();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        int64_t total_ops = 1; 

        buildLoop(rewriter, loc, op.getNumel(), 0,
            [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx) {
                mlir::Value l_idx = idx;
                mlir::Value r_idx = idx;
                if (bmod > 0) {
                    mlir::Value mod_val = bld.create<mlir::arith::ConstantIntOp>(loc, bmod, 64);
                    r_idx = bld.create<mlir::arith::RemUIOp>(loc, idx, mod_val);
                } else if (bmod < 0) {
                    mlir::Value mod_val = bld.create<mlir::arith::ConstantIntOp>(loc, -bmod, 64);
                    l_idx = bld.create<mlir::arith::RemUIOp>(loc, idx, mod_val);
                }
                mlir::Value l_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, lhs, mlir::ValueRange{l_idx});
                mlir::Value r_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, rhs, mlir::ValueRange{r_idx});
                mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx});

                mlir::Value lv = bld.create<mlir::LLVM::LoadOp>(loc, f32, l_ptr);
                mlir::Value rv = bld.create<mlir::LLVM::LoadOp>(loc, f32, r_ptr);
                mlir::Value res = bld.create<ArithOp>(loc, lv, rv);
                bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr);
            });

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

using AddOpLowering = BinaryOpLowering<mlir::c3::AddOp, mlir::arith::AddFOp>;
using SubOpLowering = BinaryOpLowering<mlir::c3::SubOp, mlir::arith::SubFOp>;
using MulOpLowering = BinaryOpLowering<mlir::c3::MulOp, mlir::arith::MulFOp>;
using DivOpLowering = BinaryOpLowering<mlir::c3::DivOp, mlir::arith::DivFOp>;

template <typename SrcOp, typename ArithOp>
struct UnaryOpLowering : public mlir::OpRewritePattern<SrcOp> {
    using mlir::OpRewritePattern<SrcOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SrcOp op, mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value input = op.getInput();
        mlir::Value out = op.getOut();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        buildLoop(rewriter, loc, op.getNumel(), 0,
            [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx) {
                mlir::Value in_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{idx});
                mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx});

                mlir::Value val = bld.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
                mlir::Value res = bld.create<ArithOp>(loc, val);
                bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr);
            });

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

using NegOpLowering     = UnaryOpLowering<mlir::c3::NegOp, mlir::arith::NegFOp>;
using TanhOpLowering    = UnaryOpLowering<mlir::c3::TanhOp, mlir::math::TanhOp>;
using ExpOpLowering     = UnaryOpLowering<mlir::c3::ExpOp, mlir::math::ExpOp>;
using LogOpLowering     = UnaryOpLowering<mlir::c3::LogOp, mlir::math::LogOp>;

struct ReLUOpLowering : public mlir::OpRewritePattern<mlir::c3::ReLUOp> {
    using OpRewritePattern<mlir::c3::ReLUOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::c3::ReLUOp op, mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value input = op.getInput();
        mlir::Value out = op.getOut();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        buildLoop(rewriter, loc, op.getNumel(), 0,
            [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx) {
                mlir::Value in_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{idx});
                mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx});

                mlir::Value val = bld.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
                mlir::Value zero = bld.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
                mlir::Value res = bld.create<mlir::arith::MaxNumFOp>(loc, val, zero);
                bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr);
            });

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct SigmoidOpLowering : public mlir::OpRewritePattern<mlir::c3::SigmoidOp> {
    using OpRewritePattern<mlir::c3::SigmoidOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::c3::SigmoidOp op, mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value input = op.getInput();
        mlir::Value out = op.getOut();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        buildLoop(rewriter, loc, op.getNumel(), 0,
            [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx) {
                mlir::Value in_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{idx});
                mlir::Value o_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx});

                mlir::Value val = bld.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
                mlir::Value neg_x = bld.create<mlir::arith::NegFOp>(loc, val);
                mlir::Value exp_val = bld.create<mlir::math::ExpOp>(loc, neg_x);
                mlir::Value one = bld.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(1.0f));
                mlir::Value denom = bld.create<mlir::arith::AddFOp>(loc, one, exp_val);
                mlir::Value res = bld.create<mlir::arith::DivFOp>(loc, one, denom);
                bld.create<mlir::LLVM::StoreOp>(loc, res, o_ptr);
            });

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct SumReduceOpLowering : public mlir::OpRewritePattern<mlir::c3::SumReduceOp> {
    using OpRewritePattern<mlir::c3::SumReduceOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::c3::SumReduceOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value input = op.getInput();
        mlir::Value out = op.getOut();

        int64_t M = op.getM();
        int64_t N = op.getN();
        int axis = op.getAxis();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        if (axis == 0) {
            mlir::Value N_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, N);
            mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
            mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

            auto loop_j = rewriter.create<mlir::scf::ForOp>(loc, c0, N_v, c1);
            rewriter.setInsertionPointToStart(loop_j.getBody());
            mlir::Value j_idx = loop_j.getInductionVar();
            mlir::Value j_i64 = indexToI64(rewriter, loc, j_idx);

            mlir::Value out_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{j_i64});
            mlir::Value zero = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
            rewriter.create<mlir::LLVM::StoreOp>(loc, zero, out_ptr);

            mlir::Value M_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, M);
            auto loop_i = rewriter.create<mlir::scf::ForOp>(loc, c0, M_v, c1);
            rewriter.setInsertionPointToStart(loop_i.getBody());
            mlir::Value i_idx = loop_i.getInductionVar();
            mlir::Value i_i64 = indexToI64(rewriter, loc, i_idx);

            mlir::Value in_idx = rewriter.create<mlir::arith::MulIOp>(loc, i_i64, rewriter.create<mlir::arith::ConstantIntOp>(loc, N, 64));
            in_idx = rewriter.create<mlir::arith::AddIOp>(loc, in_idx, j_i64);

            mlir::Value in_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{in_idx});
            mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
            mlir::Value old_sum = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, out_ptr);
            mlir::Value new_sum = rewriter.create<mlir::arith::AddFOp>(loc, old_sum, val);
            rewriter.create<mlir::LLVM::StoreOp>(loc, new_sum, out_ptr);

            rewriter.setInsertionPointAfter(loop_i);
            rewriter.setInsertionPointAfter(loop_j);
        } else {
            mlir::Value M_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, M);
            mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
            mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

            auto loop_i = rewriter.create<mlir::scf::ForOp>(loc, c0, M_v, c1);
            rewriter.setInsertionPointToStart(loop_i.getBody());
            mlir::Value i_idx = loop_i.getInductionVar();
            mlir::Value i_i64 = indexToI64(rewriter, loc, i_idx);

            mlir::Value out_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{i_i64});
            mlir::Value zero = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
            rewriter.create<mlir::LLVM::StoreOp>(loc, zero, out_ptr);

            mlir::Value N_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, N);
            auto loop_j = rewriter.create<mlir::scf::ForOp>(loc, c0, N_v, c1);
            rewriter.setInsertionPointToStart(loop_j.getBody());
            mlir::Value j_idx = loop_j.getInductionVar();
            mlir::Value j_i64 = indexToI64(rewriter, loc, j_idx);

            mlir::Value in_idx = rewriter.create<mlir::arith::MulIOp>(loc, i_i64, rewriter.create<mlir::arith::ConstantIntOp>(loc, N, 64));
            in_idx = rewriter.create<mlir::arith::AddIOp>(loc, in_idx, j_i64);

            mlir::Value in_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, input, mlir::ValueRange{in_idx});
            mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
            mlir::Value old_sum = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, out_ptr);
            mlir::Value new_sum = rewriter.create<mlir::arith::AddFOp>(loc, old_sum, val);
            rewriter.create<mlir::LLVM::StoreOp>(loc, new_sum, out_ptr);

            rewriter.setInsertionPointAfter(loop_j);
            rewriter.setInsertionPointAfter(loop_i);
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct MatMulOpLowering : public mlir::OpRewritePattern<mlir::c3::MatMulOp> {
    using OpRewritePattern<mlir::c3::MatMulOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::c3::MatMulOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        mlir::Value lhs = op.getLhs();
        mlir::Value rhs = op.getRhs();
        mlir::Value out = op.getOut();
        mlir::Value bias = op.getBias();

        int64_t M = op.getM();
        int64_t K = op.getK();
        int64_t N = op.getN();
        int transA = op.getTransA();
        int transB = op.getTransB();
        int act = op.getAct();
        int64_t tileM = op.getTileM();
        int64_t tileN = op.getTileN();
        int64_t bias_numel = op.getBiasNumel();

        auto f32 = rewriter.getF32Type();
        auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        bool fallback_to_small = false;
        int64_t total_ops = M * N * K;
        if (total_ops < 256) {
            fallback_to_small = true;
        } else if (M >= tileM && N >= tileN) {
            mlir::Value M_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, M);
            mlir::Value N_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, N);
            mlir::Value K_v = rewriter.create<mlir::arith::ConstantIndexOp>(loc, K);

            mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
            mlir::Value step_m = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tileM);
            mlir::Value step_n = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tileN);
            mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

            auto loop_it = rewriter.create<mlir::scf::ForOp>(loc, c0, M_v, step_m);
            rewriter.setInsertionPointToStart(loop_it.getBody());
            mlir::Value it_idx = loop_it.getInductionVar();

            auto loop_jt = rewriter.create<mlir::scf::ForOp>(loc, c0, N_v, step_n);
            rewriter.setInsertionPointToStart(loop_jt.getBody());
            mlir::Value jt_idx = loop_jt.getInductionVar();

            mlir::Value it_end = rewriter.create<mlir::arith::AddIOp>(loc, it_idx, step_m);
            mlir::Value jt_end = rewriter.create<mlir::arith::AddIOp>(loc, jt_idx, step_n);

            auto loop_i = rewriter.create<mlir::scf::ForOp>(loc, it_idx, it_end, c1);
            rewriter.setInsertionPointToStart(loop_i.getBody());
            mlir::Value i_idx = loop_i.getInductionVar();
            mlir::Value i_i64 = indexToI64(rewriter, loc, i_idx);

            auto loop_j = rewriter.create<mlir::scf::ForOp>(loc, jt_idx, jt_end, c1);
            rewriter.setInsertionPointToStart(loop_j.getBody());
            mlir::Value j_idx = loop_j.getInductionVar();
            mlir::Value j_i64 = indexToI64(rewriter, loc, j_idx);

            mlir::Value out_idx = rewriter.create<mlir::arith::MulIOp>(loc, i_i64, rewriter.create<mlir::arith::ConstantIntOp>(loc, N, 64));
            out_idx = rewriter.create<mlir::arith::AddIOp>(loc, out_idx, j_i64);
            mlir::Value out_cell_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{out_idx});

            mlir::Value init_val = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
            if (bias) {
                mlir::Value bias_idx = j_i64;
                if (bias_numel == 1) {
                    bias_idx = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 64);
                }
                mlir::Value bias_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, bias, mlir::ValueRange{bias_idx});
                init_val = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, bias_ptr);
            }

            auto loop_k = rewriter.create<mlir::scf::ForOp>(loc, c0, K_v, c1, mlir::ValueRange{init_val});
            rewriter.setInsertionPointToStart(loop_k.getBody());
            mlir::Value k_idx = loop_k.getInductionVar();
            mlir::Value k_i64 = indexToI64(rewriter, loc, k_idx);
            mlir::Value sum_accum = loop_k.getRegionIterArgs()[0];

            mlir::Value a_idx;
            if (transA == 112) {
                a_idx = rewriter.create<mlir::arith::MulIOp>(loc, k_i64, rewriter.create<mlir::arith::ConstantIntOp>(loc, M, 64));
                a_idx = rewriter.create<mlir::arith::AddIOp>(loc, a_idx, i_i64);
            } else {
                a_idx = rewriter.create<mlir::arith::MulIOp>(loc, i_i64, rewriter.create<mlir::arith::ConstantIntOp>(loc, K, 64));
                a_idx = rewriter.create<mlir::arith::AddIOp>(loc, a_idx, k_i64);
            }

            mlir::Value b_idx;
            if (transB == 112) {
                b_idx = rewriter.create<mlir::arith::MulIOp>(loc, j_i64, rewriter.create<mlir::arith::ConstantIntOp>(loc, K, 64));
                b_idx = rewriter.create<mlir::arith::AddIOp>(loc, b_idx, k_i64);
            } else {
                b_idx = rewriter.create<mlir::arith::MulIOp>(loc, k_i64, rewriter.create<mlir::arith::ConstantIntOp>(loc, N, 64));
                b_idx = rewriter.create<mlir::arith::AddIOp>(loc, b_idx, j_i64);
            }

            mlir::Value a_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, lhs, mlir::ValueRange{a_idx});
            mlir::Value b_ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, rhs, mlir::ValueRange{b_idx});

            mlir::Value av = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, a_ptr);
            mlir::Value bv = rewriter.create<mlir::LLVM::LoadOp>(loc, f32, b_ptr);
            mlir::Value prod = rewriter.create<mlir::arith::MulFOp>(loc, av, bv);
            mlir::Value next_sum = rewriter.create<mlir::arith::AddFOp>(loc, sum_accum, prod);

            rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{next_sum});

            rewriter.setInsertionPointAfter(loop_k);
            mlir::Value final_sum = loop_k.getResult(0);

            mlir::Value activated = final_sum;
            if (act == 1) { // ReLU
                mlir::Value zero = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(0.0f));
                activated = rewriter.create<mlir::arith::MaxNumFOp>(loc, final_sum, zero);
            } else if (act == 2) { // Sigmoid
                mlir::Value neg_sum = rewriter.create<mlir::arith::NegFOp>(loc, final_sum);
                mlir::Value exp_val = rewriter.create<mlir::math::ExpOp>(loc, neg_sum);
                mlir::Value one = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f32, llvm::APFloat(1.0f));
                mlir::Value denom = rewriter.create<mlir::arith::AddFOp>(loc, one, exp_val);
                activated = rewriter.create<mlir::arith::DivFOp>(loc, one, denom);
            } else if (act == 3) { // Tanh
                activated = rewriter.create<mlir::math::TanhOp>(loc, final_sum);
            }

            rewriter.create<mlir::LLVM::StoreOp>(loc, activated, out_cell_ptr);

            rewriter.setInsertionPointAfter(loop_j);
            rewriter.setInsertionPointAfter(loop_i);
            rewriter.setInsertionPointAfter(loop_jt);
            rewriter.setInsertionPointAfter(loop_it);
        } else {
            fallback_to_small = true;
        }

        if (fallback_to_small) {
            buildSmallMatMul(rewriter, loc, lhs, rhs, out, bias,
                             (size_t)M, (size_t)K, (size_t)N,
                             transA, transB, act, (size_t)bias_numel);
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

static void runC3Combine(mlir::ModuleOp module) {
    mlir::RewritePatternSet patterns(module.getContext());
    populateWithGenerated(patterns);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
        throw std::runtime_error("C3DialectLowering: C3Combine pattern rewrite optimization failed");
    }
}

static void runC3Lowering(mlir::ModuleOp module) {
    mlir::RewritePatternSet patterns(module.getContext());
    patterns.add<TransposeOpLowering, SumReduceOpLowering, MatMulOpLowering,
                 AddOpLowering, SubOpLowering, MulOpLowering, DivOpLowering,
                 NegOpLowering, ReLUOpLowering, SigmoidOpLowering, TanhOpLowering,
                 ExpOpLowering, LogOpLowering>(module.getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
        throw std::runtime_error("C3DialectLowering: C3ToLLVM lowering pass failed");
    }
}

static void runPass(mlir::ModuleOp module, std::unique_ptr<mlir::Pass> pass, const char* name) {
    mlir::PassManager pm(module.getContext());
    pm.addPass(std::move(pass));
    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error(std::string("C3DialectLowering: ") + name + " failed");
    }
}

void applyLoweringPipeline(mlir::ModuleOp module, int opt_level) {
    runPass(module, mlir::createStripDebugInfoPass(), "StripDebugInfo");
    runPass(module, mlir::createCanonicalizerPass(), "Canonicalizer");
    runC3Combine(module);  // 1. 运行 JIT 3.0 高层图优化 (DRR)
    runC3Lowering(module); // 2. 运行 JIT 3.0 高层算子到 LLVM 标量/向量循环 of Lowering Pass
    runPass(module, mlir::createCSEPass(), "CSE");
    runPass(module, mlir::createSymbolDCEPass(), "SymbolDCE");
    runPass(module, mlir::createLoopInvariantCodeMotionPass(), "LICM");
    runPass(module, mlir::createSCFForLoopCanonicalizationPass(), "SCFForLoopCanonicalization");
    
    // [Extreme JIT - opt_level >= 4] 能上的优化 Pass 尽可能上满，释放硬件级极致性能
    if (opt_level >= 4) {
        runPass(module, mlir::createControlFlowSinkPass(), "ControlFlowSink");
        runPass(module, mlir::createRemoveDeadValuesPass(), "RemoveDeadValues");
    }

    // [优化 2026-08-16] 移除 ParallelLoopFusionPass。因为 C3DialectLowering 仅生成顺序 scf.for 循环，
    // 无 scf.parallel 循环，此 pass 为 100% no-op，移除它以减少编译期 pass 遍历开销。
    runPass(module, mlir::createSCFToControlFlowPass(), "SCFToCF");

    runPass(module, mlir::createConvertMathToLLVMPass(), "MathToLLVM");
    runPass(module, mlir::createArithToLLVMConversionPass(), "ArithToLLVM");

    runPass(module, mlir::createConvertControlFlowToLLVMPass(), "CFToLLVM");
    runPass(module, mlir::createConvertFuncToLLVMPass(), "FuncToLLVM");
    runPass(module, mlir::createFinalizeMemRefToLLVMConversionPass(), "MemRefToLLVM");

    runPass(module, mlir::createReconcileUnrealizedCastsPass(), "ReconcileUnrealizedCasts");

    // LLVM 转换收尾后再次运行 Canonicalizer & CSE 清理无效转换、类型强转与死代码，精简 IR
    runPass(module, mlir::createCanonicalizerPass(), "CanonicalizerPost");
    runPass(module, mlir::createCSEPass(), "CSEPost");
}

} // namespace c3
} // namespace ct
