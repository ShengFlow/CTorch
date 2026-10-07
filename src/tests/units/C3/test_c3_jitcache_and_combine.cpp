/**
 * @file test_c3_jitcache_and_combine.cpp
 * @brief JITCache 2.0 磁盘 Bitcode 反序列化闭环与 C3 TableGen DRR 图优化代数化简验证套件
 * @date 2026-10-07
 * @author 苏璃珞 (CTorch Core Agent)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>
#include <memory>

#include "Tensor.h"
#include "C3/Graph.h"
#include "C3/C3Engine.h"
#include "C3/JITCache.h"
#include "C3/C3Dialect.h"
#include "C3/MLIRKernelGen.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

using namespace ct::c3;

namespace {

static void fillTensor(Tensor& t, const std::vector<float>& values) {
    float* p = t.data_write<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        p[i] = values[i];
    }
}

static bool tensorsAllClose(const Tensor& a, const Tensor& b, float rtol = 1e-4f, float atol = 1e-6f) {
    if (a.shape() != b.shape()) return false;
    const float* pa = a.data_read<float>();
    const float* pb = b.data_read<float>();
    size_t n = a.numel();
    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(pa[i] - pb[i]);
        float max_val = std::max(std::fabs(pa[i]), std::fabs(pb[i]));
        if (diff > atol + rtol * max_val) {
            return false;
        }
    }
    return true;
}

} // namespace

// ============================================================================
// Suite 1: JITCache 2.0 结构化元数据序列化与 ORC JIT Bitcode 直接加载
// ============================================================================

TEST(JITCache2Test, MetadataSerializationRoundTrip) {
    auto& cache = JITCache::getInstance();
    cache.evict();

    JITMetadata meta;
    meta.is_multi_node = true;
    meta.is_fused = false;
    meta.is_matmul = true;
    meta.num_inputs = 2;
    meta.M = 64;
    meta.K = 128;
    meta.N = 32;
    meta.elem_n = 2048;
    meta.scratch_size = 8192;
    meta.pool_buf_count = 3;
    meta.fused_out_shape = {64, 32};

    llvm::LLVMContext llvm_ctx;
    llvm::Module llvm_mod("dummy_meta_test", llvm_ctx);

    std::string key = "unit_test_meta_roundtrip_key";
    std::string path = cache.store(key, llvm_mod, meta);
    EXPECT_FALSE(path.empty()) << "store 应该成功并返回 bitcode 路径";

    JITMetadata loaded;
    bool ok = cache.loadMetadata(key, loaded);
    EXPECT_TRUE(ok) << "loadMetadata 应该成功解析元数据文件";

    EXPECT_EQ(loaded.is_multi_node, meta.is_multi_node);
    EXPECT_EQ(loaded.is_fused, meta.is_fused);
    EXPECT_EQ(loaded.is_matmul, meta.is_matmul);
    EXPECT_EQ(loaded.num_inputs, meta.num_inputs);
    EXPECT_EQ(loaded.M, meta.M);
    EXPECT_EQ(loaded.K, meta.K);
    EXPECT_EQ(loaded.N, meta.N);
    EXPECT_EQ(loaded.elem_n, meta.elem_n);
    EXPECT_EQ(loaded.scratch_size, meta.scratch_size);
    EXPECT_EQ(loaded.pool_buf_count, meta.pool_buf_count);
    EXPECT_EQ(loaded.fused_out_shape, meta.fused_out_shape);

    // 验证不存在的 key 返回 false
    JITMetadata dummy;
    EXPECT_FALSE(cache.loadMetadata("non_existent_key_12345", dummy));
}

TEST(JITCache2Test, EndToEndBitcodeCacheHitAndExecution) {
    if (!JITCache::isEnabled()) {
        GTEST_SKIP() << "C3 JIT Cache 被环境禁用，跳过测试";
    }

    auto& cache = JITCache::getInstance();
    cache.evict(); // 保证冷启动从零开始

    // 构造一个确定的 MatMul 计算图
    Graph g;
    auto a_desc = TensorDesc::fromShape({4, 8});
    auto b_desc = TensorDesc::fromShape({8, 4});
    auto c_desc = TensorDesc::fromShape({4, 4});

    size_t a_idx = g.addInput(a_desc);
    size_t b_idx = g.addInput(b_desc);
    size_t c_idx = g.addNode(MatMulNode{a_desc, b_desc}, {a_idx, b_idx}, c_desc);
    g.markOutput(c_idx);

    auto& engine = C3Engine::getInstance();
    CompileOptions opts;
    opts.enable_cache = false; // 绕过 RAM 引擎缓存，直接穿透至 JITCache 磁盘层
    opts.opt_level = 2;

    const uint64_t stores_before = cache.stores();
    const uint64_t hits_before = cache.hits();

    // 第一次编译：冷启动，触发 MLIR 构建 + Lowering + JITCache::store
    auto kernel_cold = engine.compile(g, opts);
    ASSERT_NE(kernel_cold, nullptr);

    const uint64_t stores_after_cold = cache.stores();
    const uint64_t hits_after_cold = cache.hits();
    EXPECT_EQ(stores_after_cold, stores_before + 1) << "冷启动应当写入 1 次 bitcode 到磁盘缓存";
    EXPECT_EQ(hits_after_cold, hits_before) << "冷启动不应产生命中";

    // 准备测试数据并执行冷启动内核
    Tensor tA(ShapeTag{}, {4, 8});
    Tensor tB(ShapeTag{}, {8, 4});
    std::vector<float> a_vals(32);
    std::vector<float> b_vals(32);
    for (size_t i = 0; i < 32; ++i) {
        a_vals[i] = static_cast<float>(i + 1) * 0.1f;
        b_vals[i] = static_cast<float>(32 - i) * 0.05f;
    }
    fillTensor(tA, a_vals);
    fillTensor(tB, b_vals);

    auto res_cold = kernel_cold->execute({tA, tB});
    ASSERT_EQ(res_cold.size(), 1u);

    // 第二次编译：热启动，跳过 MLIR Context 构建与 15+ 项 Pass，直接通过 Bitcode 加载
    auto kernel_warm = engine.compile(g, opts);
    ASSERT_NE(kernel_warm, nullptr);

    const uint64_t hits_after_warm = cache.hits();
    EXPECT_EQ(hits_after_warm, hits_after_cold + 1) << "热启动应当触发 JITCache 磁盘命中并在 ORC JIT 重建！";

    // 执行热启动内核
    auto res_warm = kernel_warm->execute({tA, tB});
    ASSERT_EQ(res_warm.size(), 1u);

    // 验证冷启动内核与热启动从 Bitcode 加载的内核输出 100% 精确一致
    EXPECT_TRUE(tensorsAllClose(res_cold[0], res_warm[0])) << "JITCache 反序列化后计算结果必须完全一致";
}

// ============================================================================
// Suite 2: C3 Dialect SSA & TableGen DRR 代数化简与结构折叠规则验证
// ============================================================================

TEST(C3CombineDRRTest, DoubleNegTensorPatternElimination) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::c3::C3Dialect, mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();

    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto tensorType = mlir::RankedTensorType::get({4}, builder.getF32Type());
    auto funcType = builder.getFunctionType({tensorType, tensorType, tensorType}, {tensorType});
    auto func = builder.create<mlir::func::FuncOp>(loc, "test_double_neg", funcType);

    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    mlir::Value x = entry->getArgument(0);
    mlir::Value dest1 = entry->getArgument(1);
    mlir::Value dest2 = entry->getArgument(2);

    // neg_tensor(neg_tensor(x, dest1), dest2)
    auto neg1 = builder.create<mlir::c3::NegTensorOp>(loc, tensorType, x, dest1);
    auto neg2 = builder.create<mlir::c3::NegTensorOp>(loc, tensorType, neg1.getOut(), dest2);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{neg2.getOut()});

    // 运行 C3Combine DRR 表驱动模式
    runC3Combine(module);

    // 检查优化结果：连续两个取负操作应全部被消去，直接返回原始输入 x
    int neg_count = 0;
    mlir::Value returned_val;
    func.walk([&](mlir::Operation* op) {
        if (llvm::isa<mlir::c3::NegTensorOp>(op)) {
            neg_count++;
        }
        if (auto retOp = llvm::dyn_cast<mlir::func::ReturnOp>(op)) {
            if (retOp.getNumOperands() > 0) {
                returned_val = retOp.getOperand(0);
            }
        }
    });

    EXPECT_EQ(neg_count, 0) << "DoubleNegTensorOptPattern 应当完全消除两层取负操作";
    EXPECT_EQ(returned_val, x) << "ReturnOp 应当直接引用原始操作数 x";
}

TEST(C3CombineDRRTest, ReluIdempotentPatternFolding) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::c3::C3Dialect, mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();

    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto tensorType = mlir::RankedTensorType::get({8}, builder.getF32Type());
    auto funcType = builder.getFunctionType({tensorType, tensorType, tensorType}, {tensorType});
    auto func = builder.create<mlir::func::FuncOp>(loc, "test_relu_idempotent", funcType);

    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    mlir::Value x = entry->getArgument(0);
    mlir::Value dest1 = entry->getArgument(1);
    mlir::Value dest2 = entry->getArgument(2);

    // relu_tensor(relu_tensor(x, dest1), dest2)
    auto relu1 = builder.create<mlir::c3::ReLUTensorOp>(loc, tensorType, x, dest1);
    auto relu2 = builder.create<mlir::c3::ReLUTensorOp>(loc, tensorType, relu1.getOut(), dest2);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{relu2.getOut()});

    // 运行 C3Combine DRR 表驱动模式
    runC3Combine(module);

    int relu_count = 0;
    func.walk([&](mlir::Operation* op) {
        if (llvm::isa<mlir::c3::ReLUTensorOp>(op)) {
            relu_count++;
        }
    });

    EXPECT_EQ(relu_count, 1) << "ReluIdempotentTensorOptPattern 应当将两层连续 ReLU 幂等折叠为单层";
}

TEST(C3CombineDRRTest, DoubleTransposePatternElimination) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::c3::C3Dialect, mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();

    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto tensorType_2x4 = mlir::RankedTensorType::get({2, 4}, builder.getF32Type());
    auto tensorType_4x2 = mlir::RankedTensorType::get({4, 2}, builder.getF32Type());
    auto funcType = builder.getFunctionType({tensorType_2x4, tensorType_4x2, tensorType_2x4}, {tensorType_2x4});
    auto func = builder.create<mlir::func::FuncOp>(loc, "test_double_transpose", funcType);

    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    mlir::Value x = entry->getArgument(0);
    mlir::Value dest1 = entry->getArgument(1);
    mlir::Value dest2 = entry->getArgument(2);

    // transpose_tensor(transpose_tensor(x, dest1, 2, 4, 0, 1), dest2, 4, 2, 0, 1)
    auto tr1 = builder.create<mlir::c3::TransposeTensorOp>(loc, tensorType_4x2, x, dest1, 2, 4, 0, 1);
    auto tr2 = builder.create<mlir::c3::TransposeTensorOp>(loc, tensorType_2x4, tr1.getOut(), dest2, 4, 2, 0, 1);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{tr2.getOut()});

    // 运行 C3Combine DRR 表驱动模式
    runC3Combine(module);

    int transpose_count = 0;
    mlir::Value returned_val;
    func.walk([&](mlir::Operation* op) {
        if (llvm::isa<mlir::c3::TransposeTensorOp>(op)) {
            transpose_count++;
        }
        if (auto retOp = llvm::dyn_cast<mlir::func::ReturnOp>(op)) {
            if (retOp.getNumOperands() > 0) {
                returned_val = retOp.getOperand(0);
            }
        }
    });

    EXPECT_EQ(transpose_count, 0) << "DoubleTransposeTensorOptPattern 应当完全消除双重转置操作";
    EXPECT_EQ(returned_val, x) << "双重转置消去后应直接返回原始张量 x";
}
