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
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
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

// ======================= Kernel 构建（LLVM 指针版本） =======================

template <typename ArithOp>
static void buildElementwiseBinary(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value a, mlir::Value b, mlir::Value out,
                                   mlir::Value n, int64_t rhs_broadcast_mod = 0,
                                   int64_t known_numel = 0) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();

    // 广播场景下不能展开（idx 可能被取模）
    int64_t effective_known = (rhs_broadcast_mod > 0) ? 0 : known_numel;

    buildLoop(builder, loc, n, effective_known,
        [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx_i64) {
            // 计算 b 的索引（支持广播）
            mlir::Value b_idx_i64 = idx_i64;
            if (rhs_broadcast_mod > 0) {
                mlir::Value mod_val = bld.create<mlir::arith::ConstantIntOp>(loc, rhs_broadcast_mod, 64);
                b_idx_i64 = bld.create<mlir::arith::RemUIOp>(loc, idx_i64, mod_val);
            }

            mlir::Value a_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, a, mlir::ValueRange{idx_i64});
            mlir::Value b_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, b, mlir::ValueRange{b_idx_i64});
            mlir::Value out_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});

            mlir::Value av = bld.create<mlir::LLVM::LoadOp>(loc, f32, a_ptr);
            mlir::Value bv = bld.create<mlir::LLVM::LoadOp>(loc, f32, b_ptr);
            mlir::Value rv = bld.create<ArithOp>(loc, av, bv);
            bld.create<mlir::LLVM::StoreOp>(loc, rv, out_ptr);
        });
}

/// Div 专用：含除零检查，零除时存储 NaN 并继续
static void buildDiv(mlir::OpBuilder& builder, mlir::Location loc,
                     mlir::Value a, mlir::Value b, mlir::Value out,
                     mlir::Value n, int64_t known_numel = 0) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();

    buildLoop(builder, loc, n, known_numel,
        [&](mlir::OpBuilder& bld, mlir::Location loc, mlir::Value idx_i64) {
            mlir::Value a_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, a, mlir::ValueRange{idx_i64});
            mlir::Value b_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, b, mlir::ValueRange{idx_i64});
            mlir::Value out_ptr = bld.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});

            mlir::Value av = bld.create<mlir::LLVM::LoadOp>(loc, f32, a_ptr);
            mlir::Value bv = bld.create<mlir::LLVM::LoadOp>(loc, f32, b_ptr);

            // 除零检查：若 bv == 0.0f，存储 NaN；否则正常除法
            mlir::Value zero = mlir::arith::ConstantFloatOp::create(bld, loc, f32, llvm::APFloat(0.0f));
            mlir::Value is_zero = bld.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, bv, zero);

            auto if_op = bld.create<mlir::scf::IfOp>(loc, f32, is_zero, true);
            bld.setInsertionPointToStart(&if_op.getThenRegion().front());
            mlir::Value nan_val = mlir::arith::ConstantFloatOp::create(
                bld, loc, f32, llvm::APFloat::getNaN(llvm::APFloat::IEEEsingle()));
            bld.create<mlir::scf::YieldOp>(loc, nan_val);

            bld.setInsertionPointToStart(&if_op.getElseRegion().front());
            mlir::Value div_result = bld.create<mlir::arith::DivFOp>(loc, av, bv);
            bld.create<mlir::scf::YieldOp>(loc, div_result);

            bld.setInsertionPointAfter(if_op);
            bld.create<mlir::LLVM::StoreOp>(loc, if_op.getResult(0), out_ptr);
        });
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

/// DEBT-NEW-7 v0.5.1+: 在 module 中声明/查找 ct_simd_* 批量函数（NEON/AVX 手动向量化）
/// 跟 getOrDeclareExpf 同样模式, 但用 C ABI void f(const float* in, float* out, size_t n)
/// 用于 buildSigmoid/buildTanh/buildExp/buildLog 调批量实现代替逐元素 expf
/// 链接: SIMDWrapper.cpp 提供 ct_simd_vexp/vlog/vsigmoid/vtanh/vgelu 实现
template <const char* SymbolName>
static mlir::LLVM::LLVMFuncOp getOrDeclareCtSimdBatchFn(mlir::OpBuilder& builder, mlir::Location loc) {
    auto* ctx = builder.getContext();
    auto module_op = builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
    if (!module_op)
        throw std::runtime_error(std::string("getOrDeclareCtSimdBatchFn: not inside a module (looking up ") + SymbolName + ")");
    auto existing = module_op.lookupSymbol<mlir::LLVM::LLVMFuncOp>(SymbolName);
    if (existing) return existing;

    auto void_type = mlir::LLVM::LLVMVoidType::get(ctx);
    auto i64_type = builder.getI64Type();
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(ctx);
    auto fn_type = mlir::LLVM::LLVMFunctionType::get(void_type, {ptr_type, ptr_type, i64_type}, false);

    auto saved_ip = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(module_op.getBody());
    auto func = builder.create<mlir::LLVM::LLVMFuncOp>(loc, SymbolName, fn_type);
    func.setVisibility(mlir::SymbolTable::Visibility::Private);
    builder.restoreInsertionPoint(saved_ip);
    return func;
}

// 模板实例化需要 char 字符串作为模板参数, 用 namespace 局部变量提供稳定地址
namespace {
    constexpr char kCtSimdVexp[]     = "ct_simd_vexp";
    constexpr char kCtSimdVlog[]     = "ct_simd_vlog";
    constexpr char kCtSimdVsigmoid[] = "ct_simd_vsigmoid";
    constexpr char kCtSimdVtanh[]    = "ct_simd_vtanh";
    constexpr char kCtSimdVgelu[]    = "ct_simd_vgelu";

    /// M1 1.2 (2026-08-09): AutoTuner → MLIR tile 参数接通
    /// thread_local cache 避免每次 module-build 都进 TuningState mutex 锁
    /// (TuningState.get() 内部 std::lock_guard, 编译期虽然不是 hot path,
    /// 但多线程并发编译会争抢; cache 让首线程 fetch 后其余线程 0 锁直读)
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
}

/// 在 module 中声明或查找已有的 cblas_sgemm 外部函数
static mlir::LLVM::LLVMFuncOp getOrDeclareSgemm(mlir::OpBuilder& builder, mlir::Location loc) {
    auto* ctx = builder.getContext();
    auto module_op = builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
    if (!module_op)
        throw std::runtime_error("buildMatMul: not inside a module");
    auto existing = module_op.lookupSymbol<mlir::LLVM::LLVMFuncOp>("cblas_sgemm");
    if (existing) return existing;

    auto void_type = mlir::LLVM::LLVMVoidType::get(ctx);
    auto i32 = builder.getI32Type();
    auto f32 = builder.getF32Type();
    auto ptr = mlir::LLVM::LLVMPointerType::get(ctx);

    auto sgemm_type = mlir::LLVM::LLVMFunctionType::get(void_type, {
        i32, i32, i32, i32, i32, i32, f32, ptr, i32, ptr, i32, f32, ptr, i32
    }, false);

    auto saved_ip = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(module_op.getBody());
    auto func = builder.create<mlir::LLVM::LLVMFuncOp>(loc, "cblas_sgemm", sgemm_type);
    func.setVisibility(mlir::SymbolTable::Visibility::Private);
    builder.restoreInsertionPoint(saved_ip);
    return func;
}

// 小矩阵 MatMul 阈值：当 M*K*N < 该值时，使用内联三重循环替代 cblas_sgemm
// 小矩阵的 BLAS 函数调用开销（参数检查、分块策略选择等）可能超过计算本身，
// 内联实现直接生成 MLIR 循环，避免调用开销且更易被 LLVM 自动向量化。
// [Fix 2026-08-09 DEBT-NEW-5 真根因修]: 根因是 buildFusedEpilogue 对列向量 bias
//   的广播索引错误 (idx%N 在 N=1 时恒 0), 已修复. 原 workaround 把阈值改成 0
//   强制全走 cblas, 但牺牲了小矩阵内联性能. 真根因修复后恢复合理阈值.
//   注意: 内联累加顺序与 cblas 数值不完全等价 (浮点不可结合), 但小矩阵 K 小,
//   误差在 1e-6 精度要求内可接受 (已实测 MLP 全 PASS). 中矩阵 (≥kTiled阈值)
//   仍走 cblas 保证精度.
static constexpr int64_t kSmallMatMulThreshold = 256;

/// 为小矩阵生成内联三重循环 MatMul（替代 cblas_sgemm 调用）
/// 生成 MLIR 代码：
///   for i in 0..M:
///     for j in 0..N:
///       sum = 0
///       for k in 0..K:
///         sum += A[i][k] * B[k][j]
///       C[i][j] = sum
static void buildSmallMatMul(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value a, mlir::Value b, mlir::Value out,
                             mlir::Value M, mlir::Value K, mlir::Value N) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();

    mlir::Value c0_i = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1_i = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value zero_f = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(0.0f));

    mlir::Value M_i = i64ToIndex(builder, loc, M);
    mlir::Value N_i = i64ToIndex(builder, loc, N);
    mlir::Value K_i = i64ToIndex(builder, loc, K);

    // === i 循环 (0..M) ===
    auto i_loop = builder.create<mlir::scf::ForOp>(loc, c0_i, M_i, c1_i);
    builder.setInsertionPointToStart(i_loop.getBody());
    mlir::Value i = i_loop.getInductionVar();
    mlir::Value i_i64 = indexToI64(builder, loc, i);

    // === j 循环 (0..N) ===
    auto j_loop = builder.create<mlir::scf::ForOp>(loc, c0_i, N_i, c1_i);
    builder.setInsertionPointToStart(j_loop.getBody());
    mlir::Value j = j_loop.getInductionVar();
    mlir::Value j_i64 = indexToI64(builder, loc, j);

    // === k 循环 (0..K) 带 reduction (sum = 0; sum += a*b) ===
    auto k_loop = builder.create<mlir::scf::ForOp>(loc, c0_i, K_i, c1_i, mlir::ValueRange{zero_f});
    builder.setInsertionPointToStart(k_loop.getBody());
    mlir::Value k = k_loop.getInductionVar();
    mlir::Value sum = k_loop.getRegion().front().getArguments().back(); // loop-carried variable
    mlir::Value k_i64 = indexToI64(builder, loc, k);

    // A[i][k] = GEP(a, i * K + k)
    mlir::Value a_idx = builder.create<mlir::arith::MulIOp>(loc, i_i64, K);
    a_idx = builder.create<mlir::arith::AddIOp>(loc, a_idx, k_i64);
    mlir::Value a_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, a, mlir::ValueRange{a_idx});
    mlir::Value a_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, a_ptr);

    // B[k][j] = GEP(b, k * N + j)
    mlir::Value b_idx = builder.create<mlir::arith::MulIOp>(loc, k_i64, N);
    b_idx = builder.create<mlir::arith::AddIOp>(loc, b_idx, j_i64);
    mlir::Value b_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, b, mlir::ValueRange{b_idx});
    mlir::Value b_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, b_ptr);

    // sum += a_val * b_val
    mlir::Value prod = builder.create<mlir::arith::MulFOp>(loc, a_val, b_val);
    mlir::Value new_sum = builder.create<mlir::arith::AddFOp>(loc, sum, prod);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{new_sum});

    builder.setInsertionPointAfter(k_loop);
    mlir::Value result = k_loop.getResult(0); // 最终的 sum

    // C[i][j] = result
    mlir::Value out_idx = builder.create<mlir::arith::MulIOp>(loc, i_i64, N);
    out_idx = builder.create<mlir::arith::AddIOp>(loc, out_idx, j_i64);
    mlir::Value out_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{out_idx});
    builder.create<mlir::LLVM::StoreOp>(loc, result, out_ptr);

    builder.setInsertionPointAfter(j_loop);
    builder.setInsertionPointAfter(i_loop);
}

// ======================= MatMul Tiling + Epilogue Fusion =======================

// 激活函数类型枚举
enum class MatMulActivation { None, ReLU, Sigmoid, Tanh };

// 前向声明
static void applyActivation(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value& val, MatMulActivation act);

/// 在 MatMul 输出上应用 fused epilogue（bias add + activation）
/// 支持原地操作（in == out）和分离操作
/// 当 known_numel 是已知的小常量（<=16）时，生成展开版本
static void buildFusedEpilogue(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value in, mlir::Value out,
                                mlir::Value n, mlir::Value bias = nullptr,
                                MatMulActivation act = MatMulActivation::None,
                                int64_t known_numel = 0,
                                mlir::Value N_for_bias = nullptr,
                                size_t bias_numel = 0,
                                size_t matM = 0, size_t matN = 0) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();

    buildLoop(builder, loc, n, known_numel,
        [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx_i64) {
            mlir::Value in_ptr = b.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, in, mlir::ValueRange{idx_i64});
            mlir::Value out_ptr = b.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});

            mlir::Value val = b.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);

            // 偏置加法（广播语义）：依据 bias 形状选择索引
            //  - bias_numel==matN (行向量 (1,N)) → j = idx % N (跨行广播, MNIST bias[N])
            //  - bias_numel==matM (列向量 (M,1)) → j = idx / N (跨列广播, MLP bias[out_dim,1])
            //  - bias_numel==1 (标量)            → j = 0
            //  - 否则 (全量 (M,N))               → j = idx
            // 【P0 DEBT-NEW-5】之前固定 idx % N, 对列向量 bias (M,1) 时 N=1 → j 恒 0
            //   → 每个元素都加 bias[0] 而非 bias[idx], 与 eager 的 x+bias 逐元素不一致
            if (bias) {
                mlir::Value j = idx_i64;
                if (bias_numel == 1) {
                    j = b.create<mlir::arith::ConstantIntOp>(loc, 0, 64); // 标量广播
                } else if (N_for_bias) {
                    if (matN > 0 && bias_numel == static_cast<size_t>(matN)) {
                        j = b.create<mlir::arith::RemUIOp>(loc, idx_i64, N_for_bias); // 行向量
                    } else if (matM > 0 && bias_numel == static_cast<size_t>(matM)) {
                        j = b.create<mlir::arith::DivUIOp>(loc, idx_i64, N_for_bias); // 列向量
                    }
                    // 否则为全量 (M,N): j = idx
                }
                mlir::Value bias_ptr = b.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, bias, mlir::ValueRange{j});
                mlir::Value bias_val = b.create<mlir::LLVM::LoadOp>(loc, f32, bias_ptr);
                val = b.create<mlir::arith::AddFOp>(loc, val, bias_val);
            }

            // 激活函数
            applyActivation(b, loc, val, act);

            b.create<mlir::LLVM::StoreOp>(loc, val, out_ptr);
        });
}

// 默认 tile 大小（后续可通过 AutoTuner 调优）
static constexpr int64_t kDefaultTileM = 32;
static constexpr int64_t kDefaultTileN = 32;
// 使用 tiling 的 MatMul 阈值上限（超过此值仍委托 cblas_sgemm）
// 【P0 修复 DEBT-NEW-5 2026-08-08】原值 65536 触发 MNIST L2 (128x256x128 = 4.2M ops)
// 走 buildTiledMatMulWithEpilogue，sum 累加顺序跟 cblas_sgemm 数值不等价
// 实测：MNIST epoch 0 一致，epoch 1 batch 0 分歧（loss 0.7218 vs 0.2120, grad 差 4x）
// 修复：降到 4096（小矩阵 inline 优化仍生效，大矩阵走 cblas_sgemm 保证精度）
static constexpr int64_t kTiledMatMulThreshold = 4096;

/// 生成 tiled MatMul + 可选的偏置融合 + 激活函数融合
///
/// 生成 MLIR 代码：
///   for i_tile in 0..M step tile_m:
///     for j_tile in 0..N step tile_n:
///       for i in i_tile..min(i_tile+tm, M):
///         for j in j_tile..min(j_tile+tn, N):
///           sum = 0
///           for k in 0..K:
///             sum += A[i][k] * B[k][j]
///           if bias: sum += bias[j]
///           if act: sum = activation(sum)
///           C[i][j] = sum
///
/// 偏置 bias 使用广播语义：bias 是 [N] 向量，对每行共享
/// 激活函数在 MatMul 和偏置加法之后应用
/// 当 tile_m <= 0 或 tile_n <= 0 时，退化为无 tiling 的版本（与 buildSmallMatMul 相同）
static void buildTiledMatMulWithEpilogue(mlir::OpBuilder& builder, mlir::Location loc,
                                          mlir::Value a, mlir::Value b, mlir::Value out,
                                          mlir::Value M, mlir::Value K, mlir::Value N,
                                          mlir::Value bias = nullptr,
                                          MatMulActivation act = MatMulActivation::None,
                                          int64_t tile_m = kDefaultTileM,
                                          int64_t tile_n = kDefaultTileN,
                                          size_t bias_numel = 0,
                                          size_t matM = 0, size_t matN = 0) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();

    mlir::Value c0_i = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1_i = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value zero_f = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(0.0f));

    mlir::Value M_i = i64ToIndex(builder, loc, M);
    mlir::Value N_i = i64ToIndex(builder, loc, N);
    mlir::Value K_i = i64ToIndex(builder, loc, K);

    // 偏置索引（广播语义）：与 buildFusedEpilogue 对齐
    //  - bias_numel==matN (行向量 (1,N)) → bias[col] (每行共享)
    //  - bias_numel==matM && matN==1 (列向量 (M,1)) → bias[row] (每列共享)
    //  - bias_numel==1 (标量)            → bias[0]
    //  - 否则 (全量 (M,N))               → bias[row*N + col]
    auto makeBiasIdx = [&](mlir::Location bloc,
                           mlir::Value i64_row, mlir::Value i64_col,
                           mlir::Value N_i64) {
        mlir::Value bidx = i64_col; // 默认行向量 bias[col]
        if (bias_numel == 1) {
            bidx = builder.create<mlir::arith::ConstantIntOp>(bloc, 0, 64); // 标量
        } else if (matM > 0 && matN == 1 && bias_numel == static_cast<size_t>(matM)) {
            bidx = i64_row; // 列向量 (M,1) → bias[row]
        } else if (matM > 0 && matN > 0 &&
                   bias_numel == static_cast<size_t>(matM * matN)) {
            // 全量 (M,N) → bias[row*N + col]
            bidx = builder.create<mlir::arith::MulIOp>(bloc, i64_row, N_i64);
            bidx = builder.create<mlir::arith::AddIOp>(bloc, bidx, i64_col);
        }
        // 否则行向量 (1,N) → bias[col]
        return bidx;
    };

    if (tile_m > 0 && tile_n > 0) {
        // ========== tiled 版本：2D tiling on M and N ==========
        mlir::Value tm = builder.create<mlir::arith::ConstantIndexOp>(loc, tile_m);
        mlir::Value tn = builder.create<mlir::arith::ConstantIndexOp>(loc, tile_n);

        // === i_tile 循环 (0..M, step=tile_m) ===
        auto i_tile_loop = builder.create<mlir::scf::ForOp>(loc, c0_i, M_i, tm);
        builder.setInsertionPointToStart(i_tile_loop.getBody());
        mlir::Value i_tile = i_tile_loop.getInductionVar();
        mlir::Value i_tile_end = builder.create<mlir::arith::AddIOp>(loc, i_tile, tm);
        i_tile_end = builder.create<mlir::arith::MinSIOp>(loc, i_tile_end, M_i);

        // === j_tile 循环 (0..N, step=tile_n) ===
        auto j_tile_loop = builder.create<mlir::scf::ForOp>(loc, c0_i, N_i, tn);
        builder.setInsertionPointToStart(j_tile_loop.getBody());
        mlir::Value j_tile = j_tile_loop.getInductionVar();
        mlir::Value j_tile_end = builder.create<mlir::arith::AddIOp>(loc, j_tile, tn);
        j_tile_end = builder.create<mlir::arith::MinSIOp>(loc, j_tile_end, N_i);

        // === i 微循环 (i_tile..i_tile_end) ===
        auto i_loop = builder.create<mlir::scf::ForOp>(loc, i_tile, i_tile_end, c1_i);
        builder.setInsertionPointToStart(i_loop.getBody());
        mlir::Value i_idx = i_loop.getInductionVar();
        mlir::Value i_i64 = indexToI64(builder, loc, i_idx);

        // === j 微循环 (j_tile..j_tile_end) ===
        auto j_loop = builder.create<mlir::scf::ForOp>(loc, j_tile, j_tile_end, c1_i);
        builder.setInsertionPointToStart(j_loop.getBody());
        mlir::Value j_idx = j_loop.getInductionVar();
        mlir::Value j_i64 = indexToI64(builder, loc, j_idx);

        // === k 循环 (0..K) 带 reduction ===
        auto k_loop = builder.create<mlir::scf::ForOp>(loc, c0_i, K_i, c1_i, mlir::ValueRange{zero_f});
        builder.setInsertionPointToStart(k_loop.getBody());
        mlir::Value k_idx = k_loop.getInductionVar();
        mlir::Value sum = k_loop.getRegion().front().getArguments().back();
        mlir::Value k_i64 = indexToI64(builder, loc, k_idx);

        // A[i][k] = GEP(a, i * K + k)
        mlir::Value a_idx = builder.create<mlir::arith::MulIOp>(loc, i_i64, K);
        a_idx = builder.create<mlir::arith::AddIOp>(loc, a_idx, k_i64);
        mlir::Value a_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, a, mlir::ValueRange{a_idx});
        mlir::Value a_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, a_ptr);

        // B[k][j] = GEP(b, k * N + j)
        mlir::Value b_idx = builder.create<mlir::arith::MulIOp>(loc, k_i64, N);
        b_idx = builder.create<mlir::arith::AddIOp>(loc, b_idx, j_i64);
        mlir::Value b_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, b, mlir::ValueRange{b_idx});
        mlir::Value b_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, b_ptr);

        // sum += a_val * b_val
        mlir::Value prod = builder.create<mlir::arith::MulFOp>(loc, a_val, b_val);
        mlir::Value new_sum = builder.create<mlir::arith::AddFOp>(loc, sum, prod);
        builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{new_sum});

        builder.setInsertionPointAfter(k_loop);
        mlir::Value result = k_loop.getResult(0);

        // 偏置加法（广播语义：bias[j] 对每行共享）
        if (bias) {
            mlir::Value bias_idx = makeBiasIdx(loc, i_i64, j_i64, N);
            mlir::Value bias_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, bias, mlir::ValueRange{bias_idx});
            mlir::Value bias_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, bias_ptr);
            result = builder.create<mlir::arith::AddFOp>(loc, result, bias_val);
        }

        // 激活函数
        applyActivation(builder, loc, result, act);

        // C[i][j] = result
        mlir::Value out_idx = builder.create<mlir::arith::MulIOp>(loc, i_i64, N);
        out_idx = builder.create<mlir::arith::AddIOp>(loc, out_idx, j_i64);
        mlir::Value out_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{out_idx});
        builder.create<mlir::LLVM::StoreOp>(loc, result, out_ptr);

        // 关闭所有循环（从内到外）
        builder.setInsertionPointAfter(j_loop);   // 仍在 i_loop 体内
        builder.setInsertionPointAfter(i_loop);   // 仍在 j_tile_loop 体内
        builder.setInsertionPointAfter(j_tile_loop); // 仍在 i_tile_loop 体内
        builder.setInsertionPointAfter(i_tile_loop);
    } else {
        // ========== 无 tiling 版本（与 buildSmallMatMul 相同，但带 epilogue 融合） ==========
        // === i 循环 (0..M) ===
        auto i_loop = builder.create<mlir::scf::ForOp>(loc, c0_i, M_i, c1_i);
        builder.setInsertionPointToStart(i_loop.getBody());
        mlir::Value i = i_loop.getInductionVar();
        mlir::Value i_i64 = indexToI64(builder, loc, i);

        // === j 循环 (0..N) ===
        auto j_loop = builder.create<mlir::scf::ForOp>(loc, c0_i, N_i, c1_i);
        builder.setInsertionPointToStart(j_loop.getBody());
        mlir::Value j = j_loop.getInductionVar();
        mlir::Value j_i64 = indexToI64(builder, loc, j);

        // === k 循环 (0..K) 带 reduction ===
        auto k_loop = builder.create<mlir::scf::ForOp>(loc, c0_i, K_i, c1_i, mlir::ValueRange{zero_f});
        builder.setInsertionPointToStart(k_loop.getBody());
        mlir::Value k = k_loop.getInductionVar();
        mlir::Value sum = k_loop.getRegion().front().getArguments().back();
        mlir::Value k_i64 = indexToI64(builder, loc, k);

        // A[i][k]
        mlir::Value a_idx = builder.create<mlir::arith::MulIOp>(loc, i_i64, K);
        a_idx = builder.create<mlir::arith::AddIOp>(loc, a_idx, k_i64);
        mlir::Value a_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, a, mlir::ValueRange{a_idx});
        mlir::Value a_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, a_ptr);

        // B[k][j]
        mlir::Value b_idx = builder.create<mlir::arith::MulIOp>(loc, k_i64, N);
        b_idx = builder.create<mlir::arith::AddIOp>(loc, b_idx, j_i64);
        mlir::Value b_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, b, mlir::ValueRange{b_idx});
        mlir::Value b_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, b_ptr);

        // sum += a_val * b_val
        mlir::Value prod = builder.create<mlir::arith::MulFOp>(loc, a_val, b_val);
        mlir::Value new_sum = builder.create<mlir::arith::AddFOp>(loc, sum, prod);
        builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{new_sum});

        builder.setInsertionPointAfter(k_loop);
        mlir::Value result = k_loop.getResult(0);

        // 偏置加法
        if (bias) {
            mlir::Value bias_idx = makeBiasIdx(loc, i_i64, j_i64, N);
            mlir::Value bias_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, bias, mlir::ValueRange{bias_idx});
            mlir::Value bias_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, bias_ptr);
            result = builder.create<mlir::arith::AddFOp>(loc, result, bias_val);
        }

        // 激活函数
        applyActivation(builder, loc, result, act);

        // C[i][j] = result
        mlir::Value out_idx = builder.create<mlir::arith::MulIOp>(loc, i_i64, N);
        out_idx = builder.create<mlir::arith::AddIOp>(loc, out_idx, j_i64);
        mlir::Value out_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{out_idx});
        builder.create<mlir::LLVM::StoreOp>(loc, result, out_ptr);

        builder.setInsertionPointAfter(j_loop);
        builder.setInsertionPointAfter(i_loop);
    }
}

/// 应用激活函数到 MLIR Value 上
static void applyActivation(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value& val, MatMulActivation act) {
    auto f32 = builder.getF32Type();
    if (act == MatMulActivation::ReLU) {
        mlir::Value zero = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(0.0f));
        val = builder.create<mlir::arith::MaxNumFOp>(loc, val, zero);
    } else if (act == MatMulActivation::Sigmoid) {
        auto expf_func = getOrDeclareExpf(builder, loc);
        mlir::Value neg = builder.create<mlir::arith::NegFOp>(loc, val);
        mlir::Value exp_val = builder.create<mlir::LLVM::CallOp>(loc, expf_func, mlir::ValueRange{neg}).getResult();
        mlir::Value one = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(1.0f));
        mlir::Value denom = builder.create<mlir::arith::AddFOp>(loc, one, exp_val);
        val = builder.create<mlir::arith::DivFOp>(loc, one, denom);
    } else if (act == MatMulActivation::Tanh) {
        auto expf_func = getOrDeclareExpf(builder, loc);
        mlir::Value exp_val = builder.create<mlir::LLVM::CallOp>(loc, expf_func, mlir::ValueRange{val}).getResult();
        mlir::Value neg = builder.create<mlir::arith::NegFOp>(loc, val);
        mlir::Value exp_neg = builder.create<mlir::LLVM::CallOp>(loc, expf_func, mlir::ValueRange{neg}).getResult();
        mlir::Value num = builder.create<mlir::arith::SubFOp>(loc, exp_val, exp_neg);
        mlir::Value denom = builder.create<mlir::arith::AddFOp>(loc, exp_val, exp_neg);
        val = builder.create<mlir::arith::DivFOp>(loc, num, denom);
    }
    // None: 不做任何操作
}

/// 调用 cblas_sgemm 执行 MatMul
/// @param beta sgemm 的 beta 参数：
///   0.0f = C = A*B（覆盖写入）
///   1.0f = C = A*B + C（累加模式，用于 bias 预填充融合）
static void buildMatMul(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::Value a, mlir::Value b, mlir::Value out,
                        mlir::Value M, mlir::Value K, mlir::Value N,
                        float beta = 0.0f) {
    auto* ctx = builder.getContext();
    auto i32 = builder.getI32Type();
    auto f32 = builder.getF32Type();
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(ctx);

    // 常量定义
    // CblasRowMajor = 101, CblasNoTrans = 111
    auto c_row_major = builder.create<mlir::arith::ConstantIntOp>(loc, 101, 32);
    auto c_no_trans  = builder.create<mlir::arith::ConstantIntOp>(loc, 111, 32);
    auto c_alpha     = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(1.0f));
    auto c_beta      = mlir::arith::ConstantFloatOp::create(builder, loc, f32, llvm::APFloat(beta));

    // i64 → i32 截断
    auto M_i32 = builder.create<mlir::LLVM::TruncOp>(loc, i32, M);
    auto K_i32 = builder.create<mlir::LLVM::TruncOp>(loc, i32, K);
    auto N_i32 = builder.create<mlir::LLVM::TruncOp>(loc, i32, N);

    // lda = K, ldb = N, ldc = N (row-major)
    auto lda = K_i32;
    auto ldb = N_i32;
    auto ldc = N_i32;

    // 声明或复用 cblas_sgemm
    auto sgemm = getOrDeclareSgemm(builder, loc);

    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //             M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    builder.create<mlir::LLVM::CallOp>(loc, sgemm, mlir::ValueRange{
        c_row_major, c_no_trans, c_no_trans,
        M_i32, N_i32, K_i32,
        c_alpha, a, lda,
        b, ldb,
        c_beta, out, ldc
    });
}

/// 将 bias 向量 [N] 广播填充到 out 矩阵 [M×N] 的每一行
/// 用于与 cblas_sgemm(beta=1.0) 配合，将 bias 加法融合进 BLAS 调用，
/// 避免 MatMul 完成后再单独回读做 bias add
static void buildBiasPrefill(mlir::OpBuilder& builder, mlir::Location loc,
                              mlir::Value out, mlir::Value bias,
                              mlir::Value M, mlir::Value N) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();

    mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value M_i = i64ToIndex(builder, loc, M);
    mlir::Value N_i = i64ToIndex(builder, loc, N);

    // for i in 0..M:
    //   for j in 0..N:
    //     out[i*N + j] = bias[j]
    auto i_loop = builder.create<mlir::scf::ForOp>(loc, c0, M_i, c1);
    builder.setInsertionPointToStart(i_loop.getBody());
    mlir::Value i = i_loop.getInductionVar();
    mlir::Value i_i64 = indexToI64(builder, loc, i);

    auto j_loop = builder.create<mlir::scf::ForOp>(loc, c0, N_i, c1);
    builder.setInsertionPointToStart(j_loop.getBody());
    mlir::Value j = j_loop.getInductionVar();
    mlir::Value j_i64 = indexToI64(builder, loc, j);

    // out[i*N + j] = bias[i*N + j]（使用 flat index，与 buildFusedEpilogue 一致）
    mlir::Value out_idx = builder.create<mlir::arith::MulIOp>(loc, i_i64, N);
    out_idx = builder.create<mlir::arith::AddIOp>(loc, out_idx, j_i64);
    mlir::Value out_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{out_idx});
    mlir::Value bias_ptr = builder.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, bias, mlir::ValueRange{out_idx});
    mlir::Value bias_val = builder.create<mlir::LLVM::LoadOp>(loc, f32, bias_ptr);
    builder.create<mlir::LLVM::StoreOp>(loc, bias_val, out_ptr);

    builder.setInsertionPointAfter(j_loop);
    builder.setInsertionPointAfter(i_loop);
}

static void buildNegate(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::Value in, mlir::Value out, mlir::Value n,
                        int64_t known_numel = 0) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();

    buildLoop(builder, loc, n, known_numel,
        [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx_i64) {
            mlir::Value in_ptr = b.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, in, mlir::ValueRange{idx_i64});
            mlir::Value out_ptr = b.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});

            mlir::Value iv = b.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
            mlir::Value rv = b.create<mlir::arith::NegFOp>(loc, iv);
            b.create<mlir::LLVM::StoreOp>(loc, rv, out_ptr);
        });
}

static void buildReLU(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value in, mlir::Value out, mlir::Value n,
                      int64_t known_numel = 0) {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto f32 = builder.getF32Type();

    buildLoop(builder, loc, n, known_numel,
        [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx_i64) {
            mlir::Value in_ptr = b.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, in, mlir::ValueRange{idx_i64});
            mlir::Value out_ptr = b.create<mlir::LLVM::GEPOp>(loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});

            mlir::Value iv = b.create<mlir::LLVM::LoadOp>(loc, f32, in_ptr);
            mlir::Value zero = mlir::arith::ConstantFloatOp::create(b, loc, f32, llvm::APFloat(0.0f));
            mlir::Value rv = b.create<mlir::arith::MaxNumFOp>(loc, iv, zero);
            b.create<mlir::LLVM::StoreOp>(loc, rv, out_ptr);
        });
}

static void buildSigmoid(mlir::OpBuilder& builder, mlir::Location loc,
                         mlir::Value in, mlir::Value out, mlir::Value n,
                         int64_t known_numel = 0) {
    // DEBT-NEW-7 v0.5.1+: 直接调 ct_simd_vsigmoid 批量实现 (NEON/AVX 向量化)
    // 跟之前逐元素 expf 不同, ct_simd_vsigmoid 一次处理 4-8 个元素
    // 调用约定: void ct_simd_vsigmoid(const float* in, float* out, size_t n)
    auto i64_type = builder.getI64Type();
    auto fn = getOrDeclareCtSimdBatchFn<kCtSimdVsigmoid>(builder, loc);
    builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{in, out, n});
    (void)i64_type; // 保留 unused 引用 (模板实装可能不需要)
    (void)known_numel; // 批量实现不依赖 known_numel
}

static void buildTanh(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value in, mlir::Value out, mlir::Value n,
                      int64_t known_numel = 0) {
    // DEBT-NEW-7 v0.5.1+: 直接调 ct_simd_vtanh 批量实现 (NEON/AVX 向量化)
    // 调用约定: void ct_simd_vtanh(const float* in, float* out, size_t n)
    auto fn = getOrDeclareCtSimdBatchFn<kCtSimdVtanh>(builder, loc);
    builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{in, out, n});
    (void)known_numel; // 批量实现不依赖 known_numel
}

// ======================= 融合 Kernel 构建 =======================

static void buildFused(mlir::OpBuilder& builder, mlir::Location loc,
                       mlir::Value inputs, mlir::Value out, mlir::Value n,
                       const std::vector<NodeVariant>& ops,
                       const std::vector<std::vector<size_t>>& op_inputs,
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
            if (op_idx > 0 && in_id == op_inputs[op_idx][0]) continue;
            referenced_nodes.insert(in_id);
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
                        mlir::Value nan_v = mlir::arith::ConstantFloatOp::create(
                            b, loc, f32, llvm::APFloat::getNaN(llvm::APFloat::IEEEsingle()));
                        b.create<mlir::scf::YieldOp>(loc, nan_v);
                        b.setInsertionPointToStart(&div_if.getElseRegion().front());
                        mlir::Value div_r = b.create<mlir::arith::DivFOp>(loc, lhs, rhs);
                        b.create<mlir::scf::YieldOp>(loc, div_r);
                        b.setInsertionPointAfter(div_if);
                        result = div_if.getResult(0);
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
                    }
                }, op);

                if (is_last) {
                    mlir::Value out_ptr = b.create<mlir::LLVM::GEPOp>(
                        loc, ptr_type, f32, out, mlir::ValueRange{idx_i64});
                    b.create<mlir::LLVM::StoreOp>(loc, result, out_ptr);
                } else {
                    prev_val = result;
                }
            }
        });
}

// ======================= 多节点融合 Kernel 构建 =======================

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
    std::unordered_map<size_t, size_t> node_to_buffer;
    std::vector<size_t> buffer_numels; // buffer index → numel
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
            buffer_numels.push_back(compute_nodes[i]->out_desc.numel);
        } else {
            node_to_buffer[node_id] = SIZE_MAX;
        }
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
        int64_t pool_bytes = (int64_t)(max_numel * sizeof(float));
        mlir::Value pool_size = builder.create<mlir::arith::ConstantIntOp>(loc, pool_bytes, 64);
        for (size_t pi = 0; pi < pool_buf_count; ++pi) {
            auto call = builder.create<mlir::LLVM::CallOp>(loc, malloc_func, mlir::ValueRange{pool_size});
            tmp_buffers.push_back(call.getResult());
        }
    }

    // 逻辑 buffer → pool buffer 的映射（交替分配，确保串行图的正确性）
    // logical_buf_idx → pool_buf_idx (0 或 1)
    std::vector<size_t> logical_to_pool(num_intermediates, SIZE_MAX);
    for (size_t i = 0; i < num_intermediates; ++i) {
        logical_to_pool[i] = i % std::max(pool_buf_count, (size_t)1);
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
            size_t pool_idx = logical_to_pool[buf_it->second];
            if (pool_idx < tmp_buffers.size()) return tmp_buffers[pool_idx];
            return out_ptr; // fallback
        }
        return out_ptr; // fallback
    };

    // 步骤 6: 生成每个计算节点的 MLIR 代码
    for (size_t ci = 0; ci < compute_nodes.size(); ++ci) {
        const Node* node = compute_nodes[ci];
        bool is_last = (ci == compute_nodes.size() - 1);
        // 确定输出 buffer：优先使用原地复用的 buffer
        mlir::Value out_buf;
        if (is_last) {
            out_buf = out_ptr;
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
            auto fn_node_n = builder.create<mlir::arith::ConstantIntOp>(loc, node_numel, 64);
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
            // 生成融合循环（传入 arg_numels 以支持广播）
            buildFusedMultiNode(builder, loc, out_buf, fn_node_n, fnode.ops, fnode.op_inputs,
                                fnode.arg_node_ids, fused_arg_ptrs, fused_arg_numels);
            continue;
        }

        // 收集输入指针
        std::vector<mlir::Value> in_ptrs;
        for (size_t in_id : node->inputs) {
            in_ptrs.push_back(getInputPtr(in_id));
        }

        // 计算广播取模（用于 element-wise binary ops）
        auto getBroadcastMod = [&](const NodeVariant& op) -> int64_t {
            auto getRhsShape = [](const NodeVariant& v) -> std::vector<size_t> {
                if (std::holds_alternative<AddNode>(v)) return std::get<AddNode>(v).rhs_desc.shape;
                if (std::holds_alternative<SubNode>(v)) return std::get<SubNode>(v).rhs_desc.shape;
                if (std::holds_alternative<MulNode>(v)) return std::get<MulNode>(v).rhs_desc.shape;
                if (std::holds_alternative<DivNode>(v)) return std::get<DivNode>(v).rhs_desc.shape;
                return {};
            };
            auto getLhsShape = [](const NodeVariant& v) -> std::vector<size_t> {
                if (std::holds_alternative<AddNode>(v)) return std::get<AddNode>(v).lhs_desc.shape;
                if (std::holds_alternative<SubNode>(v)) return std::get<SubNode>(v).lhs_desc.shape;
                if (std::holds_alternative<MulNode>(v)) return std::get<MulNode>(v).lhs_desc.shape;
                if (std::holds_alternative<DivNode>(v)) return std::get<DivNode>(v).lhs_desc.shape;
                return {};
            };
            auto lhs = getLhsShape(op);
            auto rhs = getRhsShape(op);
            if (lhs.empty() || rhs.empty() || lhs == rhs) return 0;
            size_t rhs_numel = 1;
            for (size_t d : rhs) rhs_numel *= d;
            if (rhs_numel == 1) return 1; // scalar broadcast
            if (rhs.size() == 1 && !lhs.empty() && lhs.back() == rhs[0]) {
                return (int64_t)rhs[0]; // 1D vector broadcast to last dim
            }
            return 0; // unsupported broadcast pattern
        };

        // 节点自身输出 numel（用于 element-wise 循环计数）
        int64_t node_numel = (int64_t)node->out_desc.numel;
        auto node_n = builder.create<mlir::arith::ConstantIntOp>(loc, node_numel, 64);

        if (std::holds_alternative<MatMulNode>(op)) {
            // 每个 MatMul 使用自己的 M, K, N 维度
            const auto& mm = std::get<MatMulNode>(op);
            int64_t matM = (int64_t)mm.lhs_desc.shape[0];
            int64_t matK = (int64_t)mm.lhs_desc.shape[1];
            int64_t matN = (int64_t)mm.rhs_desc.shape[1];
            auto mm_M = builder.create<mlir::arith::ConstantIntOp>(loc, matM, 64);
            auto mm_K = builder.create<mlir::arith::ConstantIntOp>(loc, matK, 64);
            auto mm_N = builder.create<mlir::arith::ConstantIntOp>(loc, matN, 64);

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

            // === 选择最佳 MatMul 策略 ===
            int64_t total_ops = matM * matK * matN;
            // M1 1.2 (2026-08-09): 读 AutoTuner 调优结果替换写死的 kDefaultTileM/N
            // 调优未跑 (tuned=false) 时 thread_local cache 持有 {0,0,0}, 落到默认 32/32
            auto& tile = currentTileCache();
            int64_t tile_m = (tile.tile_m > 0) ? tile.tile_m : kDefaultTileM;
            int64_t tile_n = (tile.tile_n > 0) ? tile.tile_n : kDefaultTileN;
            // 仅在 M 和 N 都足够大时才使用 tiling（避免 N=1 时 tiling 空转）
            bool use_tiling = (total_ops >= kSmallMatMulThreshold &&
                               total_ops < kTiledMatMulThreshold &&
                               matM >= tile_m && matN >= tile_n);
#ifdef CT_DEBUG
            fprintf(stderr, "[DBG-5D-MLP] MatMul total_ops=%lld kSmall=%lld use_tiling=%d branch=%s\n",
                    (long long)total_ops, (long long)kSmallMatMulThreshold, (int)use_tiling,
                    total_ops < kSmallMatMulThreshold ? "small_inline"
                    : (use_tiling ? "tiled_inline" : "cblas"));
#endif
            if (total_ops < kSmallMatMulThreshold) {
                // 小矩阵：使用无 tiling 的内联循环（带 epilogue 融合）
                buildTiledMatMulWithEpilogue(builder, loc, in_ptrs[0], in_ptrs[1], out_buf,
                                             mm_M, mm_K, mm_N, fused_bias_ptr, fused_act,
                                             /*tile_m=*/0, /*tile_n=*/0,
                                             fused_bias_numel, (size_t)matM, (size_t)matN);
            } else if (use_tiling) {
                // 中矩阵：使用 2D tiling 的融合版本（改善缓存利用率）
                // tile 来自 AutoTuner 调优 (M1 1.2 接通)
                buildTiledMatMulWithEpilogue(builder, loc, in_ptrs[0], in_ptrs[1], out_buf,
                                             mm_M, mm_K, mm_N, fused_bias_ptr, fused_act,
                                             tile_m, tile_n,
                                             fused_bias_numel, (size_t)matM, (size_t)matN);
            } else {
                // 大矩阵：委托 cblas_sgemm（BLAS 对大型矩阵有最优实现）
                // epilogue（bias + activation）在 sgemm 之后单独执行
                buildMatMul(builder, loc, in_ptrs[0], in_ptrs[1], out_buf, mm_M, mm_K, mm_N);
                if (fused_bias_ptr || fused_act != MatMulActivation::None) {
                    int64_t out_numel = matM * matN;
                    // 【P0 修复 DEBT-NEW-4 2026-08-08】传 mm_N 给 buildFusedEpilogue，
                    // 让 bias 索引用 idx_i64 % N（之前用 idx_i64 直接索引 1D bias → 越界）
                    if (out_numel > 0 && out_numel <= 16) {
                        buildFusedEpilogue(builder, loc, out_buf, out_buf,
                                           builder.create<mlir::arith::ConstantIntOp>(loc, out_numel, 64),
                                           fused_bias_ptr, fused_act, out_numel, mm_N,
                                           fused_bias_numel, (size_t)matM, (size_t)matN);
                    } else {
                        buildFusedEpilogue(builder, loc, out_buf, out_buf,
                                           builder.create<mlir::arith::ConstantIntOp>(loc, out_numel, 64),
                                           fused_bias_ptr, fused_act, /*known_numel=*/0, mm_N,
                                           fused_bias_numel, (size_t)matM, (size_t)matN);
                    }
                }
            }

            // 跳过被融合的后续节点
            ci += fused_skip;
        } else if (std::holds_alternative<AddNode>(op)) {
            int64_t bmod = getBroadcastMod(op);
            buildElementwiseBinary<mlir::arith::AddFOp>(builder, loc, in_ptrs[0], in_ptrs[1], out_buf, node_n, bmod);
        } else if (std::holds_alternative<SubNode>(op)) {
            int64_t bmod = getBroadcastMod(op);
            buildElementwiseBinary<mlir::arith::SubFOp>(builder, loc, in_ptrs[0], in_ptrs[1], out_buf, node_n, bmod);
        } else if (std::holds_alternative<MulNode>(op)) {
            int64_t bmod = getBroadcastMod(op);
            buildElementwiseBinary<mlir::arith::MulFOp>(builder, loc, in_ptrs[0], in_ptrs[1], out_buf, node_n, bmod);
        } else if (std::holds_alternative<DivNode>(op)) {
            buildDiv(builder, loc, in_ptrs[0], in_ptrs[1], out_buf, node_n);
        } else if (std::holds_alternative<NegNode>(op)) {
            buildNegate(builder, loc, in_ptrs[0], out_buf, node_n);
        } else if (std::holds_alternative<ReLUNode>(op)) {
            buildReLU(builder, loc, in_ptrs[0], out_buf, node_n);
        } else if (std::holds_alternative<SigmoidNode>(op)) {
            buildSigmoid(builder, loc, in_ptrs[0], out_buf, node_n);
        } else if (std::holds_alternative<TanhNode>(op)) {
            buildTanh(builder, loc, in_ptrs[0], out_buf, node_n);
        } else {
            // [Fix 2026-08-09 用户审查 P0]: 静默跳过是不允许的 (per user 审查:
            // 'Transpose/Exp/Log/SumReduce 进多节点仍静默跳过,比抛异常更危险')。
            // 改: 显式 throw,让编译失败 → compileAsync 走 fallback 路径。
            // 支持的 op 列表 (M1 路线图 9/15 + SumReduceNode):
            //   MatMul/Add/Sub/Mul/Div/Neg/ReLU/Sigmoid/Tanh + SumReduce
            // 未支持 (M2 范畴): Transpose/Exp/Log/Gt/Const
            const std::string op_name = std::visit(
                [](const auto& n) -> std::string { return typeid(n).name(); },
                op);
            throw std::runtime_error(
                "MLIRKernelGen: unsupported op in multi-node graph: " + op_name +
                " (per MLIR backend 完整化路线图, M2 范畴 v0.5.3+ 实装)");
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
        else if (std::holds_alternative<SigmoidNode>(op))
            buildSigmoid(builder, loc, a, out, n);
        else if (std::holds_alternative<TanhNode>(op))
            buildTanh(builder, loc, a, out, n);
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

static void runPass(mlir::ModuleOp module, std::unique_ptr<mlir::Pass> pass, const char* name) {
    mlir::PassManager pm(module.getContext());
    pm.addPass(std::move(pass));
    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error(std::string("MLIRKernelGen: ") + name + " failed");
    }
}

static void applyLoweringPipeline(mlir::ModuleOp module) {
    runPass(module, mlir::createCanonicalizerPass(), "Canonicalizer");
    runPass(module, mlir::createCSEPass(), "CSE");
    runPass(module, mlir::createLoopInvariantCodeMotionPass(), "LICM");
    runPass(module, mlir::createSCFToControlFlowPass(), "SCFToCF");

    runPass(module, mlir::createArithToLLVMConversionPass(), "ArithToLLVM");

    runPass(module, mlir::createConvertControlFlowToLLVMPass(), "CFToLLVM");
    runPass(module, mlir::createConvertFuncToLLVMPass(), "FuncToLLVM");
    runPass(module, mlir::createFinalizeMemRefToLLVMConversionPass(), "MemRefToLLVM");

    runPass(module, mlir::createReconcileUnrealizedCastsPass(), "ReconcileUnrealizedCasts");
}

// ======================= 主入口 =======================

GeneratedKernel generateFromGraphMLIR(const Graph& graph, int opt_level) {
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

    auto context = std::make_shared<mlir::MLIRContext>(reg);
    context->loadDialect<mlir::arith::ArithDialect>();
    context->loadDialect<mlir::math::MathDialect>();
    context->loadDialect<mlir::scf::SCFDialect>();
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::memref::MemRefDialect>();
    context->loadDialect<mlir::LLVM::LLVMDialect>();

    auto module = buildMLIRModule(*context, graph);
    applyLoweringPipeline(*module);

    // 在 lowering pipeline 后注册翻译接口，ExecutionEngine 需要它们来翻译 LLVM IR
    mlir::registerBuiltinDialectTranslation(*context);
    mlir::registerLLVMDialectTranslation(*context);

    // 创建 TargetMachine 以启用 LLVM 自动向量化（NEON/SIMD）
    auto tm = std::shared_ptr<llvm::TargetMachine>(
        llvm::EngineBuilder()
            .setEngineKind(llvm::EngineKind::JIT)
            .selectTarget());

    mlir::ExecutionEngineOptions engineOpts;
    if (tm) {
        // 【DEBT-NEW-5 实验 2026-08-08 23:15】先置空 transformer（不做 LLVM pass），
        // 隔离 MLIR-level vs LLVM-level 数值差异来源
        engineOpts.transformer = {};
    } else {
        engineOpts.transformer = {};
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
