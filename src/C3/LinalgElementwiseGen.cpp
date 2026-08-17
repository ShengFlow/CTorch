/**
 * @file LinalgElementwiseGen.cpp
 * @generation JIT-2.0 声明式单算子后端（Linalg 逐元素实现）
 * @brief linalg.generic 声明式逐元素 kernel 生成器（移植自 exp_linalg_elementwise PoC）
 *
 * 技术要点（见 STATUS_CONTEXT 4.9）：
 *  1. 动态 memref 必须用 `ShapedType::kDynamic`（INT64_MIN）创建，不能写字面量 -1；
 *  2. `FinalizeMemRefToLLVMConversionPass` 把 memref<?xf32> 展开成 5 个标量参数，
 *     `invokePacked` 需按展开后的标量逐个传指针（每 memref 5 个指针）；
 *  3. Lowering pipeline：linalg-to-loops → scf-to-cf → arith-to-llvm → math-to-llvm
 *     → cf-to-llvm → func-to-llvm → memref-to-llvm → reconcile-unrealized-casts。
 *
 * @date 2026/08/15
 */

#include "C3/LinalgElementwiseGen.h"
#include "C3/JITCache.h"
#include "MLIRKernelGen.h"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>  // affine.apply → arith.remsi 自定义 pattern
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/Passes.h>  // 提供 createConvertFuncToLLVMPass 等工厂函数 + createLowerAffinePass

#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Target/TargetMachine.h>

namespace ct {
namespace c3 {

// ======================= 算子元信息 =======================

bool isUnaryElementwiseOp(ElementwiseOp op) {
    switch (op) {
    case ElementwiseOp::ReLU:
    case ElementwiseOp::Sigmoid:
    case ElementwiseOp::Tanh:
    case ElementwiseOp::Exp:
    case ElementwiseOp::Log:
        return true;
    case ElementwiseOp::Add:
    case ElementwiseOp::Sub:
    case ElementwiseOp::Mul:
        return false;
    }
    return false;
}

const char* elementwiseOpName(ElementwiseOp op) {
    switch (op) {
    case ElementwiseOp::ReLU: return "ReLU";
    case ElementwiseOp::Sigmoid: return "Sigmoid";
    case ElementwiseOp::Tanh: return "Tanh";
    case ElementwiseOp::Exp: return "Exp";
    case ElementwiseOp::Log: return "Log";
    case ElementwiseOp::Add: return "Add";
    case ElementwiseOp::Sub: return "Sub";
    case ElementwiseOp::Mul: return "Mul";
    }
    return "Unknown";
}

size_t elementwiseOpNumInputs(ElementwiseOp op) {
    return isUnaryElementwiseOp(op) ? 1 : 2;
}

// ======================= memref<?xf32> 的 ABI 辅助 =======================
// FinalizeMemRefToLLVMConversionPass 会把 memref<?xf32> 函数参数展开成 5 个标量：
//   (alloc_ptr, aligned_ptr, offset, size, stride)
// 因此 invokePacked 的 packed 数组里每个 memref 需要 5 个指针。
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

// ======================= MLIR 模块构建 =======================

namespace {

/// 在 func 的 entry block 里构建 linalg.generic 逐元素算子（dest-style）。
/// 一元签名: void kernel(memref<?xf32> in, memref<?xf32> out)
/// 二元签名: void kernel(memref<?xf32> a, memref<?xf32> b, memref<?xf32> out)
/// rhs_mod 控制 b 的 indexing map（循环域由输出 identity map 决定，仍迭代 n 次）：
///   0      同尺寸 identity map `d0 -> d0`
///   1      标量广播投影 map `d0 -> 0`（b size=1）
///   k>1    周期广播投影 map `d0 -> (d0 mod k)`（b 为 1D vector size=k，沿主维度周期复用）
void buildLinalgElementwiseFunc(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::MLIRContext& context, ElementwiseOp op,
                                size_t num_inputs, int rhs_mod) {
    auto f32Type = builder.getF32Type();
    auto memrefType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, f32Type);

    std::vector<mlir::Type> arg_types(num_inputs + 1, memrefType);
    auto funcType = builder.getFunctionType(arg_types, {});
    auto func = builder.create<mlir::func::FuncOp>(loc, "c3_kernel", funcType);
    for (size_t i = 0; i < arg_types.size(); ++i) {
        func.setArgAttr(i, "llvm.noalias", builder.getUnitAttr());
    }

    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    std::vector<mlir::Value> memrefs;
    for (size_t i = 0; i < num_inputs + 1; ++i) {
        memrefs.push_back(entry->getArgument(static_cast<unsigned>(i)));
    }
    mlir::Value out_memref = memrefs.back();

    // Identity 索引映射（逐元素，1D）
    mlir::AffineExpr d0 = builder.getAffineDimExpr(0);
    auto identityMap = mlir::AffineMap::get(1, 0, {d0}, &context);
    // 标量广播：b 投影到常量 0（`d0 -> 0`），不参与循环域尺寸计算
    auto zeroMap = mlir::AffineMap::get(1, 0,
                                        {mlir::getAffineConstantExpr(0, &context)}, &context);
    // 周期广播（k>1）：b 投影到 `d0 mod k`，使 1D vector（size=k）沿主维度周期复用
    mlir::AffineMap modMap;
    if (rhs_mod > 1) {
        modMap = mlir::AffineMap::get(1, 0,
                                      {d0 % mlir::getAffineConstantExpr(rhs_mod, &context)},
                                      &context);
    }
    std::vector<mlir::AffineMap> indexingMaps;
    indexingMaps.reserve(num_inputs + 1);
    for (size_t i = 0; i < num_inputs; ++i) {
        if (rhs_mod > 0 && i == 1) {
            // 二元第二输入广播：标量投影 (mod==1) 或周期投影 (mod==k>1)
            indexingMaps.push_back(rhs_mod == 1 ? zeroMap : modMap);
        } else {
            indexingMaps.push_back(identityMap);
        }
    }
    indexingMaps.push_back(identityMap); // 输出始终 identity
    std::vector<mlir::utils::IteratorType> iteratorTypes{
        mlir::utils::IteratorType::parallel};

    mlir::ImplicitLocOpBuilder iBuilder(loc, builder);
    std::vector<mlir::Value> inputs(memrefs.begin(), memrefs.end() - 1);

    iBuilder.create<mlir::linalg::GenericOp>(
        mlir::TypeRange{},
        mlir::ValueRange{inputs},
        mlir::ValueRange{out_memref},
        indexingMaps,   // std::vector<AffineMap> → ArrayRef<AffineMap> 隐式转换
        iteratorTypes,  // std::vector<IteratorType> → ArrayRef<IteratorType>
        [&](mlir::OpBuilder& b, mlir::Location regionLoc, mlir::ValueRange args) {
            mlir::Value result;
            switch (op) {
            case ElementwiseOp::ReLU: {
                mlir::Value in_val = args[0];
                mlir::Value zero = b.create<mlir::arith::ConstantFloatOp>(
                    regionLoc, f32Type, llvm::APFloat(0.0f));
                result = b.create<mlir::arith::MaxNumFOp>(regionLoc, in_val, zero);
                break;
            }
            case ElementwiseOp::Sigmoid: {
                mlir::Value x = args[0];
                mlir::Value neg_x = b.create<mlir::arith::NegFOp>(regionLoc, x);
                mlir::Value exp_neg_x = b.create<mlir::math::ExpOp>(regionLoc, neg_x);
                mlir::Value one = b.create<mlir::arith::ConstantFloatOp>(
                    regionLoc, f32Type, llvm::APFloat(1.0f));
                mlir::Value denom = b.create<mlir::arith::AddFOp>(regionLoc, one, exp_neg_x);
                result = b.create<mlir::arith::DivFOp>(regionLoc, one, denom);
                break;
            }
            case ElementwiseOp::Tanh:
                result = b.create<mlir::math::TanhOp>(regionLoc, args[0]);
                break;
            case ElementwiseOp::Exp:
                result = b.create<mlir::math::ExpOp>(regionLoc, args[0]);
                break;
            case ElementwiseOp::Log:
                result = b.create<mlir::math::LogOp>(regionLoc, args[0]);
                break;
            case ElementwiseOp::Add:
                result = b.create<mlir::arith::AddFOp>(regionLoc, args[0], args[1]);
                break;
            case ElementwiseOp::Sub:
                result = b.create<mlir::arith::SubFOp>(regionLoc, args[0], args[1]);
                break;
            case ElementwiseOp::Mul:
                result = b.create<mlir::arith::MulFOp>(regionLoc, args[0], args[1]);
                break;
            }
            b.create<mlir::linalg::YieldOp>(regionLoc, mlir::ValueRange{result});
        });

    builder.setInsertionPointToEnd(entry);
    builder.create<mlir::func::ReturnOp>(loc);
}

/// 构建 MLIR module：一个 c3_kernel func 内嵌 linalg.generic
mlir::OwningOpRef<mlir::ModuleOp> buildLinalgModule(mlir::MLIRContext& context,
                                                    ElementwiseOp op,
                                                    size_t num_inputs,
                                                    int rhs_mod) {
    auto loc = mlir::UnknownLoc::get(&context);
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    buildLinalgElementwiseFunc(builder, loc, context, op, num_inputs, rhs_mod);
    if (mlir::failed(mlir::verify(module))) {
        module->emitError();
        module->dump();
        throw std::runtime_error(std::string("LinalgElementwiseGen: module verification failed for ")
                                 + elementwiseOpName(op));
    }
    return module;
}

/// 把周期广播的 `affine.apply (d0/s0) -> (X mod k)` 降成 `arith.remsi %x, cst`。
/// 为什么不用共享库的 createLowerAffinePass：该 pass 在 run 时会把自己的依赖方言
/// （含 memref::MemRefDialect，来自共享库的 TypeID）append 进 context registry，
/// 与本 TU 实例化的 memref 方言 TypeID 不一致 → "Trying to register different
/// dialects for the same namespace: memref" fatal abort。linalg-to-loops 对
/// `d0 mod k` 生成的 affine.apply 形状固定为「1 个输入 + 结果 = 单一表达式 mod 常量」，
/// 自定义 pattern 足够且完全自包含（头文件类型一致，无 DSO TypeID 问题）。
struct AffineApplyToArithPattern : public mlir::OpRewritePattern<mlir::affine::AffineApplyOp> {
    using mlir::OpRewritePattern<mlir::affine::AffineApplyOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::affine::AffineApplyOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        mlir::AffineMap map = op.getAffineMap();
        if (map.getNumInputs() != 1 || map.getNumResults() != 1)
            return mlir::failure();
        mlir::AffineExpr expr = map.getResult(0);
        if (expr.getKind() != mlir::AffineExprKind::Mod)
            return mlir::failure();
        auto mod = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
        // LHS 必须是对应唯一输入的维度/符号（否则表达式更复杂，放弃重写）
        mlir::AffineExpr lhs = mod.getLHS();
        if (!mlir::isa<mlir::AffineDimExpr>(lhs) && !mlir::isa<mlir::AffineSymbolExpr>(lhs))
            return mlir::failure();
        auto cst = mlir::dyn_cast<mlir::AffineConstantExpr>(mod.getRHS());
        if (!cst)
            return mlir::failure();
        mlir::Value rhs = rewriter.create<mlir::arith::ConstantIndexOp>(
            op.getLoc(), cst.getValue());
        mlir::Value rem = rewriter.create<mlir::arith::RemSIOp>(
            op.getLoc(), op.getOperand(0), rhs);
        rewriter.replaceOp(op, rem);
        return mlir::success();
    }
};

/// 标准 linalg → loops → SCF → LLVM lowering pipeline（PoC 4.9 验证顺序）
void applyLinalgLoweringPipeline(mlir::ModuleOp module) {
    // 阶段 0.5：Linalg 级别特化与内联优化
    {
        mlir::PassManager pm(module.getContext());
        pm.addPass(mlir::createLinalgInlineScalarOperandsPass());
        pm.addPass(mlir::createLinalgSpecializeGenericOpsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        if (mlir::failed(pm.run(module))) {
            throw std::runtime_error("LinalgElementwiseGen: Linalg optimization failed");
        }
    }
    // 阶段 1：linalg.generic → loops（周期广播的 `d0 mod k` 在此产生 affine.apply）
    {
        mlir::PassManager pm(module.getContext());
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        if (mlir::failed(pm.run(module))) {
            throw std::runtime_error("LinalgElementwiseGen: linalg-to-loops failed");
        }
    }
    // 阶段 2：手动把 affine.apply (d0/s0) -> (X mod k) 降成 arith.remsi
    {
        mlir::RewritePatternSet patterns(module.getContext());
        patterns.add<AffineApplyToArithPattern>(module.getContext());
        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
            throw std::runtime_error("LinalgElementwiseGen: affine.apply lowering failed");
        }
    }
    // 阶段 3：scf → cf → LLVM
    {
        mlir::PassManager pm(module.getContext());
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
            throw std::runtime_error("LinalgElementwiseGen: lowering pipeline failed");
        }
    }
}

/// 创建 ExecutionEngine（LLVM O3 优化 transformer + Aggressive codegen）
/// @param cache_graph AOT 缓存 key 的语义串（hash 前），需能区分 (op, opt_level, rhs_mod) 与后端版本
/// @param builder_slot 由调用方长期持有的 std::function 槽位（本函数只负责填充）。
///        必须 long-lived：ExecutionEngine 对 llvmModuleBuilder 是延迟回调（create 返回后
///        首次 materialize 才调用），若用本函数栈上的局部 std::function 会在 create 返回后
///        销毁 → function_ref 悬垂 → 段错误（2026-08-15 已踩坑）。
std::unique_ptr<mlir::ExecutionEngine> createEngine(
    mlir::ModuleOp module, int opt_level, const std::string& cache_graph,
    std::function<std::unique_ptr<llvm::Module>(mlir::Operation*, llvm::LLVMContext&)>&
        builder_slot) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // [诊断] C3_LINALG_EW_TRACE=1 时打印 AOT 缓存 hit/miss 路径细节
    const bool aot_trace = [] {
        const char* v = std::getenv("C3_LINALG_EW_TRACE");
        return v != nullptr && v[0] == '1';
    }();

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
    engineOpts.jitCodeGenOptLevel = (opt_level >= 3)
        ? llvm::CodeGenOptLevel::Aggressive
        : (opt_level == 2) ? llvm::CodeGenOptLevel::Default
        : (opt_level == 1) ? llvm::CodeGenOptLevel::Less
        : llvm::CodeGenOptLevel::None;

    // [管线① 2026-08-15] linalg AOT 磁盘持久化缓存（JITCache 2.0 read path）。
    // 关键点（已确认的坑）：
    //   a) ExecutionEngine 对 llvmModuleBuilder 是【延迟回调】：create 返回之后、首次
    //      materialize（lookup/execute 触发）时才调用。因此 createEngine 栈上的局部
    //      std::function 会在 create 返回后被销毁 → function_ref 悬垂 → 段错误。
    //      → builder 必须由调用方持有的 long-lived 对象保存（此处经引用写入
    //        builder_slot，由 LinalgElementwiseKernel::Impl 持有，生命周期 ≥ kernel
    //        ≥ materialize 时机，且线程安全：每个 kernel 各有一份槽位）。
    //   b) llvmModuleBuilder 收到的是 create 内部新建的 LLVMContext，随后 module 被
    //      ThreadSafeModule(module, ctx) 绑定 —— 必须在 builder 里用传入的 ctx 构造
    //      module，否则 context 不一致会出问题。
    //   c) 值捕获 module（ModuleOp 内部只是 Operation* 包装），且该 Operation 的所有权
    //      由调用方持有到 materialize 之后（Impl::heldModule 持有）。
    //   - 命中：loadBitcode(create 的 ctx)  跳过 MLIR build + lowering + translate
    //   - 未命中：translate(create 的 ctx) + store bitcode（store 的是未优化 IR，与命中
    //     load 的形态一致，transformer 对两条路径一致应用）
    // 逃生开关：C3_JIT_CACHE_DISABLE=1（复用 JITCache::isEnabled()）。
    if (JITCache::isEnabled()) {
        try {
            std::string jit_key = JITCache::makeKey(cache_graph, opt_level);
            std::string bc_path = JITCache::getInstance().lookup(jit_key);

            if (!bc_path.empty()) {
                builder_slot = [bc_path, aot_trace](mlir::Operation*, llvm::LLVMContext& ctx) {
                    if (aot_trace)
                        fprintf(stderr, "[AOT-DEBUG] builder: HIT path begin\n");
                    auto m = JITCache::getInstance().loadBitcode(bc_path, ctx);
                    if (aot_trace)
                        fprintf(stderr, "[AOT-DEBUG] builder: HIT path load=%s\n",
                                m ? "OK" : "NULL");
                    return m;
                };
            } else {
                builder_slot = [module, jit_key, aot_trace](mlir::Operation*, llvm::LLVMContext& ctx) {
                    if (aot_trace)
                        fprintf(stderr, "[AOT-DEBUG] builder: MISS path begin, key='%s'\n",
                                jit_key.c_str());
                    auto llvm_module = mlir::translateModuleToLLVMIR(module, ctx);
                    if (aot_trace)
                        fprintf(stderr, "[AOT-DEBUG] builder: translate -> %s\n",
                                llvm_module ? "OK" : "NULLPTR");
                    if (llvm_module) {
                        auto st = JITCache::getInstance().store(jit_key, *llvm_module);
                        if (aot_trace)
                            fprintf(stderr, "[AOT-DEBUG] builder: store -> '%s'\n", st.c_str());
                    }
                    return llvm_module;
                };
            }
            engineOpts.llvmModuleBuilder = builder_slot;
        } catch (...) {
            // 磁盘/缓存异常静默回退到默认 translate 路径
        }
    }

    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOpts);
    if (!maybeEngine) {
        llvm::errs() << "[linalg-debug] createEngine failed for cache_graph="
                     << cache_graph << ": " << llvm::toString(maybeEngine.takeError())
                     << "\n";
        throw std::runtime_error("LinalgElementwiseGen: failed to create ExecutionEngine");
    }
    return std::move(*maybeEngine);
}

} // namespace

// ======================= LinalgElementwiseKernel =======================

struct LinalgElementwiseKernel::Impl {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context;

    // 生命周期管理（关键）：ExecutionEngine 对 llvmModuleBuilder 是【延迟回调】（首次
    // materialize 时才调用），且 builder 值捕获的 MLIR ModuleOp 必须存活到那次调用。
    // 因此 heldModule / aotBuilder 都必须在 engine 之前声明 → 析构顺序逆序：engine
    // 先析构（JIT 已完成，不再需要 MLIR IR），再析构 heldModule / aotBuilder。
    mlir::OwningOpRef<mlir::ModuleOp> heldModule;  ///< 持有 MLIR module（builder 延迟 translate 用）
    std::function<std::unique_ptr<llvm::Module>(mlir::Operation*, llvm::LLVMContext&)>
        aotBuilder;  ///< AOT builder 槽位（由 createEngine 填充，随 kernel 存活）

    std::unique_ptr<mlir::ExecutionEngine> engine;

    Impl() : context(registry) {
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::math::MathDialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::memref::MemRefDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
        registry.insert<mlir::linalg::LinalgDialect>();
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();
    }
};

LinalgElementwiseKernel::LinalgElementwiseKernel(ElementwiseOp op, int opt_level,
                                                 int rhs_mod)
    : impl_(std::make_unique<Impl>()), op_(op), num_inputs_(elementwiseOpNumInputs(op)),
      rhs_mod_(rhs_mod) {

    // module 所有权交给 Impl::heldModule 持有：ExecutionEngine 对 builder 是延迟回调
    // （首次 materialize 才 translate），该 Operation 必须存活到那次调用之后（引擎析构）。
    impl_->heldModule = buildLinalgModule(impl_->context, op_, num_inputs_, rhs_mod_);
    mlir::ModuleOp module = *impl_->heldModule;
    applyLinalgLoweringPipeline(module);
    // [诊断] C3_LINALG_EW_TRACE=1 时打印 lowering 后的 LLVM IR（仅周期广播等有参考价值）
    const char* ew_trace = std::getenv("C3_LINALG_EW_TRACE");
    if (rhs_mod_ > 1 && ew_trace != nullptr && ew_trace[0] == '1') {
        llvm::errs() << "[linalg-debug] after lowering (rhs_mod=" << rhs_mod_
                     << "):\n";
        module.dump();
    }

    // 缓存 key 语义串：区分 (op, opt_level, rhs_mod)，保持与 JITCache 1.0 graph key 格式一致
    std::string cache_graph = std::string("linalg_ew_") + elementwiseOpName(op_)
                              + "_ol" + std::to_string(opt_level)
                              + "_rm" + std::to_string(rhs_mod_);
    impl_->engine = createEngine(module, opt_level, cache_graph, impl_->aotBuilder);
    if (!impl_->engine->lookup("c3_kernel")) {
        throw std::runtime_error("LinalgElementwiseGen: lookup c3_kernel failed");
    }
}

LinalgElementwiseKernel::~LinalgElementwiseKernel() = default;
LinalgElementwiseKernel::LinalgElementwiseKernel(LinalgElementwiseKernel&&) noexcept = default;
LinalgElementwiseKernel& LinalgElementwiseKernel::operator=(LinalgElementwiseKernel&&) noexcept = default;

void LinalgElementwiseKernel::execute(const float* const* in_ptrs, float* out_ptr,
                                      size_t n) const {
    // rhs_mod 控制 rhs memref 元素个数：0=同尺寸 n，1=标量 1，k>1=周期 k
    const size_t rhs_elem = (rhs_mod_ == 0) ? n : static_cast<size_t>(rhs_mod_);
    const size_t num_memrefs = num_inputs_ + 1;

    // 注意：descriptor 对象必须活到 invokePacked 返回之后（args 里存的是成员地址），
    // 所以全部放在函数级数组里，不能放进循环体内。
    MemRefDesc descs[3];
    void* args[15];
    int arg_idx = 0;
    for (size_t i = 0; i < num_inputs_; ++i) {
        // 二元第二输入走广播元素数（标量 1 / 周期 k），其余同尺寸 n
        size_t elem_count = (i == 1 && rhs_mod_ > 0) ? rhs_elem : n;
        descs[i] = MemRefDesc{const_cast<float*>(in_ptrs[i]), const_cast<float*>(in_ptrs[i]),
                              0, {static_cast<int64_t>(elem_count)}, {1}};
        appendMemRefDescArgs(descs[i], args, arg_idx);
    }
    descs[num_inputs_] = MemRefDesc{out_ptr, out_ptr, 0, {static_cast<int64_t>(n)}, {1}};
    appendMemRefDescArgs(descs[num_inputs_], args, arg_idx);

    (void)num_memrefs;
    auto err = impl_->engine->invokePacked("c3_kernel", args);
    if (err) {
        throw std::runtime_error("LinalgElementwiseGen: invokePacked failed: "
                                 + llvm::toString(std::move(err)));
    }
}

// ======================= 共享 kernel 缓存工厂 =======================

std::shared_ptr<LinalgElementwiseKernel> getCachedLinalgKernel(
    ElementwiseOp op, int opt_level, int rhs_mod) {
    static const bool cache_disabled = [] {
        const char* v = std::getenv("C3_LINALG_CACHE");
        return v != nullptr && std::string(v) == "0";
    }();
    if (cache_disabled) {
        return std::make_shared<LinalgElementwiseKernel>(op, opt_level, rhs_mod);
    }

    // 缓存 key: "opName_optLevel_rhsMod" → shared_ptr
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<LinalgElementwiseKernel>> cache;

    std::string key = std::string(elementwiseOpName(op))
                      + "_" + std::to_string(opt_level)
                      + "_" + std::to_string(rhs_mod);

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (auto sp = it->second.lock()) {
            return sp; // 缓存命中，复用
        }
        // weak_ptr 已过期，清理后重建
        cache.erase(it);
    }

    auto kernel = std::make_shared<LinalgElementwiseKernel>(op, opt_level, rhs_mod);
    cache[key] = kernel;
    return kernel;
}

std::vector<float> runLinalgElementwise(ElementwiseOp op,
                                        const std::vector<std::vector<float>>& inputs,
                                        size_t n) {
    LinalgElementwiseKernel kernel(op);
    std::vector<const float*> in_ptrs;
    in_ptrs.reserve(inputs.size());
    for (const auto& in : inputs) {
        in_ptrs.push_back(in.data());
    }
    std::vector<float> out(n);
    kernel.execute(in_ptrs.data(), out.data(), n);
    return out;
}

} // namespace c3
} // namespace ct
