/**
 * @file LinalgElementwiseGen.cpp
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

#include <cmath>
#include <cstdlib>
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
#include <mlir/IR/Verifier.h>
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
#include <mlir/Conversion/Passes.h>  // 提供 createConvertFuncToLLVMPass 等工厂函数

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
/// rhs_broadcast=true 时 b 的 indexing map 取 `d0 -> 0`（标量投影，b size=1），
/// 循环域由输出 identity map 决定（仍迭代 n 次），实现声明式标量广播。
void buildLinalgElementwiseFunc(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::MLIRContext& context, ElementwiseOp op,
                                size_t num_inputs, bool rhs_broadcast) {
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
    std::vector<mlir::AffineMap> indexingMaps;
    indexingMaps.reserve(num_inputs + 1);
    for (size_t i = 0; i < num_inputs; ++i) {
        // 二元标量广播：第二个输入用投影 map
        indexingMaps.push_back(rhs_broadcast && i == 1 ? zeroMap : identityMap);
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
                                                    bool rhs_broadcast) {
    auto loc = mlir::UnknownLoc::get(&context);
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    buildLinalgElementwiseFunc(builder, loc, context, op, num_inputs, rhs_broadcast);
    if (mlir::failed(mlir::verify(module))) {
        module->emitError();
        module->dump();
        throw std::runtime_error(std::string("LinalgElementwiseGen: module verification failed for ")
                                 + elementwiseOpName(op));
    }
    return module;
}

/// 标准 linalg → loops → SCF → LLVM lowering pipeline（PoC 4.9 验证顺序）
void applyLinalgLoweringPipeline(mlir::ModuleOp module) {
    mlir::PassManager pm(module.getContext());
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
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

/// 创建 ExecutionEngine（LLVM O3 优化 transformer + Aggressive codegen）
std::unique_ptr<mlir::ExecutionEngine> createEngine(mlir::ModuleOp module,
                                                    int opt_level) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

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

    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOpts);
    if (!maybeEngine) {
        throw std::runtime_error("LinalgElementwiseGen: failed to create ExecutionEngine");
    }
    return std::move(*maybeEngine);
}

} // namespace

// ======================= LinalgElementwiseKernel =======================

struct LinalgElementwiseKernel::Impl {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context;
    std::unique_ptr<mlir::ExecutionEngine> engine;

    Impl() : context(registry) {
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::math::MathDialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::memref::MemRefDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
        registry.insert<mlir::linalg::LinalgDialect>();
    }
};

LinalgElementwiseKernel::LinalgElementwiseKernel(ElementwiseOp op, int opt_level,
                                                 bool rhs_broadcast)
    : impl_(std::make_unique<Impl>()), op_(op), num_inputs_(elementwiseOpNumInputs(op)),
      rhs_broadcast_(rhs_broadcast) {
    impl_->context.loadDialect<mlir::arith::ArithDialect>();
    impl_->context.loadDialect<mlir::math::MathDialect>();
    impl_->context.loadDialect<mlir::scf::SCFDialect>();
    impl_->context.loadDialect<mlir::func::FuncDialect>();
    impl_->context.loadDialect<mlir::memref::MemRefDialect>();
    impl_->context.loadDialect<mlir::LLVM::LLVMDialect>();
    impl_->context.loadDialect<mlir::linalg::LinalgDialect>();

    auto module = buildLinalgModule(impl_->context, op_, num_inputs_, rhs_broadcast_);
    applyLinalgLoweringPipeline(*module);

    mlir::registerBuiltinDialectTranslation(impl_->context);
    mlir::registerLLVMDialectTranslation(impl_->context);

    impl_->engine = createEngine(*module, opt_level);
    if (!impl_->engine->lookup("c3_kernel")) {
        throw std::runtime_error("LinalgElementwiseGen: lookup c3_kernel failed");
    }
}

LinalgElementwiseKernel::~LinalgElementwiseKernel() = default;
LinalgElementwiseKernel::LinalgElementwiseKernel(LinalgElementwiseKernel&&) noexcept = default;
LinalgElementwiseKernel& LinalgElementwiseKernel::operator=(LinalgElementwiseKernel&&) noexcept = default;

void LinalgElementwiseKernel::execute(const float* const* in_ptrs, float* out_ptr,
                                      size_t n) const {
    // 标量广播时 rhs（in_ptrs[1]）的 memref size=1，out 仍为 n
    const size_t num_memrefs = num_inputs_ + 1;

    // 注意：descriptor 对象必须活到 invokePacked 返回之后（args 里存的是成员地址），
    // 所以全部放在函数级数组里，不能放进循环体内。
    MemRefDesc descs[3];
    void* args[15];
    int arg_idx = 0;
    for (size_t i = 0; i < num_inputs_; ++i) {
        size_t elem_count = (i == 1 && rhs_broadcast_) ? 1 : n;
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
    ElementwiseOp op, int opt_level, bool rhs_broadcast) {
    static const bool cache_disabled = [] {
        const char* v = std::getenv("C3_LINALG_CACHE");
        return v != nullptr && std::string(v) == "0";
    }();
    if (cache_disabled) {
        return std::make_shared<LinalgElementwiseKernel>(op, opt_level, rhs_broadcast);
    }

    // 缓存 key: "opName_optLevel_flag" → shared_ptr
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<LinalgElementwiseKernel>> cache;

    std::string key = std::string(elementwiseOpName(op))
                      + "_" + std::to_string(opt_level)
                      + "_" + (rhs_broadcast ? "1" : "0");

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (auto sp = it->second.lock()) {
            return sp; // 缓存命中，复用
        }
        // weak_ptr 已过期，清理后重建
        cache.erase(it);
    }

    auto kernel = std::make_shared<LinalgElementwiseKernel>(op, opt_level, rhs_broadcast);
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
