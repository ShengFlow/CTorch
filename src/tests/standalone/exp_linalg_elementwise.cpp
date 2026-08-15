/**
 * @file exp_linalg_elementwise.cpp
 * @brief PoC: linalg.generic 声明式逐元素算子 → 标准 lowering → JIT 执行
 *
 * 验证「flat pointer → memref → linalg.generic → one-shot bufferize/lower → LLVM JIT」
 * 完整链路的技术可行性，为后续统一替换 10+ 个 if-else 手写分支提供参考。
 *
 * 测试内容：
 *   1. 构建 linalg.generic ReLU kernel（最简单的非平凡逐元素算子）
 *   2. 运行 linalg → loops → 标准 lowering pipeline
 *   3. ExecutionEngine JIT 编译执行
 *   4. 验证输出与手写参考一致
 *
 * @date 2026/08/14
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
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
#include <mlir/Conversion/Passes.h>

#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>

using namespace mlir;

// ======================= 参考实现 =======================

/// 手写 ReLU 参考
static void ref_relu(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = (in[i] > 0) ? in[i] : 0.0f;
}

/// 手写 Add 参考
static void ref_add(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

/// 手写 Sigmoid 参考
static void ref_sigmoid(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-in[i]));
}

// ======================= MLIR 模块构建 =======================

/// 验证结果
static bool verifyApproxEqual(const float* expected, const float* actual, size_t n,
                              float eps = 1e-5f) {
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(expected[i] - actual[i]) > eps) {
            std::cerr << "  Mismatch at [" << i << "]: expected=" << expected[i]
                      << ", actual=" << actual[i] << "\n";
            return false;
        }
    }
    return true;
}

/// 打印一维 memref 的辅助函数
static void printMemref1D(const std::string& label, const float* data, size_t n) {
    std::cout << "  " << label << " [";
    for (size_t i = 0; i < std::min(n, size_t(8)); ++i) std::cout << data[i] << " ";
    if (n > 8) std::cout << "... ";
    std::cout << "]\n";
}

// ======================= JIT 调用 ABI 辅助 =======================

/// memref<?xf32> 的 descriptor 布局（与 LLVM lowering 后的展开参数一一对应）：
///   {alloc_ptr, aligned_ptr, offset, size[0], stride[0]}
/// FinalizeMemRefToLLVMConversionPass 会把每个 memref<?xf32> 函数参数展开成
/// 5 个标量参数（ptr, ptr, i64, i64, i64）。因此 ExecutionEngine 的包装函数
/// `_mlir_c3_kernel(void**)` 期望 packed 数组里每 5 个指针对应一个 memref。
struct MemRefDesc {
    float* alloc;
    float* aligned;
    int64_t offset;
    int64_t sizes[1];
    int64_t strides[1];
};

/// 把一个 MemRefDesc 展开成 5 个标量指针（alloc, aligned, offset, size, stride），
/// 追加到 args 数组。调用方需保证 args 有足够空间。
static void appendMemRefDescArgs(const MemRefDesc& desc, void** args, int& idx) {
    args[idx++] = const_cast<float**>(&desc.alloc);
    args[idx++] = const_cast<float**>(&desc.aligned);
    args[idx++] = const_cast<int64_t*>(&desc.offset);
    args[idx++] = const_cast<int64_t*>(&desc.sizes[0]);
    args[idx++] = const_cast<int64_t*>(&desc.strides[0]);
}

// ======================= 核心测试 =======================

/// 构建 ReLU 的 linalg.generic kernel 并执行
/// Kernel 签名: void kernel(memref<?xf32> input, memref<?xf32> output)
static bool testLinalgGenericReLU(size_t n) {
    std::cout << "\n--- linalg.generic ReLU (n=" << n << ") ---\n";

    // 1. 准备输入数据
    std::vector<float> input(n);
    std::vector<float> output(n, -1.0f);  // 初始化为 -1
    for (size_t i = 0; i < n; ++i) input[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;

    // 2. 计算参考结果
    std::vector<float> expected(n);
    ref_relu(input.data(), expected.data(), n);

    // 3. 构建 MLIR 模块
    MLIRContext context;
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<linalg::LinalgDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
    context.getOrLoadDialect<math::MathDialect>();

    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    auto module = builder.create<ModuleOp>(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // 函数类型: (memref<?xf32>, memref<?xf32>) -> ()
    // 注意：动态维度必须用 ShapedType::kDynamic（INT64_MIN），不能用字面量 -1。
    // 若用 -1，MLIR 会将其当作「静态形状 -1」，触发 IndexingMapOpInterface 的
    // 静态边界检查，报 'unexpected result less than 0'。
    auto f32Type = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32Type);
    auto funcType = builder.getFunctionType({memrefType, memrefType}, {});
    auto func = builder.create<func::FuncOp>(loc, "c3_kernel", funcType);
    func.setArgAttr(0, "llvm.noalias", builder.getUnitAttr());
    func.setArgAttr(1, "llvm.noalias", builder.getUnitAttr());

    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value input_memref = entry->getArgument(0);
    Value output_memref = entry->getArgument(1);

    // 获取元素数量 = dim(input, 0)
    Value n_val = builder.create<memref::DimOp>(loc, input_memref, 0);

    // 4. 构建 linalg.generic 实现 ReLU
    // 使用 Identity 索引映射（逐元素，1D）
    AffineExpr d0 = builder.getAffineDimExpr(0);
    auto identityMap = AffineMap::get(1, 0, {d0}, &context);

    // linalg.generic 的 indexing_maps: [input_map, output_map] 都是 identity
    // iterator_types: ["parallel"]
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
    SmallVector<utils::IteratorType> iteratorTypes = {utils::IteratorType::parallel};

    // 使用 ImplicitLocOpBuilder 简化 region 构建
    ImplicitLocOpBuilder iBuilder(loc, builder);

    // 构建 linalg.generic op
    // 注意: 使用 destination-style 重载, 输入和输出都是 memref
    auto genericOp = iBuilder.create<linalg::GenericOp>(
        TypeRange{},                       // 无 result type（dest-style）
        ValueRange{input_memref},          // inputs
        ValueRange{output_memref},         // outputs
        indexingMaps,                      // affine maps
        iteratorTypes,                     // iterator types
        [&](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] = input element, args[1] = output element (输出不需要读取)
            Value in_val = args[0];
            // ReLU: max(in, 0)
            Value zero = b.create<arith::ConstantFloatOp>(loc, f32Type, llvm::APFloat(0.0f));
            Value cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, in_val, zero);
            Value result = b.create<arith::SelectOp>(loc, cmp, in_val, zero);
            b.create<linalg::YieldOp>(loc, ValueRange{result});
        });

    // 5. 添加 return
    builder.setInsertionPointToEnd(entry);
    builder.create<func::ReturnOp>(loc);

    // 6. 验证模块
    if (failed(verify(module))) {
        std::cerr << "  FAIL: module verification failed\n";
        module->dump();
        return false;
    }

    // 7. 构建 lowering pipeline
    // 使用标准的 linalg → loops → SCF → LLVM 降低路径
    PassManager pm(&context);
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (failed(pm.run(module))) {
        std::cerr << "  FAIL: lowering pipeline failed\n";
        return false;
    }

    // 可选: 打印 lowering 后的 module
    // module->dump();

    // 8. JIT 编译执行
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);

    // 创建 TargetMachine（启用 LLVM 优化）
    auto tm = std::shared_ptr<llvm::TargetMachine>(
        llvm::EngineBuilder()
            .setEngineKind(llvm::EngineKind::JIT)
            .setOptLevel(llvm::CodeGenOptLevel::Aggressive)
            .selectTarget());

    std::function<llvm::Error(llvm::Module*)> opt_transformer =
        tm ? mlir::makeOptimizingTransformer(3, 0, tm.get())
           : std::function<llvm::Error(llvm::Module*)>();

    ExecutionEngineOptions engineOpts;
    engineOpts.transformer = opt_transformer;
    engineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

    auto maybeEngine = ExecutionEngine::create(module, engineOpts);
    if (!maybeEngine) {
        std::cerr << "  FAIL: ExecutionEngine::create failed\n";
        return false;
    }

    // 9. 准备 packed 参数并调用
    // 对于 memref<?xf32>，需要构建 MemRefDescriptor 结构
    // 使用 `invokePacked` 需要传递 MemRefDescriptor 的指针
    // 结构体: {allocated_ptr, aligned_ptr, offset, size[0], stride[0]}
    // 但更简单的方式: 通过 `lookup` 获取函数指针直接调用 LLVM 函数
    auto expectedPtr = maybeEngine->get()->lookup("c3_kernel");
    if (!expectedPtr) {
        std::cerr << "  FAIL: lookup c3_kernel failed\n";
        return false;
    }

    // 使用全局 MemRefDesc（见文件顶部 ABI 辅助节）

    MemRefDesc input_desc;
    input_desc.alloc = input.data();
    input_desc.aligned = input.data();
    input_desc.offset = 0;
    input_desc.sizes[0] = static_cast<int64_t>(n);
    input_desc.strides[0] = 1;

    MemRefDesc output_desc;
    output_desc.alloc = output.data();
    output_desc.aligned = output.data();
    output_desc.offset = 0;
    output_desc.sizes[0] = static_cast<int64_t>(n);
    output_desc.strides[0] = 1;

    // 每个 memref 展开成 5 个标量参数（alloc, aligned, offset, size, stride）
    void* args[10];
    int arg_idx = 0;
    appendMemRefDescArgs(input_desc, args, arg_idx);
    appendMemRefDescArgs(output_desc, args, arg_idx);
    auto err = maybeEngine->get()->invokePacked("c3_kernel", args);
    if (err) {
        std::cerr << "  FAIL: invokePacked failed: " << toString(std::move(err)) << "\n";
        return false;
    }

    // 10. 验证结果
    printMemref1D("input:", input.data(), n);
    printMemref1D("expected:", expected.data(), n);
    printMemref1D("output:", output.data(), n);

    if (!verifyApproxEqual(expected.data(), output.data(), n)) {
        std::cerr << "  FAIL: results mismatch\n";
        return false;
    }

    std::cout << "  PASSED\n";
    return true;
}

/// 构建 Add 的 linalg.generic kernel
/// Kernel 签名: void kernel(memref<?xf32> a, memref<?xf32> b, memref<?xf32> out)
static bool testLinalgGenericAdd(size_t n) {
    std::cout << "\n--- linalg.generic Add (n=" << n << ") ---\n";

    std::vector<float> a(n), b(n), out(n, -1.0f);
    for (size_t i = 0; i < n; ++i) {
        a[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        b[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }

    std::vector<float> expected(n);
    ref_add(a.data(), b.data(), expected.data(), n);

    MLIRContext context;
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<linalg::LinalgDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
    context.getOrLoadDialect<math::MathDialect>();

    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    auto module = builder.create<ModuleOp>(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto f32Type = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32Type);
    auto funcType = builder.getFunctionType({memrefType, memrefType, memrefType}, {});
    auto func = builder.create<func::FuncOp>(loc, "c3_kernel", funcType);
    func.setArgAttr(0, "llvm.noalias", builder.getUnitAttr());
    func.setArgAttr(1, "llvm.noalias", builder.getUnitAttr());
    func.setArgAttr(2, "llvm.noalias", builder.getUnitAttr());

    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value a_memref = entry->getArgument(0);
    Value b_memref = entry->getArgument(1);
    Value out_memref = entry->getArgument(2);

    // 构建 linalg.generic 实现 Add
    AffineExpr d0 = builder.getAffineDimExpr(0);
    auto identityMap = AffineMap::get(1, 0, {d0}, &context);

    ImplicitLocOpBuilder iBuilder(loc, builder);
    auto genericOp = iBuilder.create<linalg::GenericOp>(
        TypeRange{},
        ValueRange{a_memref, b_memref},
        ValueRange{out_memref},
        SmallVector<AffineMap>{identityMap, identityMap, identityMap},
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value result = b.create<arith::AddFOp>(loc, args[0], args[1]);
            b.create<linalg::YieldOp>(loc, ValueRange{result});
        });

    builder.setInsertionPointToEnd(entry);
    builder.create<func::ReturnOp>(loc);

    if (failed(verify(module))) {
        std::cerr << "  FAIL: module verification failed\n";
        module->dump();
        return false;
    }

    // Lowering pipeline
    PassManager pm(&context);
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (failed(pm.run(module))) {
        std::cerr << "  FAIL: lowering pipeline failed\n";
        return false;
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);

    auto tm = std::shared_ptr<llvm::TargetMachine>(
        llvm::EngineBuilder()
            .setEngineKind(llvm::EngineKind::JIT)
            .setOptLevel(llvm::CodeGenOptLevel::Aggressive)
            .selectTarget());

    std::function<llvm::Error(llvm::Module*)> opt_transformer =
        tm ? mlir::makeOptimizingTransformer(3, 0, tm.get())
           : std::function<llvm::Error(llvm::Module*)>();

    ExecutionEngineOptions engineOpts;
    engineOpts.transformer = opt_transformer;
    engineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

    auto maybeEngine = ExecutionEngine::create(module, engineOpts);
    if (!maybeEngine) {
        std::cerr << "  FAIL: ExecutionEngine::create failed\n";
        return false;
    }

    MemRefDesc a_desc{a.data(), a.data(), 0, {static_cast<int64_t>(n)}, {1}};
    MemRefDesc b_desc{b.data(), b.data(), 0, {static_cast<int64_t>(n)}, {1}};
    MemRefDesc out_desc{out.data(), out.data(), 0, {static_cast<int64_t>(n)}, {1}};

    // 每个 memref 展开成 5 个标量参数（alloc, aligned, offset, size, stride）
    void* args[15];
    int arg_idx = 0;
    appendMemRefDescArgs(a_desc, args, arg_idx);
    appendMemRefDescArgs(b_desc, args, arg_idx);
    appendMemRefDescArgs(out_desc, args, arg_idx);
    auto err = maybeEngine->get()->invokePacked("c3_kernel", args);
    if (err) {
        std::cerr << "  FAIL: invokePacked failed: " << toString(std::move(err)) << "\n";
        return false;
    }

    printMemref1D("a:", a.data(), n);
    printMemref1D("b:", b.data(), n);
    printMemref1D("expected:", expected.data(), n);
    printMemref1D("output:", out.data(), n);

    if (!verifyApproxEqual(expected.data(), out.data(), n)) {
        std::cerr << "  FAIL: results mismatch\n";
        return false;
    }

    std::cout << "  PASSED\n";
    return true;
}

/// 构建 Sigmoid 的 linalg.generic kernel
/// Kernel 签名: void kernel(memref<?xf32> input, memref<?xf32> output)
static bool testLinalgGenericSigmoid(size_t n) {
    std::cout << "\n--- linalg.generic Sigmoid (n=" << n << ") ---\n";

    std::vector<float> input(n), output(n, -1.0f);
    for (size_t i = 0; i < n; ++i) input[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;

    std::vector<float> expected(n);
    ref_sigmoid(input.data(), expected.data(), n);

    MLIRContext context;
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<linalg::LinalgDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
    context.getOrLoadDialect<math::MathDialect>();

    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    auto module = builder.create<ModuleOp>(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto f32Type = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32Type);
    auto funcType = builder.getFunctionType({memrefType, memrefType}, {});
    auto func = builder.create<func::FuncOp>(loc, "c3_kernel", funcType);
    func.setArgAttr(0, "llvm.noalias", builder.getUnitAttr());
    func.setArgAttr(1, "llvm.noalias", builder.getUnitAttr());

    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value input_memref = entry->getArgument(0);
    Value output_memref = entry->getArgument(1);

    // Sigmoid: 1 / (1 + exp(-x))
    // 需要 math::ExpOp 求指数
    AffineExpr d0 = builder.getAffineDimExpr(0);
    auto identityMap = AffineMap::get(1, 0, {d0}, &context);

    ImplicitLocOpBuilder iBuilder(loc, builder);
    auto genericOp = iBuilder.create<linalg::GenericOp>(
        TypeRange{},
        ValueRange{input_memref},
        ValueRange{output_memref},
        SmallVector<AffineMap>{identityMap, identityMap},
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value x = args[0];
            Value neg_x = b.create<arith::NegFOp>(loc, x);
            Value exp_neg_x = b.create<math::ExpOp>(loc, neg_x);
            Value one = b.create<arith::ConstantFloatOp>(loc, f32Type, llvm::APFloat(1.0f));
            Value denom = b.create<arith::AddFOp>(loc, one, exp_neg_x);
            Value result = b.create<arith::DivFOp>(loc, one, denom);
            b.create<linalg::YieldOp>(loc, ValueRange{result});
        });

    builder.setInsertionPointToEnd(entry);
    builder.create<func::ReturnOp>(loc);

    if (failed(verify(module))) {
        std::cerr << "  FAIL: module verification failed\n";
        module->dump();
        return false;
    }

    PassManager pm(&context);
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (failed(pm.run(module))) {
        std::cerr << "  FAIL: lowering pipeline failed\n";
        return false;
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);

    auto tm = std::shared_ptr<llvm::TargetMachine>(
        llvm::EngineBuilder()
            .setEngineKind(llvm::EngineKind::JIT)
            .setOptLevel(llvm::CodeGenOptLevel::Aggressive)
            .selectTarget());

    std::function<llvm::Error(llvm::Module*)> opt_transformer =
        tm ? mlir::makeOptimizingTransformer(3, 0, tm.get())
           : std::function<llvm::Error(llvm::Module*)>();

    ExecutionEngineOptions engineOpts;
    engineOpts.transformer = opt_transformer;
    engineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

    auto maybeEngine = ExecutionEngine::create(module, engineOpts);
    if (!maybeEngine) {
        std::cerr << "  FAIL: ExecutionEngine::create failed\n";
        return false;
    }

    MemRefDesc input_desc{input.data(), input.data(), 0, {static_cast<int64_t>(n)}, {1}};
    MemRefDesc output_desc{output.data(), output.data(), 0, {static_cast<int64_t>(n)}, {1}};

    // 每个 memref 展开成 5 个标量参数（alloc, aligned, offset, size, stride）
    void* args[10];
    int arg_idx = 0;
    appendMemRefDescArgs(input_desc, args, arg_idx);
    appendMemRefDescArgs(output_desc, args, arg_idx);
    auto err = maybeEngine->get()->invokePacked("c3_kernel", args);
    if (err) {
        std::cerr << "  FAIL: invokePacked failed: " << toString(std::move(err)) << "\n";
        return false;
    }

    printMemref1D("input:", input.data(), n);
    printMemref1D("expected:", expected.data(), n);
    printMemref1D("output:", output.data(), n);

    if (!verifyApproxEqual(expected.data(), output.data(), n)) {
        std::cerr << "  FAIL: results mismatch\n";
        return false;
    }

    std::cout << "  PASSED\n";
    return true;
}

// ======================= 主入口 =======================

int main() {
    std::cout << "===========================================\n";
    std::cout << "  linalg.generic 逐元素算子 PoC\n";
    std::cout << "===========================================\n";

    size_t sizes[] = {16, 128, 1024, 1048576};  // 小到大数据规模
    int num_sizes = 4;

    bool all_passed = true;

    for (int i = 0; i < num_sizes; ++i) {
        all_passed &= testLinalgGenericReLU(sizes[i]);
    }

    for (int i = 0; i < num_sizes; ++i) {
        all_passed &= testLinalgGenericAdd(sizes[i]);
    }

    for (int i = 0; i < num_sizes; ++i) {
        all_passed &= testLinalgGenericSigmoid(sizes[i]);
    }

    std::cout << "\n===========================================\n";
    if (all_passed) {
        std::cout << "  ALL TESTS PASSED (12/12)\n";
    } else {
        std::cout << "  SOME TESTS FAILED\n";
    }
    std::cout << "===========================================\n";

    return all_passed ? 0 : 1;
}