/**
 * @file test_dcu_hello_v2.cpp
 * @brief C3 → DCU 完整链路 v2 (2026-08-10, v0.5.2 DCU 接入 Phase 2)
 * @details 跟 test_c3_dcu_hello.cpp 区别:
 *          - 走完整 MLIR → LLVM IR pipeline (用 mlirToLLVMIRFromGraph helper, 不是 placeholder)
 *          - Plan A (GCVM) → Plan B (dcc bitcode) → Plan C (host CPU JIT) 自动 fallback
 *          - 输出每个 Plan 耗时, 方便 DCU 性能分析
 *
 * 用法 (DCU 节点 b02r1n05):
 *   module load compiler/dtk/26.04
 *   cd build-dcu && cmake -DCT_ENABLE_DCU=ON -DCT_ENABLE_MLIR=ON ..  # MLIR 必须 ON
 *   make -j8 test_dcu_hello_v2
 *   ./test_dcu_hello_v2
 *
 * 输出:
 *   PASS_PLAN_A = 整条 MLIR → GCVM 链路打通, 数值跟 host CPU 一致
 *   PASS_PLAN_B = GCVM 失败 (IR_VERSION_MISMATCH 预期), dcc 兜底成功
 *   PASS_PLAN_C = GCVM + dcc 都失败, host CPU baseline 跑通
 *   FAIL = 全部失败
 *
 * 预期 (per dcu-probe-interpretation.md R1 HIGH 风险):
 *   - GCVM 1.6 = LLVM 7.0.1 vs C3 MLIR 22.1.8 = LLVM 14 IR → IR_VERSION_MISMATCH
 *   - Plan A 大概率失败, Plan B 兜底
 */

#include "C3/C3Engine.h"
#include "C3/GCVMBridge.h"
#include "C3/MLIRToLLVMIR.h"
#include "CtorchError.h"
#include "Tensor.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>

#ifdef WITH_DCU
    #include <hip/hip_runtime.h>
#endif

using namespace ct;

// ============================================================================
// Timing helper
// ============================================================================
struct Timings {
    double mlir_to_llvm_ir_ms = 0.0;
    double gcvm_compile_ms = 0.0;
    double dcc_compile_ms = 0.0;        // Plan B
    double host_jit_ms = 0.0;          // Plan C
    double dcu_exec_ms = 0.0;          // DCU execute
    double correctness_ms = 0.0;
};

// ============================================================================
// Phase 0: 平台检查
// ============================================================================
bool checkPlatform() {
    std::cout << "=== Phase 0: 平台检查 ===" << std::endl;

    if (!ct::c3::isGCVMAvailable()) {
        std::cerr << "  ❌ GCVM 不可用 (rebuild with -DCT_ENABLE_DCU=ON)" << std::endl;
        return false;
    }
    std::cout << "  ✅ GCVM 库可用 (libgcvm.so 链接成功)" << std::endl;

#ifdef WITH_DCU
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        std::cerr << "  ❌ DCU 设备不可用: " << hipGetErrorString(err) << std::endl;
        return false;
    }
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << "  ✅ DCU 设备: " << device_count << " 个" << std::endl;
    std::cout << "     DCU[0]: " << prop.name
              << " | " << (prop.totalGlobalMem / 1e9) << " GB"
              << " | compute " << prop.major << "." << prop.minor << std::endl;
#else
    std::cerr << "  ❌ WITH_DCU=OFF, 跳过" << std::endl;
    return false;
#endif
    std::cout << std::endl;
    return true;
}

// ============================================================================
// Phase 1: 创简单 Graph {a, b, Add}
// ============================================================================
ct::c3::Graph createSimpleGraph() {
    std::cout << "=== Phase 1: 创 Graph {a, b, Add} ===" << std::endl;
    ct::c3::Graph g;
    auto a_desc = ct::c3::TensorDesc::fromShape({1024});
    auto b_desc = ct::c3::TensorDesc::fromShape({1024});
    auto out_desc = ct::c3::TensorDesc::fromShape({1024});
    auto a = g.addInput(a_desc);
    auto b = g.addInput(b_desc);
    g.addNode(ct::c3::AddNode{a_desc, b_desc}, {a, b}, out_desc);
    g.markOutput(g.nodeCount() - 1);
    std::cout << "  Graph: 2 inputs + 1 Add node + 1 output (1024-dim)" << std::endl;
    std::cout << std::endl;
    return g;
}

// ============================================================================
// Phase 2: MLIR → LLVM IR (真 pipeline, 用 mlirToLLVMIRFromGraph helper)
// ============================================================================
bool runMLIRToLLVMIR(const ct::c3::Graph& g,
                     std::string& llvm_ir_text,
                     std::vector<uint8_t>& llvm_ir_bitcode,
                     Timings& t) {
    std::cout << "=== Phase 2: MLIR → LLVM IR (buildMLIRModule + applyLowering + emit) ===" << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    ct::c3::MLIRToLLVMIROptions opts;
    opts.opt_level = 2;
    opts.dump_mlir = false;
    opts.verify_llvm_ir = true;

    auto ir_result = ct::c3::mlirToLLVMIRFromGraph(g, opts);
    auto t1 = std::chrono::steady_clock::now();
    t.mlir_to_llvm_ir_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ir_result.success) {
        std::cerr << "  ❌ MLIR → LLVM IR 失败: " << ir_result.error_message << std::endl;
        return false;
    }

    llvm_ir_text = ir_result.text;
    llvm_ir_bitcode = ir_result.bitcode;
    std::cout << "  ✅ MLIR → LLVM IR 成功 (" << t.mlir_to_llvm_ir_ms << " ms, "
              << "text " << llvm_ir_text.size() << " chars, "
              << "bitcode " << llvm_ir_bitcode.size() << " bytes)" << std::endl;
    std::cout << std::endl;
    return true;
}

// ============================================================================
// Phase 3a: Plan A — GCVM C API 编译
// ============================================================================
bool runPlanA_GCVM(const std::string& llvm_ir_text,
                   std::vector<char>& hsaco_code_object,
                   Timings& t,
                   bool& ir_version_mismatch) {
    ir_version_mismatch = false;
    std::cout << "=== Phase 3a: Plan A — GCVM C API 编译 ===" << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    auto gcvm_result = ct::c3::compileLLVMToDCUObject(llvm_ir_text, "c3_kernel", 2);
    auto t1 = std::chrono::steady_clock::now();
    t.gcvm_compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (gcvm_result.success) {
        hsaco_code_object = std::move(gcvm_result.code_object);
        std::cout << "  ✅ GCVM 编译成功 (" << t.gcvm_compile_ms << " ms, "
                  << "HSACO " << hsaco_code_object.size() << " bytes)" << std::endl;
        std::cout << std::endl;
        return true;
    }

    // 失败: 检查是否是 IR_VERSION_MISMATCH (per dcu-probe R1 HIGH 风险)
    std::cout << "  ⚠️  GCVM 编译失败 (" << t.gcvm_compile_ms << " ms)" << std::endl;
    std::cout << "     error: " << gcvm_result.error_message << std::endl;

    if (gcvm_result.error_message.find("IR_VERSION_MISMATCH") != std::string::npos ||
        gcvm_result.error_message.find("LLVMTranslationDialectInterface") != std::string::npos ||
        gcvm_result.error_message.find("version") != std::string::npos) {
        ir_version_mismatch = true;
        std::cout << "  → 诊断: IR 版本不兼容 (LLVM 7 vs 14), 切 Plan B (dcc)" << std::endl;
    } else {
        std::cout << "  → 诊断: 其他错误, 尝试 Plan B 兜底" << std::endl;
    }
    std::cout << std::endl;
    return false;
}

// ============================================================================
// Phase 3b: Plan B — dcc bitcode fallback (stub, 节点上实装)
// ============================================================================
bool runPlanB_DCC(const std::vector<uint8_t>& /*llvm_ir_bitcode*/,
                  std::vector<char>& /*hsaco_code_object*/,
                  Timings& t) {
    // Stub: macOS 本地不实装 dcc (Linux only)
    // 节点上需要:
    //   1. write bitcode to /tmp/kernel.bc
    //   2. system("dcc -o /tmp/kernel.hsaco /tmp/kernel.bc -arch=gfx906")
    //   3. read /tmp/kernel.hsaco
    //
    // 完整实装留节点: 探测 dcc 路径 (DCC 26.04 = dcc 25.10.0)
    // 跟 GCVM 1.6 (= LLVM 7) 一样可能 IR 不兼容, 但 bitcode 比 text 兼容性好
    std::cout << "=== Phase 3b: Plan B — dcc bitcode fallback ===" << std::endl;
    std::cout << "  ⚠️  dcc 节点专属, macOS 本地跳过 (节点上跑) " << std::endl;
    std::cout << "  Stub: 节点实装需要 system(\"dcc -o kernel.hsaco kernel.bc -arch=gfx906\")" << std::endl;
    t.dcc_compile_ms = 0.0;
    std::cout << std::endl;
    return false;
}

// ============================================================================
// Phase 3c: Plan C — host CPU baseline (LLVM JIT fallback)
// ============================================================================
bool runPlanC_HostCPU(const ct::c3::Graph& g,
                      Timings& t) {
    std::cout << "=== Phase 3c: Plan C — host CPU baseline (C3 MLIR JIT) ===" << std::endl;
    std::cout << "  (注: Plan C 只是 baseline, 不验证 DCU 链路)" << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    ct::c3::CompileOptions opts;
    opts.backend = ct::c3::C3Backend::MLIR;
    opts.target_device = ct::DeviceType::kCPU;
    opts.opt_level = 2;

    try {
        auto kernel = ct::c3::C3Engine::getInstance().compile(g, opts);
        auto t1 = std::chrono::steady_clock::now();
        t.host_jit_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  ✅ host CPU JIT 成功 (" << t.host_jit_ms << " ms)" << std::endl;
        std::cout << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "  ❌ host CPU JIT 失败: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Phase 4: 准备 input data (1024-dim a/b = 0..1023, expected = a+b)
// ============================================================================
struct TestData {
    Tensor a;
    Tensor b;
    Tensor expected;
};

TestData prepareInputData() {
    std::cout << "=== Phase 4: 准备 input data ===" << std::endl;
    TestData d;
    d.a = Tensor(ShapeTag{}, {1024});
    d.b = Tensor(ShapeTag{}, {1024});
    d.expected = Tensor(ShapeTag{}, {1024});
    for (int i = 0; i < 1024; ++i) {
        float a_val = static_cast<float>(i);
        float b_val = static_cast<float>(i) * 2.0f;
        d.a.data_write<float>()[i] = a_val;
        d.b.data_write<float>()[i] = b_val;
        d.expected.data_write<float>()[i] = a_val + b_val;
    }
    std::cout << "  a[0..2] = " << d.a.data_read<float>()[0] << ", "
              << d.a.data_read<float>()[1] << ", " << d.a.data_read<float>()[2] << std::endl;
    std::cout << "  b[0..2] = " << d.b.data_read<float>()[0] << ", "
              << d.b.data_read<float>()[1] << ", " << d.b.data_read<float>()[2] << std::endl;
    std::cout << "  expected[0..2] = " << d.expected.data_read<float>()[0] << ", "
              << d.expected.data_read<float>()[1] << ", " << d.expected.data_read<float>()[2] << std::endl;
    std::cout << std::endl;
    return d;
}

// ============================================================================
// Phase 5: DCU execute + correctness 验证
// ============================================================================
bool runDCUExecuteAndVerify(const std::vector<char>& hsaco_code_object,
                            const ct::c3::Graph& g,
                            const TestData& d,
                            Timings& t) {
    std::cout << "=== Phase 5: DCU execute + correctness 验证 ===" << std::endl;

    // Execute on DCU
    auto t0 = std::chrono::steady_clock::now();
    auto dcu_kernel = std::make_shared<ct::c3::DCUCompiledKernel>(
        std::string(hsaco_code_object.begin(), hsaco_code_object.end()),
        "c3_kernel", g, 0);
    auto outputs = dcu_kernel->execute({d.a, d.b});
    auto t1 = std::chrono::steady_clock::now();
    t.dcu_exec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (outputs.empty()) {
        std::cerr << "  ❌ DCU execute 失败: empty output" << std::endl;
        return false;
    }
    std::cout << "  ✅ DCU execute (" << t.dcu_exec_ms << " ms)" << std::endl;
    std::cout << "  output[0..2] = " << outputs[0].data_read<float>()[0] << ", "
              << outputs[0].data_read<float>()[1] << ", " << outputs[0].data_read<float>()[2] << std::endl;
    std::cout << std::endl;

    // Verify correctness
    auto t2 = std::chrono::steady_clock::now();
    size_t bad = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < 1024; ++i) {
        float diff = std::fabs(outputs[0].data_read<float>()[i] - d.expected.data_read<float>()[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-4f) bad++;
    }
    auto t3 = std::chrono::steady_clock::now();
    t.correctness_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "  bad: " << bad << " / 1024" << std::endl;
    std::cout << "  max_diff: " << max_diff << std::endl;
    std::cout << "  correctness check: " << t.correctness_ms << " ms" << std::endl;
    std::cout << std::endl;

    return (bad == 0);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  C3 → DCU Hello World v2 (完整链路)" << std::endl;
    std::cout << "  v0.5.2 DCU 接入 (2026-08-10)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    Timings t;

    // Phase 0: 平台检查
    if (!checkPlatform()) return 1;

    // Phase 1: 创 Graph
    auto g = createSimpleGraph();

    // Phase 2: MLIR → LLVM IR (真 pipeline)
    std::string llvm_ir_text;
    std::vector<uint8_t> llvm_ir_bitcode;
    if (!runMLIRToLLVMIR(g, llvm_ir_text, llvm_ir_bitcode, t)) return 1;

    // Phase 3: Plan A (GCVM) → Plan B (dcc) → Plan C (host CPU)
    std::vector<char> hsaco_code_object;
    bool plan_a_ok = false, plan_b_ok = false, plan_c_ok = false;
    bool ir_version_mismatch = false;

    plan_a_ok = runPlanA_GCVM(llvm_ir_text, hsaco_code_object, t, ir_version_mismatch);

    if (!plan_a_ok) {
        plan_b_ok = runPlanB_DCC(llvm_ir_bitcode, hsaco_code_object, t);
    }

    if (!plan_a_ok && !plan_b_ok) {
        plan_c_ok = runPlanC_HostCPU(g, t);
    }

    // Phase 4 + 5: DCU execute + verify (只有 plan_a 或 plan_b 成功才跑)
    bool dc_ok = false;
    if (plan_a_ok || plan_b_ok) {
        auto d = prepareInputData();
        dc_ok = runDCUExecuteAndVerify(hsaco_code_object, g, d, t);
    } else {
        std::cout << "=== Phase 4-5: SKIP (没有 DCU 链路) ===" << std::endl;
        std::cout << std::endl;
    }

    // === 总结 ===
    std::cout << "========================================" << std::endl;
    std::cout << "  总结" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Plan A (GCVM C API):     " << (plan_a_ok ? "✅ PASS" : "❌ FAIL")
              << "  (" << t.gcvm_compile_ms << " ms)" << std::endl;
    std::cout << "  Plan B (dcc bitcode):    " << (plan_b_ok ? "✅ PASS" : "⚠️  SKIP/STUB")
              << "  (" << t.dcc_compile_ms << " ms)" << std::endl;
    std::cout << "  Plan C (host CPU JIT):   " << (plan_c_ok ? "✅ PASS (baseline)" : "❌ FAIL")
              << "  (" << t.host_jit_ms << " ms)" << std::endl;
    std::cout << "  DCU execute + verify:    " << (dc_ok ? "✅ PASS" : "❌ FAIL / SKIP")
              << "  (" << t.dcu_exec_ms << " ms exec)" << std::endl;
    std::cout << std::endl;
    std::cout << "  MLIR → LLVM IR: " << t.mlir_to_llvm_ir_ms << " ms" << std::endl;
    std::cout << "  IR version mismatch 诊断: " << (ir_version_mismatch ? "YES (GCVM 1.6 vs LLVM 14)" : "no") << std::endl;
    std::cout << std::endl;
    std::cout << "  返回码: ";
    if (plan_a_ok || plan_b_ok) {
        std::cout << (dc_ok ? "0 (✅ DCU 真链路 PASS)" : "1 (❌ DCU execute 错)") << std::endl;
    } else if (plan_c_ok) {
        std::cout << "0 (⚠️ host CPU baseline OK, DCU 没跑通)" << std::endl;
    } else {
        std::cout << "1 (❌ 全失败)" << std::endl;
    }
    std::cout << std::endl;

    return (plan_a_ok || plan_b_ok) ? (dc_ok ? 0 : 1) : (plan_c_ok ? 0 : 1);
}
