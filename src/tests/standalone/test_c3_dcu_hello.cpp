/**
 * @file test_c3_dcu_hello.cpp
 * @brief C3 → DCU "Hello World" 测试 (v0.5 DCU 接入 Phase 2)
 * @details 1 个 Add kernel 在 DCU 上跑通, 验证整条链路
 *          C3 Graph {a, b, Add} → MLIR → LLVM IR → GCVM → Code Object → DCU execute
 *
 * 用法 (DCU 节点 b02r4n13):
 *   module load compiler/dtk/24.04
 *   cd build-dcu && cmake -DWITH_DCU=ON .. && make -j8 test_c3_dcu_hello
 *   ./test_c3_dcu_hello
 *
 * 输出: PASS = 整条链路打通 (C3 → MLIR → GCVM → DCU execute 数值跟 eager 一致)
 *       FAIL = 哪一环失败 (带详细 error message)
 *
 * 探针 (probe-dcu-dtk24.sh) 跑通后实装 GCVM C API 真实签名, 当前是 stub
 */
#include "C3/C3Engine.h"
#include "C3/C3KernelRegistry.h"
#include "C3/GCVMBridge.h"
#include "C3/DCUCompiledKernel.h"
#include "CtorchError.h"
#include "Tensor.h"

#include <iostream>
#include <iomanip>
#include <cmath>

#ifdef WITH_DCU
    #include <hip/hip_runtime.h>
#endif

using namespace ct;

int main() {
    std::cout << "=== C3 → DCU Hello World Test ===" << std::endl;
    std::cout << std::endl;

    // === 检查 GCVM 可用性 ===
    if (!ct::c3::isGCVMAvailable()) {
        std::cerr << "❌ GCVM 不可用 (rebuild with -DWITH_DCU=ON)" << std::endl;
        return 1;
    }
    std::cout << "✅ GCVM 可用" << std::endl;
    std::cout << std::endl;

    // === 检查 DCU 设备 ===
#ifdef WITH_DCU
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        std::cerr << "❌ hipGetDeviceCount failed or no DCU devices: "
                  << hipGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "✅ DCU 设备数: " << device_count << std::endl;
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << "   DCU[0] 名称: " << prop.name << std::endl;
    std::cout << "   DCU[0] 显存: " << (prop.totalGlobalMem / 1e9) << " GB" << std::endl;
    std::cout << "   DCU[0] compute capability: " << prop.major << "." << prop.minor << std::endl;
#else
    std::cerr << "❌ WITH_DCU=OFF, 跳过" << std::endl;
    return 1;
#endif
    std::cout << std::endl;

    // === Phase 2a: 创建简单 Graph {a, b, Add} ===
    std::cout << "--- Phase 2a: 创建 Graph {a, b, Add} ---" << std::endl;
    ct::c3::Graph g;
    auto a_desc = ct::c3::TensorDesc::fromShape({1024});
    auto b_desc = ct::c3::TensorDesc::fromShape({1024});
    auto out_desc = ct::c3::TensorDesc::fromShape({1024});

    auto a = g.addInput(a_desc);
    auto b = g.addInput(b_desc);
    auto add = g.addNode(ct::c3::AddNode{a_desc, b_desc}, {a, b}, out_desc);
    g.markOutput(g.nodeCount() - 1);
    std::cout << "  Graph: 2 inputs + 1 Add node + 1 output" << std::endl;
    std::cout << std::endl;

    // === Phase 2b: C3 compile (MLIR backend) ===
    std::cout << "--- Phase 2b: C3 compile (MLIR backend) ---" << std::endl;
    ct::c3::CompileOptions opts;
    opts.backend = ct::c3::C3Backend::MLIR;
    opts.target_device = ct::DeviceType::kDCU;  // ⚠️ 触发 DCU 路径 (需要 C3Engine 扩展)

    // TODO: probe-adjust: 当前 C3Engine 不支持 target_device=kDCU
    // 临时: 用 macOS MLIR 编译, 然后手动接 GCVM
    std::shared_ptr<ct::c3::CompiledKernel> kernel;
    try {
        kernel = ct::c3::C3Engine::getInstance().compile(g, opts);
        std::cout << "  C3 compile ✅" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  C3 compile ❌: " << e.what() << std::endl;
        return 1;
    }
    std::cout << std::endl;

    // === Phase 2c: MLIR → LLVM IR → GCVM ===
    // TODO: probe-adjust: 用真实 LLVM Module (从 MLIRKernelGen 拿)
    // 当前: 占位字符串
    std::cout << "--- Phase 2c: MLIR → LLVM IR → GCVM ---" << std::endl;
    std::string fake_llvm_ir = "; placeholder LLVM IR (待 probe-adjust)\n"
                                "define float @c3_kernel(float* %a, float* %b, float* %out, i64 %n) {\n"
                                "  ret float 0.0\n"
                                "}\n";
    auto gcvvm_result = ct::c3::compileLLVMToDCUObject(fake_llvm_ir, "c3_kernel", 2);
    if (!gcvvm_result.success) {
        std::cerr << "  GCVM compile ❌: " << gcvvm_result.error_message << std::endl;
        std::cerr << "  (注意: 当前 GCVM API 是 stub, 真实函数名待 probe-dcu-dtk24.sh 探针后调整)"
                  << std::endl;
        return 1;
    }
    std::cout << "  GCVM compile ✅ (Code Object size: " << gcvvm_result.code_object.size() << " bytes)" << std::endl;
    std::cout << std::endl;

    // === Phase 2d: 准备 input data ===
    std::cout << "--- Phase 2d: 准备 input data ---" << std::endl;
    Tensor a_host(ShapeTag{}, {1024});
    Tensor b_host(ShapeTag{}, {1024});
    Tensor expected(ShapeTag{}, {1024});
    for (int i = 0; i < 1024; ++i) {
        a_host.data_write<float>()[i] = static_cast<float>(i);
        b_host.data_write<float>()[i] = static_cast<float>(i) * 2.0f;
        expected.data_write<float>()[i] = static_cast<float>(i) + static_cast<float>(i) * 2.0f;
    }
    std::cout << "  a[0..2] = " << a_host.data_read<float>()[0] << ", "
              << a_host.data_read<float>()[1] << ", " << a_host.data_read<float>()[2] << std::endl;
    std::cout << "  b[0..2] = " << b_host.data_read<float>()[0] << ", "
              << b_host.data_read<float>()[1] << ", " << b_host.data_read<float>()[2] << std::endl;
    std::cout << "  expected[0..2] = " << expected.data_read<float>()[0] << ", "
              << expected.data_read<float>()[1] << ", " << expected.data_read<float>()[2] << std::endl;
    std::cout << std::endl;

    // === Phase 2e: DCU execute ===
    std::cout << "--- Phase 2e: DCU execute (via DCUCompiledKernel) ---" << std::endl;
    auto dcu_kernel = std::make_shared<ct::c3::DCUCompiledKernel>(
        gcvvm_result.code_object, "c3_kernel", g, 0);
    auto outputs = dcu_kernel->execute({a_host, b_host});
    if (outputs.empty()) {
        std::cerr << "  DCU execute ❌: empty output" << std::endl;
        return 1;
    }
    std::cout << "  DCU execute ✅" << std::endl;
    std::cout << "  output[0..2] = " << outputs[0].data_read<float>()[0] << ", "
              << outputs[0].data_read<float>()[1] << ", " << outputs[0].data_read<float>()[2] << std::endl;
    std::cout << std::endl;

    // === Phase 2f: 验证 correctness ===
    std::cout << "--- Phase 2f: 验证 correctness (DCU output vs expected) ---" << std::endl;
    size_t bad = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < 1024; ++i) {
        float d = std::fabs(outputs[0].data_read<float>()[i] - expected.data_read<float>()[i]);
        max_diff = std::max(max_diff, d);
        if (d > 1e-4f) bad++;
    }
    std::cout << "  bad: " << bad << " / 1024" << std::endl;
    std::cout << "  max_diff: " << max_diff << std::endl;
    if (bad == 0) {
        std::cout << "  ✅ ALL CORRECT - 整条链路打通!" << std::endl;
    } else {
        std::cout << "  ❌ CORRECTNESS FAIL - 探针后调整" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "=== Test End ===" << std::endl;
    return (bad == 0) ? 0 : 1;
}
