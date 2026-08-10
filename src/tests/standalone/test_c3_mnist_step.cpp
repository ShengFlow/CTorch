/**
 * @file test_c3_mnist_step.cpp
 * @brief C3 MNIST 训练步骤验证：C3 编译 MLP 前向图 vs Eager 前向
 * @details 验证项：
 *          1. 构建 3 层 MLP 前向图 (784→256→128→10) 并用 C3 编译
 *          2. C3 MLIR / Handwritten 后端输出与 Eager 输出一致
 *          3. C3 编译的融合图（fuse）输出与 Eager 一致
 * @date 2026/8/2
 */

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/Graph.h"
#include "C3/C3Engine.h"
#include "C3/C3KernelRegistry.h"
#include "Ctools.h"
#include "kernels/kernels.h"
#include "mnist/mnist_loader.h"
#include "ctQALS/Random.h"

using namespace ct;
using namespace ct::c3;

// ======================= 辅助函数 =======================

static bool allClose(const Tensor& a, const Tensor& b, float rtol = 1e-4f, float atol = 1e-5f) {
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

static float maxDiff(const Tensor& a, const Tensor& b) {
    const float* pa = a.data_read<float>();
    const float* pb = b.data_read<float>();
    float max_d = 0.0f;
    for (size_t i = 0; i < a.numel(); ++i) {
        float d = std::fabs(pa[i] - pb[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

static void xavierInit(Tensor& W, ctQALS::rng::Xoshiro256PlusPlus& rng,
                       size_t fan_in, size_t fan_out) {
    float std = std::sqrt(2.0f / (fan_in + fan_out));
    float* data = W.data_write<float>();
    for (size_t i = 0; i < W.numel(); ++i) {
        float r = 2.0f * rng.uniform_f32() - 1.0f;
        data[i] = r * std;
    }
}

// ======================= Eager 前向参考实现 =======================

static Tensor eagerForward(const std::vector<Tensor>& params, const Tensor& x) {
    // params: W1, b1, W2, b2, W3, b3
    Tensor z1 = x.matmul(params[0]) + params[1];
    Tensor h1 = z1.relu();
    Tensor z2 = h1.matmul(params[2]) + params[3];
    Tensor h2 = z2.relu();
    Tensor logits = h2.matmul(params[4]) + params[5];
    return logits;
}

// ======================= C3 图构建 =======================

static Graph buildMLPGraph(size_t batch_size) {
    Graph g;

    // 输入描述符
    auto x_desc   = TensorDesc::fromShape({batch_size, 784});
    auto w1_desc  = TensorDesc::fromShape({784, 256});
    auto b1_desc  = TensorDesc::fromShape({256});
    auto w2_desc  = TensorDesc::fromShape({256, 128});
    auto b2_desc  = TensorDesc::fromShape({128});
    auto w3_desc  = TensorDesc::fromShape({128, 10});
    auto b3_desc  = TensorDesc::fromShape({10});

    // 中间张量描述符
    auto z1_desc  = TensorDesc::fromShape({batch_size, 256});
    auto h1_desc  = TensorDesc::fromShape({batch_size, 256});
    auto z2_desc  = TensorDesc::fromShape({batch_size, 128});
    auto h2_desc  = TensorDesc::fromShape({batch_size, 128});
    auto out_desc = TensorDesc::fromShape({batch_size, 10});

    // 输入节点
    size_t x  = g.addInput(x_desc);
    size_t w1 = g.addInput(w1_desc);
    size_t b1 = g.addInput(b1_desc);
    size_t w2 = g.addInput(w2_desc);
    size_t b2 = g.addInput(b2_desc);
    size_t w3 = g.addInput(w3_desc);
    size_t b3 = g.addInput(b3_desc);

    // Layer 1: z1 = MatMul(x, W1) + b1 → h1 = ReLU(z1)
    size_t mm1 = g.addNode(MatMulNode{x_desc, w1_desc}, {x, w1}, z1_desc);
    size_t a1  = g.addNode(AddNode{z1_desc, b1_desc}, {mm1, b1}, z1_desc);
    size_t r1  = g.addNode(ReLUNode{z1_desc}, {a1}, h1_desc);

    // Layer 2: z2 = MatMul(h1, W2) + b2 → h2 = ReLU(z2)
    size_t mm2 = g.addNode(MatMulNode{h1_desc, w2_desc}, {r1, w2}, z2_desc);
    size_t a2  = g.addNode(AddNode{z2_desc, b2_desc}, {mm2, b2}, z2_desc);
    size_t r2  = g.addNode(ReLUNode{z2_desc}, {a2}, h2_desc);

    // Layer 3: logits = MatMul(h2, W3) + b3
    size_t mm3 = g.addNode(MatMulNode{h2_desc, w3_desc}, {r2, w3}, out_desc);
    size_t a3  = g.addNode(AddNode{out_desc, b3_desc}, {mm3, b3}, out_desc);

    g.markOutput(a3);
    return g;
}

// ======================= 主测试 =======================

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();

    std::cout << "=== C3 MNIST MLP 前向图验证 ===" << std::endl;
    std::cout << std::endl;

    // 加载 MNIST 数据
    MNISTLoader loader(".", DeviceType::kCPU);
    Tensor train_images, train_labels;
    loader.load_training_data(train_images, train_labels);

    const size_t batch_size = 128;

    // 准备输入 batch
    Tensor batch_x(ShapeTag{}, {batch_size, 784}, DType::kFloat, DeviceType::kCPU);
    std::memcpy(batch_x.data_write<float>(), train_images.data_read<float>(),
                batch_size * 784 * sizeof(float));

    // 初始化参数
    ctQALS::rng::Xoshiro256PlusPlus rng(42);
    std::vector<Tensor> params(6);
    params[0] = Tensor(ShapeTag{}, {784, 256}, DType::kFloat, DeviceType::kCPU);  // W1
    params[1] = Tensor(ShapeTag{}, {256}, DType::kFloat, DeviceType::kCPU);       // b1
    params[2] = Tensor(ShapeTag{}, {256, 128}, DType::kFloat, DeviceType::kCPU);  // W2
    params[3] = Tensor(ShapeTag{}, {128}, DType::kFloat, DeviceType::kCPU);       // b2
    params[4] = Tensor(ShapeTag{}, {128, 10}, DType::kFloat, DeviceType::kCPU);   // W3
    params[5] = Tensor(ShapeTag{}, {10}, DType::kFloat, DeviceType::kCPU);        // b3

    xavierInit(params[0], rng, 784, 256);
    xavierInit(params[2], rng, 256, 128);
    xavierInit(params[4], rng, 128, 10);
    params[1].zero(); params[3].zero(); params[5].zero();

    // ==================== Eager 基准 ====================
    std::cout << "--- Eager 基准前向 ---" << std::endl;
    Tensor eager_out = eagerForward(params, batch_x);
    std::cout << "  Eager logits shape: [";
    for (size_t d : eager_out.shape()) std::cout << d << " ";
    std::cout << "]" << std::endl;
    std::cout << std::endl;

    // ==================== C3 编译与执行 ====================
    auto testBackend = [&](C3Backend backend, const char* name, bool enable_fusion) {
        std::cout << "--- C3 " << name << (enable_fusion ? " (fused)" : "") << " ---" << std::endl;

        Graph g = buildMLPGraph(batch_size);
        std::cout << "  Graph nodes: " << g.nodeCount()
                  << " inputs: " << g.inputCount()
                  << " outputs: " << g.outputCount() << std::endl;

        CompileOptions opts;
        opts.backend = backend;
        opts.target_device = DeviceType::kCPU;
        opts.enable_fusion = enable_fusion;
        opts.enable_cache = false;
        opts.enable_autotune = false;

        auto& engine = C3Engine::getInstance();
        auto kernel = engine.compile(g, opts);

        if (!kernel) {
            std::cerr << "  COMPILE FAILED!" << std::endl;
            return false;
        }

        std::cout << "  Compile OK" << std::endl;

        // 执行
        std::vector<Tensor> inputs = {batch_x, params[0], params[1], params[2],
                                      params[3], params[4], params[5]};
        auto results = kernel->execute(inputs);

        if (results.size() != 1) {
            std::cerr << "  Expected 1 output, got " << results.size() << std::endl;
            return false;
        }

        // 比较输出
        bool ok = allClose(eager_out, results[0]);
        float max_d = maxDiff(eager_out, results[0]);

        std::cout << "  Output shape: [";
        for (size_t d : results[0].shape()) std::cout << d << " ";
        std::cout << "]" << std::endl;
        std::cout << "  Max diff vs Eager: " << std::scientific << std::setprecision(6) << max_d << std::endl;
        std::cout << "  Result: " << (ok ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;

        return ok;
    };

    bool all_pass = true;

    // MLIR 后端（无融合）
    all_pass &= testBackend(C3Backend::MLIR, "MLIR", false);

    // MLIR 后端（融合）
    all_pass &= testBackend(C3Backend::MLIR, "MLIR", true);

    // Handwritten 后端（无融合）
    all_pass &= testBackend(C3Backend::Handwritten, "Handwritten", false);

    // Handwritten 后端（融合）
    all_pass &= testBackend(C3Backend::Handwritten, "Handwritten", true);

    // ==================== 总结 ====================
    std::cout << "=== " << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << " ===" << std::endl;

    return all_pass ? 0 : 1;
}