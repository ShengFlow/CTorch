/**
 * @file test_debug_fused.cpp
 * @brief 调试：直接编译 fused graph 并运行 kernel，验证 FusedNode 检测
 */
#include <iostream>
#include <cmath>
#include "Tensor.h"
#include "C3/C3Engine.h"
#include "C3/Graph.h"

using namespace ct;
using namespace ct::c3;

static void fillRandom(Tensor& t, float scale = 0.1f) {
    float* data = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        data[i] = scale * std::sin(static_cast<float>(i) * 0.1f);
    }
}

int main() {
    std::cout << "=== Debug: 直接编译 fused graph ===" << std::endl;

    const size_t M = 32, K = 32, N = 32;
    Tensor X(ShapeTag{}, {M, K});
    Tensor W(ShapeTag{}, {K, N});
    Tensor B(ShapeTag{}, {M, N});
    fillRandom(X);
    fillRandom(W);
    fillRandom(B);

    // 直接构建图（与 C3HotPathManager::buildFusedGraph 相同逻辑）
    Graph g;
    auto in_desc = TensorDesc::fromShape({M, K});
    auto w_desc = TensorDesc::fromShape({K, N});
    auto b_desc = TensorDesc::fromShape({M, N});
    auto out_desc = TensorDesc::fromShape({M, N});

    size_t in1 = g.addInput(in_desc);
    size_t w1 = g.addInput(w_desc);
    size_t b1 = g.addInput(b_desc);
    size_t mm_node = g.addNode(MatMulNode{in_desc, w_desc}, {in1, w1}, out_desc);
    size_t add_node = g.addNode(AddNode{out_desc, b_desc}, {mm_node, b1}, out_desc);
    size_t sig_node = g.addNode(SigmoidNode{out_desc}, {add_node}, out_desc);
    g.markOutput(sig_node);
    std::cout << "Graph: " << g.nodeCount() << " nodes, " << g.inputCount() << " inputs" << std::endl;
    std::cout << "Graph inputs: [";
    for (size_t in_id : g.inputs()) std::cout << in_id << " ";
    std::cout << "]" << std::endl;

    // 编译
    auto& engine = C3Engine::getInstance();
    CompileOptions opts;
    opts.enable_fusion = true;
    std::cout << "\nCompiling..." << std::endl;
    auto kernel = engine.compile(g, opts);
    if (!kernel) {
        std::cout << "FAILED to compile" << std::endl;
        return 1;
    }
    std::cout << "Kernel compiled: type=" << typeid(*kernel).name() << std::endl;

    // 直接执行 kernel
    std::vector<Tensor> inputs = {X, W, B};
    Tensor out(ShapeTag{}, {M, N});
    std::vector<Tensor> outputs = {out};

    std::cout << "\nExecuting kernel..." << std::endl;
    auto results = kernel->execute({X, W, B});
    std::cout << "Results size: " << results.size() << std::endl;

    if (results.empty()) {
        std::cout << "FAILED: kernel returned no results" << std::endl;
        return 1;
    }

    // 计算 eager 参考结果
    Tensor mm_ref = matMul(X, W);
    Tensor add_ref = mm_ref + B;
    Tensor sig_ref = add_ref.sigmoid();

    const float* fused_data = results[0].data_read<float>();
    const float* ref_data = sig_ref.data_read<float>();
    size_t numel = results[0].numel();

    double max_diff = 0.0;
    int bad_count = 0;
    std::cout << "First 5 elements:" << std::endl;
    for (size_t i = 0; i < 5 && i < numel; ++i) {
        printf("  [%zu] fused=%.6f ref=%.6f\n", i, fused_data[i], ref_data[i]);
    }

    for (size_t i = 0; i < numel; ++i) {
        double diff = std::fabs((double)fused_data[i] - (double)ref_data[i]);
        double max_val = std::max(std::fabs((double)fused_data[i]), std::fabs((double)ref_data[i]));
        if (diff > 1e-4 + 1e-4 * max_val) {
            bad_count++;
            if (bad_count <= 5) {
                printf("  MISMATCH[%zu]: fused=%.6f ref=%.6f\n", i, fused_data[i], ref_data[i]);
            }
        }
        if (diff > max_diff) max_diff = diff;
    }

    bool correct = (bad_count == 0);
    std::cout << "Result: " << (correct ? "PASS" : "FAIL")
              << " bad=" << bad_count << "/" << numel
              << " max_diff=" << max_diff << std::endl;

    return correct ? 0 : 1;
}