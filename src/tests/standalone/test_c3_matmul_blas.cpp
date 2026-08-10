/**
 * @file test_c3_matmul_blas.cpp
 * @brief 验证 C3 MatMul BLAS 委托的正确性和性能
 * @details 对比 C3 Handwritten/MLIR 后端与 Eager MatMul 的结果正确性和性能
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/Graph.h"
#include "C3/C3Engine.h"
#include "C3/C3KernelRegistry.h"

using namespace ct;
using namespace ct::c3;

static bool allClose(const Tensor& a, const Tensor& b, float rtol = 1e-4f, float atol = 1e-5f) {
    if (a.shape() != b.shape()) return false;
    const float* pa = a.data_read<float>();
    const float* pb = b.data_read<float>();
    size_t n = a.numel();
    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(pa[i] - pb[i]);
        float max_val = std::max(std::fabs(pa[i]), std::fabs(pb[i]));
        if (diff > atol + rtol * max_val) {
            std::cerr << "  MISMATCH at [" << i << "]: eager=" << pa[i] << " c3=" << pb[i] << " diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

static Tensor makeRandomTensor(const std::vector<size_t>& shape, unsigned seed = 42) {
    Tensor t(ShapeTag{}, shape);
    float* p = t.data_write<float>();
    size_t n = t.numel();
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        p[i] = (float)(rand() % 100) / 100.0f;
    }
    return t;
}

static double benchmark(const std::function<void()>& fn, int warmup = 5, int iterations = 20) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    return us;
}

int main() {
    std::cout << "=== C3 MatMul BLAS 委托验证 ===" << std::endl;
    std::cout << std::endl;

    // 测试不同形状的 MatMul
    struct Shape {
        size_t M, K, N;
        const char* name;
    };
    std::vector<Shape> shapes = {
        {64, 64, 64, "64x64x64"},
        {128, 128, 128, "128x128x128"},
        {256, 256, 256, "256x256x256"},
        {512, 512, 512, "512x512x512"},
        {256, 512, 256, "256x512x256"},
        {1024, 1024, 1024, "1024x1024x1024"},
        {1, 1024, 512, "1x1024x512"},
    };

    for (const auto& shape : shapes) {
        std::cout << "--- Shape: " << shape.name << " (M=" << shape.M << " K=" << shape.K << " N=" << shape.N << ") ---" << std::endl;

        // 创建输入张量
        Tensor a = makeRandomTensor({shape.M, shape.K}, 1);
        Tensor b = makeRandomTensor({shape.K, shape.N}, 2);

        // Eager 基准
        Tensor eager_out;
        double eager_us = benchmark([&]() {
            eager_out = matMul(a, b);
        });
        std::cout << "  Eager:          " << std::setw(10) << std::fixed << std::setprecision(1) << eager_us << " us" << std::endl;

        // --- C3 Handwritten 后端 ---
        {
            // 构建图
            Graph g;
            auto a_desc = TensorDesc::fromShape({shape.M, shape.K});
            auto b_desc = TensorDesc::fromShape({shape.K, shape.N});
            auto out_desc = TensorDesc::fromShape({shape.M, shape.N});
            auto a_id = g.addInput(a_desc);
            auto b_id = g.addInput(b_desc);
            auto c_id = g.addNode(MatMulNode{a_desc, b_desc}, {a_id, b_id}, out_desc);
            g.markOutput(c_id);

            // 编译
            CompileOptions opts;
            opts.backend = C3Backend::Handwritten;
            opts.enable_cache = false;
            opts.enable_fusion = false;
            opts.enable_autotune = false;

            auto kernel = C3Engine::getInstance().compile(g, opts);
            if (!kernel) {
                std::cerr << "  C3 Handwritten: 编译失败!" << std::endl;
                continue;
            }

            // 正确性验证
            auto result = kernel->execute({a, b});
            bool correct = allClose(eager_out, result[0]);
            std::cout << "  C3 Handwritten: " << (correct ? "正确" : "错误") << std::endl;

            // 性能
            double c3_us = benchmark([&]() {
                kernel->execute({a, b});
            });
            double ratio = eager_us / c3_us;
            std::cout << "                  " << std::setw(10) << std::fixed << std::setprecision(1) << c3_us << " us"
                      << "  (x" << std::setprecision(2) << ratio << " vs Eager)" << std::endl;
        }

        // --- C3 MLIR 后端 ---
        {
            Graph g;
            auto a_desc = TensorDesc::fromShape({shape.M, shape.K});
            auto b_desc = TensorDesc::fromShape({shape.K, shape.N});
            auto out_desc = TensorDesc::fromShape({shape.M, shape.N});
            auto a_id = g.addInput(a_desc);
            auto b_id = g.addInput(b_desc);
            auto c_id = g.addNode(MatMulNode{a_desc, b_desc}, {a_id, b_id}, out_desc);
            g.markOutput(c_id);

            CompileOptions opts;
            opts.backend = C3Backend::MLIR;
            opts.enable_cache = false;
            opts.enable_fusion = false;
            opts.enable_autotune = false;

            auto kernel = C3Engine::getInstance().compile(g, opts);
            if (!kernel) {
                std::cerr << "  C3 MLIR:        编译失败!" << std::endl;
                continue;
            }

            auto result = kernel->execute({a, b});
            bool correct = allClose(eager_out, result[0]);
            std::cout << "  C3 MLIR:        " << (correct ? "正确" : "错误") << std::endl;

            double c3_us = benchmark([&]() {
                kernel->execute({a, b});
            });
            double ratio = eager_us / c3_us;
            std::cout << "                  " << std::setw(10) << std::fixed << std::setprecision(1) << c3_us << " us"
                      << "  (x" << std::setprecision(2) << ratio << " vs Eager)" << std::endl;
        }

        std::cout << std::endl;
    }

    // === Sigmoid / Tanh 激活函数验证 ===
    std::cout << "=== Sigmoid / Tanh 激活函数验证 ===" << std::endl;
    std::cout << std::endl;

    struct ActShape {
        size_t N;
        const char* name;
    };
    std::vector<ActShape> act_shapes = {
        {64, "64"},
        {256, "256"},
        {1024, "1024"},
    };

    auto testActivation = [&](const char* act_name, auto makeNode, auto eagerFunc) {
        for (const auto& shape : act_shapes) {
            std::cout << "--- " << act_name << " N=" << shape.name << " ---" << std::endl;

            Tensor x = makeRandomTensor({shape.N}, 3);

            // Eager 基准
            Tensor eager_out;
            double eager_us = benchmark([&]() {
                eager_out = eagerFunc(x);
            });
            std::cout << "  Eager:          " << std::setw(10) << std::fixed << std::setprecision(1) << eager_us << " us" << std::endl;

            // 测试两个后端
            auto testBackend = [&](C3Backend backend, const char* backend_name) {
                Graph g;
                auto in_desc = TensorDesc::fromShape({shape.N});
                auto in_id = g.addInput(in_desc);
                auto out_desc = TensorDesc::fromShape({shape.N});
                auto node_id = g.addNode(makeNode(in_desc), {in_id}, out_desc);
                g.markOutput(node_id);

                CompileOptions opts;
                opts.backend = backend;
                opts.enable_cache = false;
                opts.enable_fusion = false;
                opts.enable_autotune = false;

                auto kernel = C3Engine::getInstance().compile(g, opts);
                if (!kernel) {
                    std::cerr << "  C3 " << backend_name << ": 编译失败!" << std::endl;
                    return;
                }

                auto result = kernel->execute({x});
                bool correct = allClose(eager_out, result[0], 1e-3f, 1e-4f);
                std::cout << "  C3 " << backend_name << ": " << (correct ? "正确" : "错误") << std::endl;

                double c3_us = benchmark([&]() {
                    kernel->execute({x});
                });
                double ratio = eager_us / c3_us;
                std::cout << "                  " << std::setw(10) << std::fixed << std::setprecision(1) << c3_us << " us"
                          << "  (x" << std::setprecision(2) << ratio << " vs Eager)" << std::endl;
            };

            testBackend(C3Backend::Handwritten, "Handwritten");
            testBackend(C3Backend::MLIR, "MLIR");

            std::cout << std::endl;
        }
    };

    // Sigmoid
    testActivation("Sigmoid",
        [](const TensorDesc& d) { return SigmoidNode{d}; },
        [](const Tensor& x) -> Tensor { return x.sigmoid(); }
    );

    // Tanh
    testActivation("Tanh",
        [](const TensorDesc& d) { return TanhNode{d}; },
        [](const Tensor& x) -> Tensor { return x.tanh(); }
    );

    // ======================= 常量折叠 + 死代码消除验证 =======================

    std::cout << "\n=== Graph 常量折叠 & 死代码消除 ===\n" << std::endl;

    auto scalar_desc = TensorDesc::fromShape({1});

    // 测试 1: Add(c1, c2) → Const(c1+c2)
    {
        Graph g;
        auto c1 = g.addConstant(3.0, scalar_desc);
        auto c2 = g.addConstant(4.0, scalar_desc);
        auto add_id = g.addNode(AddNode{scalar_desc, scalar_desc}, {c1, c2}, scalar_desc);
        g.markOutput(add_id);

        auto opt = g.canonicalize().eliminateDeadCode();
        std::cout << "  [ConstFold] Add(3,4): " << opt.toString();
        bool ok = opt.nodeCount() == 1 && opt.outputCount() == 1;
        std::cout << "  Result: " << (ok ? "PASS" : "FAIL") << "\n" << std::endl;
    }

    // 测试 2: Mul(c1, c2) → Const(c1*c2)
    {
        Graph g;
        auto c1 = g.addConstant(2.5, scalar_desc);
        auto c2 = g.addConstant(4.0, scalar_desc);
        auto mul_id = g.addNode(MulNode{scalar_desc, scalar_desc}, {c1, c2}, scalar_desc);
        g.markOutput(mul_id);

        auto opt = g.canonicalize().eliminateDeadCode();
        std::cout << "  [ConstFold] Mul(2.5,4): " << opt.toString();
        bool ok = opt.nodeCount() == 1 && opt.outputCount() == 1;
        std::cout << "  Result: " << (ok ? "PASS" : "FAIL") << "\n" << std::endl;
    }

    // 测试 3: Add(x, 0) → x (identity rule)
    {
        Graph g;
        auto x = g.addInput(TensorDesc::fromShape({64}));
        auto zero = g.addConstant(0.0, scalar_desc);
        auto add_id = g.addNode(AddNode{TensorDesc::fromShape({64}), scalar_desc}, {x, zero}, TensorDesc::fromShape({64}));
        g.markOutput(add_id);

        auto opt = g.canonicalize().eliminateDeadCode();
        std::cout << "  [ConstFold] Add(x,0): " << opt.toString();
        // 应该只有一个输入+一个输出（Add 被折叠为恒等映射，zero 被 DCE 移除）
        bool ok = opt.inputCount() == 1 && opt.outputCount() == 1 && opt.nodeCount() == 1;
        std::cout << "  Result: " << (ok ? "PASS" : "FAIL") << "\n" << std::endl;
    }

    // 测试 4: 死代码消除 — 未被引用的节点应被移除
    {
        Graph g;
        auto x = g.addInput(TensorDesc::fromShape({64}));
        // 创建一个死节点（不被任何输出引用）
        auto dead = g.addNode(NegNode{TensorDesc::fromShape({64})}, {x}, TensorDesc::fromShape({64}));
        // 输出直接引用输入
        g.markOutput(x);

        std::cout << "  [DCE] Before: " << g.toString();
        auto opt = g.eliminateDeadCode();
        std::cout << "  [DCE] After:  " << opt.toString();
        // 死节点应被移除，只剩输入和输出
        bool ok = opt.nodeCount() == 1;
        std::cout << "  Result: " << (ok ? "PASS" : "FAIL") << "\n" << std::endl;
    }

    // 测试 5: 死代码消除 — 多个输出，部分路径存活
    {
        Graph g;
        auto x = g.addInput(TensorDesc::fromShape({64}));
        auto used = g.addNode(ReLUNode{TensorDesc::fromShape({64})}, {x}, TensorDesc::fromShape({64}));
        auto dead = g.addNode(NegNode{TensorDesc::fromShape({64})}, {x}, TensorDesc::fromShape({64}));
        g.markOutput(used);

        std::cout << "  [DCE] Multi-out Before: " << g.toString();
        auto opt = g.eliminateDeadCode();
        std::cout << "  [DCE] Multi-out After:  " << opt.toString();
        // used 和 x 存活，dead 被移除
        bool ok = opt.nodeCount() == 2; // input + ReLU
        std::cout << "  Result: " << (ok ? "PASS" : "FAIL") << "\n" << std::endl;
    }

    // 测试 6: 常量折叠 + 死代码消除组合
    {
        Graph g;
        auto x = g.addInput(TensorDesc::fromShape({64}));
        auto c1 = g.addConstant(2.0, scalar_desc);
        auto c2 = g.addConstant(3.0, scalar_desc);
        // c1*c2 应该被折叠为 Const(6.0)
        auto mul_id = g.addNode(MulNode{scalar_desc, scalar_desc}, {c1, c2}, scalar_desc);
        // x * Const(6.0) 不能折叠（x 不是常量）
        auto out_id = g.addNode(MulNode{TensorDesc::fromShape({64}), scalar_desc}, {x, mul_id}, TensorDesc::fromShape({64}));
        g.markOutput(out_id);

        std::cout << "  [Pipeline] Before: " << g.toString();
        auto opt = g.canonicalize().eliminateDeadCode();
        std::cout << "  [Pipeline] After:  " << opt.toString();
        // c1 和 c2 被折叠为 Mul(c1,c2)→Const(6.0)，死代码被消除
        // 预期：x(INPUT) + Const(6.0) + Mul(x, 6.0)
        bool ok = opt.nodeCount() == 3; // input + Const(6.0) + Mul
        std::cout << "  Result: " << (ok ? "PASS" : "FAIL") << "\n" << std::endl;
    }

    // 测试 7: 嵌套常量折叠 — Add(Add(c1,c2), c3) → Const(c1+c2+c3)
    {
        Graph g;
        auto c1 = g.addConstant(1.0, scalar_desc);
        auto c2 = g.addConstant(2.0, scalar_desc);
        auto c3 = g.addConstant(3.0, scalar_desc);
        auto inner = g.addNode(AddNode{scalar_desc, scalar_desc}, {c1, c2}, scalar_desc);
        auto outer = g.addNode(AddNode{scalar_desc, scalar_desc}, {inner, c3}, scalar_desc);
        g.markOutput(outer);

        auto opt = g.canonicalize().eliminateDeadCode();
        std::cout << "  [ConstFold] Add(Add(1,2),3): " << opt.toString();
        // 嵌套折叠：inner 先折叠为 Const(3.0)，outer 再折叠为 Const(6.0)
        bool ok = opt.nodeCount() == 1 && opt.outputCount() == 1;
        std::cout << "  Result: " << (ok ? "PASS" : "FAIL") << "\n" << std::endl;
    }

    return 0;
}