/**
 * @file test_c3_compile_merged_pgo.cpp
 * @brief 验证 compileMergedPGO 多层融合 + PGO 异步升级端到端流程
 * @details 覆盖：
 *   1. compileMergedPGO 返回 PGOCompiledKernel（dynamic_cast 验证）
 *   2. 首次 execute 走 Eager 解释执行（与 Eager 调度器结果一致）
 *   3. 触发 O2 异步编译（promote 后 isPromoted=true）
 *   4. O2 编译完成后切换到 O2 merged kernel（结果仍与 Eager 一致）
 *   5. 缓存命中：重复 compileMergedPGO 返回相同 PGOCompiledKernel
 *   6. 顺序版 compileMergedPGOSequential 工作正常
 *
 * @date 2026/8/3
 */

#include <iostream>
#include <cmath>
#include <chrono>
#include <future>
#include <thread>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/Graph.h"
#include "C3/GraphMerger.h"
#include "C3/C3Engine.h"
#include "C3/PGOManager.h"

using namespace ct;
using namespace ct::c3;

static void fillTensor(Tensor& t, const std::vector<float>& vals) {
    float* data = t.data_write<float>();
    size_t n = std::min(vals.size(), t.numel());
    for (size_t i = 0; i < n; ++i) data[i] = vals[i];
}

static bool tensorsAllClose(const Tensor& a, const Tensor& b, float eps = 1e-4f) {
    if (a.shape() != b.shape()) return false;
    const float* pa = a.data_read<float>();
    const float* pb = b.data_read<float>();
    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::fabs(pa[i] - pb[i]) > eps) return false;
    }
    return true;
}

// 构建单层 FC+ReLU 子图：in -> MatMul(w) -> Add(b) -> [ReLU]
static Graph buildFCReLU(const std::vector<size_t>& in_shape,
                          const std::vector<size_t>& w_shape,
                          const std::vector<size_t>& b_shape,
                          const std::vector<size_t>& out_shape,
                          bool add_relu = true) {
    auto in_desc = TensorDesc::fromShape(in_shape);
    auto w_desc = TensorDesc::fromShape(w_shape);
    auto b_desc = TensorDesc::fromShape(b_shape);
    auto out_desc = TensorDesc::fromShape(out_shape);

    Graph g;
    size_t in = g.addInput(in_desc);
    size_t w = g.addInput(w_desc);
    size_t b = g.addInput(b_desc);
    // FC = in @ w：节点 lhs_desc=in_desc, rhs_desc=w_desc，输入顺序 {in, w}
    // 调用方应保证 w_shape = {K, N}（pre-transposed），bias 与 out 的最后一维匹配。
    size_t mm = g.addNode(MatMulNode{in_desc, w_desc}, {in, w}, out_desc);
    size_t add = g.addNode(AddNode{out_desc, b_desc}, {mm, b}, out_desc);
    size_t out_id = add;
    if (add_relu) {
        out_id = g.addNode(ReLUNode{out_desc}, {add}, out_desc);
    }
    g.markOutput(out_id);
    return g;
}

// 用 Eager 调度器逐层执行 MLP，得到参考结果
static Tensor eagerMLP(const std::vector<Graph>& layers,
                        const std::vector<Tensor>& inputs) {
    auto& sched = CtorchScheduler::getInstance();
    // inputs 顺序：[x, w1, b1, w2, b2, ...]
    // 每层 3 个输入：x/w/b（第一层）或 prev/w/b（后续层）
    size_t cur = inputs[0].numel() > 0 ? 0 : 0;  // 当前值在 values 中的"指针"
    // 简化：按层顺序 dispatch
    // layer i：inputs = {prev, w_i, b_i}
    Tensor prev = inputs[0];
    for (size_t i = 0; i < layers.size(); ++i) {
        Tensor w = inputs[1 + 2 * i];
        Tensor b = inputs[2 + 2 * i];
        // FC = prev @ w + b（与 buildFCReLU 节点语义保持一致）
        Tensor mm = sched.dispatch<op::MatMul>(prev, w);
        Tensor sum = sched.dispatch<op::Add>(mm, b);
        // 检查该层是否带 ReLU
        bool has_relu = false;
        for (const auto& node : layers[i].nodes()) {
            if (std::holds_alternative<ReLUNode>(node.op)) {
                has_relu = true;
                break;
            }
        }
        if (has_relu) {
            prev = sched.dispatch<op::ReLU>(sum);
        } else {
            prev = sum;
        }
    }
    return prev;
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();  // 触发 kernel 注册
    C3Engine::getInstance().clearCache();  // 干净状态

    std::cout << "=== C3 compileMergedPGO 多层融合 + PGO 端到端测试 ===" << std::endl;

    int passed = 0, failed = 0;

    // ======================= 测试 1: compileMergedPGO 返回 PGOCompiledKernel =======================
    {
        std::vector<Graph> subs = {
            buildFCReLU({2, 4}, {4, 8}, {8}, {2, 8}, true),
            buildFCReLU({2, 8}, {8, 4}, {4}, {2, 4}, false),
        };
        MergeSpec spec = GraphMerger::makeSequentialSpec(subs);

        auto kernel = C3Engine::getInstance().compileMergedPGO(subs, spec, {});
        if (!kernel) {
            std::cout << "  FAIL: compileMergedPGO returned nullptr\n";
            ++failed;
        } else {
            auto* pgo = dynamic_cast<PGOCompiledKernel*>(kernel.get());
            if (pgo) {
                std::cout << "  PASS: compileMergedPGO 返回 PGOCompiledKernel（首次未升级）\n";
                ++passed;
            } else {
                std::cout << "  FAIL: compileMergedPGO 返回的不是 PGOCompiledKernel\n";
                ++failed;
            }

            // 初始未升级
            if (!pgo->isPromoted()) {
                std::cout << "  PASS: 初始 isPromoted() == false\n";
                ++passed;
            } else {
                std::cout << "  FAIL: 初始 isPromoted() 应为 false\n";
                ++failed;
            }
        }
    }

    // ======================= 测试 2: 端到端 — 解释执行 + 升级到 O2/Ofast =======================
    {
        std::vector<Graph> subs = {
            buildFCReLU({2, 4}, {4, 8}, {8}, {2, 8}, true),
            buildFCReLU({2, 8}, {8, 4}, {4}, {2, 4}, false),
        };
        MergeSpec spec = GraphMerger::makeSequentialSpec(subs);

        // 准备输入：[x, w1, b1, w2, b2]
        // 第一层：[2,4] @ [4,8] + [8] = [2,8]
        // 第二层：[2,8] @ [8,4] + [4] = [2,4]
        Tensor x(ShapeTag{}, {2, 4});
        Tensor w1(ShapeTag{}, {4, 8});
        Tensor b1(ShapeTag{}, {8});
        Tensor w2(ShapeTag{}, {8, 4});
        Tensor b2(ShapeTag{}, {4});
        fillTensor(x,  {0.5f, -1.0f, 2.0f, 0.0f, 0.3f, 0.8f, -0.5f, 1.5f});
        fillTensor(w1, {0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, 0.7f, -0.8f,
                        0.9f, -1.0f, 0.11f, 0.22f, -0.33f, 0.44f, 0.55f, -0.66f,
                        0.77f, -0.88f, 0.99f, -1.11f, 0.12f, 0.23f, -0.34f, 0.45f,
                        0.56f, -0.67f, 0.78f, -0.89f, 0.91f, -1.01f, 0.13f, 0.24f});
        fillTensor(b1, {0.1f, -0.1f, 0.2f, -0.2f, 0.3f, -0.3f, 0.4f, -0.4f});
        fillTensor(w2, {0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, 0.7f, -0.8f,
                        0.9f, -1.0f, 0.11f, 0.22f, -0.33f, 0.44f, 0.55f, -0.66f,
                        0.77f, -0.88f, 0.99f, -1.11f, 0.12f, 0.23f, -0.34f, 0.45f,
                        0.56f, -0.67f, 0.78f, -0.89f, 0.91f, -1.01f, 0.13f, 0.24f});
        fillTensor(b2, {0.1f, -0.1f, 0.2f, -0.2f});

        // Eager 参考
        Tensor eager_out = eagerMLP(subs, {x, w1, b1, w2, b2});

        // PGO 融合编译
        auto kernel = C3Engine::getInstance().compileMergedPGO(subs, spec, {});
        auto* pgo = dynamic_cast<PGOCompiledKernel*>(kernel.get());
        if (!pgo) {
            std::cout << "  FAIL: 端到端 — kernel 不是 PGOCompiledKernel\n";
            ++failed;
        } else {
            // 首次 execute（解释执行）
            auto out1 = kernel->execute({x, w1, b1, w2, b2});
            if (tensorsAllClose(out1[0], eager_out)) {
                std::cout << "  PASS: 首次 execute（Eager 解释）结果与 Eager MLP 一致\n";
                ++passed;
            } else {
                std::cout << "  FAIL: 首次 execute 结果不匹配\n";
                ++failed;
            }

            // 强制同步升级（O2 + Ofast）
            pgo->promote();

            if (pgo->isPromoted()) {
                std::cout << "  PASS: promote() 后 isPromoted() == true\n";
                ++passed;
            } else {
                std::cout << "  FAIL: promote() 后仍未升级\n";
                ++failed;
            }

            if (pgo->o2Kernel()) {
                std::cout << "  PASS: O2 merged kernel 已就绪\n";
                ++passed;
            } else {
                std::cout << "  FAIL: O2 kernel 未生成\n";
                ++failed;
            }

            if (pgo->ofastKernel()) {
                std::cout << "  PASS: Ofast merged kernel 已就绪\n";
                ++passed;
            } else {
                std::cout << "  FAIL: Ofast kernel 未生成\n";
                ++failed;
            }

            // 升级后再次 execute（走 O2/Ofast）
            auto out2 = kernel->execute({x, w1, b1, w2, b2});
            if (tensorsAllClose(out2[0], eager_out)) {
                std::cout << "  PASS: 升级后 execute（走 O2/Ofast merged kernel）结果一致\n";
                ++passed;
            } else {
                std::cout << "  FAIL: 升级后 execute 结果不匹配\n";
                ++failed;
            }
        }
    }

    // ======================= 测试 3: 缓存命中 — 重复调用返回相同 PGOCompiledKernel =======================
    {
        std::vector<Graph> subs = {
            buildFCReLU({2, 3}, {3, 5}, {5}, {2, 5}, true),
            buildFCReLU({2, 5}, {5, 3}, {3}, {2, 3}, false),
        };
        MergeSpec spec = GraphMerger::makeSequentialSpec(subs);

        auto k1 = C3Engine::getInstance().compileMergedPGO(subs, spec, {});
        auto k2 = C3Engine::getInstance().compileMergedPGO(subs, spec, {});

        if (k1.get() == k2.get()) {
            std::cout << "  PASS: 缓存命中 — 重复调用返回相同 kernel 实例\n";
            ++passed;
        } else {
            std::cout << "  FAIL: 缓存未命中 — 两次调用返回不同实例\n";
            ++failed;
        }
    }

    // ======================= 测试 4: compileMergedPGOSequential 简化版 =======================
    {
        std::vector<Graph> subs = {
            buildFCReLU({1, 3}, {3, 4}, {4}, {1, 4}, true),
            buildFCReLU({1, 4}, {4, 2}, {2}, {1, 2}, false),
        };

        Tensor x(ShapeTag{}, {1, 3});
        Tensor w1(ShapeTag{}, {3, 4});
        Tensor b1(ShapeTag{}, {4});
        Tensor w2(ShapeTag{}, {4, 2});
        Tensor b2(ShapeTag{}, {2});
        fillTensor(x,  {0.5f, -1.0f, 0.8f});
        fillTensor(w1, {0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f, -1.0f, 0.11f, 0.22f});
        fillTensor(b1, {0.1f, -0.1f, 0.2f, -0.2f});
        fillTensor(w2, {0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, 0.7f, -0.8f});
        fillTensor(b2, {0.1f, -0.1f});

        Tensor eager_out = eagerMLP(subs, {x, w1, b1, w2, b2});

        auto kernel = C3Engine::getInstance().compileMergedPGOSequential(subs, {});
        auto* pgo = dynamic_cast<PGOCompiledKernel*>(kernel.get());
        if (!pgo) {
            std::cout << "  FAIL: compileMergedPGOSequential 返回非 PGOCompiledKernel\n";
            ++failed;
        } else {
            auto out = kernel->execute({x, w1, b1, w2, b2});
            if (tensorsAllClose(out[0], eager_out)) {
                std::cout << "  PASS: compileMergedPGOSequential execute 结果与 Eager 一致\n";
                ++passed;
            } else {
                std::cout << "  FAIL: compileMergedPGOSequential execute 结果不匹配\n";
                ++failed;
            }
        }
    }

    // ======================= 测试 5: 验证 pgo_mode=false 时不走 PGO 包装 =======================
    {
        std::vector<Graph> subs = {
            buildFCReLU({1, 3}, {3, 4}, {4}, {1, 4}, true),
            buildFCReLU({1, 4}, {4, 2}, {2}, {1, 2}, false),
        };
        MergeSpec spec = GraphMerger::makeSequentialSpec(subs);

        auto kernel = C3Engine::getInstance().compileMerged(subs, spec, {});
        auto* pgo = dynamic_cast<PGOCompiledKernel*>(kernel.get());
        if (pgo == nullptr) {
            std::cout << "  PASS: 默认 pgo_mode=false 时 compileMerged 不走 PGO 包装\n";
            ++passed;
        } else {
            std::cout << "  FAIL: 默认 compileMerged 不应返回 PGOCompiledKernel\n";
            ++failed;
        }
    }

    // ======================= 测试 6: PGOManager 统计反映 merged kernel 注册 =======================
    {
        auto stats_before = PGOManager::getInstance().getStats();
        size_t before = stats_before.total_registered;

        std::vector<Graph> subs = {
            buildFCReLU({2, 2}, {2, 3}, {3}, {2, 3}, true),
            buildFCReLU({2, 3}, {3, 1}, {1}, {2, 1}, false),
        };
        MergeSpec spec = GraphMerger::makeSequentialSpec(subs);

        auto kernel = C3Engine::getInstance().compileMergedPGO(subs, spec, {});
        (void)kernel;  // 触发注册

        auto stats_after = PGOManager::getInstance().getStats();
        if (stats_after.total_registered > before) {
            std::cout << "  PASS: PGOManager 注册数增加（" << before << " → "
                      << stats_after.total_registered << "）\n";
            ++passed;
        } else {
            std::cout << "  FAIL: PGOManager 注册数未增加（" << before
                      << " → " << stats_after.total_registered << "）\n";
            ++failed;
        }
    }

    // ======================= 小结 =======================
    std::cout << "\n结果: " << passed << " passed, " << failed << " failed" << std::endl;
    int ret = failed > 0 ? 1 : 0;
    // 显式 shutdown：等待所有 std::async 后台编译完成（包含 C3Engine + PGOManager），
    // 避免单例析构时后台线程 lock 已析构 mutex。
    C3Engine::getInstance().shutdown();
    // 显式清空缓存：在 main 退出前释放所有持有 MLIR ExecutionEngine 的 kernel，
    // 否则 LLVM 全局析构（GDBJITRegistrationListener 的 recursive_mutex）会在
    // MLIR ExecutionEngine 析构之后发生，导致 lock 已析构 mutex 失败。
    // 这是 LLVM 库 + MLIR ExecutionEngine 共享指针生命周期的已知交互问题。
    C3Engine::getInstance().clearCache();
    return ret;
}
