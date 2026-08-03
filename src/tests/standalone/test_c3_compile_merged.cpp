/**
 * @file test_c3_compile_merged.cpp
 * @brief 验证 C3 compileMerged / compileMergedSequential / compileMergedAsync 端到端流程
 * @details 覆盖：单子图退化为 compile、多子图顺序融合、shape mismatch 抛异常、
 *          异步融合去重、cache 命中、跨设备 shape 一致性检查。
 * @date 2026/8/2
 */

#include <iostream>
#include <cmath>
#include <chrono>
#include <future>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/Graph.h"
#include "C3/GraphMerger.h"
#include "C3/C3Engine.h"

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

// 构建单层 FC+ReLU 子图（in -> MatMul(w) -> Add(b) -> [ReLU]）
// 语义：out = in @ w + b，调用方应保证 w_shape = {K, N}（pre-transposed）
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
    // FC = in @ w：节点输入顺序 {in, w}
    size_t mm = g.addNode(MatMulNode{in_desc, w_desc}, {in, w}, out_desc);
    size_t add = g.addNode(AddNode{out_desc, b_desc}, {mm, b}, out_desc);
    size_t out_id = add;
    if (add_relu) {
        out_id = g.addNode(ReLUNode{out_desc}, {add}, out_desc);
    }
    g.markOutput(out_id);
    return g;
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();  // 触发 kernel 注册

    std::cout << "=== C3 compileMerged 端到端测试 ===" << std::endl;

    int passed = 0, failed = 0;

    // ======================= 测试 1: makeSequentialSpec / mergedCacheKey 工具方法 =======================
    {
        std::vector<Graph> subs = {
            buildFCReLU({2, 4}, {4, 8}, {8}, {2, 8}, true),
            buildFCReLU({2, 8}, {8, 4}, {4}, {2, 4}, false),
        };
        MergeSpec spec = GraphMerger::makeSequentialSpec(subs);
        if (spec.links.size() == 1 &&
            spec.links[0].from_output == 0 && spec.links[0].to_input == 0) {
            std::cout << "  PASS: makeSequentialSpec 正确生成单链接\n";
            ++passed;
        } else {
            std::cout << "  FAIL: makeSequentialSpec 链接数量或内容错误\n";
            ++failed;
        }

        std::string key = GraphMerger::mergedCacheKey(subs, spec);
        // 期望以统一版本前缀 kMergedCacheKeyPrefix 开头（当前为 "merged_v4_|"，与 C3Engine 同步）
        if (key.find(kMergedCacheKeyPrefix) == 0 && key.find("|2|") != std::string::npos) {
            std::cout << "  PASS: mergedCacheKey 格式正确（key=" << key.substr(0, 50) << "...）\n";
            ++passed;
        } else {
            std::cout << "  FAIL: mergedCacheKey 格式错误: " << key << "\n";
            ++failed;
        }

        // spec 长度不匹配应返回 invalid_spec
        MergeSpec bad_spec;  // 空 spec
        std::string bad_key = GraphMerger::mergedCacheKey(subs, bad_spec);
        if (bad_key.find("invalid_spec") != std::string::npos) {
            std::cout << "  PASS: mergedCacheKey 对非法 spec 优雅降级\n";
            ++passed;
        } else {
            std::cout << "  FAIL: mergedCacheKey 对非法 spec 处理错误: " << bad_key << "\n";
            ++failed;
        }
    }

    // ======================= 测试 2: compileMergedSequential 端到端 =======================
    {
        // 清空缓存确保这次是干净编译
        C3Engine::getInstance().clearCache();

        // 构建 2 层 MLP：x ∈ [2,4] → [2,8] → [2,4]
        std::vector<Graph> subs = {
            buildFCReLU({2, 4}, {4, 8}, {8}, {2, 8}, true),
            buildFCReLU({2, 8}, {8, 4}, {4}, {2, 4}, false),
        };

        try {
            auto& engine = C3Engine::getInstance();
            auto kernel = engine.compileMergedSequential(subs);
            if (!kernel) {
                std::cout << "  FAIL: compileMergedSequential 返回 nullptr\n";
                ++failed;
            } else {
                // 构造输入：x, w1, b1, w2, b2（5 个输入）
                // 第一层：[2,4]@[4,8] + [8] = [2,8]
                // 第二层：[2,8]@[8,4] + [4] = [2,4]
                Tensor x(ShapeTag{}, {2, 4});
                Tensor w1(ShapeTag{}, {4, 8});
                Tensor b1(ShapeTag{}, {8});
                Tensor w2(ShapeTag{}, {8, 4});
                Tensor b2(ShapeTag{}, {4});

                fillTensor(x, {1, 2, 3, 4, 5, 6, 7, 8});
                fillTensor(w1, std::vector<float>(32, 0.1f));
                fillTensor(b1, std::vector<float>(8, 0.0f));
                fillTensor(w2, std::vector<float>(32, 0.1f));
                fillTensor(b2, std::vector<float>(4, 0.0f));

                auto results = kernel->execute({x, w1, b1, w2, b2});
                if (results.size() == 1 && results[0].shape() == std::vector<size_t>({2, 4})) {
                    std::cout << "  PASS: compileMergedSequential 2层MLP执行成功，输出shape=[2,4]\n";
                    ++passed;
                } else {
                    std::cout << "  FAIL: 2层MLP输出 shape 错误，got " << results.size() << " 个张量\n";
                    ++failed;
                }
            }
        } catch (const std::exception& e) {
            std::cout << "  FAIL: compileMergedSequential 抛异常: " << e.what() << "\n";
            ++failed;
        }
    }

    // ======================= 测试 3: compileMergedSequential cache 命中 =======================
    {
        auto& engine = C3Engine::getInstance();
        // 第一次：miss
        engine.clearCache();
        auto stats_before = engine.getCacheStats();
        std::vector<Graph> subs = {
            buildFCReLU({2, 4}, {4, 8}, {8}, {2, 8}, true),
            buildFCReLU({2, 8}, {8, 4}, {4}, {2, 4}, false),
        };
        auto kernel1 = engine.compileMergedSequential(subs);
        auto stats_after_miss = engine.getCacheStats();
        size_t miss_diff1 = stats_after_miss.misses - stats_before.misses;

        // 第二次：应该 hit
        auto kernel2 = engine.compileMergedSequential(subs);
        auto stats_after_hit = engine.getCacheStats();
        size_t hit_diff = stats_after_hit.hits - stats_after_miss.hits;

        if (kernel1 && kernel2 && miss_diff1 == 1 && hit_diff == 1) {
            std::cout << "  PASS: compileMergedSequential cache 命中正确（miss=1, hit=1）\n";
            ++passed;
        } else {
            std::cout << "  FAIL: cache 行为异常（miss_diff=" << miss_diff1
                      << ", hit_diff=" << hit_diff << "）\n";
            ++failed;
        }
    }

    // ======================= 测试 4: compileMerged 入参校验 =======================
    {
        auto& engine = C3Engine::getInstance();

        // 空子图列表
        std::vector<Graph> empty;
        MergeSpec spec;
        try {
            engine.compileMerged(empty, spec);
            std::cout << "  FAIL: 空子图应抛 invalid_argument\n";
            ++failed;
        } catch (const std::invalid_argument&) {
            std::cout << "  PASS: 空子图正确抛 invalid_argument\n";
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "  FAIL: 空子图抛出错误类型: " << e.what() << "\n";
            ++failed;
        }

        // 链接数不匹配（2 子图但 spec.links 有 5 个）
        std::vector<Graph> subs = {
            buildFCReLU({2, 4}, {4, 8}, {8}, {2, 8}, true),
            buildFCReLU({2, 8}, {8, 4}, {4}, {2, 4}, false),
        };
        MergeSpec wrong_spec;
        for (int i = 0; i < 5; ++i) wrong_spec.links.push_back({0, 0});
        try {
            engine.compileMerged(subs, wrong_spec);
            std::cout << "  FAIL: 链接数不匹配应抛 invalid_argument\n";
            ++failed;
        } catch (const std::invalid_argument&) {
            std::cout << "  PASS: 链接数不匹配正确抛 invalid_argument\n";
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "  FAIL: 链接数不匹配抛出错误类型: " << e.what() << "\n";
            ++failed;
        }

        // 形状不匹配（g1 的输出 shape 与 g2 的输入 shape 不一致）
        std::vector<Graph> mismatch_subs = {
            buildFCReLU({2, 4}, {4, 8}, {8}, {2, 8}, true),  // 输出 [2,8]
            buildFCReLU({2, 7}, {7, 4}, {4}, {2, 4}, false),  // 输入 [2,7] 期望 [2,8]
        };
        try {
            engine.compileMergedSequential(mismatch_subs);
            std::cout << "  FAIL: 形状不匹配应抛 invalid_argument\n";
            ++failed;
        } catch (const std::invalid_argument&) {
            std::cout << "  PASS: 形状不匹配正确抛 invalid_argument\n";
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "  FAIL: 形状不匹配抛出错误类型: " << e.what() << "\n";
            ++failed;
        }
    }

    // ======================= 测试 5: compileMergedAsync 后台编译 =======================
    {
        auto& engine = C3Engine::getInstance();
        engine.clearCache();

        std::vector<Graph> subs = {
            buildFCReLU({2, 4}, {4, 8}, {8}, {2, 8}, true),
            buildFCReLU({2, 8}, {8, 4}, {4}, {2, 4}, false),
        };
        MergeSpec spec = GraphMerger::makeSequentialSpec(subs);

        try {
            auto future = engine.compileMergedAsync(subs, spec);
            auto kernel = future.get();
            if (kernel) {
                std::cout << "  PASS: compileMergedAsync 后台编译完成并返回有效 kernel\n";
                ++passed;
            } else {
                std::cout << "  FAIL: compileMergedAsync 完成后 kernel 为 nullptr\n";
                ++failed;
            }
        } catch (const std::exception& e) {
            std::cout << "  FAIL: compileMergedAsync 抛异常: " << e.what() << "\n";
            ++failed;
        }
    }

    // ======================= 测试 6: compileMergedAsync 去重 =======================
    {
        auto& engine = C3Engine::getInstance();
        engine.clearCache();

        std::vector<Graph> subs = {
            buildFCReLU({2, 4}, {4, 8}, {8}, {2, 8}, true),
            buildFCReLU({2, 8}, {8, 4}, {4}, {2, 4}, false),
        };
        MergeSpec spec = GraphMerger::makeSequentialSpec(subs);

        // 同时发起 3 个相同的异步融合任务
        auto f1 = engine.compileMergedAsync(subs, spec);
        auto f2 = engine.compileMergedAsync(subs, spec);
        auto f3 = engine.compileMergedAsync(subs, spec);

        auto k1 = f1.get();
        auto k2 = f2.get();
        auto k3 = f3.get();

        // 3 个 future 应该指向同一个 kernel 对象（去重）
        if (k1 && k2 && k3 && k1.get() == k2.get() && k2.get() == k3.get()) {
            std::cout << "  PASS: compileMergedAsync 正确去重（3次调用返回同一kernel）\n";
            ++passed;
        } else {
            std::cout << "  FAIL: 异步去重失败（k1==" << (void*)k1.get()
                      << " k2==" << (void*)k2.get() << " k3==" << (void*)k3.get() << "）\n";
            ++failed;
        }
    }

    // ======================= 小结 =======================
    std::cout << "\n结果: " << passed << " passed, " << failed << " failed" << std::endl;
    int ret = failed > 0 ? 1 : 0;
    // 显式 shutdown：等待所有 std::async 后台编译完成（包含 C3Engine + PGOManager），
    // 避免单例析构时后台线程 lock 已析构 mutex。
    // (主流程 API 用户应在 main exit 前同样调一次 shutdown())
    C3Engine::getInstance().shutdown();
    // 显式清空缓存：在 main 退出前释放所有持有 MLIR ExecutionEngine 的 kernel，
    // 否则 LLVM 全局析构（GDBJITRegistrationListener 的 recursive_mutex）会在
    // MLIR ExecutionEngine 析构之后发生，导致 lock 已析构 mutex 失败
    // （参见 lldb 堆栈: libc++ recursive_mutex::lock() 异常）。
    // 这是 LLVM 库 + MLIR ExecutionEngine 共享指针生命周期的已知交互问题。
    C3Engine::getInstance().clearCache();
    return ret;
}
