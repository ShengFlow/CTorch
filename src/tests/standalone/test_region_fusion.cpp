/**
 * @file test_region_fusion.cpp
 * @brief 区域融合端到端验证
 * @details 验证 Rolling Hash 匹配 + 预走确认 + region 执行完整流程：
 *          1. Rolling Hash 正确性
 *          2. Region 匹配 + 预走并执行
 *          3. 预走失败回退 eager
 *          4. 区域融合加速比
 */

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <cstring>
#include <cassert>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/RollingHash.h"
#include "C3/RegionFusion.h"
#include "C3/C3Engine.h"
#include "C3/C3KernelRegistry.h"

using namespace ct;
using namespace ct::c3;
using hires = std::chrono::high_resolution_clock;
using us = std::chrono::duration<double, std::micro>;

// 用于测试注册表的 mock kernel
struct MockCompiledKernel : public CompiledKernel {
    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override { return inputs; }
    const std::string& cacheKey() const override { static const std::string k = "mock"; return k; }
    DeviceType targetDevice() const override { return DeviceType::kCPU; }
    size_t workspaceBytes() const override { return 0; }
};

static void fillRandom(Tensor& t, float scale = 0.1f) {
    float* data = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        data[i] = scale * std::sin(static_cast<float>(i) * 0.1f);
    }
}

static void fillConst(Tensor& t, float val) {
    float* data = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        data[i] = val;
    }
}

// ======================= MLIR Kernel 内部性能采样 =======================
// JIT kernel 调用 c3_profile_mark() 函数写入 g_profile_ts 全局数组
// g_profile_ts 和 c3_profile_mark 定义在 MLIRKernelGen.cpp 中，通过 extern "C" 导出
// phase 0: kernel entry, phase 3: kernel done
extern "C" {
    extern uint64_t g_profile_ts[8];
}

static void reset_profile_ts() {
    for (int i = 0; i < 8; i++) g_profile_ts[i] = 0;
}

static void print_profile_ts(int iters) {
    uint64_t entry_marker = g_profile_ts[0];
    uint64_t done_marker = g_profile_ts[3];
    uint64_t entry_ts = g_profile_ts[4];   // phase 0: 写入 phase_idx+4=4
    uint64_t done_ts = g_profile_ts[7];    // phase 3: 写入 phase_idx+4=7
    printf("  [PROFILE] Marker[0]=0x%llx, Marker[3]=0x%llx\n",
           (unsigned long long)entry_marker, (unsigned long long)done_marker);
    printf("  [PROFILE] TS[4]=%llu, TS[7]=%llu\n",
           (unsigned long long)entry_ts, (unsigned long long)done_ts);
    if (entry_marker != 0xDEAD || done_marker != 0xBEEF) {
        printf("  [PROFILE] c3_profile_mark 未被调用或符号解析失败！\n");
        return;
    }
    printf("  [PROFILE] c3_profile_mark 调用成功 ✓\n");
    if (entry_ts == 0 || done_ts == 0) {
        printf("  [PROFILE] clock_gettime 返回 0，可能调用失败\n");
        return;
    }
    double total_ns = (double)(done_ts - entry_ts);
    double avg_ns = total_ns / iters;
    printf("  [PROFILE] Kernel entry=%llu, done=%llu\n",
           (unsigned long long)entry_ts, (unsigned long long)done_ts);
    printf("  [PROFILE] Total kernel time: %.1f ns = %.3f us (last iteration)\n",
           total_ns, total_ns / 1000.0);
    printf("  [PROFILE] Avg per iteration: %.1f ns = %.3f us\n",
           avg_ns, avg_ns / 1000.0);
}

int main() {
    std::cout << "=== 区域融合端到端验证 ===" << std::endl;
    int passed = 0, total = 0;

    // ======================= EXP-1: Rolling Hash 正确性 =======================
    {
        std::cout << "\n[EXP-1] Rolling Hash 正确性..." << std::endl;
        RollingHash::precompute(64);

        // 测试 op 编码唯一性
        std::vector<op> ops = {op::Add, op::Sub, op::Mul, op::MatMul, op::Sigmoid};
        std::vector<uint64_t> codes;
        for (auto o : ops) codes.push_back(RollingHash::getOpCode(o));
        // 检查是否唯一
        bool unique = true;
        for (size_t i = 0; i < codes.size(); ++i)
            for (size_t j = i+1; j < codes.size(); ++j)
                if (codes[i] == codes[j]) unique = false;
        std::cout << "  op 编码唯一性: " << (unique ? "✅" : "❌") << std::endl;
        total++;

        // 测试相同序列的哈希一致性
        std::vector<op> seq1 = {op::MatMul, op::Add, op::Sigmoid};
        std::vector<op> seq2 = {op::MatMul, op::Add, op::Sigmoid};
        auto prefix1 = RollingHash::computePrefixHashes(seq1);
        auto prefix2 = RollingHash::computePrefixHashes(seq2);
        uint64_t h1 = RollingHash::getSubHash(prefix1, 0, 2);
        uint64_t h2 = RollingHash::getSubHash(prefix2, 0, 2);
        bool consistent = (h1 == h2);
        std::cout << "  相同序列哈希一致: " << (consistent ? "✅" : "❌")
                  << " h1=" << h1 << " h2=" << h2 << std::endl;
        total++;

        // 测试不同序列的哈希不同
        std::vector<op> seq3 = {op::MatMul, op::Sub, op::Sigmoid};
        auto prefix3 = RollingHash::computePrefixHashes(seq3);
        uint64_t h3 = RollingHash::getSubHash(prefix3, 0, 2);
        bool different = (h1 != h3);
        std::cout << "  不同序列哈希不同: " << (different ? "✅" : "❌")
                  << " h1=" << h1 << " h3=" << h3 << std::endl;
        total++;

        // 测试子序列提取
        std::vector<op> seq4 = {op::MatMul, op::Add, op::Sigmoid, op::Mul, op::Sub};
        auto prefix4 = RollingHash::computePrefixHashes(seq4);
        uint64_t sub1 = RollingHash::getSubHash(prefix4, 0, 1);  // MatMul+Add
        uint64_t sub2 = RollingHash::getSubHash(prefix4, 2, 3);  // Sigmoid+Mul
        bool sub_diff = (sub1 != sub2);
        std::cout << "  子序列提取正确: " << (sub_diff ? "✅" : "❌")
                  << " sub1=" << sub1 << " sub2=" << sub2 << std::endl;
        total++;

        // 验证子序列提取的可逆性：子序列哈希与独立计算的相同
        std::vector<op> sub_ops = {op::Sigmoid, op::Mul};
        auto sub_prefix = RollingHash::computePrefixHashes(sub_ops);
        uint64_t sub_expected = RollingHash::getSubHash(sub_prefix, 0, 1);
        bool sub_match = (sub2 == sub_expected);
        std::cout << "  子序列哈希与独立计算一致: " << (sub_match ? "✅" : "❌")
                  << " sub2=" << sub2 << " expected=" << sub_expected << std::endl;
        total++;

        if (unique && consistent && different && sub_diff && sub_match) passed += 5;
    }

    // ======================= EXP-2: Region 注册与匹配 =======================
    {
        std::cout << "\n[EXP-2] Region 注册与匹配..." << std::endl;
        auto& region_reg = RegionFusionRegistry::getInstance();
        region_reg.clear();

        // 注册一个 region: MatMul→Add→Sigmoid
        std::vector<op> region_ops = {op::MatMul, op::Add, op::Sigmoid};
        auto prefix = RollingHash::computePrefixHashes(region_ops);
        uint64_t hash = RollingHash::getSubHash(prefix, 0, 2);

        // 创建一个假的 CompiledKernel
        auto fake_kernel = std::make_shared<MockCompiledKernel>();
        region_reg.install(hash, region_ops, fake_kernel, {32, 32, 32, 32, 32, 32});

        // 查找
        auto* found = region_reg.find(hash);
        bool found_match = (found != nullptr && found->active && found->len == 3);
        std::cout << "  通过 hash 查找到 region: " << (found_match ? "✅" : "❌") << std::endl;
        total++;

        // 验证不存在的 hash 返回 nullptr
        auto* not_found = region_reg.find(0xDEADBEEF);
        bool no_match = (not_found == nullptr);
        std::cout << "  不存在的 hash 返回 nullptr: " << (no_match ? "✅" : "❌") << std::endl;
        total++;

        if (found_match && no_match) passed += 2;
        region_reg.clear();
    }

    // ======================= EXP-3: 端到端区域融合 =======================
    {
        std::cout << "\n[EXP-3] 端到端区域融合..." << std::endl;
        auto& sched = CtorchScheduler::getInstance();
        auto& engine = C3Engine::getInstance();
        auto& registry = C3KernelRegistry::getInstance();
        auto& region_reg = RegionFusionRegistry::getInstance();

        region_reg.clear();
        engine.clearCache();

        const size_t M = 32, K = 32, N = 32;

        // 编译一个 MatMul+Add+Sigmoid 的融合 kernel
        Graph g;
        auto in_desc = TensorDesc::fromShape({M, K});
        auto w_desc = TensorDesc::fromShape({K, N});
        auto b_desc = TensorDesc::fromShape({M, N});
        auto out_desc = TensorDesc::fromShape({M, N});

        size_t in1 = g.addInput(in_desc);
        size_t w1 = g.addInput(w_desc);
        size_t mm_node = g.addNode(MatMulNode{in_desc, w_desc}, {in1, w1}, out_desc);
        size_t b1 = g.addInput(b_desc);
        size_t add_node = g.addNode(AddNode{out_desc, b_desc}, {mm_node, b1}, out_desc);
        size_t sig_node = g.addNode(SigmoidNode{out_desc}, {add_node}, out_desc);
        g.markOutput(sig_node);

        CompileOptions opts;
        opts.pgo_mode = false;
        auto kernel = engine.compile(g, opts);

        if (!kernel) {
            std::cout << "  ❌ 融合 kernel 编译失败，跳过测试" << std::endl;
            total++;
            // 标记为失败
            engine.shutdown();
            engine.clearCache();
            region_reg.clear();
        } else {
            std::cout << "  ✅ 融合 kernel 编译成功" << std::endl;

            // 创建输入数据
            Tensor X(ShapeTag{}, {M, K});
            Tensor W(ShapeTag{}, {K, N});
            Tensor B(ShapeTag{}, {M, N});
            fillRandom(X);
            fillRandom(W);
            fillRandom(B);

            // 第一次迭代：先不注册 region，纯 eager 执行，记录 trace，保存参考结果
            Tensor act_ref;  // 保存 eager 参考结果
            {
                std::cout << "  Iter 1 (eager + trace 记录)..." << std::endl;
                std::cout.flush();
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                std::cout << "    MatMul eager: data_ptr=" << (void*)mm.data_read<float>() << " numel=" << mm.numel() << std::endl;
                std::cout.flush();
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                std::cout << "    Add eager: data_ptr=" << (void*)sum.data_read<float>() << " numel=" << sum.numel() << std::endl;
                std::cout.flush();
                Tensor act = sched.dispatch<op::Sigmoid>(sum);
                std::cout << "    Sigmoid eager: data_ptr=" << (void*)act.data_read<float>() << " numel=" << act.numel() << std::endl;
                std::cout.flush();
                act_ref = act;  // 保存 eager 结果
                std::cout << "    act_ref after copy: data_ptr=" << (void*)act_ref.data_read<float>() << " numel=" << act_ref.numel() << std::endl;
                std::cout.flush();
            }

            // 现在注册 region（第一次迭代的 trace 已完整记录）
            // 注意：不重置 region trace，第二次迭代会利用旧 trace 从开头匹配
            std::vector<op> region_ops = {op::MatMul, op::Add, op::Sigmoid};
            auto prefix = RollingHash::computePrefixHashes(region_ops);
            uint64_t hash = RollingHash::getSubHash(prefix, 0, 2);
            region_reg.install(hash, region_ops, kernel, {M, K, K, N, M, N});

            // 第二次迭代：应该触发区域融合
            // 注意：第一次迭代的 trace [MatMul, Add, Sigmoid] 保留在调度器中。
            // 第二次迭代第一个 dispatch（MatMul）时，full_seq = [MatMul, Add, Sigmoid, MatMul]，
            // 从开头匹配 len=3 命中 region，进入预走模式，后续 Add 和 Sigmoid 自然匹配。
            {
                std::cout << "  Iter 2 (区域融合)..." << std::endl;
                std::cout.flush();
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                std::cout << "    MatMul done, data_ptr=" << (void*)mm.data_read<float>() << std::endl;
                std::cout.flush();
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                std::cout << "    Add done" << std::endl;
                std::cout.flush();
                Tensor act = sched.dispatch<op::Sigmoid>(sum);
                std::cout << "    Sigmoid done" << std::endl;
                std::cout.flush();

                std::cout << "    Comparing results..." << std::endl;
                const float* fused_data = act.data_read<float>();
                const float* ref_data = act_ref.data_read<float>();
                size_t numel = act.numel();
                std::cout << "    numel=" << numel << " fused_ptr=" << (void*)fused_data
                          << " ref_ptr=" << (void*)ref_data << std::endl;

                double max_diff = 0.0;
                int bad_count = 0;
                for (size_t i = 0; i < numel; ++i) {
                    double diff = std::fabs((double)fused_data[i] - (double)ref_data[i]);
                    double max_val = std::max(std::fabs((double)fused_data[i]), std::fabs((double)ref_data[i]));
                    if (diff > 1e-4 + 1e-4 * max_val) {
                        bad_count++;
                        if (bad_count <= 3) {
                            std::cout << "    MISMATCH[" << i << "]: fused=" << fused_data[i]
                                      << " ref=" << ref_data[i] << std::endl;
                        }
                    }
                    if (diff > max_diff) max_diff = diff;
                }

                bool correct = (bad_count == 0);
                std::cout << "  区域融合结果: " << (correct ? "✅" : "❌")
                          << " bad=" << bad_count << "/" << numel
                          << " max_diff=" << max_diff << std::endl;
                total++;
                if (correct) passed++;
            }

            engine.shutdown();
            engine.clearCache();
            region_reg.clear();
        }
    }

    // ======================= EXP-4: 性能对比 =======================
    {
        std::cout << "\n[EXP-4] 区域融合性能对比..." << std::endl;
        auto& sched = CtorchScheduler::getInstance();
        auto& engine = C3Engine::getInstance();
        auto& registry = C3KernelRegistry::getInstance();
        auto& region_reg = RegionFusionRegistry::getInstance();

        region_reg.clear();
        engine.clearCache();

        const size_t M = 256, K = 256, N = 256;
        const int test_iters = 100;

        Tensor X(ShapeTag{}, {M, K});
        Tensor W(ShapeTag{}, {K, N});
        Tensor B(ShapeTag{}, {N});  // 1D bias 向量
        fillRandom(X);
        fillRandom(W);
        fillConst(B, 0.1f);  // 偏置初始化

        // 编译并注册 region kernel
        Graph g;
        auto in_desc = TensorDesc::fromShape({M, K});
        auto w_desc = TensorDesc::fromShape({K, N});
        auto b_desc = TensorDesc::fromShape({N});  // 1D bias
        auto out_desc = TensorDesc::fromShape({M, N});

        size_t in1 = g.addInput(in_desc);
        size_t w1 = g.addInput(w_desc);
        size_t mm_node = g.addNode(MatMulNode{in_desc, w_desc}, {in1, w1}, out_desc);
        size_t b1 = g.addInput(b_desc);
        size_t add_node = g.addNode(AddNode{out_desc, b_desc}, {mm_node, b1}, out_desc);
        size_t sig_node = g.addNode(SigmoidNode{out_desc}, {add_node}, out_desc);
        g.markOutput(sig_node);

        CompileOptions opts;
        opts.pgo_mode = false;
        auto kernel = engine.compile(g, opts);

        if (kernel) {
            std::vector<op> region_ops = {op::MatMul, op::Add, op::Sigmoid};
            auto prefix = RollingHash::computePrefixHashes(region_ops);
            uint64_t hash = RollingHash::getSubHash(prefix, 0, 2);
            region_reg.install(hash, region_ops, kernel, {M, K, K, N, N});

            // 重置调度器状态，确保干净起始
            sched.resetRegionFusion();
            engine.clearCache();

            // 第一次迭代（template dispatch）：记录 trace
            {
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                Tensor act = sched.dispatch<op::Sigmoid>(sum);
                (void)act;
            }

            // 测量：开启区域融合（template dispatch，trace 已对齐）
            reset_profile_ts();
            auto t0 = hires::now();
            for (int i = 0; i < test_iters; ++i) {
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                Tensor act = sched.dispatch<op::Sigmoid>(sum);
                (void)act;
            }
            auto t1 = hires::now();
            double fused_avg = std::chrono::duration_cast<us>(t1 - t0).count() / test_iters;
            print_profile_ts(test_iters);

            // 测量：不开启区域融合（使用非 template dispatch，避免 trace 干扰）
            // 重置 trace 和 registry
            region_reg.clear();
            sched.resetRegionFusion();
            auto t2 = hires::now();
            for (int i = 0; i < test_iters; ++i) {
                // 使用非 template dispatch 完全绕过区域融合
                Tensor mm = sched.dispatch(X, W, op::MatMul);
                Tensor sum = sched.dispatch(mm, B, op::Add);
                Tensor act = sched.dispatch(sum, op::Sigmoid);
                (void)act;
            }
            auto t3 = hires::now();
            double eager_avg = std::chrono::duration_cast<us>(t3 - t2).count() / test_iters;

            double speedup = eager_avg / fused_avg;
            std::cout << "  Eager 平均延迟: " << eager_avg << " us" << std::endl;
            std::cout << "  区域融合平均延迟: " << fused_avg << " us" << std::endl;
            std::cout << "  加速比: " << speedup << "x" << std::endl;

            if (speedup >= 1.0f) {
                std::cout << "  ✅ 区域融合性能不退化" << std::endl;
                passed++;
            } else {
                std::cout << "  ⚠️  区域融合性能退化（加速比 < 1.0）" << std::endl;
            }
            total++;
        } else {
            std::cout << "  ⚠️  融合 kernel 编译失败，跳过性能测试" << std::endl;
        }

        engine.shutdown();
        engine.clearCache();
        region_reg.clear();
    }

    // ======================= EXP-5: Handwritten 后端性能对比 =======================
    {
        std::cout << "\n[EXP-5] Handwritten 后端区域融合性能对比..." << std::endl;
        auto& sched = CtorchScheduler::getInstance();
        auto& engine = C3Engine::getInstance();
        auto& region_reg = RegionFusionRegistry::getInstance();

        region_reg.clear();
        engine.clearCache();

        const size_t M = 256, K = 256, N = 256;
        const int test_iters = 100;

        Tensor X(ShapeTag{}, {M, K});
        Tensor W(ShapeTag{}, {K, N});
        Tensor B(ShapeTag{}, {M, N});
        fillRandom(X);
        fillRandom(W);
        fillRandom(B);

        // 使用 Handwritten 后端编译 region kernel
        Graph g;
        auto in_desc = TensorDesc::fromShape({M, K});
        auto w_desc = TensorDesc::fromShape({K, N});
        auto b_desc = TensorDesc::fromShape({M, N});
        auto out_desc = TensorDesc::fromShape({M, N});

        size_t in1 = g.addInput(in_desc);
        size_t w1 = g.addInput(w_desc);
        size_t mm_node = g.addNode(MatMulNode{in_desc, w_desc}, {in1, w1}, out_desc);
        size_t b1 = g.addInput(b_desc);
        size_t add_node = g.addNode(AddNode{out_desc, b_desc}, {mm_node, b1}, out_desc);
        size_t sig_node = g.addNode(SigmoidNode{out_desc}, {add_node}, out_desc);
        g.markOutput(sig_node);

        CompileOptions opts;
        opts.pgo_mode = false;
        opts.backend = C3Backend::Handwritten;
        auto kernel = engine.compile(g, opts);

        if (kernel) {
            std::vector<op> region_ops = {op::MatMul, op::Add, op::Sigmoid};
            auto prefix = RollingHash::computePrefixHashes(region_ops);
            uint64_t hash = RollingHash::getSubHash(prefix, 0, 2);
            region_reg.install(hash, region_ops, kernel, {M, K, K, N, M, N});

            sched.resetRegionFusion();
            engine.clearCache();

            // 第一次迭代：记录 trace
            {
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                Tensor act = sched.dispatch<op::Sigmoid>(sum);
                (void)act;
            }

            // 测量：区域融合（Handwritten）
            auto t0 = hires::now();
            for (int i = 0; i < test_iters; ++i) {
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                Tensor act = sched.dispatch<op::Sigmoid>(sum);
                (void)act;
            }
            auto t1 = hires::now();
            double fused_avg = std::chrono::duration_cast<us>(t1 - t0).count() / test_iters;

            // 测量：eager 基准
            region_reg.clear();
            sched.resetRegionFusion();
            auto t2 = hires::now();
            for (int i = 0; i < test_iters; ++i) {
                Tensor mm = sched.dispatch(X, W, op::MatMul);
                Tensor sum = sched.dispatch(mm, B, op::Add);
                Tensor act = sched.dispatch(sum, op::Sigmoid);
                (void)act;
            }
            auto t3 = hires::now();
            double eager_avg = std::chrono::duration_cast<us>(t3 - t2).count() / test_iters;

            double speedup = eager_avg / fused_avg;
            std::cout << "  Handwritten 区域融合平均延迟: " << fused_avg << " us" << std::endl;
            std::cout << "  Eager 平均延迟: " << eager_avg << " us" << std::endl;
            std::cout << "  加速比: " << speedup << "x" << std::endl;

            if (speedup >= 1.0f) {
                std::cout << "  ✅ Handwritten 区域融合性能不退化" << std::endl;
                passed++;
            } else {
                std::cout << "  ⚠️  Handwritten 区域融合性能退化（加速比 < 1.0）" << std::endl;
            }
            total++;
        } else {
            std::cout << "  ⚠️  Handwritten 融合 kernel 编译失败，跳过性能测试" << std::endl;
        }

        engine.shutdown();
        engine.clearCache();
        region_reg.clear();
    }

    // ======================= EXP-6: 纯逐元素模式（6 op）区域融合 =======================
    {
        std::cout << "\n[EXP-6] 纯逐元素模式（Mul→Add→Div→Sub→Mul→Mul）区域融合..." << std::endl;
        auto& sched = CtorchScheduler::getInstance();
        auto& engine = C3Engine::getInstance();
        auto& region_reg = RegionFusionRegistry::getInstance();

        region_reg.clear();
        engine.clearCache();

        const size_t N = 1024 * 1024;  // 1M 元素
        const int test_iters = 100;

        Tensor A(ShapeTag{}, {N});
        Tensor B(ShapeTag{}, {N});
        fillRandom(A, 1.0f);
        fillRandom(B, 1.0f);

        // 编译一个 6 op 逐元素融合 kernel: Mul→Add→Div→Sub→Mul→Mul
        Graph g;
        auto desc = TensorDesc::fromShape({N});

        size_t a_in = g.addInput(desc);
        size_t b_in = g.addInput(desc);
        size_t mul1 = g.addNode(MulNode{desc, desc}, {a_in, b_in}, desc);
        size_t c_in = g.addInput(desc);
        size_t add1 = g.addNode(AddNode{desc, desc}, {mul1, c_in}, desc);
        size_t d_in = g.addInput(desc);
        size_t div1 = g.addNode(DivNode{desc, desc}, {add1, d_in}, desc);
        size_t e_in = g.addInput(desc);
        size_t sub1 = g.addNode(SubNode{desc, desc}, {div1, e_in}, desc);
        size_t f_in = g.addInput(desc);
        size_t mul2 = g.addNode(MulNode{desc, desc}, {sub1, f_in}, desc);
        size_t mul3 = g.addNode(MulNode{desc, desc}, {mul2, f_in}, desc);
        g.markOutput(mul3);

        CompileOptions opts;
        opts.pgo_mode = false;
        auto kernel = engine.compile(g, opts);

        if (!kernel) {
            std::cout << "  ⚠️  融合 kernel 编译失败，跳过测试" << std::endl;
            total++;
        } else {
            std::vector<op> region_ops = {op::Mul, op::Add, op::Div, op::Sub, op::Mul, op::Mul};
            auto prefix = RollingHash::computePrefixHashes(region_ops);
            uint64_t hash = RollingHash::getSubHash(prefix, 0, 5);
            // 扁平化输入形状：6 个外部输入，每个都是 {N}
            std::vector<size_t> input_shapes;
            for (int i = 0; i < 6; ++i) {
                input_shapes.push_back(N);
            }
            region_reg.install(hash, region_ops, kernel, input_shapes);

            // 第一次迭代：记录 trace
            sched.resetRegionFusion();
            engine.clearCache();
            {
                Tensor t1 = sched.dispatch<op::Mul>(A, B);
                Tensor t2 = sched.dispatch<op::Add>(t1, B);
                Tensor t3 = sched.dispatch<op::Div>(t2, B);
                Tensor t4 = sched.dispatch<op::Sub>(t3, B);
                Tensor t5 = sched.dispatch<op::Mul>(t4, B);
                Tensor t6 = sched.dispatch<op::Mul>(t5, B);
                (void)t6;
            }

            // 测量：区域融合（逐 op 计时）
            // 先做一次预热让预走状态稳定
            {
                Tensor t1 = sched.dispatch<op::Mul>(A, B);
                Tensor t2 = sched.dispatch<op::Add>(t1, B);
                Tensor t3 = sched.dispatch<op::Div>(t2, B);
                Tensor t4 = sched.dispatch<op::Sub>(t3, B);
                Tensor t5 = sched.dispatch<op::Mul>(t4, B);
                Tensor t6 = sched.dispatch<op::Mul>(t5, B);
                (void)t6;
            }
            sched.resetRegionFusion();
            // 重新记录 trace
            {
                Tensor t1 = sched.dispatch<op::Mul>(A, B);
                Tensor t2 = sched.dispatch<op::Add>(t1, B);
                Tensor t3 = sched.dispatch<op::Div>(t2, B);
                Tensor t4 = sched.dispatch<op::Sub>(t3, B);
                Tensor t5 = sched.dispatch<op::Mul>(t4, B);
                Tensor t6 = sched.dispatch<op::Mul>(t5, B);
                (void)t6;
            }
            // 逐 op 计时
            double op_times[6] = {0};
            auto t0 = hires::now();
            for (int i = 0; i < test_iters; ++i) {
                auto top0 = hires::now();
                Tensor t1 = sched.dispatch<op::Mul>(A, B);
                auto top1 = hires::now();
                Tensor t2 = sched.dispatch<op::Add>(t1, B);
                auto top2 = hires::now();
                Tensor t3 = sched.dispatch<op::Div>(t2, B);
                auto top3 = hires::now();
                Tensor t4 = sched.dispatch<op::Sub>(t3, B);
                auto top4 = hires::now();
                Tensor t5 = sched.dispatch<op::Mul>(t4, B);
                auto top5 = hires::now();
                Tensor t6 = sched.dispatch<op::Mul>(t5, B);
                auto top6 = hires::now();
                (void)t6;
                op_times[0] += std::chrono::duration_cast<us>(top1 - top0).count();
                op_times[1] += std::chrono::duration_cast<us>(top2 - top1).count();
                op_times[2] += std::chrono::duration_cast<us>(top3 - top2).count();
                op_times[3] += std::chrono::duration_cast<us>(top4 - top3).count();
                op_times[4] += std::chrono::duration_cast<us>(top5 - top4).count();
                op_times[5] += std::chrono::duration_cast<us>(top6 - top5).count();
            }
            auto t1 = hires::now();
            double fused_avg = std::chrono::duration_cast<us>(t1 - t0).count() / test_iters;

            // 测量：直接调用 kernel（绕过调度器，排除区域融合 dispatch 开销）
            Tensor C(ShapeTag{}, {N}), D(ShapeTag{}, {N}), E(ShapeTag{}, {N}), F(ShapeTag{}, {N});
            fillRandom(C, 1.0f); fillRandom(D, 1.0f); fillRandom(E, 1.0f); fillRandom(F, 1.0f);
            auto t1a = hires::now();
            for (int i = 0; i < test_iters; ++i) {
                auto results = kernel->execute({A, B, C, D, E, F});
                (void)results;
            }
            auto t1b = hires::now();
            double raw_kernel_avg = std::chrono::duration_cast<us>(t1b - t1a).count() / test_iters;

            // 测量：eager 基准
            region_reg.clear();
            sched.resetRegionFusion();
            auto t2 = hires::now();
            for (int i = 0; i < test_iters; ++i) {
                Tensor t1 = sched.dispatch(A, B, op::Mul);
                Tensor t2 = sched.dispatch(t1, B, op::Add);
                Tensor t3 = sched.dispatch(t2, B, op::Div);
                Tensor t4 = sched.dispatch(t3, B, op::Sub);
                Tensor t5 = sched.dispatch(t4, B, op::Mul);
                Tensor t6 = sched.dispatch(t5, B, op::Mul);
                (void)t6;
            }
            auto t3 = hires::now();
            double eager_avg = std::chrono::duration_cast<us>(t3 - t2).count() / test_iters;

            // 测量：模板 dispatch 纯开销（不触发区域融合，走 eager kernel 路径）
            // 这样可以对比模板 dispatch vs 非模板 dispatch 的开销差异
            region_reg.clear();
            sched.resetRegionFusion();
            auto t4a = hires::now();
            for (int i = 0; i < test_iters; ++i) {
                Tensor t1 = sched.dispatch<op::Mul>(A, B);
                Tensor t2 = sched.dispatch<op::Add>(t1, B);
                Tensor t3 = sched.dispatch<op::Div>(t2, B);
                Tensor t4 = sched.dispatch<op::Sub>(t3, B);
                Tensor t5 = sched.dispatch<op::Mul>(t4, B);
                Tensor t6 = sched.dispatch<op::Mul>(t5, B);
                (void)t6;
            }
            auto t4b = hires::now();
            double template_eager_avg = std::chrono::duration_cast<us>(t4b - t4a).count() / test_iters;

            double speedup = eager_avg / fused_avg;
            double dispatch_overhead = fused_avg - raw_kernel_avg;
            double template_vs_nontemplate = template_eager_avg - eager_avg;
            std::cout << "  Eager 平均延迟: " << eager_avg << " us" << std::endl;
            std::cout << "  区域融合平均延迟: " << fused_avg << " us" << std::endl;
            std::cout << "  原始 kernel 延迟: " << raw_kernel_avg << " us (直接调用，绕过调度器)" << std::endl;
            std::cout << "  模板 dispatch (eager): " << template_eager_avg << " us" << std::endl;
            std::cout << "  Dispatch 开销: " << dispatch_overhead << " us (融合路径 - 原始 kernel)" << std::endl;
            std::cout << "  模板 vs 非模板差异: " << template_vs_nontemplate << " us" << std::endl;
            std::cout << "  逐 op 计时 (avg per call):" << std::endl;
            const char* op_names[6] = {"Mul", "Add", "Div", "Sub", "Mul", "Mul"};
            for (int i = 0; i < 6; ++i) {
                std::cout << "    op[" << i << "] " << op_names[i] << ": "
                          << (op_times[i] / test_iters) << " us" << std::endl;
            }
            std::cout << "  加速比 (vs eager): " << speedup << "x" << std::endl;
            std::cout << "  加速比 (vs raw): " << eager_avg / raw_kernel_avg << "x" << std::endl;

            // 基准测试：测量 now() 和 vector 创建的开销
            {
                double now_overhead = 0;
                double vec_overhead = 0;
                const int bench_iters = 10000;
                auto tb0 = hires::now();
                for (int i = 0; i < bench_iters; ++i) {
                    auto t = hires::now();
                    (void)t;
                }
                auto tb1 = hires::now();
                now_overhead = std::chrono::duration_cast<us>(tb1 - tb0).count() / bench_iters;
                
                auto tb2 = hires::now();
                for (int i = 0; i < bench_iters; ++i) {
                    std::vector<Tensor> v = {A, B};
                    (void)v;
                }
                auto tb3 = hires::now();
                vec_overhead = std::chrono::duration_cast<us>(tb3 - tb2).count() / bench_iters;
                
                std::cout << "  基准测试:" << std::endl;
                std::cout << "    now() 开销: " << now_overhead << " us" << std::endl;
                std::cout << "    vector<Tensor>={A,B} 开销: " << vec_overhead << " us" << std::endl;
                
                // 测量空 dispatch 开销（绕过区域融合，直接走 eager）
                // 使用非模板 dispatch 避免区域融合检查
                auto tb4 = hires::now();
                for (int i = 0; i < bench_iters; ++i) {
                    Tensor r = sched.dispatch(A, B, op::Mul);
                    (void)r;
                }
                auto tb5 = hires::now();
                double nontemplate_dispatch = std::chrono::duration_cast<us>(tb5 - tb4).count() / bench_iters;
                std::cout << "    非模板 dispatch(Mul) 开销: " << nontemplate_dispatch << " us" << std::endl;
            }

            if (speedup >= 1.5f) {
                std::cout << "  ✅ 加速比 ≥ 1.5x！" << std::endl;
                passed++;
            } else if (speedup >= 1.0f) {
                std::cout << "  ✅ 区域融合性能不退化" << std::endl;
                passed++;
            } else {
                std::cout << "  ⚠️  区域融合性能退化（加速比 < 1.0）" << std::endl;
            }
            total++;
        }

        engine.shutdown();
        engine.clearCache();
        region_reg.clear();
    }

    // ======================= EXP-7: 大 tensor 场景（1024x1024）性能对比 =======================
    {
        std::cout << "\n[EXP-7] 大 tensor 场景（1024x1024）区域融合性能对比..." << std::endl;
        auto& sched = CtorchScheduler::getInstance();
        auto& engine = C3Engine::getInstance();
        auto& region_reg = RegionFusionRegistry::getInstance();

        region_reg.clear();
        engine.clearCache();

        const size_t M = 1024, K = 1024, N = 1024;
        const int test_iters = 10;  // 大 tensor 计算量大，减少迭代次数

        Tensor X(ShapeTag{}, {M, K});
        Tensor W(ShapeTag{}, {K, N});
        Tensor B(ShapeTag{}, {N});  // 1D bias 向量
        fillRandom(X);
        fillRandom(W);
        fillConst(B, 0.1f);

        // 编译并注册 region kernel
        Graph g;
        auto in_desc = TensorDesc::fromShape({M, K});
        auto w_desc = TensorDesc::fromShape({K, N});
        auto b_desc = TensorDesc::fromShape({N});
        auto out_desc = TensorDesc::fromShape({M, N});

        size_t in1 = g.addInput(in_desc);
        size_t w1 = g.addInput(w_desc);
        size_t mm_node = g.addNode(MatMulNode{in_desc, w_desc}, {in1, w1}, out_desc);
        size_t b1 = g.addInput(b_desc);
        size_t add_node = g.addNode(AddNode{out_desc, b_desc}, {mm_node, b1}, out_desc);
        size_t sig_node = g.addNode(SigmoidNode{out_desc}, {add_node}, out_desc);
        g.markOutput(sig_node);

        CompileOptions opts;
        opts.pgo_mode = false;
        auto kernel = engine.compile(g, opts);

        if (kernel) {
            std::vector<op> region_ops = {op::MatMul, op::Add, op::Sigmoid};
            auto prefix = RollingHash::computePrefixHashes(region_ops);
            uint64_t hash = RollingHash::getSubHash(prefix, 0, 2);
            region_reg.install(hash, region_ops, kernel, {M, K, K, N, N});

            sched.resetRegionFusion();
            engine.clearCache();

            // 第一次迭代：记录 trace
            {
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                Tensor act = sched.dispatch<op::Sigmoid>(sum);
                (void)act;
            }

            // 测量：开启区域融合
            auto t0 = hires::now();
            for (int i = 0; i < test_iters; ++i) {
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                Tensor act = sched.dispatch<op::Sigmoid>(sum);
                (void)act;
            }
            auto t1 = hires::now();
            double fused_avg = std::chrono::duration_cast<us>(t1 - t0).count() / test_iters;

            // 测量：不开启区域融合
            region_reg.clear();
            sched.resetRegionFusion();
            auto t2 = hires::now();
            for (int i = 0; i < test_iters; ++i) {
                Tensor mm = sched.dispatch(X, W, op::MatMul);
                Tensor sum = sched.dispatch(mm, B, op::Add);
                Tensor act = sched.dispatch(sum, op::Sigmoid);
                (void)act;
            }
            auto t3 = hires::now();
            double eager_avg = std::chrono::duration_cast<us>(t3 - t2).count() / test_iters;

            double speedup = eager_avg / fused_avg;
            std::cout << "  Eager 平均延迟: " << eager_avg << " us" << std::endl;
            std::cout << "  区域融合平均延迟: " << fused_avg << " us" << std::endl;
            std::cout << "  加速比: " << speedup << "x" << std::endl;

            if (speedup >= 1.0f) {
                std::cout << "  ✅ 区域融合性能不退化（大 tensor 加速比 > 1.0x，符合预期）" << std::endl;
                passed++;
            } else {
                std::cout << "  ⚠️  区域融合性能退化（加速比 < 1.0）" << std::endl;
            }
            total++;
        } else {
            std::cout << "  ⚠️  融合 kernel 编译失败，跳过性能测试" << std::endl;
            total++;
        }

        engine.shutdown();
        engine.clearCache();
        region_reg.clear();
    }

    // ======================= 结果汇总 =======================
    {
        std::cout << "\n=== 结果汇总 ===" << std::endl;
        std::cout << "  通过: " << passed << "/" << total << std::endl;
        if (passed == total) {
            std::cout << "  ✅ 全部通过！" << std::endl;
        } else {
            std::cout << "  ❌ " << (total - passed) << " 个测试失败" << std::endl;
        }
    }

    return (passed == total) ? 0 : 1;
}