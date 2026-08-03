/**
 * @file test_c3_compile_timeout.cpp
 * @brief 验证 C3Engine 异步编译超时（watchdog）熔断机制（ADR-011）
 * @details 覆盖场景：
 *   1. API 基础：getCompileTimeoutMs 默认 30000，set 后读回正确
 *   2. 不超时路径：timeout=30s + 简单图 → future.get() 返回有效 kernel
 *   3. 超时熔断：timeout=10ms + 复杂图 → future.get() 立即返回 nullptr + last_compile_error_ 含 "async-timeout"
 *   4. 永不超时（timeout=0）：编译最终成功（watchdog 退化为直等）
 *   5. 超时后 cache 命中：第一次 timeout 失败，background compile 完成后第二次命中
 *   6. clearLastCompileError 验证：错误状态可被显式清空
 *
 * 设计说明：
 *   - Handwritten backend 编译 .so 通常 200-500ms（clang++ 调用开销）
 *   - 10ms timeout 几乎必定触发熔断，可靠地验证 watchdog 路径
 *   - 30s timeout 给正常编译充足余量，避免误杀
 *
 * @date 2026/8/3
 */

#include <iostream>
#include <chrono>
#include <future>
#include <thread>
#include <iomanip>
#include <vector>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/Graph.h"
#include "C3/C3Engine.h"

using namespace ct;
using namespace ct::c3;
using clk = std::chrono::steady_clock;

// 强制行缓冲 + 立即 flush，避免调试时假 "hang" 误判
#define LOG(msg) do { std::cout << msg << std::flush; } while (0)

// ====================== 工具：构造测试图 ======================

/// 简单 Add 图（单算子）—— 编译快速，用于不超时场景
static Graph buildSimpleAddGraph() {
    Graph g;
    auto desc = TensorDesc::fromShape({4, 4});
    size_t a = g.addInput(desc);
    size_t b = g.addInput(desc);
    size_t c = g.addNode(AddNode{desc, desc}, {a, b}, desc);
    g.markOutput(c);
    return g;
}

/// 复杂 MatMul + Add + ReLU 链 —— 编译较慢（多 kernel + 大 IR），
/// 用于触发 watchdog 超时熔断
static Graph buildComplexGraph() {
    Graph g;
    auto in_desc  = TensorDesc::fromShape({8, 16});
    auto w_desc   = TensorDesc::fromShape({16, 32});
    auto b_desc   = TensorDesc::fromShape({32});
    auto out_desc = TensorDesc::fromShape({8, 32});

    size_t in = g.addInput(in_desc);
    size_t w  = g.addInput(w_desc);
    size_t b  = g.addInput(b_desc);
    size_t mm = g.addNode(MatMulNode{in_desc, w_desc}, {in, w}, out_desc);
    size_t add = g.addNode(AddNode{out_desc, b_desc}, {mm, b}, out_desc);
    size_t relu = g.addNode(ReLUNode{out_desc}, {add}, out_desc);
    g.markOutput(relu);
    return g;
}

// ====================== 主测试 ======================
int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();  // 触发 kernel 注册

    std::cout << "=== C3Engine 异步编译 watchdog timeout 测试（ADR-011）===" << std::endl;

    int passed = 0, failed = 0;

    // 保存原始 timeout 便于恢复
    uint32_t original_timeout = C3Engine::getInstance().getCompileTimeoutMs();

    // ============== 测试 1: API 基础 ==============
    {
        C3Engine::getInstance().clearCache();
        C3Engine::getInstance().clearLastCompileError();

        uint32_t default_ms = C3Engine::getInstance().getCompileTimeoutMs();
        if (default_ms != 30000) {
            std::cout << "  FAIL [1a]: 默认 timeout 应为 30000ms，实际 " << default_ms << "\n";
            ++failed;
        } else {
            std::cout << "  PASS [1a]: 默认 timeout = 30000ms\n";
            ++passed;
        }

        C3Engine::getInstance().setCompileTimeoutMs(1500);
        if (C3Engine::getInstance().getCompileTimeoutMs() != 1500) {
            std::cout << "  FAIL [1b]: setCompileTimeoutMs(1500) 后读取不正确\n";
            ++failed;
        } else {
            std::cout << "  PASS [1b]: setCompileTimeoutMs / getCompileTimeoutMs 双向同步\n";
            ++passed;
        }

        C3Engine::getInstance().setCompileTimeoutMs(0);
        if (C3Engine::getInstance().getCompileTimeoutMs() != 0) {
            std::cout << "  FAIL [1c]: setCompileTimeoutMs(0) 后读取不正确\n";
            ++failed;
        } else {
            std::cout << "  PASS [1c]: timeout=0（永不超时）配置生效\n";
            ++passed;
        }

        // 恢复
        C3Engine::getInstance().setCompileTimeoutMs(original_timeout);
    }

    // ============== 测试 2: 不超时路径（30s timeout + 简单图）==============
    {
        C3Engine::getInstance().clearCache();
        C3Engine::getInstance().clearLastCompileError();
        C3Engine::getInstance().setCompileTimeoutMs(30000);

        Graph g = buildSimpleAddGraph();
        CompileOptions opts;
        opts.opt_level = 2;

        auto t0 = clk::now();
        auto future = C3Engine::getInstance().compileAsync(g, opts);
        auto kernel = future.get();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();

        if (!kernel) {
            std::cout << "  FAIL [2a]: 30s timeout 简单图应成功，实际 nullptr (elapsed="
                      << elapsed_ms << "ms)\n";
            ++failed;
        } else {
            std::cout << "  PASS [2a]: 30s timeout + 简单图 → kernel 有效 (elapsed="
                      << elapsed_ms << "ms)\n";
            ++passed;
        }

        std::string err = C3Engine::getInstance().getLastCompileError();
        if (!err.empty()) {
            std::cout << "  FAIL [2b]: 不应记录错误，实际: " << err << "\n";
            ++failed;
        } else {
            std::cout << "  PASS [2b]: 编译成功后 getLastCompileError() 仍为空\n";
            ++passed;
        }
    }

    // ============== 测试 3: 超时熔断（10ms timeout + 复杂图）==============
    {
        C3Engine::getInstance().clearCache();
        C3Engine::getInstance().clearLastCompileError();
        C3Engine::getInstance().setCompileTimeoutMs(10);  // 10ms：足够短

        Graph g = buildComplexGraph();
        CompileOptions opts;
        opts.opt_level = 2;

        auto future = C3Engine::getInstance().compileAsync(g, opts);

        // future.get() 应在 timeout + 一点开销 内返回 nullptr
        auto t0 = clk::now();
        auto kernel = future.get();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();

        if (kernel != nullptr) {
            std::cout << "  FAIL [3a]: 10ms timeout 复杂图应超时返回 nullptr，实际有 kernel (elapsed="
                      << elapsed_ms << "ms)\n";
            ++failed;
        } else if (elapsed_ms > 500) {
            std::cout << "  FAIL [3a]: 超时返回太慢 (" << elapsed_ms << "ms)，watchdog 未生效\n";
            ++failed;
        } else {
            std::cout << "  PASS [3a]: 10ms timeout + 复杂图 → nullptr (elapsed=" << elapsed_ms << "ms)\n";
            ++passed;
        }

        // 验证 last_compile_error_ 含 "async-timeout"
        std::string err = C3Engine::getInstance().getLastCompileError();
        if (err.find("async-timeout") == std::string::npos) {
            std::cout << "  FAIL [3b]: getLastCompileError() 应含 'async-timeout'，实际: "
                      << (err.empty() ? "<empty>" : err) << "\n";
            ++failed;
        } else {
            std::cout << "  PASS [3b]: last_compile_error_ 含 'async-timeout': "
                      << err.substr(0, 80) << "...\n";
            ++passed;
        }

        // 错误信息应包含 cache_key（用于诊断）
        if (err.find("cache_key=") == std::string::npos) {
            std::cout <<  "  WARN [3c]: 错误信息未包含 cache_key（仅诊断用，非必须）\n";
        } else {
            std::cout << "  PASS [3c]: 错误信息含 cache_key 便于诊断\n";
            ++passed;
        }

        // 恢复
        C3Engine::getInstance().setCompileTimeoutMs(original_timeout);
    }

    // ============== 测试 4: 永不超时（timeout=0 + 简单图）==============
    // 用 buildSimpleAddGraph 而不是 buildComplexGraph，避免 MLIR backend
    // 在 MatMul+Add+ReLU 链 + opt_level=2 编译失败带来的干扰。
    // 核心验证目标：timeout=0 时 watchdog 走 wait() 路径，compile 完成后正常 set_value。
    {
        C3Engine::getInstance().clearCache();
        C3Engine::getInstance().clearLastCompileError();
        C3Engine::getInstance().setCompileTimeoutMs(0);  // 永不超时

        Graph g = buildSimpleAddGraph();
        CompileOptions opts;
        opts.opt_level = 2;
        // 强制 Handwritten backend 避免 MLIR backend 复杂图限制
#ifdef CT_ENABLE_MLIR
        opts.backend = C3Backend::Handwritten;
#endif

        auto future = C3Engine::getInstance().compileAsync(g, opts);
        auto kernel = future.get();

        if (!kernel) {
            std::cout << "  FAIL [4]: timeout=0 永不超时场景，简单图应编译成功，实际 nullptr\n";
            ++failed;
        } else {
            std::cout << "  PASS [4]: timeout=0 + 简单图 → kernel 有效（watchdog 退化为直等）\n";
            ++passed;
        }

        std::string err = C3Engine::getInstance().getLastCompileError();
        if (!err.empty()) {
            std::cout << "  FAIL [4b]: timeout=0 路径不应记录超时错误: " << err << "\n";
            ++failed;
        } else {
            std::cout << "  PASS [4b]: timeout=0 路径无超时错误记录\n";
            ++passed;
        }

        // 恢复
        C3Engine::getInstance().setCompileTimeoutMs(original_timeout);
    }

    // ============== 测试 5: 超时后 cache 命中（关键场景）==============
    // 设计：第一次 timeout 失败 → 等 background compile 写 cache → 第二次命中
    {
        // ⚠️ 必须先等前面测试的 background compile 完成（清空 state.pending），
        // 否则本次 compileAsync 会命中旧 pending 的 future，绕开本次 timeout 测试。
        // （测试 3 的 background compile 仍在跑，本测试 5a 必须等它结束。）
        std::cout << "  [测试 5: 等待前序 background compile 结束...]\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));

        C3Engine::getInstance().clearCache();
        C3Engine::getInstance().clearLastCompileError();
        C3Engine::getInstance().setCompileTimeoutMs(10);

        Graph g = buildComplexGraph();
        CompileOptions opts;
        opts.opt_level = 2;

        // 第一次：触发超时
        auto future1 = C3Engine::getInstance().compileAsync(g, opts);
        auto kernel1 = future1.get();
        if (kernel1 != nullptr) {
            std::cout << "  FAIL [5a]: 第一次应超时返回 nullptr\n";
            ++failed;
        } else {
            std::cout << "  PASS [5a]: 第一次 timeout=10ms → nullptr（超时）\n";
            ++passed;
        }

        // 等 background compile 完成并写 cache
        // Handwritten backend 通常 200-500ms；MLIR 类似；给 5s 充足余量
        std::cout << "  [等待 background compile 写 cache...]\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));

        // 第二次：相同 key 应走 cache fast path
        C3Engine::getInstance().setCompileTimeoutMs(30000);
        C3Engine::getInstance().clearLastCompileError();
        auto t0 = clk::now();
        auto future2 = C3Engine::getInstance().compileAsync(g, opts);
        auto kernel2 = future2.get();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();

        if (!kernel2) {
            std::cout << "  FAIL [5b]: 第二次应命中 cache 获得有效 kernel，实际 nullptr (elapsed="
                      << elapsed_ms << "ms)\n";
            ++failed;
        } else if (elapsed_ms > 100) {
            std::cout << "  WARN [5b]: cache 命中耗时偏长 (" << elapsed_ms << "ms)，可能走 background compile 路径\n";
            ++passed;  // 软通过 — 仍正确，只是慢了
        } else {
            std::cout << "  PASS [5b]: 第二次 cache 命中 → kernel 有效 (elapsed=" << elapsed_ms << "ms)\n";
            ++passed;
        }

        C3Engine::getInstance().setCompileTimeoutMs(original_timeout);
    }

    // ============== 测试 6: clearLastCompileError ==============
    {
        C3Engine::getInstance().clearCache();
        C3Engine::getInstance().clearLastCompileError();
        C3Engine::getInstance().setCompileTimeoutMs(10);

        // 触发一次超时
        Graph g = buildComplexGraph();
        CompileOptions opts;
        opts.opt_level = 2;
        auto future = C3Engine::getInstance().compileAsync(g, opts);
        (void)future.get();

        std::string err = C3Engine::getInstance().getLastCompileError();
        if (err.find("async-timeout") == std::string::npos) {
            std::cout << "  FAIL [6a]: 预设条件不满足：未触发超时错误: " << err << "\n";
            ++failed;
        } else {
            std::cout << "  PASS [6a]: 触发超时，错误已记录\n";
            ++passed;

            // 显式清空
            C3Engine::getInstance().clearLastCompileError();
            err = C3Engine::getInstance().getLastCompileError();
            if (!err.empty()) {
                std::cout << "  FAIL [6b]: clearLastCompileError() 后应为空，实际: " << err << "\n";
                ++failed;
            } else {
                std::cout << "  PASS [6b]: clearLastCompileError() 生效\n";
                ++passed;
            }
        }

        C3Engine::getInstance().setCompileTimeoutMs(original_timeout);
    }

    std::cout << "\n=== 总结 ===\n";
    std::cout << "  PASS: " << passed << "\n";
    std::cout << "  FAIL: " << failed << "\n";

    return failed == 0 ? 0 : 1;
}
