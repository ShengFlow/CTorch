/**
 * @file C3HotPathManager.h
 * @generation SHARED 跨代热路径检测与编译触发
 * @brief 热路径自动 C3 编译管理器
 * @details 在调度器 dispatch 路径中自动检测热路径，
 *          触发后台 C3 编译，编译完成后自动安装到 C3KernelRegistry。
 *
 *          ### 工作流程
 *
 *          ```
 *          dispatch(op, a, b)
 *            ├─ 1. 查 C3KernelRegistry → 命中 → 执行 C3 kernel
 *            ├─ 2. 记录调用 → call_count++
 *            ├─ 3. call_count >= hot_threshold ?
 *            │     ├─ 是 → 提交异步 C3 编译
 *            │     │      └─ 编译完成 → 自动 install 到 registry
 *            │     └─ 否 → 继续
 *            └─ 4. 执行 eager kernel
 *          ```
 *
 *          ### 节流策略
 *          - 同一 (op, shape) 在 hot_threshold 次调用后触发编译
 *          - 编译触发后进入 cooldown_sec 冷却期
 *          - 待编译队列达到 max_pending 时进入背压状态
 *
 * @date 2026/08/03
 */

#ifndef CTORCH_C3_C3_HOT_PATH_MANAGER_H
#define CTORCH_C3_C3_HOT_PATH_MANAGER_H

#include "C3/C3KernelRegistry.h"
#include "C3/C3Engine.h"
#include "C3/Graph.h"
#include "C3/RegionFusion.h"
#include "C3/C3Config.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "Ctools.h"
#include "Tensor.h"

namespace ct {
namespace c3 {

// ============================================================
// C3HotPathManager 配置
// ============================================================

struct HotPathConfig {
    /// 触发编译的调用次数阈值（达到此值后触发异步编译）
    size_t hot_threshold = 5;
    /// 同一 key 编译后的冷却期（秒），避免重复触发
    size_t cooldown_sec = 60;
    /// 最大待编译任务数（背压阈值）
    size_t max_pending = 16;
    /// 编译超时时间（ms），传递给 C3Engine
    uint32_t compile_timeout_ms = 30000;
    /// 是否启用日志输出
    bool verbose = false;
};

// ============================================================
// C3HotPathManager
// ============================================================

class C3HotPathManager {
public:
    static C3HotPathManager& instance() {
        static C3HotPathManager mgr;
        return mgr;
    }

    // ======================= 配置 =======================

    void configure(const HotPathConfig& cfg) {
        std::lock_guard<std::mutex> lk(cfg_mutex_);
        config_ = cfg;
    }

    HotPathConfig getConfig() const {
        std::lock_guard<std::mutex> lk(cfg_mutex_);
        return config_;
    }

    // ======================= 统计 =======================

    struct Stats {
        size_t calls_tracked = 0;         ///< 总记录调用次数
        size_t compilations_triggered = 0; ///< 已触发的编译次数
        size_t pending_compiles = 0;       ///< 当前待编译数
        size_t cooldown_hits = 0;          ///< 冷却期内被忽略的次数
        size_t backpressure_hits = 0;      ///< 背压被忽略的次数
    };

    Stats getStats() const {
        Stats s;
        s.calls_tracked = calls_tracked_.load(std::memory_order_relaxed);
        s.compilations_triggered = compilations_triggered_.load(std::memory_order_relaxed);
        s.pending_compiles = pending_compiles_.load(std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mutex_);
            s.cooldown_hits = cooldown_hits_;
            s.backpressure_hits = backpressure_hits_;
        }
        return s;
    }

    /// 单次 dispatch 记录（用于测试直接构造图）
    struct DispatchRecord {
        op op_type;
        std::vector<size_t> shape;      ///< 拼接形状 lhs+rhs（融合检测用，兼容旧逻辑）
        std::vector<size_t> lhs_shape;  ///< 实际左输入形状（单 kernel 安装用）
        std::vector<size_t> rhs_shape;  ///< 实际右输入形状（单 kernel 安装用，unary 为空）
        std::chrono::steady_clock::time_point timestamp;
    };

    // ======================= 生命周期管理 =======================

    /// 等待所有待处理的编译任务完成（必须在 main 退出前调用）
    /// 调用后不再接受新的编译提交（新的 recordCall 只计数不触发编译）。
    /// 应在 C3Engine::shutdown() / clearCache() 之前调用，
    /// 因为 HotPathManager 的后台任务会调用 C3Engine 编译接口。
    /// 完整退出序列：HotPathManager::shutdown() → C3Engine::shutdown() → C3Engine::clearCache()
    void shutdown() {
        // 1. 设置关闭标志，阻止新任务提交
        shutting_down_.store(true, std::memory_order_release);

        // 2. 取出所有待完成的 future
        std::vector<std::future<void>> futures;
        {
            std::lock_guard<std::mutex> lk(futures_mutex_);
            futures = std::move(pending_futures_);
        }

        // 3. 逐个等待，超时 30s（与 C3Engine/PGOManager 保持一致）
        for (auto& f : futures) {
            if (f.valid()) {
                auto st = f.wait_for(std::chrono::seconds(30));
                if (st == std::future_status::ready) {
                    try { f.get(); } catch (...) {}
                } else {
                    // 30s 超时仍未完成，静默放弃（避免阻塞退出）
                    // 注意：超时后 future 被销毁但其线程可能仍在运行，
                    // 这是已知的 trade-off —— 总比 detach 后完全失控好。
                    CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL,
                        ErrorType::UNKNOWN,
                        "C3HotPathManager::shutdown: background compile did not finish in 30s, "
                        "future abandoned (may cause UAF if main exits before thread finishes)");
                }
            }
        }
    }

    /// 查询是否处于关闭状态
    bool isShuttingDown() const {
        return shutting_down_.load(std::memory_order_acquire);
    }

    /// 等待所有待处理的编译任务完成（用于测试场景）
    void waitForPendingCompiles() {
        while (pending_compiles_.load(std::memory_order_acquire) > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // 额外等待确保编译任务完全完成并注册
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // ======================= 核心接口 =======================

    /**
     * @brief 记录一次调用并检查是否需要触发编译
     * @param op_type 算子类型
     * @param dev 目标设备
     * @param lhs_shape 左输入形状（unary 用 a.shape()）
     * @param rhs_shape 右输入形状（unary 省略）
     */
    void recordCall(op op_type, DeviceType dev, const std::vector<size_t>& lhs_shape,
                    const std::vector<size_t>& rhs_shape = {},
                    bool in_autograd = false) {
        // M1 1.4 (2026-08-09): 热路径检测总开关短路
        // hotPathTrackingEnabled() 关闭时 (CT_C3_DISABLE_HOTPATH 编译宏
        // 或 C3_DISABLE_HOTPATH=1 运行时 env) recordCall 直接 O(1) 退出,
        // 跳过 fprintf debug log / 双重 mutex / RingBuffer 写 / 编译触发
        // (C3Config.h 的 hotPathTrackingEnabled() 自身有 static cache,
        //  第一次调用后是 O(1) 函数返回, 0 额外开销)
        if (!ct::c3::hotPathTrackingEnabled()) return;

        calls_tracked_.fetch_add(1, std::memory_order_relaxed);

        if (dev == DeviceType::kMPS) return; // MPS 暂不纳入 C3 编译

        // shutdown 后不再触发新的编译任务，只更新计数
        if (shutting_down_.load(std::memory_order_acquire)) {
            return;
        }

        // 拼接 shape（epoch 用：融合检测 + 编译触发 key）
        std::vector<size_t> shape = lhs_shape;
        shape.insert(shape.end(), rhs_shape.begin(), rhs_shape.end());

        // 记录到 RingBuffer，用于融合检测
        {
            std::lock_guard<std::mutex> lk(rb_mutex_);
            DispatchRecord rec{op_type, shape, lhs_shape, rhs_shape,
                               std::chrono::steady_clock::now()};
            recent_dispatches_.push_back(rec);
            if (recent_dispatches_.size() > kMaxRingBufferSize) {
                recent_dispatches_.pop_front();
            }
        }

        // [Dev 2026-08-09 inAutogradScope 短路] 训练态跳过单 kernel 编译触发
        //   训练 (autograd scope) 时单 kernel 编译 ROI 极低: kernel 编译完可能下一 epoch
        //   shape 就变了/反向走完一次就丢. 仍保留 RingBuffer 写入 (tryFuseRecentDispatches
        //   region fusion 检测需要历史 trace).
        //   实测 53848 dispatch/epoch × mutex_ 锁 + entries_[key] hash ≈ ~200ms/epoch 浪费.
        if (in_autograd) {
            HotPathConfig cfg = getConfig();
            tryFuseRecentDispatches(dev, cfg);
            return;
        }

        size_t key = hashShapeKey(shape, op_type, dev);

        HotPathConfig cfg = getConfig();

        bool should_compile = false;

        {
            std::lock_guard<std::mutex> lk(mutex_);

            auto& entry = entries_[key];
            entry.call_count++;
            // [Dev] v0.5.2+ (2026-08-09): 2 个 fprintf debug log 包到 #ifdef CT_DEBUG
            // release build (NDEBUG) 自动从 CT_DEBUG 推导 OFF,0 成本
            // 实测每个 fprintf 含 stderr flush 1-2us,53848 dispatch/epoch 省 ~200ms/epoch
            #ifdef CT_DEBUG
            fprintf(stderr, "[DBG] recordCall ENTRY op=%d shape_size=%zu cc=%zu compiling=%d pending=%zu\n",
                    (int)op_type, shape.size(), entry.call_count, (int)entry.compiling,
                    pending_compiles_.load(std::memory_order_relaxed));
            #endif

            // 检查是否在冷却期
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - entry.last_compile_time).count();
            if (elapsed < static_cast<decltype(elapsed)>(cfg.cooldown_sec)) {
                cooldown_hits_++;
                return;
            }

            // 检查背压
            if (pending_compiles_.load(std::memory_order_relaxed) >= cfg.max_pending) {
                backpressure_hits_++;
                return;
            }

            // 达到阈值，标记需要编译（在锁外执行，避免死锁）
            if (entry.call_count >= cfg.hot_threshold && !entry.compiling) {
                entry.compiling = true;
                entry.call_count = 0;
                pending_compiles_.fetch_add(1, std::memory_order_relaxed);
                compilations_triggered_.fetch_add(1, std::memory_order_relaxed);
                should_compile = true;
                #ifdef CT_DEBUG
                fprintf(stderr, "[DBG] recordCall TRIGGER op=%d shape_size=%zu cc=%zu\n",
                        (int)op_type, shape.size(), entry.call_count);
                #endif
            }
        } // mutex_ 释放

        if (should_compile) {
            // 先检查是否有可融合的多算子模式
            tryFuseRecentDispatches(dev, cfg);

            // 异步提交单算子编译任务
            submitCompileAsync(op_type, dev, shape, lhs_shape, rhs_shape, key, cfg);
        }
    }

    /**
     * @brief 清除所有跟踪状态
     */
    void clear() {
        std::lock_guard<std::mutex> lk(mutex_);
        entries_.clear();
        std::lock_guard<std::mutex> rblk(rb_mutex_);
        recent_dispatches_.clear();
        calls_tracked_.store(0, std::memory_order_relaxed);
        compilations_triggered_.store(0, std::memory_order_relaxed);
        pending_compiles_.store(0, std::memory_order_relaxed);
        cooldown_hits_ = 0;
        backpressure_hits_ = 0;
    }

private:
    C3HotPathManager() = default;
    ~C3HotPathManager() {
        // 防御性清理：若 shutdown() 未被显式调用，尝试安全清理 pending futures
        // 避免静态析构期访问已销毁对象（如 LLVM GDBJITRegistrationListener 的 mutex）
        shutting_down_.store(true, std::memory_order_release);
        std::vector<std::future<void>> futures;
        {
            std::lock_guard<std::mutex> lk(futures_mutex_);
            futures = std::move(pending_futures_);
        }
        for (auto& f : futures) {
            if (f.valid()) {
                auto st = f.wait_for(std::chrono::milliseconds(100));
                if (st == std::future_status::ready) {
                    try { f.get(); } catch (...) {}
                }
                // 超时则静默放弃，避免阻塞退出
            }
        }
    }
    C3HotPathManager(const C3HotPathManager&) = delete;
    C3HotPathManager& operator=(const C3HotPathManager&) = delete;

    // ======================= 内部结构 =======================

    struct HotEntry {
        size_t call_count = 0;
        bool compiling = false;
        std::chrono::steady_clock::time_point last_compile_time;
    };

    static constexpr size_t kMaxRingBufferSize = 32;

    // ======================= 哈希工具 =======================

    /// 计算 (shape, op_type, dev) 的哈希 key
    static size_t hashShapeKey(const std::vector<size_t>& shape,
                               op op_type, DeviceType dev) {
        size_t h = static_cast<size_t>(op_type) ^ (static_cast<size_t>(dev) << 8);
        h ^= 0x9e3779b9 + (h << 6) + (h >> 2);
        for (auto s : shape) {
            h ^= s + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }

    // ======================= Graph 构建 =======================

    /// 构建单算子 Graph
    static Graph buildGraphForOp(op op_type, const std::vector<size_t>& shape,
                                 const std::vector<size_t>& lhs_shape,
                                 const std::vector<size_t>& rhs_shape) {
        Graph g;

        switch (op_type) {
        case op::Sigmoid: {
            auto desc = TensorDesc::fromShape(lhs_shape);
            size_t in = g.addInput(desc);
            size_t out = g.addNode(SigmoidNode{desc}, {in}, desc);
            g.markOutput(out);
            break;
        }
        case op::Tanh: {
            auto desc = TensorDesc::fromShape(lhs_shape);
            size_t in = g.addInput(desc);
            size_t out = g.addNode(TanhNode{desc}, {in}, desc);
            g.markOutput(out);
            break;
        }
        case op::ReLU: {
            auto desc = TensorDesc::fromShape(lhs_shape);
            size_t in = g.addInput(desc);
            size_t out = g.addNode(ReLUNode{desc}, {in}, desc);
            g.markOutput(out);
            break;
        }
        // 二元算子：包含两个输入，使用真实 lhs/rhs 形状
        case op::Add: {
            TensorDesc lhs_desc = TensorDesc::fromShape(lhs_shape);
            TensorDesc rhs_desc = TensorDesc::fromShape(rhs_shape.empty() ? lhs_shape : rhs_shape);
            size_t a = g.addInput(lhs_desc);
            size_t b = g.addInput(rhs_desc);
            size_t out = g.addNode(AddNode{lhs_desc, rhs_desc}, {a, b}, lhs_desc);
            g.markOutput(out);
            break;
        }
        case op::Sub: {
            TensorDesc lhs_desc = TensorDesc::fromShape(lhs_shape);
            TensorDesc rhs_desc = TensorDesc::fromShape(rhs_shape.empty() ? lhs_shape : rhs_shape);
            size_t a = g.addInput(lhs_desc);
            size_t b = g.addInput(rhs_desc);
            size_t out = g.addNode(SubNode{lhs_desc, rhs_desc}, {a, b}, lhs_desc);
            g.markOutput(out);
            break;
        }
        case op::Mul: {
            TensorDesc lhs_desc = TensorDesc::fromShape(lhs_shape);
            TensorDesc rhs_desc = TensorDesc::fromShape(rhs_shape.empty() ? lhs_shape : rhs_shape);
            size_t a = g.addInput(lhs_desc);
            size_t b = g.addInput(rhs_desc);
            size_t out = g.addNode(MulNode{lhs_desc, rhs_desc}, {a, b}, lhs_desc);
            g.markOutput(out);
            break;
        }
        case op::Div: {
            TensorDesc lhs_desc = TensorDesc::fromShape(lhs_shape);
            TensorDesc rhs_desc = TensorDesc::fromShape(rhs_shape.empty() ? lhs_shape : rhs_shape);
            size_t a = g.addInput(lhs_desc);
            size_t b = g.addInput(rhs_desc);
            size_t out = g.addNode(DivNode{lhs_desc, rhs_desc}, {a, b}, lhs_desc);
            g.markOutput(out);
            break;
        }
        case op::Neg: {
            auto desc = TensorDesc::fromShape(lhs_shape);
            size_t in = g.addInput(desc);
            size_t out = g.addNode(NegNode{desc}, {in}, desc);
            g.markOutput(out);
            break;
        }
        case op::MatMul: {
            // MatMul: shape 是 {M, K, K, N} 格式
            // 拆分为 lhs(M,K), rhs(K,N), out(M,N)
            if (shape.size() >= 4) {
                size_t M = shape[0], K1 = shape[1], K2 = shape[2], N = shape[3];
                if (K1 == K2) {
                    TensorDesc lhs_desc = TensorDesc::fromShape({M, K1});
                    TensorDesc rhs_desc = TensorDesc::fromShape({K2, N});
                    TensorDesc out_desc = TensorDesc::fromShape({M, N});

                    size_t a = g.addInput(lhs_desc);
                    size_t b = g.addInput(rhs_desc);
                    size_t out = g.addNode(MatMulNode{lhs_desc, rhs_desc}, {a, b}, out_desc);
                    g.markOutput(out);
                }
            }
            break;
        }
        default:
            // 不支持的操作，返回空图
            break;
        }

        return g;
    }

    /// 从 DispatchRecord 创建 NodeVariant
    static NodeVariant makeNodeVariant(const DispatchRecord& rec) {
        auto desc = TensorDesc::fromShape(rec.shape);
        switch (rec.op_type) {
        case op::Add: {
            // shape_sig = lhs_shape + rhs_shape（拼接）
            // 提取 lhs: 前半部分，rhs: 后半部分
            auto out_shape = extractBinOpOutShape(rec.shape);
            auto lhs = TensorDesc::fromShape(out_shape);
            std::vector<size_t> rhs_shape;
            if (rec.shape.size() > out_shape.size()) {
                rhs_shape.assign(rec.shape.begin() + (ptrdiff_t)out_shape.size(), rec.shape.end());
            } else {
                rhs_shape = out_shape;
            }
            auto rhs = TensorDesc::fromShape(rhs_shape);
            return AddNode{lhs, rhs};
        }
        case op::Sub:
            return SubNode{desc, desc};
        case op::Mul:
            return MulNode{desc, desc};
        case op::Div:
            return DivNode{desc, desc};
        case op::MatMul: {
            if (rec.shape.size() >= 4) {
                auto lhs = TensorDesc::fromShape({rec.shape[0], rec.shape[1]});
                auto rhs = TensorDesc::fromShape({rec.shape[2], rec.shape[3]});
                return MatMulNode{lhs, rhs};
            }
            return MatMulNode{desc, desc};
        }
        case op::Neg:
            return NegNode{desc};
        case op::ReLU:
            return ReLUNode{desc};
        case op::Tanh:
            return TanhNode{desc};
        case op::Sigmoid:
            return SigmoidNode{desc};
        default:
            return SigmoidNode{desc}; // fallback
        }
    }

    /// 判断是否为 C3 支持的算子
    static bool isSupportedOp(op op_type) {
        // [位掩码 2026-08-11] C3 支持算子集合 → 单 uint64 位掩码, O(1) 查表,
        // 消除 switch 的分支预测失败惩罚 (submitCompileAsync 每次触发时调用).
        static_assert(static_cast<size_t>(op::kCount) <= 64,
                      "op::kCount exceeds uint64 bitmask capacity");
        static constexpr uint64_t kSupportedMask =
              (1ull << static_cast<size_t>(op::Add))
            | (1ull << static_cast<size_t>(op::Sub))
            | (1ull << static_cast<size_t>(op::Mul))
            | (1ull << static_cast<size_t>(op::Div))
            | (1ull << static_cast<size_t>(op::MatMul))
            | (1ull << static_cast<size_t>(op::Neg))
            | (1ull << static_cast<size_t>(op::ReLU))
            | (1ull << static_cast<size_t>(op::Tanh))
            | (1ull << static_cast<size_t>(op::Sigmoid));
        return (kSupportedMask >> static_cast<size_t>(op_type)) & 1ull;
    }

    // ======================= 异步编译 =======================

    /// 尝试融合最近的 dispatch 序列，检测多算子模式
    void tryFuseRecentDispatches(DeviceType dev, const HotPathConfig& cfg) {
        std::lock_guard<std::mutex> lk(rb_mutex_);
        if (recent_dispatches_.size() < 3) return; // 至少需要 3 个 dispatch

        // 检查最近的序列是否匹配已知融合模式
        // 模式1: MatMul + Add + Sigmoid (FCWithActivation)
        // 模式2: MatMul + Add + ReLU (FCWithReLU)
        auto checkPattern = [&](const std::deque<DispatchRecord>& seq) -> bool {
            if (seq.size() < 3) return false;

            const auto& last3_0 = seq[seq.size() - 3];
            const auto& last3_1 = seq[seq.size() - 2];
            const auto& last3_2 = seq[seq.size() - 1];

            // MatMul + Add + Sigmoid 模式
            if (last3_0.op_type == op::MatMul &&
                last3_1.op_type == op::Add &&
                last3_2.op_type == op::Sigmoid) {
                // 检查 shape 兼容性: MatMul 输出 (M,N) == Add 输入 == Sigmoid 输入
                if (last3_0.shape.size() >= 4 && last3_1.shape.size() >= 2 && last3_2.shape.size() >= 1) {
                    size_t M = last3_0.shape[0];
                    size_t N = last3_0.shape[3];
                    if (last3_1.shape[0] == M && last3_1.shape[1] == N) {
                        // 提交融合编译
                        submitFusedCompileAsync({last3_0, last3_1, last3_2}, dev, cfg, "MatMul+Add+Sigmoid");
                        return true;
                    }
                }
            }

            // MatMul + Add + ReLU 模式
            if (last3_0.op_type == op::MatMul &&
                last3_1.op_type == op::Add &&
                last3_2.op_type == op::ReLU) {
                if (last3_0.shape.size() >= 4 && last3_1.shape.size() >= 2 && last3_2.shape.size() >= 1) {
                    size_t M = last3_0.shape[0];
                    size_t N = last3_0.shape[3];
                    if (last3_1.shape[0] == M && last3_1.shape[1] == N) {
                        submitFusedCompileAsync({last3_0, last3_1, last3_2}, dev, cfg, "MatMul+Add+ReLU");
                        return true;
                    }
                }
            }

            // MatMul + Sigmoid 模式 (无 bias)
            if (last3_0.op_type == op::MatMul &&
                last3_1.op_type == op::Sigmoid) {
                if (last3_0.shape.size() >= 4 && last3_1.shape.size() >= 1) {
                    size_t M = last3_0.shape[0];
                    size_t N = last3_0.shape[3];
                    bool shape_ok = false;
                    if (last3_1.shape.size() >= 2) {
                        shape_ok = (last3_1.shape[0] == M && last3_1.shape[1] == N);
                    } else if (last3_1.shape.size() == 1) {
                        shape_ok = (last3_1.shape[0] == M * N);
                    }
                    if (shape_ok) {
                        submitFusedCompileAsync({last3_0, last3_1}, dev, cfg, "MatMul+Sigmoid");
                        return true;
                    }
                }
            }

            // MatMul + ReLU 模式 (无 bias)
            if (last3_0.op_type == op::MatMul &&
                last3_1.op_type == op::ReLU) {
                if (last3_0.shape.size() >= 4 && last3_1.shape.size() >= 1) {
                    size_t M = last3_0.shape[0];
                    size_t N = last3_0.shape[3];
                    bool shape_ok = false;
                    if (last3_1.shape.size() >= 2) {
                        shape_ok = (last3_1.shape[0] == M && last3_1.shape[1] == N);
                    } else if (last3_1.shape.size() == 1) {
                        shape_ok = (last3_1.shape[0] == M * N);
                    }
                    if (shape_ok) {
                        submitFusedCompileAsync({last3_0, last3_1}, dev, cfg, "MatMul+ReLU");
                        return true;
                    }
                }
            }

            return false;
        };

        // 检查最近的 dispatch 序列
        if (!checkPattern(recent_dispatches_)) {
            // 也可以检查更长的历史序列中是否有重复的模式
            // 这里简化处理，只检查最近的
        }
    }

    /// 从 DispatchRecord 的 shape 提取 MatMul 的 M,K,N 维度
    /// shape 格式: {M, K, K, N} (从 dispatch 时合并的 shape_sig 来)
    static bool extractMatMulDims(const std::vector<size_t>& shape,
                                   size_t& M, size_t& K, size_t& N) {
        if (shape.size() < 4) return false;
        M = shape[0]; K = shape[1]; N = shape[3];
        return true;
    }

    /// 从 DispatchRecord 的 shape 提取二元算子的 out_shape
    /// 对于 MatMul, shape 是 {M,K,K,N}, out_shape = {M,N}
    /// 对于逐元素二元算子, shape 是 {M,N,M,N}, out_shape = {M,N}
    static std::vector<size_t> extractBinOpOutShape(const std::vector<size_t>& shape) {
        if (shape.size() >= 4) {
            // shape = {M, K, K, N} 或 {M, N, M, N}
            return {shape[0], shape[shape.size() - 1]};
        }
        if (shape.size() >= 2) {
            return {shape[0], shape[1]};
        }
        return shape;
    }

    /// 判断一个 op 是一元算子还是二元算子
    static bool isUnaryOp(op op_type) {
        // [位掩码 2026-08-11] 一元算子集合 → uint64 位掩码, O(1) 查表,
        // 消除 switch 的分支预测失败惩罚 (buildFusedGraph 构建时调用).
        static_assert(static_cast<size_t>(op::kCount) <= 64,
                      "op::kCount exceeds uint64 bitmask capacity");
        static constexpr uint64_t kUnaryMask =
              (1ull << static_cast<size_t>(op::Neg))
            | (1ull << static_cast<size_t>(op::ReLU))
            | (1ull << static_cast<size_t>(op::Tanh))
            | (1ull << static_cast<size_t>(op::Sigmoid))
            | (1ull << static_cast<size_t>(op::GELU))
            | (1ull << static_cast<size_t>(op::LReLU))
            | (1ull << static_cast<size_t>(op::Log))
            | (1ull << static_cast<size_t>(op::Exp))
            | (1ull << static_cast<size_t>(op::Abs))
            | (1ull << static_cast<size_t>(op::Sin))
            | (1ull << static_cast<size_t>(op::Cos));
        return (kUnaryMask >> static_cast<size_t>(op_type)) & 1ull;
    }

public:
    /// 测试入口：直接复用 buildFusedGraph 的图构造逻辑，跳过预走机制
    static Graph buildFusedGraphForTest(const std::vector<DispatchRecord>& records,
                                         const std::string& pattern_name) {
        return buildFusedGraph(records, pattern_name);
    }

private:
    /// 构建融合 Graph 的核心方法
    /// @param records 按执行顺序排列的 DispatchRecord 序列
    /// @param pattern_name 融合模式名（用于生成 KernelShapeInfo）
    /// @return 正确连接的 Graph
    static Graph buildFusedGraph(const std::vector<DispatchRecord>& records,
                                 const std::string& pattern_name) {
        Graph g;

        // 为每个 record 预先计算: 输入描述符、输出描述符、需要的输入数量
        struct NodeInfo {
            TensorDesc out_desc;
            std::vector<TensorDesc> input_descs; // 外部输入（非链式中间结果）
            std::vector<bool> input_is_external; // 标记每个输入是否为外部
        };

        std::vector<NodeInfo> infos;
        for (const auto& rec : records) {
            NodeInfo info;

            if (rec.op_type == op::MatMul) {
                size_t M, K, N;
                if (extractMatMulDims(rec.shape, M, K, N)) {
                    TensorDesc lhs_desc = TensorDesc::fromShape({M, K});
                    TensorDesc rhs_desc = TensorDesc::fromShape({K, N});
                    info.input_descs = {lhs_desc, rhs_desc};
                    info.input_is_external = {true, true};
                    info.out_desc = TensorDesc::fromShape({M, N});
                } else {
                    info.out_desc = TensorDesc::fromShape(rec.shape);
                    info.input_descs = {info.out_desc, info.out_desc};
                    info.input_is_external = {true, true};
                }
            } else if (!isUnaryOp(rec.op_type)) {
                // 二元逐元素算子
                auto out_shape = extractBinOpOutShape(rec.shape);
                info.out_desc = TensorDesc::fromShape(out_shape);
                // 第一个输入 = out_shape，第二个输入从 shape 剩余部分提取
                // shape_sig = lhs_shape + rhs_shape（拼接），所以 rhs_shape = shape[out_shape.size():]
                std::vector<size_t> rhs_shape;
                if (rec.shape.size() > out_shape.size()) {
                    rhs_shape.assign(rec.shape.begin() + (ptrdiff_t)out_shape.size(), rec.shape.end());
                } else {
                    rhs_shape = out_shape;
                }
                info.input_descs = {info.out_desc, TensorDesc::fromShape(rhs_shape)};
                // 第一个输入可能是链式的（由后续设置），第二个是外部的
                info.input_is_external = {false, true};
            } else {
                // 一元算子
                auto out_shape = extractBinOpOutShape(rec.shape);
                info.out_desc = TensorDesc::fromShape(out_shape);
                // 一元算子的输入来自 chain
                info.input_descs = {info.out_desc};
                info.input_is_external = {false};
            }
            infos.push_back(std::move(info));
        }

        // 对于第一个节点，所有输入都是外部的
        if (!infos.empty()) {
            for (size_t j = 0; j < infos[0].input_is_external.size(); ++j) {
                infos[0].input_is_external[j] = true;
            }
        }

        // 添加所有外部输入
        // 跟踪每个输入对应的 node_id (用于链式连接)
        std::vector<size_t> input_node_ids; // 外部输入的 node_id 列表
        std::vector<std::pair<size_t, size_t>> input_to_node; // (input_index, record_index)

        // 为每个 record 的每个外部输入添加 Graph 输入
        for (size_t ri = 0; ri < records.size(); ++ri) {
            for (size_t ii = 0; ii < infos[ri].input_descs.size(); ++ii) {
                if (infos[ri].input_is_external[ii]) {
                    size_t in_id = g.addInput(infos[ri].input_descs[ii]);
                    input_node_ids.push_back(in_id);
                    input_to_node.push_back({in_id, ri});
                }
            }
        }

        // 添加计算节点并连接
        size_t prev_node_id = SIZE_MAX;
        for (size_t ri = 0; ri < records.size(); ++ri) {
            const auto& rec = records[ri];
            const auto& info = infos[ri];

            // 收集此节点的输入 node_ids
            std::vector<size_t> node_input_ids;

            if (ri == 0) {
                // 第一个节点：所有输入都是外部的
                for (size_t ii = 0; ii < info.input_descs.size(); ++ii) {
                    // 找到对应的外部输入 ID
                    // 对于 MatMul: input_descs[0]=lhs, input_descs[1]=rhs
                    // 我们按添加顺序获取前 N 个外部输入
                    size_t ext_idx = ii; // 因为第一个节点的所有输入都是外部的，且按顺序添加
                    if (ext_idx < input_node_ids.size()) {
                        node_input_ids.push_back(input_node_ids[ext_idx]);
                    }
                }
            } else {
                // 后续节点：第一个输入来自 chain（上一个节点的输出），其余为外部输入
                node_input_ids.push_back(prev_node_id);

                // 收集剩余的外部输入
                for (size_t ii = 1; ii < info.input_descs.size(); ++ii) {
                    // 查找剩余的外部输入
                    size_t ext_idx = 0;
                    for (size_t rj = 0; rj < ri; ++rj) {
                        ext_idx += infos[rj].input_descs.size();
                    }
                    // 跳过第一个节点已消费的
                    ext_idx += (ii - 1); // 因为当前节点的第一个输入是 chain 内部的
                    if (ext_idx < input_node_ids.size()) {
                        node_input_ids.push_back(input_node_ids[ext_idx]);
                    }
                }
            }

            // 创建节点
            size_t node_id = g.addNode(makeNodeVariant(rec), node_input_ids, info.out_desc);
            prev_node_id = node_id;
        }

        // 标记最后一个节点为输出
        if (prev_node_id != SIZE_MAX) {
            g.markOutput(prev_node_id);
        }

        return g;
    }

    /// 提交融合编译任务（通用版，支持 2 或 3 个算子）
    void submitFusedCompileAsync(const std::vector<DispatchRecord>& records,
                                 DeviceType dev, const HotPathConfig& cfg,
                                 const std::string& pattern_name) {
        // 计算融合 key
        size_t fused_key = 0;
        for (const auto& r : records) {
            fused_key ^= hashShapeKey(r.shape, r.op_type, dev);
        }

        // 检查是否已经在编译，并原子地标记为 compiling 以防并发重复触发
        {
            std::lock_guard<std::mutex> lk(mutex_);
            auto it = entries_.find(fused_key);
            if (it != entries_.end() && it->second.compiling) return;
            entries_[fused_key].compiling = true;
        }

        pending_compiles_.fetch_add(1, std::memory_order_relaxed);
        compilations_triggered_.fetch_add(1, std::memory_order_relaxed);

        if (cfg.verbose) {
            CtorchError::log(ErrorLevel::INFO, ErrorPlatform::kGENERAL,
                ErrorType::UNKNOWN,
                "C3HotPathManager: 检测到融合模式: " + pattern_name);
        }

        // 使用 std::async 启动后台任务，future 纳入 pending_futures_ 管理
        // 这样 shutdown() 可以等待所有任务完成，避免进程退出时 UAF
        auto future = std::async(std::launch::async, [this, records, fused_key, pattern_name]() {
            // 使用新的 Graph 构建方法
            Graph g = buildFusedGraph(records, pattern_name);

            // 编译选项：DEBT-NEW-7 性能优化（v0.5.1+）
            // MatMul-rooted region 迁移至 MLIR 真实内存级 True JIT 后端（M2 阶段优化 2026-08-14）：
            //   - 使用 buildMatMul 直接在 MLIR 中调用外部 cblas_sgemm，保留 100% AMX 极致物理性能与精度等价。
            //   - 后置 epilogue 算子（bias add + ReLU）完全走 MLIR 向量化 + Host 托管暂存区（Scratchpad）零分配路径！
            //   - 彻底消灭动态堆内存分配，不再使用 dlopen 与外部 clang++ 进程，实现真正的 True JIT。
            CompileOptions opts;
            opts.backend = C3Backend::MLIR;
#ifdef CT_DEBUG
            {
                std::ostringstream oss;
                oss << "[C3-HotPath] buildFusedGraph returned, pattern=" << pattern_name
                    << " total_nodes=" << g.nodes().size()
                    << " inputs=" << g.inputs().size();
                for (size_t i = 0; i < g.nodes().size(); ++i) {
                    oss << " n" << i << ":op" << g.nodes()[i].op.index()
                        << "(in=" << g.nodes()[i].inputs.size() << ")";
                }
                CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN, oss.str());
            }
#endif

            // 编译(强制 Handwritten backend,见上 CompileOptions opts 的注释)
            auto& engine = C3Engine::getInstance();
            auto kernel = engine.compile(g, opts);

            if (kernel) {
                KernelShapeInfo info;
                if (!records.empty()) {
                    // 设置 shape 信息
                    if (records[0].op_type == op::MatMul && records[0].shape.size() >= 4) {
                        info.lhs_shape = {records[0].shape[0], records[0].shape[1]};
                        info.rhs_shape = {records[0].shape[2], records[0].shape[3]};
                    } else {
                        info.lhs_shape = records[0].shape;
                    }
                    // 最后一个记录的 out_shape
                    const auto& last = records.back();
                    info.out_shape = extractBinOpOutShape(last.shape);
                }
                info.fused_pattern = pattern_name;

                // 使用 C3Engine 编译
                C3KernelRegistry::getInstance().installFused(kernel, records.back().op_type, info);

                // 同时安装到 RegionFusionRegistry，启用预走匹配。
                // 附带读写次数成本模型评估：仅值得融合的 pattern 才会被激活。
                std::vector<op> op_seq;
                std::vector<size_t> out_numels;
                for (const auto& rec : records) {
                    op_seq.push_back(rec.op_type);
                    // 提取每个算子的输出 numel（MatMul: {M,K,K,N}->{M,N}，其余按 out_shape）
                    std::vector<size_t> os = extractBinOpOutShape(rec.shape);
                    size_t numel = 1;
                    for (auto s : os) numel *= s;
                    out_numels.push_back(numel);
                }
                // 首个 op 的输入形状：用于预走匹配时的形状校验，
                // 避免反向传播（形状不同）错误匹配前向注册的区域。
                std::vector<std::vector<size_t>> first_input_shapes;
                if (!records.empty()) {
                    const auto& r0 = records[0];
                    if (r0.op_type == op::MatMul && r0.shape.size() >= 4) {
                        first_input_shapes = {{r0.shape[0], r0.shape[1]},
                                              {r0.shape[2], r0.shape[3]}};
                    } else if (r0.shape.size() >= 2) {
                        size_t half = r0.shape.size() / 2;
                        first_input_shapes.push_back(
                            {r0.shape.begin(), r0.shape.begin() + (ptrdiff_t)half});
                        first_input_shapes.push_back(
                            {r0.shape.begin() + (ptrdiff_t)half, r0.shape.end()});
                    }
                }
                RegionFusionRegistry::getInstance().installWithCost(
                    op_seq, kernel, out_numels, first_input_shapes);

                if (getConfig().verbose) {
                    CtorchError::log(ErrorLevel::INFO, ErrorPlatform::kGENERAL,
                        ErrorType::UNKNOWN,
                        "C3HotPathManager: 融合编译完成: " + pattern_name);
                }
            }

            pending_compiles_.fetch_sub(1, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lk(mutex_);
                auto it = entries_.find(fused_key);
                if (it != entries_.end()) {
                    it->second.compiling = false;
                    it->second.last_compile_time = std::chrono::steady_clock::now();
                }
            }
        });

        // 将 future 加入统一管理列表
        {
            std::lock_guard<std::mutex> lk(futures_mutex_);
            pending_futures_.push_back(std::move(future));
        }
    }

    /// 异步提交 C3 编译任务 (并发双管线 Tier 1/2)
    void submitCompileAsync(op op_type, DeviceType dev,
                            const std::vector<size_t>& shape,
                            const std::vector<size_t>& lhs_shape,
                            const std::vector<size_t>& rhs_shape,
                            size_t key, const HotPathConfig& cfg) {
        if (!isSupportedOp(op_type)) {
            // 不支持的操作，释放 pending 计数
            pending_compiles_.fetch_sub(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lk(mutex_);
            auto it = entries_.find(key);
            if (it != entries_.end()) {
                it->second.compiling = false;
            }
            return;
        }

        Graph g = buildGraphForOp(op_type, shape, lhs_shape, rhs_shape);

        // 防御：空图（无输出节点）跳过编译，避免安装损坏的 kernel
        if (g.outputCount() == 0) {
            pending_compiles_.fetch_sub(1, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lk(mutex_);
                auto it = entries_.find(key);
                if (it != entries_.end()) {
                    it->second.compiling = false;
                }
            }
            return;
        }

        // 保存原始 timeout 并设置临时 timeout
        auto& engine = C3Engine::getInstance();
        uint32_t original_timeout = engine.getCompileTimeoutMs();
        engine.setCompileTimeoutMs(cfg.compile_timeout_ms);

        // ======================= 1. 提交 Tier 1 JIT (快速优化管线) =======================
        CompileOptions opts_fast;
        opts_fast.opt_level = 2; // O2 JIT, 极速编译响应
        if (op_type == op::MatMul) {
            opts_fast.backend = C3Backend::Handwritten;
        }
        auto compile_future_fast = engine.compileAsync(g, opts_fast);

        // ======================= 2. 提交 Tier 2 JIT (极限优化管线) =======================
        CompileOptions opts_extreme;
        opts_extreme.opt_level = 4; // Ofast JIT, Passes 全开 (如 Loop Unroll-and-Jam, 寄存器最大复用)
        if (op_type == op::MatMul) {
            opts_extreme.backend = C3Backend::Handwritten;
        }
        auto compile_future_extreme = engine.compileAsync(g, opts_extreme);

        // 恢复 timeout
        engine.setCompileTimeoutMs(original_timeout);

        // 后台辅助 lambda，处理单个编译 future 的解析和注册
        auto run_install_task = [this, op_type, shape, lhs_shape, rhs_shape](CompileFuture compile_future, std::string pipeline_name) {
            auto kernel = compile_future.get();
            if (kernel) {
                // 安装到 C3KernelRegistry
                KernelShapeInfo info;
                if (op_type == op::MatMul && shape.size() >= 4) {
                    // MatMul: shape={M, K, K, N}
                    size_t M = shape[0], K = shape[1], N = shape[3];
                    info.is_matmul = true;
                    info.M = M; info.K = K; info.N = N;
                    info.lhs_shape = {M, K};
                    info.rhs_shape = {K, N};
                    info.out_shape = {M, N};
                } else {
                    // 单算子统一使用真实输入形状，保证注册 key 与执行期 key 一致
                    info.lhs_shape = lhs_shape;
                    info.rhs_shape = rhs_shape;
                    info.out_shape = lhs_shape;
                }
                C3KernelRegistry::getInstance().install(op_type, DeviceType::kCPU, kernel, info);

                if (getConfig().verbose) {
                    CtorchError::log(ErrorLevel::INFO, ErrorPlatform::kGENERAL,
                        ErrorType::UNKNOWN,
                        "C3HotPathManager: [" + pipeline_name + "] 编译完成并注册: op=" +
                        std::to_string(static_cast<int>(op_type)) +
                        " opt_level=" + std::to_string(kernel->optLevel()) +
                        " shape=[" + shapeToString(shape) + "]");
                }
            }
        };

        // 启动两个并发等待线程
        auto future_fast = std::async(std::launch::async, [run_install_task, compile_future_fast = std::move(compile_future_fast)]() mutable {
            run_install_task(std::move(compile_future_fast), "Tier 1 JIT (Fast)");
        });

        auto future_extreme = std::async(std::launch::async, [this, run_install_task, compile_future_extreme = std::move(compile_future_extreme), key]() mutable {
            run_install_task(std::move(compile_future_extreme), "Tier 2 JIT (Extreme)");
            
            // 极限管线作为生命周期的收尾，负责释放全局 pending 计数和设置 compiling 状态
            pending_compiles_.fetch_sub(1, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lk(mutex_);
                auto it = entries_.find(key);
                if (it != entries_.end()) {
                    it->second.compiling = false;
                    it->second.last_compile_time = std::chrono::steady_clock::now();
                }
            }
        });

        // 将并发 future 纳入全局生命周期管理，防止进程退出 UAF
        {
            std::lock_guard<std::mutex> lk(futures_mutex_);
            pending_futures_.push_back(std::move(future_fast));
            pending_futures_.push_back(std::move(future_extreme));
        }
    }

    static std::string shapeToString(const std::vector<size_t>& shape) {
        std::string s;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) s += ",";
            s += std::to_string(shape[i]);
        }
        return s;
    }

    // ======================= 成员变量 =======================

    mutable std::mutex mutex_;
    mutable std::mutex cfg_mutex_;
    mutable std::mutex rb_mutex_;
    mutable std::mutex futures_mutex_;  ///< 保护 pending_futures_
    HotPathConfig config_;

    std::unordered_map<size_t, HotEntry> entries_;

    // RingBuffer for fusion detection
    std::deque<DispatchRecord> recent_dispatches_;

    // 待管理的异步编译任务 future（替代 detach 线程）
    std::vector<std::future<void>> pending_futures_;

    std::atomic<size_t> calls_tracked_{0};
    std::atomic<size_t> compilations_triggered_{0};
    std::atomic<size_t> pending_compiles_{0};
    std::atomic<bool> shutting_down_{false};
    size_t cooldown_hits_ = 0;
    size_t backpressure_hits_ = 0;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_C3_HOT_PATH_MANAGER_H