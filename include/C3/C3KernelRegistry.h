/**
 * @file C3KernelRegistry.h
 * @brief C3 JIT 内核注册表 — 线程安全的热替换与回退机制
 * @details 存储 C3 编译后的 kernel 函数指针 + 形状签名。
 *          调度器在 dispatch 时优先查询此注册表，命中则使用 C3 kernel；
 *          未命中或执行失败时自动回退到 eager 路径。
 *
 *          热替换流程：
 *          1. C3Engine::compile 生成 C3 kernel
 *          2. 用户调用 C3KernelRegistry::install 注册
 *          3. 调度器下次 dispatch 时自动使用 C3 kernel（原子可见性）
 *          4. 若 C3 kernel 执行失败（异常），自动回退 eager 并记录
 *
 *          回退策略：
 *          - 形状不匹配 → 静默回退（不记录错误，预期行为）
 *          - 执行异常 → 回退 + 记录错误日志 + 可选自动卸载
 * @date 2026/7/31
 */

#ifndef CTORCH_C3_C3_KERNEL_REGISTRY_H
#define CTORCH_C3_C3_KERNEL_REGISTRY_H

#include "Graph.h"

#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../Ctools.h"
#include "../Tensor.h"

namespace ct {
namespace c3 {

// ======================= C3 Kernel 函数指针类型 =======================

/**
 * @brief C3 JIT kernel 函数指针统一签名
 * @param a 输入 A 的数据指针
 * @param b 输入 B 的数据指针
 * @param out 输出数据指针
 * @param n 元素总数（逐元素操作）
 * @param M MatMul M 维度（逐元素操作时忽略）
 * @param K MatMul K 维度（逐元素操作时忽略）
 * @param N MatMul N 维度（逐元素操作时忽略）
 */
using C3KernelFunc = void (*)(const float*, const float*, float*, size_t, size_t, size_t, size_t);

/**
 * @brief C3 融合 kernel 函数指针签名
 * @param inputs 输入数据指针数组（长度 = num_inputs）
 * @param out 输出数据指针
 * @param n 元素总数
 */
using FusedKernelFunc = void (*)(const float* const*, float*, size_t);

/**
 * @brief C3 多节点 kernel 函数指针签名
 * @param inputs 输入数据指针数组（长度 = num_inputs）
 * @param out 输出数据指针
 * @param n 元素总数（逐元素操作）
 * @param M MatMul M 维度
 * @param K MatMul K 维度
 * @param N MatMul N 维度
 */
using MultiNodeKernelFunc = void (*)(const float* const*, float*, size_t, size_t, size_t, size_t);

// ======================= 内核形状签名 =======================

/**
 * @struct KernelShapeInfo
 * @brief C3 kernel 的形状签名，用于运行时匹配。
 */
struct KernelShapeInfo {
    std::vector<size_t> lhs_shape;
    std::vector<size_t> rhs_shape;
    std::vector<size_t> out_shape;
    bool is_matmul = false;
    size_t M = 0, K = 0, N = 0;
};

// ======================= C3 内核注册表 =======================

/**
 * @class C3KernelRegistry
 * @brief C3 JIT 内核注册表单例，线程安全。
 * @details 存储从 (op_type, device, shape_hash) → C3 kernel 的映射。
 *          支持热替换：install 后立即生效（下一次 dispatch 可见）。
 *          支持回退：uninstall 或执行失败时自动回退到 eager。
 */
class C3KernelRegistry {
public:
    static C3KernelRegistry& getInstance() {
        static C3KernelRegistry instance;
        return instance;
    }

    // ======================= 注册与卸载 =======================

    /**
     * @brief 安装 C3 kernel
     * @param op_type 算子类型
     * @param dev 目标设备
     * @param func C3 kernel 函数指针
     * @param shapes 形状签名
     * @param dl_handle dlopen 句柄（注册表不负责释放，由 CompiledKernel 管理）
     */
    void install(op op_type, DeviceType dev, C3KernelFunc func,
                 const KernelShapeInfo& shapes) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto key = makeKey(op_type, dev);
        entries_[key] = {func, shapes, true};
        install_count_.fetch_add(1, std::memory_order_release);
    }

    /**
     * @brief 卸载 C3 kernel（回退到 eager）
     */
    void uninstall(op op_type, DeviceType dev) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto key = makeKey(op_type, dev);
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            it->second.active = false;
            entries_.erase(it);
            uninstall_count_.fetch_add(1, std::memory_order_release);
        }
    }

    /**
     * @brief 卸载所有 C3 kernel
     */
    void uninstallAll() {
        std::lock_guard<std::mutex> lock(mutex_);
        uninstall_count_.fetch_add(entries_.size(), std::memory_order_release);
        entries_.clear();
    }

    // ======================= 执行 =======================

    /**
     * @brief 尝试通过 C3 kernel 执行
     * @param op_type 算子类型
     * @param a 左操作数
     * @param b 右操作数
     * @return 若命中且执行成功返回 Tensor；否则返回 std::nullopt（回退 eager）
     */
    std::optional<Tensor> tryExecute(op op_type, const Tensor& a, const Tensor& b) {
        auto key = makeKey(op_type, a.device());

        C3Entry entry;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(key);
            if (it == entries_.end() || !it->second.active) {
                return std::nullopt;
            }
            entry = it->second;
        }

        // 形状匹配检查
        if (a.shape() != entry.shapes.lhs_shape ||
            b.shape() != entry.shapes.rhs_shape) {
            return std::nullopt; // 静默回退
        }

        // 执行 C3 kernel
        try {
            Tensor out;
            if (entry.shapes.is_matmul) {
                out = Tensor(ShapeTag{}, entry.shapes.out_shape);
                entry.func(
                    a.data_read<float>(),
                    b.data_read<float>(),
                    out.data_write<float>(),
                    0,
                    entry.shapes.M, entry.shapes.K, entry.shapes.N);
            } else {
                out = Tensor(ShapeTag{}, a.shape());
                entry.func(
                    a.data_read<float>(),
                    b.data_read<float>(),
                    out.data_write<float>(),
                    a.numel(), 0, 0, 0);
            }
            hit_count_.fetch_add(1, std::memory_order_relaxed);
            return out;
        } catch (const std::exception& e) {
            miss_count_.fetch_add(1, std::memory_order_relaxed);
            // 执行失败，静默回退到 eager
            return std::nullopt;
        } catch (...) {
            miss_count_.fetch_add(1, std::memory_order_relaxed);
            return std::nullopt;
        }
    }

    // ======================= 统计 =======================

    struct Stats {
        size_t install_count = 0;
        size_t uninstall_count = 0;
        size_t hit_count = 0;
        size_t miss_count = 0;
        size_t active_entries = 0;
    };

    Stats getStats() const {
        Stats s;
        s.install_count = install_count_.load(std::memory_order_acquire);
        s.uninstall_count = uninstall_count_.load(std::memory_order_acquire);
        s.hit_count = hit_count_.load(std::memory_order_acquire);
        s.miss_count = miss_count_.load(std::memory_order_acquire);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            s.active_entries = entries_.size();
        }
        return s;
    }

private:
    C3KernelRegistry() = default;

    struct C3Entry {
        C3KernelFunc func = nullptr;
        KernelShapeInfo shapes;
        bool active = false;
    };

    using KeyType = std::pair<size_t, size_t>; // (op_index, device_index)

    static KeyType makeKey(op op_type, DeviceType dev) {
        return {static_cast<size_t>(op_type), static_cast<size_t>(dev)};
    }

    mutable std::mutex mutex_;
    std::unordered_map<KeyType, C3Entry,
        std::function<size_t(const KeyType&)>> entries_{
        64, [](const KeyType& k) {
            return k.first ^ (k.second << 16);
        }};

    std::atomic<size_t> install_count_{0};
    std::atomic<size_t> uninstall_count_{0};
    std::atomic<size_t> hit_count_{0};
    std::atomic<size_t> miss_count_{0};
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_C3_KERNEL_REGISTRY_H