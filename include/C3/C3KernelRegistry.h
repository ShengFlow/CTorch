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
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../Ctools.h"
#include "../Tensor.h"

// Forward declaration — CompiledKernel is defined in C3Engine.h
namespace ct {
namespace c3 {
class CompiledKernel;
} // namespace c3
} // namespace ct

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
    std::string fused_pattern;  // 融合模式名（如 "MatMul+Add+Sigmoid"），空字符串表示非融合
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
        auto key = makeKey(op_type, dev, shapes);
        entries_[key] = {func, shapes, true};
        install_count_.fetch_add(1, std::memory_order_release);
        fprintf(stderr, "[DBG] INSTALL op=%d dev=%d key3=%zu lhs=[%s] rhs=[%s]\n",
                (int)op_type, (int)dev, key.third,
                shapeDebug(shapes.lhs_shape).c_str(), shapeDebug(shapes.rhs_shape).c_str());
    }

    /**
     * @brief 卸载 C3 kernel（回退到 eager）
     */
    void uninstall(op op_type, DeviceType dev) {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t removed = 0;
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->first.first == static_cast<size_t>(op_type) &&
                it->first.second == static_cast<size_t>(dev)) {
                it->second.active = false;
                it = entries_.erase(it);
                ++removed;
            } else {
                ++it;
            }
        }
        if (removed > 0) {
            uninstall_count_.fetch_add(removed, std::memory_order_release);
        }
    }

    /**
     * @brief 卸载所有 C3 kernel
     */
    void uninstallAll() {
        std::lock_guard<std::mutex> lock(mutex_);
        uninstall_count_.fetch_add(entries_.size(), std::memory_order_release);
        entries_.clear();
        // 同时清理 fused_entries_ 和 backward_entries_，避免其持有的
        // CompiledKernel（内含 LLVM ExecutionEngine）在静态析构阶段触发
        // LLVM 后端清理逻辑导致 null pointer dereference (getCopyToParts crash)。
        // 调用方应在 main() 返回前调用本方法，参见 C3Engine.h 退出序列文档。
        fused_entries_.clear();
        backward_entries_.clear();
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
        auto key = makeKeyFromShapes(op_type, a.device(), a.shape(), b.shape());

        C3Entry entry;
        bool found = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(key);
            if (it == entries_.end() || !it->second.active) {
                found = false;
            } else {
                found = true;
                entry = it->second;
            }
        }
        fprintf(stderr, "[DBG] tryBin op=%d key3=%zu found=%d a=[%s] b=[%s]\n",
                (int)op_type, key.third, (int)found,
                shapeDebug(a.shape()).c_str(), shapeDebug(b.shape()).c_str());
        if (!found) return std::nullopt;

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

            // 输出形状验证：C3 kernel 输出形状必须与注册时记录的预期形状一致
            if (!validateOutputShape(op_type, a.device(), out, entry.shapes.out_shape)) {
                return std::nullopt; // 形状不匹配，已卸载
            }

#ifdef CT_DEBUG
            {
                // 输出 C3 kernel 执行后的前 5 个元素值（用于验证正确性）
                const float* out_data = out.data_read<float>();
                size_t n = std::min(size_t(5), out.numel());
                fprintf(stderr, "[C3-VALIDATE] binary op=%d out=[", (int)op_type);
                for (size_t i = 0; i < out.shape().size(); ++i) {
                    if (i > 0) fprintf(stderr, ",");
                    fprintf(stderr, "%zu", out.shape()[i]);
                }
                fprintf(stderr, "] first_%zu=[", n);
                for (size_t i = 0; i < n; ++i) {
                    if (i > 0) fprintf(stderr, ",");
                    fprintf(stderr, "%.6f", out_data[i]);
                }
                fprintf(stderr, "]\n");
            }
#endif
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

    /**
     * @brief 尝试通过 C3 kernel 执行（unary 版本）
     * @param op_type 算子类型
     * @param a 输入
     * @return 若命中且执行成功返回 Tensor；否则返回 std::nullopt（回退 eager）
     */
    std::optional<Tensor> tryExecuteUnary(op op_type, const Tensor& a) {
        auto key = makeKeyFromShapes(op_type, a.device(), a.shape(), {});

        C3Entry entry;
        bool found = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(key);
            if (it == entries_.end() || !it->second.active) {
                found = false;
            } else {
                found = true;
                entry = it->second;
            }
        }
        fprintf(stderr, "[DBG] tryUnary op=%d key3=%zu found=%d a=[%s]\n",
                (int)op_type, key.third, (int)found,
                shapeDebug(a.shape()).c_str());
        if (!found) return std::nullopt;

        // 形状匹配检查
        if (a.shape() != entry.shapes.lhs_shape) {
            return std::nullopt; // 静默回退
        }

        // 执行 C3 kernel
        try {
            Tensor out = Tensor(ShapeTag{}, a.shape());
            entry.func(
                a.data_read<float>(),
                nullptr,
                out.data_write<float>(),
                a.numel(), 0, 0, 0);
            hit_count_.fetch_add(1, std::memory_order_relaxed);

            // 输出形状验证：C3 kernel 输出形状必须与注册时记录的预期形状一致
            if (!validateOutputShape(op_type, a.device(), out, entry.shapes.out_shape)) {
                return std::nullopt; // 形状不匹配，已卸载
            }

#ifdef CT_DEBUG
            {
                // 输出 C3 kernel 执行后的前 5 个元素值（用于验证正确性）
                const float* out_data = out.data_read<float>();
                const float* in_data = a.data_read<float>();
                size_t n = std::min(size_t(5), out.numel());
                fprintf(stderr, "[C3-VALIDATE] unary op=%d out=[", (int)op_type);
                for (size_t i = 0; i < out.shape().size(); ++i) {
                    if (i > 0) fprintf(stderr, ",");
                    fprintf(stderr, "%zu", out.shape()[i]);
                }
                fprintf(stderr, "] in_first_%zu=[", n);
                for (size_t i = 0; i < n; ++i) {
                    if (i > 0) fprintf(stderr, ",");
                    fprintf(stderr, "%.6f", in_data[i]);
                }
                fprintf(stderr, "] out_first_%zu=[", n);
                for (size_t i = 0; i < n; ++i) {
                    if (i > 0) fprintf(stderr, ",");
                    fprintf(stderr, "%.6f", out_data[i]);
                }
                fprintf(stderr, "]\n");
            }
#endif
            return out;
        } catch (const std::exception& e) {
            miss_count_.fetch_add(1, std::memory_order_relaxed);
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
        size_t bypass_count = 0;   ///< DEBT-NEW-7 H2 fix:被 inAutogradScope guard 跳过的次数
        size_t active_entries = 0;
        size_t fused_entries = 0;
        size_t backward_entries = 0;
    };

    /**
     * @brief 记录一次 guard bypass（DEBT-NEW-7 H2 fix 计数器）
     * @details 由调度器 inAutogradScope guard 在检测到 autograd 上下文时调用，
     *          表明本次本可以走 c3 single kernel 路径但被主动跳过。
     *          通过 bypass_count vs hit_count + miss_count 的对比，
     *          可以验证 H2 fix 是否在工作：
     *            - 训练期间：bypass >> (hit + miss),准确率高
     *            - 推理期间：hit + miss >> bypass,加速生效
     */
    void recordBypass() {
        bypass_count_.fetch_add(1, std::memory_order_relaxed);
    }

    Stats getStats() const {
        Stats s;
        s.install_count = install_count_.load(std::memory_order_acquire);
        s.uninstall_count = uninstall_count_.load(std::memory_order_acquire);
        s.hit_count = hit_count_.load(std::memory_order_acquire);
        s.miss_count = miss_count_.load(std::memory_order_acquire);
        s.bypass_count = bypass_count_.load(std::memory_order_acquire);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            s.active_entries = entries_.size();
            s.fused_entries = fused_entries_.size();
            s.backward_entries = backward_entries_.size();
        }
        return s;
    }

    // ======================= 融合 kernel =======================

    /**
     * @brief 安装融合 kernel
     * @param kernel 编译好的融合 kernel
     * @param op_type 主算子类型
     * @param shapes 形状签名（含 fused_pattern）
     */
    void installFused(std::shared_ptr<CompiledKernel> kernel,
                      op op_type, const KernelShapeInfo& shapes) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string key = makeFusedKey(op_type, shapes);
        fused_entries_[key] = {std::move(kernel), shapes, true};
        install_count_.fetch_add(1, std::memory_order_release);
    }

    /**
     * @brief 尝试执行融合 kernel（声明，实现在 .cpp 中）
     * @param op_type 算子类型
     * @param inputs 输入张量列表
     * @return 若命中且执行成功返回输出 Tensor；否则返回 nullopt
     */
    std::optional<Tensor> tryExecuteFused(op op_type,
                                           const std::vector<Tensor>& inputs);

    // ======================= Backward kernel =======================

    /**
     * @brief 安装 backward kernel
     * @param backward_key 唯一标识 backward 子图的 key（如 "ReLU|shape:1024"）
     * @param kernel 编译好的 CompiledKernel
     * @param grad_shape 下游梯度形状
     * @param out_shape 上游梯度形状（输出形状）
     */
    void installBackward(const std::string& backward_key,
                         std::shared_ptr<CompiledKernel> kernel,
                         const std::vector<size_t>& grad_shape,
                         const std::vector<size_t>& out_shape) {
        std::lock_guard<std::mutex> lock(mutex_);
        backward_entries_[backward_key] = {std::move(kernel), grad_shape, out_shape, true};
        install_count_.fetch_add(1, std::memory_order_release);
    }

    /**
     * @brief 尝试执行 backward kernel
     * @param backward_key 唯一标识 backward 子图的 key
     * @param grad 下游梯度张量
     * @param forward_inputs forward 阶段的输入张量（如 [x] 用于 ReLU backward）
     * @return 若命中返回上游梯度列表；否则返回 nullopt
     */
    std::optional<std::vector<Tensor>> tryExecuteBackward(
        const std::string& backward_key, const Tensor& grad,
        const std::vector<Tensor>& forward_inputs = {});

    /**
     * @brief 快速检查某 backward key 是否存在且 active（不执行，仅用于查找预判）
     * @param backward_key 反向子图/融合的唯一 key
     * @return true 若注册表中存在该 key 且 active=true
     */
    bool hasBackwardKey(const std::string& backward_key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = backward_entries_.find(backward_key);
        return it != backward_entries_.end() && it->second.active;
    }

    /**
     * @brief 根据 op 序列查找匹配的融合 kernel
     * @param op_seq 算子序列（如 {MatMul, Add, Sigmoid}）
     * @param dev 目标设备
     * @param first_input_shape 首个算子的输入形状（用于精确匹配，可选）
     * @return 匹配的 kernel + shapes；未找到返回 nullopt
     */
    std::optional<std::pair<std::shared_ptr<CompiledKernel>, KernelShapeInfo>>
    findFusedKernelForSequence(const std::vector<op>& op_seq, DeviceType dev,
                               const std::vector<size_t>& first_input_shape = {});

    /**
     * @brief 根据首个算子查找匹配的融合 kernel
     * @param op_type 首个算子类型
     * @param input_shape 首个算子的输入形状
     * @param dev 目标设备
     * @return 匹配的 kernel + shapes；未找到返回 nullopt
     */
    std::optional<std::pair<std::shared_ptr<CompiledKernel>, KernelShapeInfo>>
    findFusedKernelForFirstOp(op op_type, const std::vector<size_t>& input_shape,
                              DeviceType dev);

    /**
     * @brief 用原始输入执行融合 kernel
     * @param kernel 融合 kernel
     * @param inputs 原始输入张量列表
     * @param shapes 形状签名
     * @return 执行结果
     */
    Tensor executeFusedWithInputs(std::shared_ptr<CompiledKernel> kernel,
                                   const std::vector<Tensor>& inputs,
                                   const KernelShapeInfo& shapes);

    /**
     * @brief 卸载融合 kernel
     */
    void uninstallFused(op op_type, const KernelShapeInfo& shapes) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string key = makeFusedKey(op_type, shapes);
        auto it = fused_entries_.find(key);
        if (it != fused_entries_.end()) {
            it->second.active = false;
            fused_entries_.erase(it);
            uninstall_count_.fetch_add(1, std::memory_order_release);
        }
    }

    /**
     * @brief 获取融合 entries 数量
     */
    size_t fusedEntryCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return fused_entries_.size();
    }

private:
    C3KernelRegistry() = default;

    struct C3Entry {
        C3KernelFunc func = nullptr;
        KernelShapeInfo shapes;
        bool active = false;
    };

    struct FusedEntry {
        std::shared_ptr<CompiledKernel> kernel;
        KernelShapeInfo shapes;
        bool active = false;
    };

    struct BackwardEntry {
        std::shared_ptr<CompiledKernel> kernel;
        std::vector<size_t> grad_shape;
        std::vector<size_t> out_shape;
        bool active = false;
    };

    // (op_index, device_index, shape_hash)
    // shape_hash 区分同一算子不同形状的 kernel，避免多形状互相覆盖（H2 缺陷根因）。
    struct KeyType {
        size_t first;
        size_t second;
        size_t third;

        bool operator==(const KeyType& o) const {
            return first == o.first && second == o.second && third == o.third;
        }
    };

    struct KeyHash {
        size_t operator()(const KeyType& k) const {
            // 组合哈希，避免 (op,dev) 相同仅 shape 不同时碰撞
            size_t h = k.first;
            h ^= (k.second << 16) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= (k.third << 24) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    /// 形状组合 → 哈希签名（lhs + rhs，用于区分同一算子的不同形状）
    static size_t hashShapes(const std::vector<size_t>& lhs,
                             const std::vector<size_t>& rhs) {
        size_t h = 1469598103934665603ull; // FNV offset basis
        auto mix = [&h](size_t v) {
            h ^= v;
            h *= 1099511628211ull; // FNV prime
        };
        mix(lhs.size());
        for (auto s : lhs) mix(s);
        mix(rhs.size());
        for (auto s : rhs) mix(s);
        return h;
    }

    static size_t hashShapeInfo(const KernelShapeInfo& shapes) {
        return hashShapes(shapes.lhs_shape, shapes.rhs_shape);
    }

    static KeyType makeKey(op op_type, DeviceType dev, const KernelShapeInfo& shapes) {
        return {static_cast<size_t>(op_type), static_cast<size_t>(dev),
                hashShapeInfo(shapes)};
    }

    /// 从运行时张量构造 key（tryExecute 侧，与 install 侧 hashShapeInfo 对齐）
    static KeyType makeKeyFromShapes(op op_type, DeviceType dev,
                                     const std::vector<size_t>& lhs,
                                     const std::vector<size_t>& rhs) {
        return {static_cast<size_t>(op_type), static_cast<size_t>(dev),
                hashShapes(lhs, rhs)};
    }

    static std::string makeFusedKey(op op_type, const KernelShapeInfo& shapes) {
        std::string key = shapes.fused_pattern;
        key += "|";
        key += std::to_string(static_cast<size_t>(op_type));
        for (auto s : shapes.lhs_shape) key += ":" + std::to_string(s);
        return key;
    }

    /// 形状向量转字符串（用于调试日志）
    static std::string shapeToString(const std::vector<size_t>& shape) {
        if (shape.empty()) return "[]";
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) s += ",";
            s += std::to_string(shape[i]);
        }
        s += "]";
        return s;
    }

    static std::string shapeDebug(const std::vector<size_t>& shape) {
        return shapeToString(shape);
    }

    /// 输出形状验证：C3 kernel 执行后检查输出形状是否匹配预期
    /// 若不匹配，打印调试日志并卸载该 kernel
    /// @return true 表示形状匹配，false 表示不匹配（已卸载）
    bool validateOutputShape(op op_type, DeviceType dev,
                              const Tensor& out,
                              const std::vector<size_t>& expected_shape) {
        if (expected_shape.empty()) return true; // 无预期形状，跳过验证
        if (out.shape() == expected_shape) return true;
        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL,
            ErrorType::DIMENSION,
            "[C3-SHAPE-CHECK] op=" + std::to_string(static_cast<int>(op_type)) +
            " expected=" + shapeToString(expected_shape) +
            " actual=" + shapeToString(out.shape()) +
            " numel=" + std::to_string(out.numel()) +
            " — uninstalling defective kernel");
        uninstall(op_type, dev);
        return false;
    }

    mutable std::mutex mutex_;
    std::unordered_map<KeyType, C3Entry, KeyHash> entries_;

    // 融合 kernel 存储
    std::unordered_map<std::string, FusedEntry> fused_entries_;

    // Backward kernel 存储（key = type_name|shape）
    std::unordered_map<std::string, BackwardEntry> backward_entries_;

    std::atomic<size_t> install_count_{0};
    std::atomic<size_t> uninstall_count_{0};
    std::atomic<size_t> hit_count_{0};
    std::atomic<size_t> miss_count_{0};
    std::atomic<size_t> bypass_count_{0};  ///< DEBT-NEW-7 H2 fix 计数器
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_C3_KERNEL_REGISTRY_H