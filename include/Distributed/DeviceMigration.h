/**
 * @file DeviceMigration.h
 * @brief 设备迁移 — 自然变换模板
 *       编译期安全的跨后端张量迁移路径
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details 本文件实现自然变换（Natural Transformation）概念：
 *          后端间的张量迁移是函子范畴中的态射。
 *
 *          核心设计原则（CTFP #15-17）：
 *          1. 自然变换 = 设备迁移: Migrate<From, To> 是 Backend 范畴中的态射
 *          2. 自然性条件: 先计算再迁移 == 先迁移再计算
 *             f(to(t)) == to(f(t)) 对所有算子 f 成立
 *          3. 中立空间: 所有跨后端迁移均通过 CPU 中立空间中转，
 *             保证聚合操作与设备迁移可交换
 *
 *          编译期安全性：只有显式模板特化的迁移路径才是合法的，
 *          未定义的路径在编译期报错。
 */

#ifndef CTORCH_DISTRIBUTED_DEVICE_MIGRATION_H
#define CTORCH_DISTRIBUTED_DEVICE_MIGRATION_H

#include "DeviceBackend.h"
#include "BackendManager.h"
#include "Tensor.h"

#include <type_traits>

namespace ct {
namespace distributed {

/**
 * @struct MigrationTraits
 * @brief 迁移路径特性 — 编译期描述迁移路径属性
 * @tparam From 源 DeviceType
 * @tparam To 目标 DeviceType
 */
template <DeviceType From, DeviceType To>
struct MigrationTraits {
    /// 该迁移路径是否被支持
    static constexpr bool supported = false;
    /// 迁移是否需要经过中立空间（CPU buffer）
    static constexpr bool needs_neutral = true;
    /// 迁移是否涉及网络传输（跨节点）
    static constexpr bool is_network = false;
    /// 迁移路径的额外开销因子（相对于 memcpy 的倍数）
    static constexpr float overhead_factor = 1.0f;
};

/**
 * @class DeviceMigration
 * @brief 自然变换模板 — 后端范畴中的态射
 *
 * 默认模板不提供实现，只有显式特化的路径才是合法的。
 * 所有迁移路径必须保证自然性条件：
 *   先计算再迁移 == 先迁移再计算
 *
 * 实现方式：通过"源后端 → 中立空间（CPU）→ 目标后端"的两步路径，
 * 保证自然性条件的成立。
 *
 * @tparam From 源 DeviceType
 * @tparam To 目标 DeviceType
 */
template <DeviceType From, DeviceType To>
class DeviceMigration {
public:
    /**
     * @brief 执行跨后端迁移
     * @param src 源张量
     * @return 在目标后端上的张量
     * @throws CtorchError 如果迁移路径不支持或迁移失败
     *
     * 默认实现：通过 CPU 中立空间中转。
     * 子类可以提供更高效的直接迁移路径。
     */
    static Tensor migrate(const Tensor& src) {
        static_assert(MigrationTraits<From, To>::supported,
                      "Migration path not supported. Specialize MigrationTraits "
                      "and DeviceMigration for this path, or use CPU neutral space.");

        // 默认实现：源 → CPU → 目标
        auto neutral = src.to(DeviceType::kCPU);
        return neutral.to(To);
    }

    /**
     * @brief 验证自然性条件
     *        先计算再迁移 == 先迁移再计算
     * @param op_type 算子类型
     * @param a 输入张量 A
     * @param b 输入张量 B
     * @return 两条路径结果的逐元素最大差异
     *
     * 用于 MVE 实验中的验证，不用于生产路径。
     */
    static float verifyNaturality(::op op_type, const Tensor& a, const Tensor& b) {
        // 路径 1: 先计算再迁移
        Tensor result_on_from;
        {
            auto backend = BackendManager::getInstance().getBackend(From);
            if (!backend) {
                CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
                    "DeviceMigration: source backend not registered");
            }
            Tensor local_result(ShapeTag{}, a.shape(), a.dtype(), From, false);
            backend->executeKernel(op_type, a, b, local_result);
            auto migrated = migrate(local_result);
            result_on_from = migrated.to(DeviceType::kCPU);
        }

        // 路径 2: 先迁移再计算
        Tensor result_on_to;
        {
            auto a_migrated = migrate(a);
            auto b_migrated = migrate(b);
            auto backend = BackendManager::getInstance().getBackend(To);
            if (!backend) {
                CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
                    "DeviceMigration: target backend not registered");
            }
            Tensor remote_result(ShapeTag{}, a_migrated.shape(), a_migrated.dtype(), To, false);
            backend->executeKernel(op_type, a_migrated, b_migrated, remote_result);
            result_on_to = remote_result.to(DeviceType::kCPU);
        }

        // 计算最大差异
        float max_diff = 0.0f;
        size_t n = std::min(result_on_from.numel(), result_on_to.numel());
        for (size_t i = 0; i < n; ++i) {
            float diff = std::abs(result_on_from.data_read<float>()[i] -
                                   result_on_to.data_read<float>()[i]);
            if (diff > max_diff) max_diff = diff;
        }
        return max_diff;
    }
};

// ======================= 基础迁移路径特化 =======================

/**
 * @brief CPU ↔ CPU 迁移（空操作）
 */
template <>
struct MigrationTraits<DeviceType::kCPU, DeviceType::kCPU> {
    static constexpr bool supported = true;
    static constexpr bool needs_neutral = false;
    static constexpr bool is_network = false;
    static constexpr float overhead_factor = 0.0f;
};

/**
 * @brief MPS → CPU 迁移
 */
template <>
struct MigrationTraits<DeviceType::kMPS, DeviceType::kCPU> {
    static constexpr bool supported = true;
    static constexpr bool needs_neutral = false;
    static constexpr bool is_network = false;
    static constexpr float overhead_factor = 1.0f;
};

/**
 * @brief CPU → MPS 迁移
 */
template <>
struct MigrationTraits<DeviceType::kCPU, DeviceType::kMPS> {
    static constexpr bool supported = true;
    static constexpr bool needs_neutral = false;
    static constexpr bool is_network = false;
    static constexpr float overhead_factor = 1.0f;
};

/**
 * @brief MPS → MPS 迁移（空操作）
 */
template <>
struct MigrationTraits<DeviceType::kMPS, DeviceType::kMPS> {
    static constexpr bool supported = true;
    static constexpr bool needs_neutral = false;
    static constexpr bool is_network = false;
    static constexpr float overhead_factor = 0.0f;
};

// ======================= 跨后端迁移路径特化（通过中立空间） =======================

/**
 * @brief MPS → CUDA 迁移（通过 CPU 中立空间）
 *
 * 由于 MPS 和 CUDA 没有直接通信路径，必须通过 CPU 缓冲区中转。
 * 这保证了自然性条件：聚合操作在 CPU 中立空间中执行，
 * 与设备迁移可交换。
 */
template <>
class DeviceMigration<DeviceType::kMPS, DeviceType::kCUDA> {
public:
    static Tensor migrate(const Tensor& src) {
        if (src.device() != DeviceType::kMPS) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
                "DeviceMigration: source tensor is not on MPS device");
        }
        // MPS → CPU（通过 Tensor.to）
        auto cpu_tensor = src.to(DeviceType::kCPU);
        // CPU → CUDA（通过 Tensor.to；CUDA 后端需先注册到 BackendManager）
        return cpu_tensor.to(DeviceType::kCUDA);
    }
};

/**
 * @brief CUDA → MPS 迁移（通过 CPU 中立空间）
 */
template <>
class DeviceMigration<DeviceType::kCUDA, DeviceType::kMPS> {
public:
    static Tensor migrate(const Tensor& src) {
        if (src.device() != DeviceType::kCUDA) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
                "DeviceMigration: source tensor is not on CUDA device");
        }
        auto cpu_tensor = src.to(DeviceType::kCPU);
        return cpu_tensor.to(DeviceType::kMPS);
    }
};

/**
 * @brief 便捷迁移函数 — 根据源和目标设备自动选择迁移路径
 * @param src 源张量
 * @param target_device 目标设备类型
 * @return 迁移后的张量
 * @throws CtorchError 如果迁移路径不支持
 */
inline Tensor migrateTensor(const Tensor& src, DeviceType target_device) {
    DeviceType src_device = src.device();

    if (src_device == target_device) {
        return src;  // 同设备无需迁移
    }

    // 通过 CPU 中立空间通用迁移
    auto cpu_tensor = src.to(DeviceType::kCPU);
    return cpu_tensor.to(target_device);
}

/**
 * @brief 跨节点迁移（涉及网络传输）
 * @param src 源张量
 * @param target_node 目标节点 ID
 * @param target_device 目标设备类型
 * @return 迁移后的张量（在目标节点上）
 *
 * @note 当前为占位实现，完整实现需集成 CommEngine 的网络传输层。
 */
inline Tensor migrateTensorAcrossNodes(const Tensor& src,
                                        uint32_t target_node,
                                        DeviceType target_device) {
    (void)target_node;  // 预留：网络传输逻辑
    return migrateTensor(src, target_device);
}

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_DEVICE_MIGRATION_H