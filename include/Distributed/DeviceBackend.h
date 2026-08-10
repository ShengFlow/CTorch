/**
 * @file DeviceBackend.h
 * @brief 后端抽象接口 — 函子范畴中的对象
 *       每个后端是一个函子 F: Op → Kernel，将算子映射到具体实现
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details 本文件定义了 DeviceBackend 抽象基类，是 Gen 2 分布式系统
 *          后端感知自然变换架构（BANT）的核心抽象层。
 *          每个后端必须实现四个核心原语：allocate/deallocate/execute/serialize
 *          以及 capability 查询接口。
 */

#ifndef CTORCH_DISTRIBUTED_DEVICE_BACKEND_H
#define CTORCH_DISTRIBUTED_DEVICE_BACKEND_H

#include "Ctools.h"
#include "CtorchError.h"
#include "Tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace ct {
namespace distributed {

/**
 * @struct BackendCapability
 * @brief 后端能力描述，用于调度决策和负载均衡
 */
struct BackendCapability {
    DeviceType device;             ///< 后端设备类型
    float compute_throughput;      ///< 计算吞吐量 (TFLOPS)
    size_t memory_bandwidth;       ///< 内存带宽 (GB/s)
    bool unified_memory;           ///< 是否统一内存架构 (MPS=true, CUDA=false)
    float numerical_precision;     ///< 相对数值精度 (float32=1.0 基准)
    size_t available_memory;       ///< 可用显存/内存 (bytes)
    uint32_t node_id;              ///< 节点 ID
    std::string backend_name;      ///< 后端名称 (如 "MPS", "CUDA", "CPU")

    /**
     * @brief 计算综合算力评分
     * @return 归一化后的评分 (0.0 ~ 1.0)
     */
    float compositeScore() const noexcept {
        // 综合评分 = 0.5 * 吞吐量归一化 + 0.3 * 精度 + 0.2 * 带宽归一化
        constexpr float kThroughputRef = 100.0f;  // 100 TFLOPS 参考值
        constexpr float kBandwidthRef = 500.0f;    // 500 GB/s 参考值
        float t_score = std::min(compute_throughput / kThroughputRef, 1.0f);
        float b_score = std::min(memory_bandwidth / static_cast<float>(kBandwidthRef), 1.0f);
        return 0.5f * t_score + 0.3f * numerical_precision + 0.2f * b_score;
    }
};

/**
 * @class DeviceBackend
 * @brief 后端抽象基类 — 函子范畴中的对象
 *
 * 每个后端是一个函子 F: Op → Kernel，将算子映射到具体实现。
 * 后端需要实现四个核心原语：
 *   - allocate/deallocate: 内存管理
 *   - executeKernel: 算子执行
 *   - serialize/deserialize: 跨后端序列化
 *   - capability: 后端能力查询
 *
 * @note 新增后端（如 ROCm、Vulkan）只需继承此类并实现全部纯虚函数，
 *       然后在 BackendManager 中注册即可，无需修改核心调度器。
 */
class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;

    // ======================= 内存管理原语 =======================

    /**
     * @brief 分配后端内存
     * @param bytes 字节数
     * @return 分配的内存指针，失败返回 nullptr
     */
    virtual void* allocate(size_t bytes) = 0;

    /**
     * @brief 释放后端内存
     * @param ptr 要释放的内存指针
     */
    virtual void deallocate(void* ptr) = 0;

    // ======================= 算子执行原语 =======================

    /**
     * @brief 执行后端算子
     * @param op_type 算子类型
     * @param a 输入张量 A
     * @param b 输入张量 B (一元算子可传空 Tensor)
     * @param out 输出张量（引用）
     */
    virtual void executeKernel(::op op_type, const Tensor& a,
                                const Tensor& b, Tensor& out) = 0;

    // ======================= 序列化原语 (Gen 2 新增) =======================

    /**
     * @brief 将张量序列化为字节流
     * @param t 输入张量
     * @return 序列化后的字节流
     * @throws CtorchError 如果序列化失败
     */
    virtual std::vector<uint8_t> serialize(const Tensor& t) = 0;

    /**
     * @brief 从字节流反序列化为张量
     * @param data 序列化后的字节流
     * @return 反序列化后的张量
     * @throws CtorchError 如果反序列化失败
     */
    virtual Tensor deserialize(const std::vector<uint8_t>& data) = 0;

    // ======================= 后端能力查询 =======================

    /**
     * @brief 查询后端能力
     * @return BackendCapability 结构体
     */
    virtual BackendCapability capability() const = 0;

    /**
     * @brief 获取后端类型
     * @return DeviceType 枚举值
     */
    virtual DeviceType deviceType() const noexcept = 0;

    /**
     * @brief 获取后端名称
     * @return 后端名称字符串
     */
    virtual const char* name() const noexcept = 0;

    // ======================= 同步原语 =======================

    /**
     * @brief 同步后端，确保所有未完成操作完成
     */
    virtual void synchronize() = 0;

    /**
     * @brief 标记缓冲区已修改（在 host 写入后通知后端）
     * @param ptr 缓冲区指针
     * @param bytes 修改的字节数
     */
    virtual void markBufferModified(void* ptr, size_t bytes) = 0;
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_DEVICE_BACKEND_H