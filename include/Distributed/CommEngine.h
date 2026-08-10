/**
 * @file CommEngine.h
 * @brief 通信引擎 — 后端感知通信、序列化、压缩与传输
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details 本文件实现 CommEngine，是 BANT 架构中"通信逻辑"的核心组件。
 *          CommEngine 封装了所有与通信相关的逻辑：
 *            - 后端感知的序列化/反序列化（使用 CDTF 协议）
 *            - 率失真自适应压缩（继承 Gen 1 的 RD-LocalSGD）
 *            - 动态拓扑管理（继承 Gen 1 的 TASS）
 *            - 跨后端数据传输
 *
 *          设计原则（分离原理 — Feedback Systems #11）：
 *          CommEngine 只负责"如何传输"，不关心"传输什么"。
 *          梯度值、优化器状态等语义信息由 DistributedOptimizer 处理。
 *
 *          CommEngine 与 DistributedOptimizer 完全解耦：
 *          CommEngine 只操作字节流（std::vector<uint8_t>），
 *          DistributedOptimizer 只操作张量（Tensor）。
 */

#ifndef CTORCH_DISTRIBUTED_COMM_ENGINE_H
#define CTORCH_DISTRIBUTED_COMM_ENGINE_H

#include "CDTF.h"
#include "DeviceBackend.h"
#include "BackendManager.h"
#include "Transport.h"
#include "Tensor.h"

#include <memory>
#include <vector>
#include <functional>
#include <chrono>
#include <unordered_map>
#include <cstdint>

namespace ct {
namespace distributed {

/**
 * @brief 节点 ID 类型
 */
using NodeId = uint32_t;

/**
 * @brief 通信回调类型 — 接收梯度后的处理函数
 */
using GradientCallback = std::function<void(NodeId source, const Tensor& gradient)>;

/**
 * @struct NodeInfo
 * @brief 节点信息，用于拓扑管理
 */
struct NodeInfo {
    NodeId id;                     ///< 节点 ID
    std::string address;           ///< 网络地址 (IP:Port)
    DeviceType backend_type;       ///< 后端类型
    float rtt_ms;                  ///< 往返延迟 (ms)
    float bandwidth_mbps;          ///< 测量带宽 (Mbps)
    bool is_active;                ///< 是否活跃
    std::chrono::steady_clock::time_point last_seen;  ///< 最后可见时间
    uint32_t compatibility_score;  ///< 后端兼容性评分 (0-100)
};

/**
 * @struct CompressionConfig
 * @brief 自适应压缩配置（继承 Gen 1 的 RD-LocalSGD）
 */
struct CompressionConfig {
    bool enable_quantization;      ///< 是否启用量化
    uint8_t quantize_bits;         ///< 量化位数 (8/16)
    bool enable_entropy_coding;    ///< 是否启用熵编码
    float target_compression_ratio; ///< 目标压缩比
    float entropy_threshold;       ///< 熵阈值，低于此值启用更强压缩

    static CompressionConfig defaultConfig() {
        return CompressionConfig{
            true,       // enable_quantization
            16,         // quantize_bits
            true,       // enable_entropy_coding
            0.5f,       // target_compression_ratio
            0.8f        // entropy_threshold
        };
    }
};

/**
 * @class CommEngine
 * @brief 通信引擎 — 后端感知的序列化、压缩与传输
 *
 * CommEngine 是 Gen 2 分布式系统的通信中枢，负责：
 * 1. 将梯度张量序列化为 CDTF 字节流
 * 2. 根据带宽和梯度熵自适应压缩
 * 3. 通过后端感知的传输层发送/接收
 * 4. 管理动态拓扑（节点发现、延迟测量、邻居选择）
 *
 * @note CommEngine 不持有任何优化器状态，所有状态由 DistributedOptimizer 管理。
 *       这种分离保证了"优化器不关心后端类型，通信层不关心张量语义"。
 */
class CommEngine {
public:
    /**
     * @brief 构造 CommEngine
     * @param local_node_id 本地节点 ID
     */
    explicit CommEngine(NodeId local_node_id);

    ~CommEngine() = default;

    // ======================= 节点管理 =======================

    /**
     * @brief 注册邻居节点
     * @param node_info 节点信息
     */
    void registerNode(const NodeInfo& node_info);

    /**
     * @brief 注销节点
     * @param node_id 节点 ID
     */
    void unregisterNode(NodeId node_id);

    /**
     * @brief 获取所有活跃节点
     * @return 节点信息列表
     */
    std::vector<NodeInfo> activeNodes() const;

    /**
     * @brief 获取本地节点 ID
     * @return 节点 ID
     */
    NodeId localNodeId() const { return _local_node_id; }

    // ======================= 梯度发送/接收 =======================

    /**
     * @brief 发送梯度到目标节点
     * @param grad 梯度张量
     * @param target 目标节点 ID
     * @param flags CDTF 序列化标志
     *
     * 自动完成：序列化 → 压缩 → 传输
     */
    void sendGradient(const Tensor& grad, NodeId target,
                      uint16_t flags = CDTF_FLAG_NONE);

    /**
     * @brief 广播梯度到所有节点
     * @param grad 梯度张量
     * @param flags CDTF 序列化标志
     */
    void broadcastGradient(const Tensor& grad,
                           uint16_t flags = CDTF_FLAG_NONE);

    /**
     * @brief 设置梯度接收回调
     * @param callback 回调函数 (source, gradient)
     */
    void setGradientCallback(GradientCallback callback) {
        _gradient_callback = std::move(callback);
    }

    // ======================= 网络传输层 =======================

    /**
     * @brief 设置 TCP 传输层
     * @param transport 传输层共享指针
     *
     * 设置后，sendGradient/broadcastGradient 将使用真正的 TCP 传输。
     * 未设置时使用本地回调占位实现（用于单机测试）。
     *
     * @note Transport 的生命周期由调用方管理，CommEngine 不持有所有权。
     */
    void setTransport(std::shared_ptr<Transport> transport) {
        _transport = std::move(transport);
    }

    /**
     * @brief 获取当前传输层
     * @return 传输层共享指针（可能为空）
     */
    std::shared_ptr<Transport> transport() const { return _transport; }

    // ======================= 压缩控制 =======================

    /**
     * @brief 设置压缩配置
     * @param config 压缩配置
     */
    void setCompressionConfig(const CompressionConfig& config) {
        _compression_config = config;
    }

    /**
     * @brief 获取当前压缩配置
     * @return 压缩配置
     */
    const CompressionConfig& compressionConfig() const {
        return _compression_config;
    }

    /**
     * @brief 根据梯度熵自适应选择压缩参数
     * @param grad 梯度张量
     * @return 自适应选择的 CDTF 标志位
     *
     * 实现 Gen 1 的 RD-LocalSGD 自适应：当梯度熵低时用更强压缩，
     * 熵高时保留更多精度。
     */
    uint16_t adaptiveCompressionFlags(const Tensor& grad);

    // ======================= 拓扑管理 (继承 TASS) =======================

    /**
     * @brief 更新节点延迟测量
     * @param node_id 节点 ID
     * @param rtt_ms 测量到的 RTT 延迟 (ms)
     */
    void updateLatency(NodeId node_id, float rtt_ms);

    /**
     * @brief 获取邻居节点列表（按延迟排序）
     * @param max_neighbors 最大邻居数（0 = 全部）
     * @return 节点 ID 列表，按延迟升序
     */
    std::vector<NodeId> getNeighbors(size_t max_neighbors = 0) const;

    /**
     * @brief 获取后端兼容的邻居节点
     * @param target_backend 目标后端类型
     * @param max_neighbors 最大邻居数
     * @return 节点 ID 列表
     *
     * 扩展自 Gen 1 TASS：邻居选择从"按延迟"扩展为"按延迟+后端兼容性"。
     */
    std::vector<NodeId> getCompatibleNeighbors(DeviceType target_backend,
                                                size_t max_neighbors = 0) const;

    // ======================= 统计信息 =======================

    /**
     * @brief 通信统计信息
     */
    struct Stats {
        size_t bytes_sent;         ///< 发送字节总数（压缩后）
        size_t bytes_received;     ///< 接收字节总数
        size_t messages_sent;      ///< 发送消息总数
        size_t messages_received;  ///< 接收消息总数
        float compression_ratio;   ///< 平均压缩比
        double avg_latency_ms;     ///< 平均延迟 (ms)

        // 压缩统计（新增）
        size_t raw_bytes_sent;     ///< 压缩前原始字节总数
        size_t compressed_bytes_sent; ///< 压缩后字节总数
        size_t compression_count;      ///< 应用压缩的次数
        size_t quantize_8_count;       ///< 8-bit 量化次数
        size_t quantize_16_count;      ///< 16-bit 量化次数

        /// 有效压缩比（raw / compressed），>1 表示压缩有效
        float effective_ratio() const {
            if (compressed_bytes_sent == 0) return 1.0f;
            return static_cast<float>(raw_bytes_sent) / compressed_bytes_sent;
        }

        void reset() {
            bytes_sent = 0;
            bytes_received = 0;
            messages_sent = 0;
            messages_received = 0;
            compression_ratio = 1.0f;
            avg_latency_ms = 0.0;
            raw_bytes_sent = 0;
            compressed_bytes_sent = 0;
            compression_count = 0;
            quantize_8_count = 0;
            quantize_16_count = 0;
        }
    };

    /**
     * @brief 获取统计信息
     * @return 当前统计信息快照
     */
    Stats stats() const { return _stats; }

    /**
     * @brief 重置统计信息
     */
    void resetStats() { _stats.reset(); }

private:
    NodeId _local_node_id;

    // 节点表
    mutable std::mutex _nodes_mtx;
    std::unordered_map<NodeId, NodeInfo> _nodes;

    // 压缩配置
    CompressionConfig _compression_config;

    // 回调
    GradientCallback _gradient_callback;

    // 传输层（可选，未设置时使用占位实现）
    std::shared_ptr<Transport> _transport;

    // 统计
    Stats _stats;

    /**
     * @brief 内部实现：发送序列化数据
     * @param data 序列化后的字节流
     * @param target 目标节点 ID
     *
     * @note 当前为占位实现，使用本地回调模拟网络传输。
     *       完整实现将使用 TCP/UDP socket 或 MPI 通信。
     */
    void transmitData(const std::vector<uint8_t>& data, NodeId target);

    /**
     * @brief 内部实现：接收数据
     * @param source 源节点 ID
     * @return 接收到的字节流
     */
    std::vector<uint8_t> receiveData(NodeId source);
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_COMM_ENGINE_H