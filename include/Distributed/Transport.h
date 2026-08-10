/**
 * @file Transport.h
 * @brief TCP 网络传输层 — 可靠的数据传输通道
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details Transport 是 Gen 2 分布式系统的网络传输层，
 *          使用 TCP socket 在节点间传输序列化后的数据。
 *
 *          设计原则：
 *          1. 传输层只关心字节流，不关心数据语义
 *          2. 使用简单的长度前缀协议自描述消息边界
 *          3. 每个连接一个 reader 线程，使用阻塞 I/O
 *          4. 线程安全：所有公共方法可被多线程安全调用
 *          5. 可替换设计：Transport 可与 CommEngine 配合使用，
 *             也可独立使用作为通用传输层
 *
 *          消息协议（Wire Protocol）：
 *          [4 bytes: payload length (uint32_t, big-endian)]
 *          [4 bytes: source node ID (uint32_t, big-endian)]
 *          [N bytes: payload (raw bytes)]
 *
 *          线程模型：
 *          ┌─────────────────────────────────────┐
 *          │ Main Thread                          │
 *          │  ├─ send() / broadcast()             │
 *          │  └─ connectToPeer() / disconnect()   │
 *          ├─────────────────────────────────────┤
 *          │ Acceptor Thread (1)                  │
 *          │  └─ accept() 新连接 → 创建 reader    │
 *          ├─────────────────────────────────────┤
 *          │ Reader Threads (per connection)      │
 *          │  └─ recv() → ReceiveCallback()       │
 *          └─────────────────────────────────────┘
 */

#ifndef CTORCH_DISTRIBUTED_TRANSPORT_H
#define CTORCH_DISTRIBUTED_TRANSPORT_H

#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <memory>

namespace ct {
namespace distributed {

/**
 * @brief 接收回调类型
 * @param source_node_id 源节点 ID
 * @param data 接收到的字节流
 */
using TransportReceiveCallback = std::function<void(uint32_t source_node_id,
                                                    const std::vector<uint8_t>& data)>;

/**
 * @class Transport
 * @brief TCP 网络传输层 — 可靠的数据传输通道
 *
 * 使用方式：
 * @code
 *   // 节点 0
 *   Transport t0(Transport::Config{0, 8000});
 *   t0.start();
 *
 *   // 节点 1
 *   Transport t1(Transport::Config{1, 8001});
 *   t1.setReceiveCallback([](uint32_t src, auto& data) {
 *       std::cout << "收到来自节点 " << src << " 的数据，大小 " << data.size() << std::endl;
 *   });
 *   t1.start();
 *
 *   // 节点 1 连接节点 0
 *   t1.connectToPeer(0, "127.0.0.1", 8000);
 *
 *   // 节点 1 发送数据
 *   t1.send(0, {0x01, 0x02, 0x03});
 * @endcode
 */
class Transport {
public:
    /**
     * @struct Config
     * @brief 传输层配置
     */
    struct Config {
        uint32_t local_node_id = 0;     ///< 本地节点 ID
        uint16_t port = 0;              ///< 监听端口 (0 = 系统分配)
        size_t max_connections = 32;    ///< 最大连接数
        size_t buffer_size = 65536;     ///< 初始接收缓冲区大小 (64KB)
    };

    /**
     * @brief 构造传输层
     * @param config 传输层配置
     */
    explicit Transport(const Config& config);
    ~Transport();

    // 禁止拷贝
    Transport(const Transport&) = delete;
    Transport& operator=(const Transport&) = delete;

    // ======================= 生命周期 =======================

    /**
     * @brief 启动传输层（开始监听）
     * @return true 启动成功
     *
     * 启动 acceptor 线程，开始监听端口接受连接。
     */
    bool start();

    /**
     * @brief 停止传输层
     *
     * 关闭所有连接，终止 acceptor 和 reader 线程。
     * 阻塞直到所有线程退出。
     */
    void stop();

    /**
     * @brief 检查传输层是否正在运行
     * @return true 运行中
     */
    bool isRunning() const { return _running.load(); }

    // ======================= 连接管理 =======================

    /**
     * @brief 连接到对等节点
     * @param node_id 目标节点 ID
     * @param host 目标主机名或 IP
     * @param port 目标端口
     * @return true 连接成功
     *
     * 尝试 TCP 连接，成功后创建 reader 线程。
     */
    bool connectToPeer(uint32_t node_id, const std::string& host, uint16_t port);

    /**
     * @brief 断开与对等节点的连接
     * @param node_id 目标节点 ID
     * @return true 断开成功
     */
    bool disconnect(uint32_t node_id);

    /**
     * @brief 检查是否已连接到指定节点
     * @param node_id 目标节点 ID
     * @return true 已连接
     */
    bool isConnected(uint32_t node_id) const;

    /**
     * @brief 获取当前连接数
     * @return 连接数
     */
    size_t numConnections() const;

    /**
     * @brief 获取所有已连接节点 ID
     * @return 节点 ID 列表
     */
    std::vector<uint32_t> connectedNodes() const;

    // ======================= 数据传输 =======================

    /**
     * @brief 发送数据到指定节点
     * @param target_node_id 目标节点 ID
     * @param data 要发送的数据
     * @return true 发送成功（数据已写入发送缓冲区）
     *
     * 自动添加消息头：[payload_len][source_node_id][payload]
     */
    bool send(uint32_t target_node_id, const std::vector<uint8_t>& data);

    /**
     * @brief 广播数据到所有已连接节点
     * @param data 要广播的数据
     * @param exclude_self 是否排除自身
     *
     * 依次发送给所有已连接节点。
     */
    void broadcast(const std::vector<uint8_t>& data, bool exclude_self = true);

    // ======================= 回调 =======================

    /**
     * @brief 设置接收数据回调
     * @param callback 回调函数
     */
    void setReceiveCallback(TransportReceiveCallback callback) {
        _receive_callback = std::move(callback);
    }

    /**
     * @brief 获取本地端口
     * @return 监听端口号
     */
    uint16_t localPort() const { return _local_port; }

    /**
     * @brief 获取本地节点 ID
     * @return 节点 ID
     */
    uint32_t localNodeId() const { return _config.local_node_id; }

private:
    /**
     * @struct PeerConnection
     * @brief 对等连接内部状态
     */
    struct PeerConnection {
        int socket_fd = -1;              ///< socket 文件描述符
        uint32_t node_id = 0;            ///< 对端节点 ID
        std::thread reader_thread;       ///< reader 线程
        std::atomic<bool> active{false}; ///< 是否活跃
    };

    // ======================= 内部线程 =======================

    /**
     * @brief Acceptor 线程主循环
     *
     * 阻塞在 accept() 上等待新连接。
     * 接受到新连接后，确定对端节点 ID 并创建 reader 线程。
     */
    void acceptorLoop();

    /**
     * @brief Reader 线程主循环
     * @param conn 连接对象指针
     *
     * 阻塞在 recv() 上读取数据。
     * 收到完整消息后调用 ReceiveCallback。
     * 连接断开或出错时退出。
     */
    void readerLoop(PeerConnection* conn);

    /**
     * @brief 接收完整消息头
     * @param fd socket 文件描述符
     * @param header 输出：消息头（8 字节）
     * @return true 接收成功
     */
    bool recvHeader(int fd, uint8_t header[8]);

    /**
     * @brief 发送完整数据（处理部分写入）
     * @param fd socket 文件描述符
     * @param data 数据指针
     * @param len 数据长度
     * @return true 发送成功
     */
    bool sendAll(int fd, const uint8_t* data, size_t len);

    /**
     * @brief 接收完整数据（处理部分读取）
     * @param fd socket 文件描述符
     * @param data 数据缓冲区
     * @param len 期望接收长度
     * @return true 接收成功
     */
    bool recvAll(int fd, uint8_t* data, size_t len);

    // ======================= 辅助 =======================

    /**
     * @brief 将 uint32_t 转换为网络字节序（大端）
     */
    static void hton32(uint32_t val, uint8_t* buf) {
        buf[0] = static_cast<uint8_t>((val >> 24) & 0xFF);
        buf[1] = static_cast<uint8_t>((val >> 16) & 0xFF);
        buf[2] = static_cast<uint8_t>((val >> 8) & 0xFF);
        buf[3] = static_cast<uint8_t>(val & 0xFF);
    }

    /**
     * @brief 将网络字节序（大端）转换为 uint32_t
     */
    static uint32_t ntoh32(const uint8_t* buf) {
        return (static_cast<uint32_t>(buf[0]) << 24)
             | (static_cast<uint32_t>(buf[1]) << 16)
             | (static_cast<uint32_t>(buf[2]) << 8)
             | static_cast<uint32_t>(buf[3]);
    }

    // ======================= 成员变量 =======================

    Config _config;
    uint16_t _local_port = 0;

    // 服务器 socket
    int _server_fd = -1;

    // 线程控制
    std::thread _acceptor_thread;
    std::atomic<bool> _running{false};

    // 连接管理
    mutable std::mutex _connections_mutex;
    std::unordered_map<uint32_t, std::unique_ptr<PeerConnection>> _connections;

    // 回调
    TransportReceiveCallback _receive_callback;
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_TRANSPORT_H