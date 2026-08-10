/**
 * @file Transport.cpp
 * @brief TCP 网络传输层实现
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 *
 * 使用 POSIX socket API（macOS/Linux 兼容）。
 * 线程模型：1 个 acceptor 线程 + 每个连接 1 个 reader 线程。
 *
 * 消息协议：
 * [4 bytes: payload length (network byte order)]
 * [4 bytes: source node ID (network byte order)]
 * [N bytes: payload]
 */

#include "Distributed/Transport.h"
#include "CtorchError.h"

#include <cstring>
#include <system_error>
#include <utility>
#include <thread>
#include <chrono>
#include <future>
#include <vector>

// POSIX socket headers
#include <sys/socket.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <netdb.h>
#include <errno.h>

namespace ct {
namespace distributed {

// ======================= 构造/析构 =======================

/// 高性能 TCP 缓冲区大小（4 MB）
static constexpr size_t kTcpBufferSize = 4 * 1024 * 1024;

/// 调优 TCP socket 参数（前向声明）
static void tuneSocket(int fd);

Transport::Transport(const Config& config)
    : _config(config)
{
}

Transport::~Transport() {
    if (_running.load()) {
        stop();
    }
}

// ======================= Socket 调优 =======================

/**
 * @brief 调优 TCP socket 参数以获得最佳性能
 *
 * - 放大发送/接收缓冲区以减少 ACK 等待
 * - 禁用 Nagle 算法避免延迟累积
 */
static void tuneSocket(int fd) {
    int optval = 1;
    // 禁用 Nagle 算法
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval));
    // 放大发送缓冲区
    int sndbuf = static_cast<int>(kTcpBufferSize);
    (void)::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
    // 放大接收缓冲区
    int rcvbuf = static_cast<int>(kTcpBufferSize);
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
}

/**
 * @brief 使用 writev 零拷贝发送 header + payload
 *
 * 相比两次 sendAll 调用，writev 减少一次系统调用并避免 header 的内存拷贝。
 * 正确处理部分写入（partial write）。
 */
static bool sendAllV(int fd, const uint8_t* header, size_t header_len,
                      const uint8_t* payload, size_t payload_len) {
    struct iovec iov[2];
    iov[0].iov_base = const_cast<uint8_t*>(header);
    iov[0].iov_len = header_len;
    int iovcnt = 1;
    if (payload_len > 0) {
        iov[1].iov_base = const_cast<uint8_t*>(payload);
        iov[1].iov_len = payload_len;
        iovcnt = 2;
    }

    size_t total = header_len + payload_len;
    size_t sent = 0;

    while (sent < total) {
        ssize_t n = ::writev(fd, iov, iovcnt);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);

        // 更新 iovec 处理部分写入
        size_t remaining = static_cast<size_t>(n);
        for (int i = 0; i < iovcnt && remaining > 0; ++i) {
            if (remaining < iov[i].iov_len) {
                iov[i].iov_base = static_cast<uint8_t*>(iov[i].iov_base) + remaining;
                iov[i].iov_len -= remaining;
                remaining = 0;
            } else {
                remaining -= iov[i].iov_len;
                iov[i].iov_len = 0;
            }
        }
    }
    return true;
}

// ======================= 生命周期 =======================

bool Transport::start() {
    if (_running.load()) {
        return true;  // 已在运行
    }

    // 创建监听 socket
    _server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_server_fd < 0) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
            std::string("Transport: failed to create socket: ") + std::strerror(errno));
        return false;
    }

    // 允许端口重用（避免 TIME_WAIT 问题）
    int optval = 1;
    if (::setsockopt(_server_fd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
            std::string("Transport: failed to set SO_REUSEADDR: ") + std::strerror(errno));
    }

    // 绑定地址
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(_config.port);

    if (::bind(_server_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
            std::string("Transport: bind failed on port ") + std::to_string(_config.port)
            + ": " + std::strerror(errno));
        ::close(_server_fd);
        _server_fd = -1;
        return false;
    }

    // 设置端口
    _local_port = _config.port;
    CtorchError::log(ErrorLevel::INFO, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
        "Transport::start() _config.port=" + std::to_string(_config.port)
        + " _local_port=" + std::to_string(_local_port));

    // 如果端口为 0，获取系统分配的实际端口
    if (_local_port == 0) {
        struct sockaddr_in bound_addr;
        for (int retry = 0; retry < 10; ++retry) {
            socklen_t addr_len = sizeof(bound_addr);
            std::memset(&bound_addr, 0, sizeof(bound_addr));
            if (::getsockname(_server_fd, reinterpret_cast<struct sockaddr*>(&bound_addr), &addr_len) == 0) {
                _local_port = ntohs(bound_addr.sin_port);
                if (_local_port > 0) {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (_local_port == 0) {
            CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
                "Transport: failed to get assigned port after bind");
            ::close(_server_fd);
            _server_fd = -1;
            return false;
        }
    }

    // 开始监听
    if (::listen(_server_fd, _config.max_connections) < 0) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
            std::string("Transport: listen failed: ") + std::strerror(errno));
        ::close(_server_fd);
        _server_fd = -1;
        return false;
    }

    _running.store(true);

    // 启动 acceptor 线程
    _acceptor_thread = std::thread(&Transport::acceptorLoop, this);

    CtorchError::log(ErrorLevel::INFO, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
        "Transport: node " + std::to_string(_config.local_node_id)
        + " listening on port " + std::to_string(_local_port));

    return true;
}

void Transport::stop() {
    if (!_running.load()) {
        return;
    }

    _running.store(false);

    // 关闭 server socket 以唤醒 acceptor 线程
    if (_server_fd >= 0) {
        ::close(_server_fd);
        _server_fd = -1;
    }

    // 关闭所有连接
    {
        std::lock_guard<std::mutex> lock(_connections_mutex);
        for (auto& [_, conn] : _connections) {
            if (conn->socket_fd >= 0) {
                conn->active.store(false);
                ::shutdown(conn->socket_fd, SHUT_RDWR);
                ::close(conn->socket_fd);
                conn->socket_fd = -1;
            }
        }
    }

    // 等待 acceptor 线程退出
    if (_acceptor_thread.joinable()) {
        _acceptor_thread.join();
    }

    // 等待所有 reader 线程退出
    {
        std::lock_guard<std::mutex> lock(_connections_mutex);
        for (auto& [_, conn] : _connections) {
            if (conn->reader_thread.joinable()) {
                conn->reader_thread.join();
            }
        }
        _connections.clear();
    }

    CtorchError::log(ErrorLevel::INFO, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
        "Transport: node " + std::to_string(_config.local_node_id) + " stopped");
}

// ======================= 连接管理 =======================

bool Transport::connectToPeer(uint32_t node_id, const std::string& host, uint16_t port) {
    {
        std::lock_guard<std::mutex> lock(_connections_mutex);
        if (_connections.find(node_id) != _connections.end()) {
            // 已经连接
            return true;
        }
    }

    // 创建 socket
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
            std::string("Transport: failed to create socket for node ") + std::to_string(node_id)
            + ": " + std::strerror(errno));
        return false;
    }

    // 解析地址
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        // 尝试 DNS 解析
        struct addrinfo hints, *res;
        std::memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        if (::getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0) {
            CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
                std::string("Transport: failed to resolve host ") + host
                + ": " + std::strerror(errno));
            ::close(fd);
            return false;
        }

        if (res) {
            auto* sa = reinterpret_cast<struct sockaddr_in*>(res->ai_addr);
            addr.sin_addr = sa->sin_addr;
            ::freeaddrinfo(res);
        } else {
            ::close(fd);
            return false;
        }
    }

    // 连接
    if (::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
            std::string("Transport: connect to node ") + std::to_string(node_id)
            + " at " + host + ":" + std::to_string(port) + " failed: " + std::strerror(errno));
        ::close(fd);
        return false;
    }

    // 发送握手消息（空 payload），让 acceptor 能立即获取 source node ID
    // 握手消息格式：[4 bytes: payload_len=0][4 bytes: source_node_id]
    {
        uint8_t handshake[8];
        hton32(0, handshake);                                    // payload_len = 0
        hton32(_config.local_node_id, handshake + 4);            // 本地节点 ID
        if (!sendAll(fd, handshake, 8)) {
            CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
                "Transport: failed to send handshake to node " + std::to_string(node_id));
            ::close(fd);
            return false;
        }
    }

    // 调优 TCP 参数（大缓冲区 + 无 Nagle）
    tuneSocket(fd);

    // 创建连接对象
    auto conn = std::make_unique<PeerConnection>();
    conn->socket_fd = fd;
    conn->node_id = node_id;
    conn->active.store(true);

    auto* conn_ptr = conn.get();

    {
        std::lock_guard<std::mutex> lock(_connections_mutex);
        _connections[node_id] = std::move(conn);
    }

    // 启动 reader 线程
    conn_ptr->reader_thread = std::thread(&Transport::readerLoop, this, conn_ptr);

    return true;
}

bool Transport::disconnect(uint32_t node_id) {
    std::unique_ptr<PeerConnection> conn;

    {
        std::lock_guard<std::mutex> lock(_connections_mutex);
        auto it = _connections.find(node_id);
        if (it == _connections.end()) {
            return false;
        }
        conn = std::move(it->second);
        _connections.erase(it);
    }

    // 关闭连接
    if (conn) {
        conn->active.store(false);
        if (conn->socket_fd >= 0) {
            ::shutdown(conn->socket_fd, SHUT_RDWR);
            ::close(conn->socket_fd);
            conn->socket_fd = -1;
        }
        if (conn->reader_thread.joinable()) {
            conn->reader_thread.join();
        }
    }

    return true;
}

bool Transport::isConnected(uint32_t node_id) const {
    std::lock_guard<std::mutex> lock(_connections_mutex);
    return _connections.find(node_id) != _connections.end();
}

size_t Transport::numConnections() const {
    std::lock_guard<std::mutex> lock(_connections_mutex);
    return _connections.size();
}

std::vector<uint32_t> Transport::connectedNodes() const {
    std::lock_guard<std::mutex> lock(_connections_mutex);
    std::vector<uint32_t> nodes;
    nodes.reserve(_connections.size());
    for (const auto& [id, _] : _connections) {
        nodes.push_back(id);
    }
    return nodes;
}

// ======================= 数据传输 =======================

bool Transport::send(uint32_t target_node_id, const std::vector<uint8_t>& data) {
    PeerConnection* conn = nullptr;

    {
        std::lock_guard<std::mutex> lock(_connections_mutex);
        auto it = _connections.find(target_node_id);
        if (it == _connections.end() || !it->second->active.load()) {
            CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
                "Transport: cannot send to node " + std::to_string(target_node_id) + " (not connected)");
            return false;
        }
        conn = it->second.get();
    }

    // 构建消息头
    // [4 bytes: payload_len][4 bytes: source_node_id][N bytes: payload]
    uint8_t header[8];
    hton32(static_cast<uint32_t>(data.size()), header);
    hton32(_config.local_node_id, header + 4);

    // 发送消息头
    // 使用 writev 零拷贝：一次系统调用发送 header + payload
    if (!sendAllV(conn->socket_fd, header, 8, data.data(), data.size())) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
            "Transport: failed to send data to node " + std::to_string(target_node_id));
        return false;
    }

    return true;
}

void Transport::broadcast(const std::vector<uint8_t>& data, bool exclude_self) {
    (void)exclude_self;

    std::vector<uint32_t> targets;
    {
        std::lock_guard<std::mutex> lock(_connections_mutex);
        targets.reserve(_connections.size());
        for (const auto& [id, conn] : _connections) {
            if (conn->active.load()) {
                targets.push_back(id);
            }
        }
    }

    // 并行发送到所有目标节点
    // 每个 send() 独立持有 _connections_mutex 查找连接，
    // 不同 socket FD 的 writev 互不干扰，线程安全。
    if (targets.size() <= 1) {
        for (auto target : targets) {
            send(target, data);
        }
    } else {
        std::vector<std::future<bool>> futures;
        futures.reserve(targets.size());
        for (auto target : targets) {
            futures.push_back(std::async(std::launch::async, [this, target, &data]() {
                return send(target, data);
            }));
        }
        // 等待所有发送完成
        for (auto& f : futures) {
            (void)f.get();
        }
    }
}

// ======================= 内部线程 =======================

void Transport::acceptorLoop() {
    while (_running.load()) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);

        int client_fd = ::accept(_server_fd,
                                 reinterpret_cast<struct sockaddr*>(&client_addr),
                                 &addr_len);

        if (client_fd < 0) {
            if (!_running.load()) {
                break;  // 正常关闭
            }
            // accept 错误（非阻塞）
            continue;
        }

        // 获取对端节点 ID：收到第一条消息才能知道
        // 先读取消息头的前 8 字节获取 source_node_id
        uint8_t header[8];
        ssize_t n = ::recv(client_fd, header, 8, MSG_PEEK);
        if (n != 8) {
            ::close(client_fd);
            continue;
        }

        uint32_t peer_node_id = ntoh32(header + 4);

        // 检查是否已经连接
        {
            std::lock_guard<std::mutex> lock(_connections_mutex);
            if (_connections.find(peer_node_id) != _connections.end()) {
                CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
                    "Transport: duplicate connection from node " + std::to_string(peer_node_id));
                ::close(client_fd);
                continue;
            }
        }

        // 调优 TCP 参数（大缓冲区 + 无 Nagle）
        tuneSocket(client_fd);

        // 创建连接对象
        auto conn = std::make_unique<PeerConnection>();
        conn->socket_fd = client_fd;
        conn->node_id = peer_node_id;
        conn->active.store(true);

        auto* conn_ptr = conn.get();

        {
            std::lock_guard<std::mutex> lock(_connections_mutex);
            _connections[peer_node_id] = std::move(conn);
        }

        // 启动 reader 线程
        conn_ptr->reader_thread = std::thread(&Transport::readerLoop, this, conn_ptr);

        CtorchError::log(ErrorLevel::INFO, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
            "Transport: accepted connection from node " + std::to_string(peer_node_id));
    }
}

void Transport::readerLoop(PeerConnection* conn) {
    while (conn->active.load() && _running.load()) {
        // 读取消息头
        uint8_t header[8];
        if (!recvHeader(conn->socket_fd, header)) {
            break;  // 连接关闭或出错
        }

        uint32_t payload_len = ntoh32(header);
        uint32_t source_node_id = ntoh32(header + 4);

        // 读取消息体
        std::vector<uint8_t> payload(payload_len);
        if (payload_len > 0) {
            if (!recvAll(conn->socket_fd, payload.data(), payload_len)) {
                break;
            }
        }

        // 调用回调
        if (_receive_callback) {
            try {
                _receive_callback(source_node_id, payload);
            } catch (const std::exception& e) {
                CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
                    std::string("Transport: receive callback error: ") + e.what());
            }
        }
    }

    // 连接结束，清理
    conn->active.store(false);
    if (conn->socket_fd >= 0) {
        ::close(conn->socket_fd);
        conn->socket_fd = -1;
    }
}

// ======================= I/O 辅助 =======================

bool Transport::recvHeader(int fd, uint8_t header[8]) {
    // 读取 8 字节消息头
    size_t total_read = 0;
    while (total_read < 8) {
        ssize_t n = ::recv(fd, header + total_read, 8 - total_read, 0);
        if (n <= 0) {
            return false;  // 连接关闭或出错
        }
        total_read += static_cast<size_t>(n);
    }
    return true;
}

bool Transport::sendAll(int fd, const uint8_t* data, size_t len) {
    size_t total_sent = 0;
    while (total_sent < len) {
        ssize_t n = ::send(fd, data + total_sent, len - total_sent, 0);
        if (n <= 0) {
            return false;
        }
        total_sent += static_cast<size_t>(n);
    }
    return true;
}

bool Transport::recvAll(int fd, uint8_t* data, size_t len) {
    size_t total_read = 0;
    while (total_read < len) {
        ssize_t n = ::recv(fd, data + total_read, len - total_read, 0);
        if (n <= 0) {
            return false;
        }
        total_read += static_cast<size_t>(n);
    }
    return true;
}

} // namespace distributed
} // namespace ct