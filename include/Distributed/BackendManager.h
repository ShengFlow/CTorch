/**
 * @file BackendManager.h
 * @brief 后端管理器 — 扩展自 AllocatorManager，统一管理 DeviceBackend 生命周期
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details BackendManager 是 AllocatorManager 的扩展升级版。
 *          原来 AllocatorManager 只管理 DeviceAllocator（内存分配器），
 *          BackendManager 将管理范围扩展到整个 DeviceBackend 生命周期，
 *          包括内存分配、算子执行、序列化、能力查询等完整后端功能。
 *
 *          设计原则（分离原理 — Feedback Systems #11）：
 *          BackendManager 只负责任意后端类型的查找和生命周期管理，
 *          不参与任何业务逻辑（优化、通信等），这些由 DistributedOptimizer
 *          和 CommEngine 分别处理。
 */

#ifndef CTORCH_DISTRIBUTED_BACKEND_MANAGER_H
#define CTORCH_DISTRIBUTED_BACKEND_MANAGER_H

#include "DeviceBackend.h"
#include "DeviceAllocator.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace ct {
namespace distributed {

/**
 * @class BackendManager
 * @brief 后端管理器 — 线程安全的 DeviceBackend 注册中心
 *
 * 采用 Meyers 单例模式，与 AllocatorManager 共享同一生命周期管理策略。
 * 新增后端只需调用 registerBackend() 注册，系统自动将其 DeviceAllocator
 * 同步注册到 AllocatorManager。
 *
 * @note 程序退出期的析构顺序：BackendManager 单例在 AllocatorManager 之后析构，
 *       确保 Storage::Deleter 在退出期仍能安全访问分配器。
 */
class BackendManager {
private:
    std::unordered_map<DeviceType, std::shared_ptr<DeviceBackend>> _backends;
    mutable std::mutex _mtx;
    uint32_t _local_node_id;

    BackendManager() : _local_node_id(0) {}

public:
    static BackendManager& getInstance() {
        static BackendManager instance;
        return instance;
    }

    BackendManager(const BackendManager&) = delete;
    BackendManager& operator=(const BackendManager&) = delete;

    // ======================= 后端注册 =======================

    /**
     * @brief 注册后端
     * @param backend 后端智能指针
     * @throws CtorchError 如果同一 DeviceType 已注册
     *
     * 注册时自动同步注册 AllocatorManager，确保旧代码路径通过
     * AllocatorManager 也能访问到正确的内存分配器。
     */
    void registerBackend(std::shared_ptr<DeviceBackend> backend);

    /**
     * @brief 注销后端
     * @param device 要注销的 DeviceType
     */
    void unregisterBackend(DeviceType device);

    // ======================= 后端查询 =======================

    /**
     * @brief 获取指定类型的后端
     * @param device 设备类型
     * @return 后端智能指针，未注册返回 nullptr
     */
    std::shared_ptr<DeviceBackend> getBackend(DeviceType device);

    /**
     * @brief 获取所有已注册的后端类型列表
     * @return DeviceType 列表
     */
    std::vector<DeviceType> registeredBackends() const;

    /**
     * @brief 获取所有已注册后端的能力信息
     * @return BackendCapability 列表
     */
    std::vector<BackendCapability> allCapabilities() const;

    /**
     * @brief 检查指定后端是否已注册
     * @param device 设备类型
     * @return true 如果已注册
     */
    bool hasBackend(DeviceType device) const;

    /**
     * @brief 获取已注册后端的数量
     * @return 后端数量
     */
    size_t backendCount() const;

    // ======================= 节点管理 =======================

    /**
     * @brief 设置本地节点 ID
     * @param node_id 节点 ID
     */
    void setLocalNodeId(uint32_t node_id) { _local_node_id = node_id; }

    /**
     * @brief 获取本地节点 ID
     * @return 节点 ID
     */
    uint32_t localNodeId() const { return _local_node_id; }

    // ======================= 后端同步 =======================

    /**
     * @brief 同步所有后端
     */
    void synchronizeAll();

    /**
     * @brief 同步指定后端
     * @param device 设备类型
     */
    void synchronize(DeviceType device);
};

/**
 * @brief 后端注册辅助类 — RAII 风格注册
 *
 * 在构造时注册后端，析构时注销。支持作用域内的临时后端注册。
 */
class ScopedBackendRegistration {
public:
    ScopedBackendRegistration(std::shared_ptr<DeviceBackend> backend)
        : _device(backend->deviceType()), _backend(std::move(backend)) {
        BackendManager::getInstance().registerBackend(_backend);
    }

    ~ScopedBackendRegistration() {
        BackendManager::getInstance().unregisterBackend(_device);
    }

    ScopedBackendRegistration(const ScopedBackendRegistration&) = delete;
    ScopedBackendRegistration& operator=(const ScopedBackendRegistration&) = delete;

private:
    DeviceType _device;
    std::shared_ptr<DeviceBackend> _backend;
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_BACKEND_MANAGER_H