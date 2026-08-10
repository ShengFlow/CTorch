/**
 * @file IAOTCache.h
 * @brief AOTCache 抽象接口 (v0.5.2 P1 解耦 refactor, 2026-08-10)
 * @details 把 AOTCache 从具体单例类抽成接口, 让 C3Engine 持 IAOTCache& 而非直连
 *          AOTCache::getInstance()。解决 P1 风险: C3Engine 强耦合具体 cache 实现,
 *          无法 mock 测试 / 无法注入替代实现 (DCU 接入时可能需要新 cache 类型)。
 *
 * 设计原则 (per ADR-NEW-1 接口隔离):
 *   - 纯虚接口, 9 个核心方法 (lookup/store/evict/... 跟 AOTCache 1:1 对应)
 *   - 静态方法 (makeKey/sha256Hex/currentBackendVersion) 留 AOTCache, 因为它们是纯函数
 *     不需要 polymorphic, 接口隔离原则
 *   - 默认实现: AOTCache::getInstance() (singleton, 行为不变)
 *   - 注入方式: C3Engine::setAOTCacheImpl(IAOTCache*) — 传 nullptr 恢复 singleton
 *
 * 兼容性:
 *   - AOTCache.h 改成 public IAOTCache (binary 兼容, 现有代码继续用)
 *   - C3Engine 6 个 facade 方法签名不变, 内部从 AOTCache::getInstance() 改为 aot_cache_override_
 *   - test_c3_aot_cache.cpp 不用改
 *
 * 跨 backend 复用:
 *   - 未来 DCUCompiledKernel 可以实现自己的 IAOTCache (缓存 HSACO)
 *   - 或用模板方式复用 AOTCache
 */
#pragma once

// 注: 不 include AOTCache.h 以避免循环依赖 (IAOTCache <- AOTCache)
//     AOTCacheStats 完整定义在本头, AOTCache.h 改 include IAOTCache.h 复用

#include <cstdint>
#include <string>

namespace ct {
namespace c3 {

/**
 * @struct AOTCacheStats
 * @brief AOT cache 运行时统计 (v0.5.2 移到此头: IAOTCache 接口需要返这个 type)
 *
 * 之前在 AOTCache.h 定义, P1 解耦 refactor 时移到 IAOTCache.h,
 * 因为接口返 AOTCacheStats 完整类型, 不能用前置声明
 */
struct AOTCacheStats {
    uint64_t hits = 0;            ///< 命中次数（避免重新编译）
    uint64_t misses = 0;          ///< miss 次数（需重新编译并写入）
    uint64_t writes = 0;          ///< 写入磁盘次数
    uint64_t evictions = 0;       ///< 显式清空次数（evictAOTCache）
    uint64_t load_failures = 0;   ///< dlopen 失败次数（fallback 到 in-memory）
    uint64_t invalidations = 0;   ///< 因 backend version / meta 不匹配而失效的次数
    uint64_t disk_errors = 0;     ///< 磁盘 I/O 错误次数（权限、空间等）
    size_t total_files = 0;       ///< 当前 .so 文件数（最近一次统计）
    size_t total_bytes = 0;       ///< 占用磁盘字节数（最近一次统计）
};

/**
 * @class IAOTCache
 * @brief AOT (Ahead-Of-Time) cache 抽象接口
 *
 * 实现要求:
 *   - 线程安全 (lookup/store 可并发调用)
 *   - 失败优雅 (返回空 string, 不抛异常 — 调用方 fallback in-memory)
 *   - 状态可重置 (evict() 清理所有)
 */
class IAOTCache {
public:
    virtual ~IAOTCache() = default;

    /**
     * @brief 查找 cache key 对应的编译产物路径 (.so / .hsaco 等)
     * @param cache_key 派生自 (graph_hash, device, opt_level, backend_version) 的 hex 字符串
     * @return 命中且 meta 校验通过 → 绝对路径; 否则空 string
     */
    [[nodiscard]] virtual std::string lookup(const std::string& cache_key) = 0;

    /**
     * @brief 写入新编译产物到 cache
     * @param cache_key 同 lookup
     * @param artifact_path 临时产物路径 (caller 提供的 .so/.hsaco)
     * @return 成功 → 最终 cache 路径; 失败 → 空 string (caller fallback in-memory)
     */
    [[nodiscard]] virtual std::string store(const std::string& cache_key,
                                            const std::string& artifact_path) = 0;

    /**
     * @brief 记录一次 load 失败 (供 stats 统计)
     */
    virtual void recordLoadFailure() = 0;

    /**
     * @brief 清空所有 cache 文件
     */
    virtual void evict() = 0;

    /**
     * @brief 获取统计信息
     */
    [[nodiscard]] virtual AOTCacheStats getStats() const = 0;

    /**
     * @brief 启用/禁用 cache
     */
    virtual void setEnabled(bool enabled) = 0;
    [[nodiscard]] virtual bool isEnabled() const = 0;

    /**
     * @brief 设置/获取 cache 目录
     */
    virtual void setCacheDir(std::string dir) = 0;
    [[nodiscard]] virtual std::string getCacheDir() const = 0;
};

}  // namespace c3
}  // namespace ct
