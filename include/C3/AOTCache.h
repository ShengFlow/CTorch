/**
 * @file AOTCache.h
 * @brief C3 AOT (Ahead-Of-Time) 持久化 .so cache
 * @details 将 HandwrittenKernelGen 编译的 .so 持久化到磁盘，实现跨进程复用。
 *          解决 C3 当前"Pure JIT"模型下冷启动必须重新编译的问题。
 *
 *  设计目标（与工业级 AOT 引擎对齐，参考 TensorRT engine cache）：
 *  - 跨进程共享：~/.c3cache/c3_<key>.so
 *  - 自动失效：backend 版本变化 → 自动重新编译
 *  - 线程安全：单例 + 内部 mutex
 *  - 优雅降级：磁盘不可用时静默回退 in-memory
 *  - 零依赖：不依赖 openssl（自带轻量级 SHA-256）
 *
 *  集成点（修改 HandwrittenKernelGen.cpp）：
 *    1. compileAndLoad() 开始时查询 AOT cache
 *    2. 命中 → dlopen + return（避免 clang++ 编译）
 *    3. 未命中 → 正常编译 → 复制 .so 到 cache → return
 *
 *  不影响：
 *    - MLIR backend（in-process JIT，不适合 AOT 持久化）
 *    - PGO 模式（PGO 走 PGOManager 自己的逻辑）
 *    - in-memory LRU cache（AOT 是 layer above）
 *
 * @date 2026/08/03
 * @see ADR-008-aot-persistent-cache
 */

#ifndef CTORCH_C3_AOT_CACHE_H
#define CTORCH_C3_AOT_CACHE_H

#include "C3Config.h"
#include "IAOTCache.h"  // [Dev] v0.5.2 P1 解耦 refactor: AOTCache 实现 IAOTCache 接口
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace ct {
namespace c3 {

/**
 * @struct AOTCacheConfig
 * @brief AOT cache 配置
 */
struct AOTCacheConfig {
    /** @brief 是否启用（默认 true） */
    bool enabled = true;
    /** @brief 自定义 cache 目录（空 = 使用 $HOME/.c3cache） */
    std::string custom_dir;
    /** @brief 最大磁盘占用字节数（默认 1 GB，0 = 不限制） */
    size_t max_bytes = 1ULL * 1024 * 1024 * 1024;
    /** @brief 最大文件数（默认 1024，0 = 不限制） */
    size_t max_files = 1024;
};

/**
 * @class AOTCache
 * @brief AOT .so 持久化缓存单例
 *
 * 用法：
 * @code
 *   auto& cache = AOTCache::getInstance();
 *   if (auto* so_path = cache.lookup(key); so_path) {
 *       // 命中：直接 dlopen
 *       void* h = dlopen(so_path->c_str(), RTLD_NOW);
 *   } else {
 *       // miss：正常编译，然后 store
 *       cache.store(key, so_path);
 *   }
 * @endcode
 */
class AOTCache : public IAOTCache {
public:
    static AOTCache& getInstance();

    /**
     * @brief 查找 cache key 对应的 .so 文件路径
     * @param cache_key 派生自 (graph_hash, device, opt_level, backend_version) 的 hex 字符串
     * @return 若命中且 meta 校验通过，返回 .so 绝对路径；否则返回空 optional（用 empty string 表示）
     * @details 线程安全。会检查 backend version 是否匹配，不匹配返回空并增加 invalidations。
     *          dlopen 由调用方负责（不在此 API 内执行，因为调用方需要拿到 func pointer）。
     */
    [[nodiscard]] std::string lookup(const std::string& cache_key) override;

    /**
     * @brief 将新编译的 .so 写入 cache
     * @param cache_key 同 lookup
     * @param so_path 临时 .so 路径（HandwrittenKernelGen 编译产物）
     * @return 成功返回最终 cache 路径（.c3cache/c3_<key>.so），失败返回空 string
     * @details 写入采用"先 .tmp 再 rename"模式避免读到半截文件。
     *          失败时（如磁盘满）返回空 string 并增加 disk_errors，调用方应继续使用 in-memory。
     */
    [[nodiscard]] std::string store(const std::string& cache_key, const std::string& so_path) override;

    /**
     * @brief 记录一次 dlopen 失败（用于统计）
     * @details 调用方在 dlopen 失败时应调用本方法，AOTCache 会增加 load_failures 计数。
     */
    void recordLoadFailure() override {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.load_failures++;
    }

    /**
     * @brief 清空所有 AOT cache 文件
     * @details 删除 ~/.c3cache/c3_* 下所有文件。
     *          不影响 in-memory cache。
     */
    void evict() override;

    /**
     * @brief 获取统计信息
     */
    [[nodiscard]] AOTCacheStats getStats() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    /**
     * @brief 启用/禁用
     */
    void setEnabled(bool enabled) override {
        std::lock_guard<std::mutex> lock(mutex_);
        config_.enabled = enabled;
    }
    [[nodiscard]] bool isEnabled() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        // 全局开关：编译期 CT_C3_DISABLE_AOT 宏或运行时 C3_DISABLE_AOT=1 均强制禁用
        return config_.enabled && aotCacheEnabled();
    }

    /**
     * @brief 设置自定义 cache 目录
     * @details 下次 lookup/store 时生效。
     *          设置为 "" 恢复默认 $HOME/.c3cache。
     */
    void setCacheDir(std::string dir) override {
        std::lock_guard<std::mutex> lock(mutex_);
        config_.custom_dir = std::move(dir);
        dir_initialized_ = false; // 强制下次 getCacheDir 重新解析
    }

    /**
     * @brief 获取当前 cache 目录
     */
    [[nodiscard]] std::string getCacheDir() const override;

    /**
     * @brief 计算 cache key（从 graph toString + 编译参数派生）
     * @param graph_str graph.toString()
     * @param device "cpu" / "mps" / "cuda"
     * @param opt_level 0/1/2/3
     * @param backend_version backend 版本字符串（HandwrittenKernelGen::kBackendVersion）
     * @return 32 字符 hex 字符串
     */
    [[nodiscard]] static std::string makeKey(
        const std::string& graph_str,
        const std::string& device,
        int opt_level,
        const std::string& backend_version);

    /**
     * @brief 轻量级 SHA-256 实现（公开供测试与外部使用）
     * @param data 输入字节
     * @return 32 字符 hex 字符串（小写）
     */
    [[nodiscard]] static std::string sha256Hex(const std::string& data);

    /**
     * @brief 获取当前 backend 版本（与 AOT meta 文件中的版本比较）
     * @details 当 HandwrittenKernelGen 的源码生成逻辑发生不兼容变更时，
     *          应递增此版本号，让所有旧 cache 失效。
     */
    [[nodiscard]] static const char* currentBackendVersion();

private:
    AOTCache() = default;

    /// 解析最终 cache 目录（priority: custom_dir > $C3_AOT_CACHE_DIR > $HOME/.c3cache）
    [[nodiscard]] static std::string effectiveCacheDir(const AOTCacheConfig& cfg);

    /// 计算 cache 目录使用情况（文件数、字节数）
    void scanDiskUsage();

    /// 内部统计字段
    mutable std::mutex mutex_;
    AOTCacheConfig config_;
    AOTCacheStats stats_;
    /// 缓存 cache_dir（避免每次 effectiveCacheDir 都读 env / getenv）
    /// mutable：const 方法 getCacheDir() 中可写（lazy 初始化）
    mutable std::string cached_dir_;
    mutable bool dir_initialized_ = false;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_AOT_CACHE_H
