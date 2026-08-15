/**
 * @file JITCache.h
 * @brief MLIR JIT 编译产物磁盘缓存
 * @details 将 MLIR JIT 编译的 LLVM IR 模块以 bitcode 形式持久化到 ~/.c3cache/，
 *          实现跨进程复用，避免每次重启重新走 MLIR → LLVM IR → lowering pipeline。
 *
 *  设计目标：
 *  - 存储 LLVM IR bitcode（.bc 文件），非 .so 文件
 *  - 缓存命中：加载 bitcode → 创建 ExecutionEngine → JIT 编译（跳过 MLIR building + lowering）
 *  - 缓存未命中：正常编译 → 保存 bitcode 到磁盘
 *  - 自动失效：JIT backend 版本变化 → 自动重新编译
 *  - 线程安全 + 优雅降级：磁盘不可用时静默回退
 *  - 复用 ~/.c3cache/ 目录（文件前缀 c3_jit_ 与其余产物区分）
 *
 * @date 2026/08/06
 */

#ifndef CTORCH_C3_JIT_CACHE_H
#define CTORCH_C3_JIT_CACHE_H

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>

// Forward declarations
namespace llvm {
class Module;
class LLVMContext;
}

namespace ct {
namespace c3 {

/**
 * @class JITCache
 * @brief MLIR JIT bitcode 磁盘缓存
 *
 * 用法：
 * @code
 *   auto& jc = JITCache::getInstance();
 *   std::string key = jc.makeKey(graph_str, opt_level);
 *   if (auto* bc_path = jc.lookup(key); bc_path) {
 *       auto mod = jc.loadBitcode(bc_path, ctx);  // 跳过 MLIR building + lowering
 *   } else {
 *       // 正常编译 → 保存 bitcode
 *       jc.store(key, *llvmModule);
 *   }
 * @endcode
 */
class JITCache {
public:
    static JITCache& getInstance();

    /**
     * @brief 查找 cache key 对应的 bitcode 文件路径
     * @param cache_key 由 makeKey() 派生
     * @return 若命中返回 .bc 文件绝对路径，否则返回空字符串
     */
    [[nodiscard]] std::string lookup(const std::string& cache_key);

    /**
     * @brief 将 LLVM 模块以 bitcode 形式写入磁盘
     * @param cache_key 同 lookup
     * @param module LLVM 模块（已 lowering 完成）
     * @return 成功返回最终 cache 路径，失败返回空字符串
     */
    [[nodiscard]] std::string store(const std::string& cache_key, llvm::Module& module);

    /**
     * @brief 从磁盘加载 bitcode 文件并反序列化为 LLVM 模块
     * @param bc_path bitcode 文件路径
     * @param ctx 用于模块创建的 LLVMContext
     * @return 解析后的 LLVM 模块，失败返回 nullptr
     */
    [[nodiscard]] std::unique_ptr<llvm::Module> loadBitcode(
        const std::string& bc_path, llvm::LLVMContext& ctx);

    /**
     * @brief 计算 cache key
     * @param graph_str graph.toString() 输出
     * @param opt_level 编译优化级别 (0-3)
     * @return 32 字符 hex 字符串
     */
    [[nodiscard]] static std::string makeKey(const std::string& graph_str, int opt_level);

    /**
     * @brief 获取当前 JIT backend 版本
     * @details 当 MLIR 代码生成逻辑发生不兼容变更时递增此版本号，
     *          让所有旧 cache 失效。
     */
    [[nodiscard]] static const char* currentJITVersion();

    /**
     * @brief 获取 cache 目录
     */
    [[nodiscard]] std::string cacheDir() const;

    /**
     * @brief 清空所有 JIT cache 文件
     */
    void evict();

    /**
     * @brief 获取命中统计
     */
    [[nodiscard]] uint64_t hits() const { return hits_; }
    [[nodiscard]] uint64_t misses() const { return misses_; }
    [[nodiscard]] uint64_t stores() const { return stores_; }

    /**
     * @brief [Dev] v0.5.2 (4) JITCache 1.0 store-only (2026-08-09):
     *        查询 JITCache 是否启用 (env var 控制)
     * @details 默认启用。设 C3_JIT_CACHE_DISABLE=1 关闭。
     *          用户测试注意 (per 洛锦 2026-08-09 提醒):
     *            - 性能测试前必须 evict() (避免命中作弊)
     *            - MLIR backend 改动后必须 evict() (旧 .bc 跟新 MLIR IR 不兼容)
     *            - 正确性测试允许 warm cache (cache deterministic)
     */
    [[nodiscard]] static bool isEnabled() {
        const char* v = std::getenv("C3_JIT_CACHE_DISABLE");
        return !(v != nullptr && v[0] == '1');
    }

    /**
     * @brief 手动记录一次 hit (1.0 store-only 期间,lookup 已确认命中但不走 read path)
     * @details 1.0 实装 lookup 走 disk check,确认有 .bc 但不实际反序列化 (read path TODO 跨 session)
     *          调用方在确认 bc_path 非空时调 recordHit,store 仍走正常 store() 计 stores_
     */
    void recordHit() { hits_.fetch_add(1, std::memory_order_relaxed); }

private:
    JITCache() = default;

    /// 计算 ~/.c3cache/ 目录路径
    [[nodiscard]] static std::string resolveCacheDir();

    mutable std::mutex mutex_;
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> stores_{0};  ///< v0.5.2 (4) store 计数器 (1.0 store-only 验证用)
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_JIT_CACHE_H