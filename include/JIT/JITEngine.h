/**
 * @file JITEngine.h
 * @brief CTorch JIT 编译引擎公共接口
 * @details 提供将计算图（Graph）编译为后端 kernel 的能力，并管理编译产物缓存。
 *          当前为公共接口层，具体 Graph 定义与 kernel 实现位于 src/JIT 模块。
 * @date 2026/7/31
 */

#ifndef CTORCH_JIT_JITENGINE_H
#define CTORCH_JIT_JITENGINE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../Ctools.h"
#include "../Tensor.h"
#include "C3KernelRegistry.h"

namespace ct {
namespace jit {

// 前向声明：计算图定义由 src/JIT 模块提供，避免在公共头文件中暴露实现细节。
class Graph;

/**
 * @struct CompileOptions
 * @brief JIT 编译选项
 * @details 控制编译目标设备、优化级别、算子融合策略与缓存行为。
 */
struct CompileOptions {
    /** @brief 目标设备，默认 CPU */
    DeviceType target_device = DeviceType::kCPU;
    /** @brief 优化级别：0=关闭优化，1=基础优化，2=积极优化（默认），3=极限优化 */
    int opt_level = 2;
    /** @brief 是否启用算子融合（默认开启） */
    bool enable_fusion = true;
    /** @brief 是否启用编译缓存（默认开启） */
    bool enable_cache = true;
    /** @brief 自定义缓存键；为空时由引擎根据图结构与选项生成 */
    std::string cache_key_override;
};

/**
 * @struct JITCacheStats
 * @brief JIT 编译缓存统计信息
 */
struct JITCacheStats {
    /** @brief 缓存条目总数 */
    size_t total_entries = 0;
    /** @brief 缓存命中次数 */
    size_t hits = 0;
    /** @brief 缓存未命中次数 */
    size_t misses = 0;
    /** @brief 被逐出的条目数 */
    size_t evictions = 0;
    /** @brief 缓存占用的近似字节数 */
    size_t bytes_used = 0;
};

/**
 * @class CompiledKernel
 * @brief 编译后的可执行 kernel 抽象
 * @details 由 JITEngine::compile() 产出，封装了后端特定的可执行代码。
 *          通过 execute() 接收输入张量并返回输出张量。
 */
class CompiledKernel {
public:
    virtual ~CompiledKernel() = default;

    /**
     * @brief 执行编译后的 kernel
     * @param inputs 输入张量列表，顺序与 Graph 输入一致
     * @return 输出张量列表，顺序与 Graph 输出一致
     * @throw std::runtime_error 当输入不匹配或执行失败时抛出
     */
    virtual std::vector<Tensor> execute(const std::vector<Tensor>& inputs) = 0;

    /** @brief 返回该 kernel 的缓存键 */
    [[nodiscard]] virtual const std::string& cacheKey() const = 0;

    /** @brief 返回该 kernel 的目标设备 */
    [[nodiscard]] virtual DeviceType targetDevice() const = 0;

    /** @brief 返回执行时所需的工作空间字节数（0 表示不需要额外内存） */
    [[nodiscard]] virtual size_t workspaceBytes() const = 0;

    /**
     * @brief 将编译产物安装到 C3 内核注册表，启用热替换
     * @param op_type 对应算子类型
     * @param shapes 形状签名
     * @return 安装成功返回 true；默认实现返回 false（未实现）
     */
    virtual bool installIntoRegistry(op op_type, const KernelShapeInfo& shapes) {
        (void)op_type; (void)shapes;
        return false;
    }
};

/**
 * @class JITEngine
 * @brief CTorch JIT 编译引擎
 * @details 单例类，负责将 Graph 编译为 CompiledKernel，并维护编译产物缓存。
 *          线程安全：所有公共方法内部同步。
 */
class JITEngine {
public:
    /** @brief 获取 JITEngine 单例实例 */
    static JITEngine& getInstance();

    /**
     * @brief 编译计算图
     * @param graph 待编译的计算图
     * @param options 编译选项
     * @return 编译后的 kernel；若缓存命中则直接返回缓存产物
     * @throw std::runtime_error 当图不合法或目标设备不支持时抛出
     */
    std::shared_ptr<CompiledKernel> compile(const Graph& graph, const CompileOptions& options = {});

    /**
     * @brief 根据缓存键获取已编译 kernel
     * @param cache_key 缓存键
     * @return 对应 kernel；若不存在则返回 nullptr
     */
    [[nodiscard]] std::shared_ptr<CompiledKernel> getKernel(const std::string& cache_key) const;

    /** @brief 查询当前缓存状态 */
    [[nodiscard]] JITCacheStats getCacheStats() const;

    /** @brief 清空全部编译缓存 */
    void clearCache();

private:
    JITEngine() = default;
    ~JITEngine() = default;
    JITEngine(const JITEngine&) = delete;
    JITEngine& operator=(const JITEngine&) = delete;
};

} // namespace jit
} // namespace ct

#endif // CTORCH_JIT_JITENGINE_H
