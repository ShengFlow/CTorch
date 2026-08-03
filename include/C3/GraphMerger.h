/**
 * @file GraphMerger.h
 * @brief 子图合并工具：将 N 个独立子图合并为单个融合图
 * @details 用于多层网络（如 MLP）的"逐层编译 + 后台全图融合"模式：
 *          1. 各层子图独立编译，独立执行（首屏快）
 *          2. 后台异步提交全图编译任务
 *          3. 全图 kernel 编译完成后原子热替换
 *
 *          合并算法：
 *          - 子图 i 的所有输入分类为：
 *            a) "子图输入"：来自子图外部（其他子图输出或用户输入）
 *            b) "子图内部"：从子图内部节点产生
 *          - 子图 i 的输出 = 子图 i+1 的"子图输入"集合中对应匹配的输入
 *          - 合并时：(子图索引, 子图内输入索引) → 融合图中的输入 ID
 *
 * @date 2026/8/2
 */

#ifndef CTORCH_C3_GRAPH_MERGER_H
#define CTORCH_C3_GRAPH_MERGER_H

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "Graph.h"

namespace ct {
namespace c3 {

/**
 * @brief 融合图缓存键的统一版本前缀
 * @details 集中管理 C3 Engine 内所有 merged cache key 的版本号。
 *          演进规则：变更此常量将使所有旧 merged 缓存失效（一次性 cache miss）。
 *          当前版本 v4 与 makeCacheKey 的 c3_v4_ 同步演进，PGO 包装与子图编译互不冲突。
 *
 *          注意：C3Engine.cpp 中的 kMergedCacheKeyPrefix = this value，
 *                GraphMerger::mergedCacheKey() 内部也使用 this value，
 *                三者必须保持一致。
 */
inline constexpr const char* kMergedCacheKeyPrefix = "merged_v4_";

/**
 * @struct MergeSpec
 * @brief 合并规格：描述如何将 N 个子图串接成单个融合图
 * @details
 *   - `sub_graphs[i].outputs()[k]` = 子图 i 的第 k 个输出
 *   - `links[i]` 描述子图 i 的输出如何连接到子图 i+1 的输入：
 *     `links[i].from_output` = 子图 i 的输出索引
 *     `links[i].to_input`    = 子图 i+1 的输入索引
 *   - 若 `links[i].to_input == SIZE_MAX`，表示子图 i 的输出不连接到任何子图，
 *     而是作为融合图的最终输出之一
 */
struct MergeLink {
    size_t from_output;  ///< 子图 i 的输出索引
    size_t to_input;     ///< 子图 i+1 的输入索引（SIZE_MAX 表示不连接，作为最终输出）
};

struct MergeSpec {
    std::vector<MergeLink> links;  ///< 链接数 = sub_graphs.size() - 1
};

/**
 * @struct MergedGraphInfo
 * @brief 合并后图的元数据
 */
struct MergedGraphInfo {
    Graph graph;                                              ///< 合并后的图
    std::vector<size_t> external_input_ids;                   ///< 融合图的外部输入 ID（按子图 i、子图内输入索引的顺序平铺）
    /// sub_graphs[i] 的第 j 个输入在融合图中的 ID
    std::vector<std::vector<size_t>> input_remap;
    /// sub_graphs[i] 的第 k 个输出在融合图中的 ID（无对应连接则为 SIZE_MAX）
    std::vector<std::vector<size_t>> output_remap;
};

/**
 * @class GraphMerger
 * @brief 子图合并工具类
 */
class GraphMerger {
public:
    /**
     * @brief 将 N 个子图按链接规格合并为单个融合图
     * @param sub_graphs 子图列表（至少 1 个）
     * @param spec 链接规格
     * @return 合并后的图与映射信息
     * @throws std::invalid_argument 子图数量与链接数量不匹配
     * @throws std::runtime_error 链接的目标输入索引越界
     */
    static MergedGraphInfo merge(const std::vector<Graph>& sub_graphs,
                                  const MergeSpec& spec);

    /**
     * @brief 简化合并：纯顺序链接的 N 个子图
     * @details 子图 i 的 outputs()[0] 自动连接到子图 i+1 的 inputs()[0]，
     *          最后一个子图的 outputs()[0] 作为融合图的最终输出。
     *          这覆盖了 MLP 的典型场景：每层是单输入单输出。
     */
    static MergedGraphInfo mergeSequential(const std::vector<Graph>& sub_graphs);

    /**
     * @brief 为顺序子图生成 MergeSpec（供 compileMerged 直接使用）
     * @details 等价于 mergeSequential 内部生成的 spec，但不实际执行合并。
     *          适用于只需要 spec 字符串以生成 cache key 的场景。
     * @return 与 mergeSequential 等价的 MergeSpec
     */
    static MergeSpec makeSequentialSpec(const std::vector<Graph>& sub_graphs);

    /**
     * @brief 从 (sub_graphs, spec) 生成稳定的字符串标识（用于 cache key 拼接）
     * @details 将子图各自的 graph.toString() 拼接并混入 spec.links 索引，
     *          保证 (sub_graphs+spec) 唯一对应一个 hash。
     *          该函数不执行实际合并，仅生成字符串；时间复杂度 O(N) + 图描述 O(N*K)。
     */
    static std::string mergedCacheKey(const std::vector<Graph>& sub_graphs,
                                       const MergeSpec& spec);

    /**
     * @brief 在执行合并前，做拓扑/一致性检查（不实际合并）
     * @details 检查项：
     *   1. 子图数量与 spec.links 数量一致
     *   2. 链接目标输入索引 / 源输出索引在子图范围内
     *   3. 链接的源输出与目标输入的 shape/dtype/device 一致
     *   4. 链接不产生"非顺序"依赖环（基于 spec.links 的纯链接图视角）
     *      — 注意：这仅检查 MergeSpec 自身（节点级环由 Graph::isValid 在合并后保证）
     * @return 错误消息（空字符串表示无错）
     */
    static std::string validate(const std::vector<Graph>& sub_graphs,
                                 const MergeSpec& spec);
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_GRAPH_MERGER_H
