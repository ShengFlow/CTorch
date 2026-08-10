/**
 * @file GraphMerger.cpp
 * @brief 子图合并实现
 * @date 2026/8/2
 */

#include "../../include/C3/GraphMerger.h"

#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace ct {
namespace c3 {

// ======================= 辅助函数 =======================

/// 拼接 TensorDesc 为字符串（用于错误信息）
static std::string descToString(const TensorDesc& d) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < d.shape.size(); ++i) {
        if (i > 0) ss << "x";
        ss << d.shape[i];
    }
    ss << "] dtype=" << static_cast<int>(d.dtype)
       << " device=" << static_cast<int>(d.device);
    return ss.str();
}

/// 检查两个 TensorDesc 是否兼容（shape/dtype/device 全部一致）
static bool descsCompatible(const TensorDesc& a, const TensorDesc& b) {
    return a.shape == b.shape && a.dtype == b.dtype && a.device == b.device;
}

// ======================= 校验 =======================

std::string GraphMerger::validate(const std::vector<Graph>& sub_graphs,
                                   const MergeSpec& spec) {
    if (sub_graphs.empty()) {
        return "GraphMerger: sub_graphs is empty";
    }
    if (sub_graphs.size() > 1 && spec.links.size() != sub_graphs.size() - 1) {
        std::ostringstream ss;
        ss << "GraphMerger: links count (" << spec.links.size()
           << ") must be sub_graphs.size() - 1 (" << (sub_graphs.size() - 1) << ")";
        return ss.str();
    }

    for (size_t i = 0; i + 1 < sub_graphs.size(); ++i) {
        const auto& link = spec.links[i];
        const auto& g_from = sub_graphs[i];
        const auto& g_to = sub_graphs[i + 1];

        if (link.from_output >= g_from.outputCount()) {
            std::ostringstream ss;
            ss << "GraphMerger: link[" << i << "].from_output ("
               << link.from_output << ") out of range (g_from has "
               << g_from.outputCount() << " outputs)";
            return ss.str();
        }
        if (link.to_input != SIZE_MAX && link.to_input >= g_to.inputCount()) {
            std::ostringstream ss;
            ss << "GraphMerger: link[" << i << "].to_input ("
               << link.to_input << ") out of range (g_to has "
               << g_to.inputCount() << " inputs)";
            return ss.str();
        }

        if (link.to_input != SIZE_MAX) {
            // 形状/dtype/device 一致性
            const auto& out_desc = g_from.node(g_from.outputs()[link.from_output]).out_desc;
            const auto& in_desc = g_to.node(g_to.inputs()[link.to_input]).out_desc;
            if (!descsCompatible(out_desc, in_desc)) {
                std::ostringstream ss;
                ss << "GraphMerger: link[" << i << "] desc mismatch: out="
                   << descToString(out_desc) << " vs in=" << descToString(in_desc);
                return ss.str();
            }
        }
    }

    return "";  // 无错
}

// ======================= 合并 =======================

MergedGraphInfo GraphMerger::merge(const std::vector<Graph>& sub_graphs,
                                    const MergeSpec& spec) {
    MergedGraphInfo info;

    // 1. 校验
    std::string err = validate(sub_graphs, spec);
    if (!err.empty()) {
        throw std::invalid_argument(err);
    }

    if (sub_graphs.size() == 1) {
        // 单子图：直接拷贝
        info.graph = sub_graphs[0];
        info.input_remap.resize(1);
        info.input_remap[0].reserve(sub_graphs[0].inputCount());
        for (size_t i = 0; i < sub_graphs[0].inputCount(); ++i) {
            info.input_remap[0].push_back(sub_graphs[0].inputs()[i]);
        }
        info.output_remap.resize(1);
        info.output_remap[0].reserve(sub_graphs[0].outputCount());
        for (size_t i = 0; i < sub_graphs[0].outputCount(); ++i) {
            info.output_remap[0].push_back(sub_graphs[0].outputs()[i]);
        }
        info.external_input_ids = info.input_remap[0];
        return info;
    }

    // 2. 准备 ID 映射
    // input_remap[i][j] = 子图 i 的第 j 个输入在融合图中的节点 ID
    // output_remap[i][k] = 子图 i 的第 k 个输出在融合图中的节点 ID
    info.input_remap.resize(sub_graphs.size());
    info.output_remap.resize(sub_graphs.size());

    // 3. 预计算所有子图的输入需要变成"外部输入"还是"内部连接到前驱输出"
    // 链接标记：input_source[i][j] = k 表示子图 i 的第 j 个输入由子图 k 的输出提供
    // 若 k == SIZE_MAX，表示子图 i 的第 j 个输入是真正的外部输入
    std::vector<std::vector<size_t>> input_source(sub_graphs.size());
    for (size_t i = 0; i < sub_graphs.size(); ++i) {
        input_source[i].assign(sub_graphs[i].inputCount(), SIZE_MAX);
    }
    for (size_t i = 0; i + 1 < sub_graphs.size(); ++i) {
        const auto& link = spec.links[i];
        if (link.to_input != SIZE_MAX) {
            input_source[i + 1][link.to_input] = i;
        }
    }

    // 4. 构建融合图：两遍算法
    //   Pass 1: 为所有子图输入分配"占位"ID（无论是否被链接覆盖）
    //   Pass 2: 复制所有子图节点，引用占位 ID
    //   Pass 3: 把"被链接覆盖"的占位 ID 重定向到前驱输出
    //   Pass 4: eliminateDeadCode 自动清理未使用的占位
    // 这样在 Pass 2 中所有 ID 都是有效的。

    // 4a. Pass 1: 为所有子图输入分配占位 ID
    //   - 无链接覆盖 → 立即创建外部输入节点
    //   - 有链接覆盖 → 创建临时占位（外部输入），最后会被 unused-eliminate 掉
    Graph& g = info.graph;
    for (size_t i = 0; i < sub_graphs.size(); ++i) {
        for (size_t j = 0; j < sub_graphs[i].inputCount(); ++j) {
            const auto& in_node = sub_graphs[i].node(sub_graphs[i].inputs()[j]);
            // 总是创建占位外部输入（即使是链接覆盖的也先占位，最后会被消除）
            size_t fid = g.addInput(in_node.out_desc);
            info.input_remap[i].push_back(fid);
        }
    }

    // 4b. 标记子图输入的 ID 映射
    std::vector<std::unordered_map<size_t, size_t>> node_id_remap(sub_graphs.size());
    for (size_t i = 0; i < sub_graphs.size(); ++i) {
        for (size_t j = 0; j < sub_graphs[i].inputCount(); ++j) {
            size_t src_in = sub_graphs[i].inputs()[j];
            node_id_remap[i][src_in] = info.input_remap[i][j];
        }
    }

    // 4c. Pass 2: 复制所有子图节点（不区分输入/非输入）
    //   但子图内部的"输入"节点在 Graph 看来也是普通节点，需要复制到 g
    //   简化处理：直接复制所有非输入节点，依赖其子图内 inputs
    for (size_t i = 0; i < sub_graphs.size(); ++i) {
        const Graph& sg = sub_graphs[i];
        for (const auto& src_node : sg.nodes()) {
            // 跳过输入节点（已在 4b 标记，但未在 g 中作为普通节点存在）
            bool is_input = false;
            for (size_t in_id : sg.inputs()) {
                if (src_node.id == in_id) { is_input = true; break; }
            }
            if (is_input) continue;

            // 重映射 inputs
            std::vector<size_t> new_inputs;
            new_inputs.reserve(src_node.inputs.size());
            for (size_t src_in : src_node.inputs) {
                auto it = node_id_remap[i].find(src_in);
                if (it == node_id_remap[i].end()) {
                    std::ostringstream ss;
                    ss << "GraphMerger: subgraph " << i << " node " << src_node.id
                       << " has unresolved input " << src_in
                       << " (topology not yet replicated)";
                    throw std::runtime_error(ss.str());
                }
                new_inputs.push_back(it->second);
            }

            // 添加节点到融合图
            size_t new_id = g.addNode(src_node.op, new_inputs, src_node.out_desc);
            node_id_remap[i][src_node.id] = new_id;
        }
    }

    // 4d. Pass 3: 把"链接覆盖"的占位 ID 替换为前驱输出
    for (size_t i = 0; i + 1 < sub_graphs.size(); ++i) {
        const auto& link = spec.links[i];
        if (link.to_input == SIZE_MAX) continue;
        size_t out_src_id = sub_graphs[i].outputs()[link.from_output];
        size_t out_fused_id = node_id_remap[i][out_src_id];
        size_t placeholder_id = info.input_remap[i + 1][link.to_input];
        g._rewriteInputRefInternal(placeholder_id, out_fused_id);
    }

    // 4e. 标记融合图输出：最后一个子图的所有输出作为最终输出
    for (size_t i = 0; i < sub_graphs.size(); ++i) {
        info.output_remap[i].assign(sub_graphs[i].outputCount(), SIZE_MAX);
    }
    for (size_t i = 0; i + 1 < sub_graphs.size(); ++i) {
        const auto& link = spec.links[i];
        size_t out_src_id = sub_graphs[i].outputs()[link.from_output];
        info.output_remap[i][link.from_output] = node_id_remap[i][out_src_id];
    }
    if (!sub_graphs.empty()) {
        size_t last = sub_graphs.size() - 1;
        for (size_t k = 0; k < sub_graphs[last].outputCount(); ++k) {
            size_t out_src_id = sub_graphs[last].outputs()[k];
            size_t fid = node_id_remap[last][out_src_id];
            info.output_remap[last][k] = fid;
            g.markOutput(fid);
        }
    }

    // 4f. 计算 external_input_ids（融合图外部输入的实际 ID）
    //    只保留无链接覆盖的那些
    std::unordered_set<size_t> seen;
    for (size_t i = 0; i < sub_graphs.size(); ++i) {
        for (size_t j = 0; j < info.input_remap[i].size(); ++j) {
            if (input_source[i][j] == SIZE_MAX) {
                if (seen.insert(info.input_remap[i][j]).second) {
                    info.external_input_ids.push_back(info.input_remap[i][j]);
                }
            }
        }
    }

    // 5. 清理：链接覆盖的占位节点不再被引用，需彻底移除（从 nodes_ 和 inputs_）
    //    使用 _eliminateDeadCodeForMergedInternal：它返回新图以及 old_to_new 映射
    // [Fix] v0.5.2 Linux build: 用 named 变量替代 structured binding
    //   DTK clang 17 OpenMP 严格模式: lambda 不能 capture structured binding
    //   (L248 下面 lambda 引用 old_to_new, 必须 named)
    auto _elim_result = g._eliminateDeadCodeForMergedInternal();
    auto& cleaned_graph = _elim_result.first;
    auto& old_to_new = _elim_result.second;
    info.graph = cleaned_graph;

    // 6. 更新所有 ID 映射到新图
    auto remap = [&](size_t old_id) -> size_t {
        auto it = old_to_new.find(old_id);
        if (it == old_to_new.end()) return SIZE_MAX;
        return it->second;
    };
    for (size_t i = 0; i < info.input_remap.size(); ++i) {
        for (size_t& id : info.input_remap[i]) {
            id = remap(id);
        }
    }
    for (size_t i = 0; i < info.output_remap.size(); ++i) {
        for (size_t& id : info.output_remap[i]) {
            id = remap(id);
        }
    }
    for (size_t& id : info.external_input_ids) {
        id = remap(id);
    }

    return info;
}

MergedGraphInfo GraphMerger::mergeSequential(const std::vector<Graph>& sub_graphs) {
    MergeSpec spec;
    spec.links.reserve(sub_graphs.size() - 1);
    for (size_t i = 0; i + 1 < sub_graphs.size(); ++i) {
        MergeLink link;
        link.from_output = 0;   // 默认：每个子图只有 1 个输出
        link.to_input = 0;      // 默认：连接到下个子图的第 0 个输入
        // 若子图 i+1 的第 0 个输入实际上不是来自前驱的（比如是用户输入），
        // 调用方应使用 merge() 显式指定 spec
        spec.links.push_back(link);
    }
    return merge(sub_graphs, spec);
}

MergeSpec GraphMerger::makeSequentialSpec(const std::vector<Graph>& sub_graphs) {
    MergeSpec spec;
    spec.links.reserve(sub_graphs.size() - 1);
    for (size_t i = 0; i + 1 < sub_graphs.size(); ++i) {
        MergeLink link;
        link.from_output = 0;
        link.to_input = 0;
        spec.links.push_back(link);
    }
    return spec;
}

std::string GraphMerger::mergedCacheKey(const std::vector<Graph>& sub_graphs,
                                         const MergeSpec& spec) {
    // 稳定字符串拼接：避免 std::ostringstream 在嵌入式平台的潜在格式差异。
    // 格式: "<kMergedCacheKeyPrefix>|<N>|<g0.toString>|<g1.toString>|...|<link0.from_output>-><link0.to_input>|..."
    // 1. 验证 spec 长度匹配（避免越界 panic）
    if (spec.links.size() != (sub_graphs.size() > 0 ? sub_graphs.size() - 1 : 0)) {
        return "merged_v1|invalid_spec|";
    }
    std::string out;
    out.reserve(64 + sub_graphs.size() * 64);
    // 使用与 C3Engine::compileMergedAsync 同步的版本前缀（kMergedCacheKeyPrefix）
    out += kMergedCacheKeyPrefix;
    out += "|";
    out += std::to_string(sub_graphs.size());
    for (const auto& g : sub_graphs) {
        out += '|';
        out += g.toString();
    }
    for (const auto& link : spec.links) {
        out += '|';
        out += std::to_string(link.from_output);
        out += "->";
        if (link.to_input == SIZE_MAX) {
            out += "END";
        } else {
            out += std::to_string(link.to_input);
        }
    }
    return out;
}

} // namespace c3
} // namespace ct
