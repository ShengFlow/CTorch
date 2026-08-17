/**
 * @file LinalgOneShotGen.h
 * @generation JIT-3.0 One-Shot 统一融合后端设计
 * @brief C3 JIT 3.0: 统一 C3-to-Linalg Lowering + Linalg Fusion + One-Shot Bufferization 极致优化管线
 * @date 2026/08/15
 */

#ifndef CTORCH_C3_LINALG_ONESHOT_GEN_H
#define CTORCH_C3_LINALG_ONESHOT_GEN_H

#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include "C3/Graph.h"

namespace ct {
namespace c3 {

/// 编译并持有基于 JIT 3.0 (One-Shot Bufferization) 统一管线的多节点融合 kernel
class LinalgOneShotKernel {
public:
    /// 编译融合 kernel
    explicit LinalgOneShotKernel(const Graph& graph, int opt_level = 3);
    ~LinalgOneShotKernel();
    LinalgOneShotKernel(const LinalgOneShotKernel&) = delete;
    LinalgOneShotKernel& operator=(const LinalgOneShotKernel&) = delete;
    LinalgOneShotKernel(LinalgOneShotKernel&&) noexcept;
    LinalgOneShotKernel& operator=(LinalgOneShotKernel&&) noexcept;

    size_t numInputs() const { return num_inputs_; }
    size_t numOutputs() const { return num_outputs_; }

    /// 执行融合 kernel
    void execute(const float* const* in_ptrs, float* const* out_ptrs, size_t n) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    size_t num_inputs_;
    size_t num_outputs_;
};

/// 共享融合 kernel 缓存工厂：同一 graph_key 只编译一次
std::shared_ptr<LinalgOneShotKernel> getCachedLinalgOneShotKernel(
    const Graph& graph, const std::string& graph_key, int opt_level = 3);

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_LINALG_ONESHOT_GEN_H
