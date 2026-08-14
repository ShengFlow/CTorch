/**
 * @file Tracer.h
 * @brief C3 JIT 图捕获层 — ProxyTensor CRTP + Tracer
 * @details ProxyTensor 使用 CRTP（奇异递归模板模式）实现编译期多态：
 *          ProxyTensorBase<Derived> 声明所有算子重载，
 *          方法体在 Tracer 完整定义之后实现（解决循环依赖）。
 *          Tracer 管理图的生命周期，支持 lambda 式 trace 和手动式 trace。
 *
 *          设计模式（ctfp #18 Kleisli 组合）：
 *          每个 recordOp 返回新 ProxyTensor，携带指向输入节点的引用，
 *          trace 结束时从最后一个操作的返回值反向遍历引用链重建 DAG。
 * @date 2026/7/31
 */

#ifndef CTORCH_C3_TRACER_H
#define CTORCH_C3_TRACER_H

#include "Graph.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ct {
namespace c3 {

// ======================= 前向声明 =======================

class Tracer;
class ProxyTensor;

// ======================= CRTP 基类：ProxyTensorBase（声明） =======================

/**
 * @class ProxyTensorBase
 * @brief ProxyTensor 的 CRTP 基类，声明所有算子重载。
 * @tparam Derived 派生类类型（必须继承自 ProxyTensorBase<Derived>）
 * @details 方法体在 Tracer 完整定义之后实现，解决 Tracer ↔ ProxyTensor 循环依赖。
 */
template <typename Derived>
class ProxyTensorBase {
public:
    // 逐元素算子
    Derived operator+(const Derived& other) const;
    Derived operator-(const Derived& other) const;
    Derived operator*(const Derived& other) const;
    Derived operator/(const Derived& other) const;
    Derived operator-() const;

    // 矩阵乘法
    Derived matmul(const Derived& other) const;

    // 激活函数
    Derived relu() const;
    Derived sigmoid() const;
    Derived tanh() const;

    // 标量操作（右操作数）
    Derived operator*(float scalar) const;
    Derived operator+(float scalar) const;
    Derived operator-(float scalar) const;
    Derived operator/(float scalar) const;

protected:
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

private:
    template <typename OpNode>
    Derived binaryOp(const Derived& other, const char* op_name) const;

    template <typename OpNode>
    Derived unaryOp(const char* op_name) const;

    template <typename OpNode>
    Derived scalarOp(float scalar, const char* op_name) const;
};

// ======================= 标量左操作数重载（声明） =======================

ProxyTensor operator*(float lhs, const ProxyTensor& rhs);
ProxyTensor operator+(float lhs, const ProxyTensor& rhs);
ProxyTensor operator-(float lhs, const ProxyTensor& rhs);
ProxyTensor operator/(float lhs, const ProxyTensor& rhs);

// ======================= ProxyTensor =======================

/**
 * @class ProxyTensor
 * @brief 图捕获代理张量，继承 CRTP 基类获得所有算子重载。
 */
class ProxyTensor : public ProxyTensorBase<ProxyTensor> {
    friend class Tracer;
    friend class ProxyTensorBase<ProxyTensor>;
    friend ProxyTensor operator*(float lhs, const ProxyTensor& rhs);
    friend ProxyTensor operator+(float lhs, const ProxyTensor& rhs);
    friend ProxyTensor operator-(float lhs, const ProxyTensor& rhs);
    friend ProxyTensor operator/(float lhs, const ProxyTensor& rhs);

public:
    ProxyTensor() : tracer_(nullptr), handle_(SIZE_MAX) {}
    operator Tensor() const = delete;

    Tracer* tracer() const { return tracer_; }
    size_t handle() const { return handle_; }
    const TensorDesc& desc() const;

protected:
    ProxyTensor(Tracer* t, size_t h) : tracer_(t), handle_(h) {}

private:
    Tracer* tracer_ = nullptr;
    size_t handle_ = SIZE_MAX;
};

// ======================= Tracer =======================

/**
 * @class Tracer
 * @brief 图捕获器，管理 ProxyTensor 的生命周期和图构建。
 */
class Tracer {
public:
    Tracer() = default;

    template <typename F>
    static Graph trace(F&& fn, const TensorDesc& desc) {
        Tracer t;
        t.begin();
        auto x = t.input(desc);
        auto result = fn(x);
        return t.end(result);
    }

    template <typename F>
    static Graph trace(F&& fn, const TensorDesc& a_desc, const TensorDesc& b_desc) {
        Tracer t;
        t.begin();
        auto a = t.input(a_desc);
        auto b = t.input(b_desc);
        auto result = fn(a, b);
        return t.end(result);
    }

    template <typename F>
    static Graph trace(F&& fn, const TensorDesc& a_desc, const TensorDesc& b_desc,
                const TensorDesc& c_desc) {
        Tracer t;
        t.begin();
        auto a = t.input(a_desc);
        auto b = t.input(b_desc);
        auto c = t.input(c_desc);
        auto result = fn(a, b, c);
        return t.end(result);
    }

    void begin() {
        graph_ = Graph();
        input_handles_.clear();
    }

    ProxyTensor input(const TensorDesc& desc) {
        size_t handle = graph_.addInput(desc);
        input_handles_.push_back(handle);
        return ProxyTensor(this, handle);
    }

    Graph end(const ProxyTensor& output) {
        graph_.markOutput(output.handle());
        return graph_;
    }

    size_t recordOp(const NodeVariant& op,
                     const std::vector<size_t>& input_ids,
                     const TensorDesc& out_desc) {
        return graph_.addNode(op, input_ids, out_desc);
    }

    const TensorDesc& getDesc(size_t handle) const {
        return graph_.node(handle).out_desc;
    }

    const Graph& graph() const { return graph_; }

private:
    Graph graph_;
    std::vector<size_t> input_handles_;
};

// ======================= ProxyTensor 延迟实现 =======================

inline const TensorDesc& ProxyTensor::desc() const {
    return tracer_->getDesc(handle_);
}

// ======================= ProxyTensorBase 模板方法实现 =======================

template <typename Derived>
Derived ProxyTensorBase<Derived>::operator+(const Derived& other) const {
    return binaryOp<AddNode>(other, "Add");
}

template <typename Derived>
Derived ProxyTensorBase<Derived>::operator-(const Derived& other) const {
    return binaryOp<SubNode>(other, "Sub");
}

template <typename Derived>
Derived ProxyTensorBase<Derived>::operator*(const Derived& other) const {
    return binaryOp<MulNode>(other, "Mul");
}

template <typename Derived>
Derived ProxyTensorBase<Derived>::operator/(const Derived& other) const {
    return binaryOp<DivNode>(other, "Div");
}

template <typename Derived>
Derived ProxyTensorBase<Derived>::operator-() const {
    return unaryOp<NegNode>("Neg");
}

template <typename Derived>
Derived ProxyTensorBase<Derived>::matmul(const Derived& other) const {
    return binaryOp<MatMulNode>(other, "MatMul");
}

// ======================= 激活函数实现 =======================

template <typename Derived>
Derived ProxyTensorBase<Derived>::relu() const {
    return unaryOp<ReLUNode>("ReLU");
}

template <typename Derived>
Derived ProxyTensorBase<Derived>::sigmoid() const {
    return unaryOp<SigmoidNode>("Sigmoid");
}

template <typename Derived>
Derived ProxyTensorBase<Derived>::tanh() const {
    return unaryOp<TanhNode>("Tanh");
}

// ======================= 标量操作实现 =======================

template <typename Derived>
Derived ProxyTensorBase<Derived>::operator*(float scalar) const {
    return scalarOp<MulNode>(scalar, "Mul");
}

template <typename Derived>
Derived ProxyTensorBase<Derived>::operator+(float scalar) const {
    return scalarOp<AddNode>(scalar, "Add");
}

template <typename Derived>
Derived ProxyTensorBase<Derived>::operator-(float scalar) const {
    return scalarOp<SubNode>(scalar, "Sub");
}

template <typename Derived>
Derived ProxyTensorBase<Derived>::operator/(float scalar) const {
    return scalarOp<DivNode>(scalar, "Div");
}

template <typename Derived>
template <typename OpNode>
Derived ProxyTensorBase<Derived>::binaryOp(const Derived& other, const char* op_name) const {
    const Derived& self = derived();
    Tracer* t = self.tracer();
    if (!t) {
        throw std::runtime_error(
            std::string("ProxyTensor::") + op_name + ": tracer is null");
    }

    const TensorDesc& lhs_desc = t->getDesc(self.handle());
    const TensorDesc& rhs_desc = t->getDesc(other.handle());

    TensorDesc out_desc = lhs_desc;
    if constexpr (std::is_same_v<OpNode, MatMulNode>) {
        out_desc = TensorDesc::fromShape(
            {lhs_desc.shape[0], rhs_desc.shape[1]},
            lhs_desc.dtype, lhs_desc.device);
    }

    size_t new_handle = t->recordOp(
        OpNode{lhs_desc, rhs_desc},
        {self.handle(), other.handle()},
        out_desc);

    return Derived(t, new_handle);
}

template <typename Derived>
template <typename OpNode>
Derived ProxyTensorBase<Derived>::unaryOp(const char* op_name) const {
    const Derived& self = derived();
    Tracer* t = self.tracer();
    if (!t) {
        throw std::runtime_error(
            std::string("ProxyTensor::") + op_name + ": tracer is null");
    }

    const TensorDesc& desc = t->getDesc(self.handle());
    size_t new_handle = t->recordOp(
        OpNode{desc}, {self.handle()}, desc);
    return Derived(t, new_handle);
}

template <typename Derived>
template <typename OpNode>
Derived ProxyTensorBase<Derived>::scalarOp(float scalar, const char* op_name) const {
    const Derived& self = derived();
    Tracer* t = self.tracer();
    if (!t) {
        throw std::runtime_error(
            std::string("ProxyTensor::") + op_name + " scalar: tracer is null");
    }

    // [Fix 2026-08-15] 按值拷贝 desc，避免 recordOp 引发 vector 扩容后引用悬垂
    TensorDesc desc = t->getDesc(self.handle());
    size_t scalar_handle = t->recordOp(
        ConstNode{static_cast<double>(scalar)}, {}, desc);
    return Derived(t, t->recordOp(
        OpNode{desc, desc},
        {scalar_handle, self.handle()},
        desc));
}

// ======================= 标量左操作数实现 =======================

inline ProxyTensor operator*(float lhs, const ProxyTensor& rhs) {
    return rhs * lhs;
}

inline ProxyTensor operator+(float lhs, const ProxyTensor& rhs) {
    return rhs + lhs;
}

inline ProxyTensor operator-(float lhs, const ProxyTensor& rhs) {
    Tracer* t = rhs.tracer();
    if (!t) throw std::runtime_error("ProxyTensor: tracer is null");
    // [Fix 2026-08-15] 按值拷贝 desc，避免 recordOp 引发 vector 扩容后引用悬垂
    TensorDesc desc = t->getDesc(rhs.handle());
    size_t lhs_handle = t->recordOp(
        ConstNode{static_cast<double>(lhs)}, {}, desc);
    size_t result_handle = t->recordOp(
        SubNode{desc, desc}, {lhs_handle, rhs.handle()}, desc);
    return ProxyTensor(t, result_handle);
}

inline ProxyTensor operator/(float lhs, const ProxyTensor& rhs) {
    Tracer* t = rhs.tracer();
    if (!t) throw std::runtime_error("ProxyTensor: tracer is null");
    // [Fix 2026-08-15] 按值拷贝 desc，避免 recordOp 引发 vector 扩容后引用悬垂
    TensorDesc desc = t->getDesc(rhs.handle());
    size_t lhs_handle = t->recordOp(
        ConstNode{static_cast<double>(lhs)}, {}, desc);
    size_t result_handle = t->recordOp(
        DivNode{desc, desc}, {lhs_handle, rhs.handle()}, desc);
    return ProxyTensor(t, result_handle);
}

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_TRACER_H