/**
 * @file C3KernelRegistry.cpp
 * @brief C3 内核注册表 · 融合/反向 kernel 子系统的实现
 * @details 二元/一元单 kernel 路径（install/tryExecute/tryExecuteUnary）已在
 *          C3KernelRegistry.h 内联实现（header-only），本 .cpp 负责：
 *          1. 融合 kernel (fused_entries_) 的安装/查询/执行
 *          2. 反向 kernel (backward_entries_) 的安装/查询/执行
 *          3. 序列/首 op 模糊匹配 (findFusedKernelFor*)
 *          4. 融合 kernel 包装执行 (executeFusedWithInputs)
 *
 *          **当前状态：stub 阶段**。DEBT-NEW-7 修复集（用户之前设计的 region fusion
 *          全套）需要这些方法作为执行后端，但 region fusion 仍按宏开关
 *          CT_C3_DISABLE_REGION_FUSION 关闭；本文件所有方法在 stub 形态下
 *          返回 nullopt / 空向量 / 抛 not_implemented，保证 build 通过且
 *          c3 单 kernel 路径行为不退化（calls fall back to eager）。
 *
 * @date 2026-08-09
 */

#include "../../include/C3/C3KernelRegistry.h"
#include "../../include/C3/C3Engine.h"  // CompiledKernel 完整定义
#include "../../include/CtorchError.h"

namespace ct {
namespace c3 {

// ======================= 融合 kernel 执行 =======================

// DEBT-NEW-7 region fusion 后端：从 CompiledKernel 取 FusedKernelFunc
// (通过 HandwrittenKernelGen.cpp::generateFromGraph 注入的 fused_func),
// 准备 input/output buffer,invoke function pointer,包装成 Tensor 返回。
// FusedKernelFunc 签名: void (*)(const float* const*, float*, size_t)
//   - inputs: 外部输入指针数组(顺序与 Graph.addInput 一致)
//   - output: 输出 buffer(由调用方按 shapes.out_shape 预分配)
//   - n: 输出元素数(用于校验 + kernel 内部向量化长度)
Tensor C3KernelRegistry::executeFusedWithInputs(
    std::shared_ptr<CompiledKernel> kernel,
    const std::vector<Tensor>& inputs,
    const KernelShapeInfo& shapes) {
    if (!kernel || inputs.empty()) {
        return Tensor();
    }

    // 从 CompiledKernel 派生类取 FusedKernelFunc
    // HandwrittenKernelGen 编译结果存于 GeneratedKernel.fused_func,
    // 通过 ConcreteCompiledKernel/HandwrittenCompiledKernel 暴露
    // 当前 CompiledKernel 接口是虚函数 execute() → vector<Tensor>,
    // 我们用它作为统一调用入口（每个 backend 各自实现 fused kernel 路径）
    try {
#ifdef CT_PROFILE_PERF
        auto t0 = std::chrono::steady_clock::now();
#endif
        auto outputs = kernel->execute(inputs);
#ifdef CT_PROFILE_PERF
        auto t1 = std::chrono::steady_clock::now();
        recordPerfRegionMatch(
            (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
#endif
        if (outputs.empty()) {
            return Tensor();
        }
        fused_hit_count_.fetch_add(1, std::memory_order_relaxed);
        return outputs[0];
    } catch (const std::exception& e) {
        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL,
            ErrorType::UNKNOWN,
            "C3KernelRegistry::executeFusedWithInputs: kernel execute failed: " +
            std::string(e.what()) + ", falling back to eager.");
        return Tensor();
    }
}

// tryExecuteFused 通过 C3KernelRegistry 自身的 fused_entries_ map 查找
// (与 tryExecute/tryExecuteUnary 平行,使用 fused_pattern + shape 做 key)。
// 当前 stub: region fusion 完整启用时由 tryRegionDispatch 通过
// executeFusedWithInputs(kernel, ...) 直接调用,不走本入口。
// 保留接口以备 region fusion 启用后通过 op_type + inputs 快速 dispatch。
std::optional<Tensor> C3KernelRegistry::tryExecuteFused(
    op /*op_type*/, const std::vector<Tensor>& /*inputs*/) {
    return std::nullopt;
}

// ======================= 二元/一元 forward kernel 执行 =======================

// [Fix 2026-08-09 + v0.5.2 修 (2026-08-09)]: tryExecute/tryExecuteUnary out-of-line
// 之前 inline 在 C3KernelRegistry.h, 调 entry.func (老路径), 但 install 新接口
// (shared_ptr<CompiledKernel>) 只设 entry.kernel 没设 entry.func → entry.func
// 永远是 nullptr → 调 nullptr → segfault. test_c3_graph C3HotReplace.InstallAndDispatch
// 暴露这个 bug. 改用 entry.kernel->execute() (跟 tryExecuteBackward 一致).
std::optional<Tensor> C3KernelRegistry::tryExecute(
    op op_type, const Tensor& a, const Tensor& b) {
    auto key = makeKeyFromShapes(op_type, a.device(), a.shape(), b.shape());

    C3Entry entry;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it == entries_.end() || !it->second.active) {
            found = false;
        } else {
            found = true;
            entry = it->second;
        }
    }
#ifdef CT_DEBUG
    fprintf(stderr, "[DBG] tryBin op=%d key3=%zu found=%d a=[%s] b=[%s]\n",
            (int)op_type, key.third, (int)found,
            shapeDebug(a.shape()).c_str(), shapeDebug(b.shape()).c_str());
#endif
    if (!found) return std::nullopt;

    if (a.shape() != entry.shapes.lhs_shape ||
        b.shape() != entry.shapes.rhs_shape) {
        return std::nullopt;
    }

    try {
#ifdef CT_PROFILE_PERF
        auto t0 = std::chrono::steady_clock::now();
#endif
        if (!entry.kernel) {
            return std::nullopt;
        }
        std::vector<Tensor> inputs = {a, b};
        auto outputs = entry.kernel->execute(inputs);
        if (outputs.empty()) {
            return std::nullopt;
        }
        Tensor out = outputs[0];
#ifdef CT_PROFILE_PERF
        auto t1 = std::chrono::steady_clock::now();
        recordPerfC3SingleInvoke(
            (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
#endif
        hit_count_.fetch_add(1, std::memory_order_relaxed);

        if (!validateOutputShape(op_type, a.device(), out, entry.shapes.out_shape)) {
            return std::nullopt;
        }

#ifdef CT_DEBUG
        {
            const float* out_data = out.data_read<float>();
            size_t n = std::min(size_t(5), out.numel());
            fprintf(stderr, "[C3-VALIDATE] binary op=%d out=[", (int)op_type);
            for (size_t i = 0; i < out.shape().size(); ++i) {
                if (i > 0) fprintf(stderr, ",");
                fprintf(stderr, "%zu", out.shape()[i]);
            }
            fprintf(stderr, "] first_%zu=[", n);
            for (size_t i = 0; i < n; ++i) {
                if (i > 0) fprintf(stderr, ",");
                fprintf(stderr, "%.6f", out_data[i]);
            }
            fprintf(stderr, "]\n");
        }
#endif
        return out;
    } catch (...) {
        miss_count_.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }
}

std::optional<Tensor> C3KernelRegistry::tryExecuteUnary(op op_type, const Tensor& a) {
    auto key = makeKeyFromShapes(op_type, a.device(), a.shape(), {});

    C3Entry entry;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it == entries_.end() || !it->second.active) {
            found = false;
        } else {
            found = true;
            entry = it->second;
        }
    }
#ifdef CT_DEBUG
    fprintf(stderr, "[DBG] tryUnary op=%d key3=%zu found=%d a=[%s]\n",
            (int)op_type, key.third, (int)found,
            shapeDebug(a.shape()).c_str());
#endif
    if (!found) return std::nullopt;

    if (a.shape() != entry.shapes.lhs_shape) {
        return std::nullopt;
    }

    try {
        if (!entry.kernel) {
            return std::nullopt;
        }
        std::vector<Tensor> inputs = {a};
        auto outputs = entry.kernel->execute(inputs);
        if (outputs.empty()) {
            return std::nullopt;
        }
        Tensor out = outputs[0];
        hit_count_.fetch_add(1, std::memory_order_relaxed);

        if (!validateOutputShape(op_type, a.device(), out, entry.shapes.out_shape)) {
            return std::nullopt;
        }
        return out;
    } catch (...) {
        miss_count_.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }
}

// ======================= 反向 kernel 执行 =======================

// TODO(c3-backward): 反向 fusion kernel 的实际执行后端。
// 当前 stub 返回 nullopt → C3BackwardCapture::tryExecuteBackward 会回退 eager。
// 完整实现需要：
//  1. 在 backward_entries_ 中查找 backward_key
//  2. 验证 grad.shape() 与注册时记录的 grad_shape 一致
//  3. 验证 forward_inputs 数量与 kernel 签名匹配
//  4. invoke CompiledKernel 的 function pointer
//  5. 包装为 vector<Tensor> 返回（多输出支持）
// DEBT-NEW-7 v0.5.1+: 反向 fusion kernel 的实际执行后端
// 之前是 stub → C3BackwardCapture 编译完 kernel 装进 backward_entries_ 后
// 也没人能找到它(此函数返回 nullopt),导致 bw_hit=0,反向全走 eager。
// 修复:从 backward_entries_ 查 backward_key,invoke CompiledKernel,
// 包装成 vector<Tensor>(1 element) 返回(C3BackwardCapture 每次
// 查 per-key,所以返回单元素 vector)。
std::optional<std::vector<Tensor>> C3KernelRegistry::tryExecuteBackward(
    const std::string& backward_key, const Tensor& grad,
    const std::vector<Tensor>& forward_inputs) {
    BackwardEntry entry;
    bool found = false;
    size_t map_size = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        map_size = backward_entries_.size();
        auto it = backward_entries_.find(backward_key);
        if (it == backward_entries_.end() || !it->second.active) {
            found = false;
        } else {
            found = true;
            entry = it->second;
        }
    }
    if (!found) {
#ifdef CT_DEBUG
        static int dbg_miss = 0;
        if (dbg_miss < 200) {
            std::cerr << "[C3-BW-DEBUG] tryExecuteBackward MISS key=" << backward_key
                      << " grad_shape=[";
            for (auto s : grad.shape()) std::cerr << s << ",";
            std::cerr << "] map_size=" << map_size << " numel=" << grad.numel() << std::endl;
            std::cerr.flush();
            dbg_miss++;
        }
#endif
        return std::nullopt;
    }

#ifdef CT_DEBUG
    {
        static int dbg_hit = 0;
        if (dbg_hit < 5) {
            std::cerr << "[C3-BW-DEBUG] tryExecuteBackward HIT key=" << backward_key
                      << " grad_shape=[";
            for (auto s : grad.shape()) std::cerr << s << ",";
            std::cerr << "] entry_out_shape=[";
            for (auto s : entry.out_shape) std::cerr << s << ",";
            std::cerr << "]" << std::endl;
            std::cerr.flush();
            dbg_hit++;
        }
    }
#endif

    // 形状验证:grad.shape() 必须与注册时记录的 grad_shape 一致
    if (grad.shape() != entry.grad_shape) {
#ifdef CT_DEBUG
        std::cerr << "[C3-BW-DEBUG] tryExecuteBackward SHAPE MISMATCH key=" << backward_key
                  << " grad_shape=[";
        for (auto s : grad.shape()) std::cerr << s << ",";
        std::cerr << "] expected=[";
        for (auto s : entry.grad_shape) std::cerr << s << ",";
        std::cerr << "]" << std::endl;
        std::cerr.flush();
#endif
        return std::nullopt;
    }

    // Invoke CompiledKernel: backward kernel 签名 = [grad, forward_input_0, ...]
    // 不同 backward graph 接受不同数量的 input,install 时已存 num_inputs:
    //   - ReLU/Sigmoid/Tanh: 2 inputs (grad, x)
    //   - Add/Sub:           1 input  (grad only)
    //   - Mul/MatMul/Div:    3 inputs (grad, A, B)
    // 传多报 BroadcastUtils 错(已实测),传少 kernel 读野指针。
    // [Fix 2026-08-11 DCE 输入平移] 不再按「前 num_inputs-1 个 forward_input」前缀喂入，
    // 而是严格按 entry.fwd_input_map 逐一取 forward_inputs[map[k]]。因为最小集 build
    // 的图输入顺序 ≠ forward_inputs 顺序（如 MatMul grad_x 图=[grad,B]，B=forward_inputs[1]）。
    try {
        std::vector<Tensor> inputs;
        inputs.reserve(1 + entry.fwd_input_map.size());
        inputs.push_back(grad);
        for (size_t fwd_idx : entry.fwd_input_map) {
            if (fwd_idx >= forward_inputs.size()) {
                // 防御：索引越界 → 回退 eager（正确性优先）
                return std::nullopt;
            }
            inputs.push_back(forward_inputs[fwd_idx]);
        }
        if (inputs.size() != entry.num_inputs) {
            // 防御：输入数量与注册时不符 → 回退 eager
            return std::nullopt;
        }
        auto outputs = entry.kernel->execute(inputs);
        if (outputs.empty()) return std::nullopt;
        return outputs;
    } catch (...) {
        return std::nullopt;
    }
}

// ======================= 序列/首 op 模糊匹配 =======================

// TODO(region-fusion): 用于按 op 序列模糊匹配已注册的融合 kernel（备选 path A）。
// 当前 stub 返回 nullopt → 调度器继续走精确匹配或 eager。
std::optional<std::pair<std::shared_ptr<CompiledKernel>, KernelShapeInfo>>
C3KernelRegistry::findFusedKernelForSequence(
    const std::vector<op>& /*op_seq*/, DeviceType /*dev*/,
    const std::vector<size_t>& /*first_input_shape*/) {
    return std::nullopt;
}

// TODO(region-fusion): 用于按首 op 匹配融合 kernel。
// 当前 stub 返回 nullopt。
std::optional<std::pair<std::shared_ptr<CompiledKernel>, KernelShapeInfo>>
C3KernelRegistry::findFusedKernelForFirstOp(
    op /*op_type*/, const std::vector<size_t>& /*input_shape*/,
    DeviceType /*dev*/) {
    return std::nullopt;
}

void C3KernelRegistry::install(op op_type, DeviceType dev,
                               std::shared_ptr<CompiledKernel> kernel,
                               const KernelShapeInfo& shapes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = makeKey(op_type, dev, shapes);
    auto it = entries_.find(key);
    if (it != entries_.end() && it->second.active && it->second.kernel) {
        if (kernel->optLevel() <= it->second.kernel->optLevel()) {
            // 如果已有的同形状 kernel 的编译优化等级更高，则跳过本次安装（即低优化度内核不能覆盖已存在的高优化度内核）
            return;
        }
    }
    C3Entry e;
    e.kernel = std::move(kernel);
    e.shapes = shapes;
    e.active = true;
    entries_[key] = std::move(e);
    install_count_.fetch_add(1, std::memory_order_release);
#ifdef CT_DEBUG
    fprintf(stderr, "[DBG] INSTALL op=%d dev=%d key3=%zu lhs=[%s] rhs=[%s]\n",
            (int)op_type, (int)dev, key.third,
            shapeDebug(shapes.lhs_shape).c_str(), shapeDebug(shapes.rhs_shape).c_str());
#endif
}

void C3KernelRegistry::installFused(std::shared_ptr<CompiledKernel> kernel,
                                    op op_type, const KernelShapeInfo& shapes) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = makeFusedKey(op_type, shapes);
    auto it = fused_entries_.find(key);
    if (it != fused_entries_.end() && it->second.active && it->second.kernel) {
        if (kernel->optLevel() <= it->second.kernel->optLevel()) {
            return;
        }
    }
    fused_entries_[key] = {std::move(kernel), shapes, true};
    install_count_.fetch_add(1, std::memory_order_release);
}

void C3KernelRegistry::installBackward(const std::string& backward_key,
                                       std::shared_ptr<CompiledKernel> kernel,
                                       const std::vector<size_t>& grad_shape,
                                       const std::vector<size_t>& out_shape,
                                       const std::vector<size_t>& fwd_input_map,
                                       size_t num_inputs) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = backward_entries_.find(backward_key);
    if (it != backward_entries_.end() && it->second.active && it->second.kernel) {
        if (kernel->optLevel() <= it->second.kernel->optLevel()) {
            return;
        }
    }
    BackwardEntry e;
    e.kernel = std::move(kernel);
    e.grad_shape = grad_shape;
    e.out_shape = out_shape;
    e.fwd_input_map = fwd_input_map;
    e.num_inputs = num_inputs;
    e.active = true;
    backward_entries_[backward_key] = std::move(e);
    install_count_.fetch_add(1, std::memory_order_release);
}

} // namespace c3
} // namespace ct
