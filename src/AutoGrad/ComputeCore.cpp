/**
 *@file ComputeCore.cpp
 *@author Beapoe
 *@brief 计算核心
 *@date 2026/2/22
 **/

#include <utility>
#include <queue>
#include <algorithm>

#include "../../include/AutoGrad/ComputeCore.h"
#include "../../include/AutoGrad/Node.h"
#include "../../include/CtorchError.h"
#include "../../include/CtorchScheduler.h"  // ct::detail::set_in_backward（DEBT-NEW-7 H2 fix）
#include "../../include/ThreadPool.h"
#include "../../include/Arena.h"
#include "../../include/AutoGrad.h"
#ifndef CT_DISABLE_C3
#include "../../include/C3/C3BackwardCapture.h"  // DEBT-NEW-7 v0.5.1+ 接通 C3 backward fusion
#endif
#ifdef __OBJC__
#include "../../src/kernels/kernels.h"
#endif

GradBucket &GradBucket::getInstance() {
    static GradBucket instance;
    return instance;
}

ssize_t GradBucket::find(const std::shared_ptr<Node> &target) {
    auto it = std::find_if(_packs.begin(), _packs.end(), [target](const GradPack &pack) {
        return pack._targetNode == target;
    });
    if (it != _packs.end())
        return std::distance(_packs.begin(), it);
    return -1;
}

ssize_t GradBucket::find(const std::shared_ptr<Node> &target, int idx) {
    auto it = std::find_if(_packs.begin(), _packs.end(), [target, idx](const GradPack &pack) {
        return pack._targetNode == target && pack._idx == idx;
    });
    if (it != _packs.end())
        return std::distance(_packs.begin(), it);
    return -1;
}

void GradBucket::add(std::vector<GradPack>&& newPacks) {
    std::lock_guard<std::mutex> lock(_mtx);
    for (auto&& pack : newPacks) {
        const ssize_t idx = find(pack._targetNode);
        if (idx != -1) {
            for (size_t i = 0; i < pack._grad.size(); ++i) {
                if (i < _packs[idx]._grad.size()) {
                    if (_packs[idx]._grad[i].numel() > 0) {
                        // 避免 MPS 异步写入完成前触发深拷贝导致梯度归零
                        Tensor add_result = _packs[idx]._grad[i] + pack._grad[i];
                        _packs[idx]._grad[i] = std::move(add_result);
                    } else {
                        _packs[idx]._grad[i] = std::move(pack._grad[i]);
                    }
                } else {
                    _packs[idx]._grad.push_back(std::move(pack._grad[i]));
                }
            }
        } else {
            pack._idx = -1;
            _packs.push_back(std::move(pack));
        }
    }
}

void GradBucket::remove(const std::shared_ptr<Node> &target) {
    std::lock_guard lock(_mtx);
    const ssize_t idx = find(target);
    if (idx != -1)
        _packs.erase(_packs.begin() + idx);
}

bool GradBucket::tryGetGrad(const std::shared_ptr<Node> &target, std::vector<Tensor> &out_grads) {
    std::lock_guard lock(_mtx);
    const ssize_t idx = find(target);
    if (idx != -1) {
        out_grads = std::move(_packs[idx]._grad);
        _packs.erase(_packs.begin() + idx);
        return true;
    }
    return false;
}

bool GradBucket::empty() {
    std::lock_guard lock(_mtx);
    return _packs.empty();
}

void GradBucket::clear() {
    std::lock_guard<std::mutex> lock(_mtx);
    _packs.clear();
}

std::vector<Tensor> GradBucket::operator[](const std::shared_ptr<Node> &target) {
    std::lock_guard<std::mutex> lock(_mtx);
    ssize_t idx = find(target);
    if (idx != -1) {
        auto result = std::move(_packs[idx]._grad);
        _packs.erase(_packs.begin() + idx);
        return result;
    }
    CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN,
                        "Trying to visit an nonexistential grad pack");
    return {};
}

std::shared_ptr<Node> ComputeCore::tryPopReadyNode() {
    std::lock_guard lock(_mtx);
    if (_readyNodes.empty())
        return nullptr;
    auto node = std::move(_readyNodes.front());
    _readyNodes.pop();
    return node;
}

ComputeCore &ComputeCore::getInstance() {
    static ComputeCore instance;
    return instance;
}

void ComputeCore::addReadyNode(std::shared_ptr<Node> node) {
    std::lock_guard lock(_mtx);
    _readyNodes.push(std::move(node));
    _cv.notify_one();
}

void ComputeCore::backward(std::shared_ptr<Node> root, bool retainGraph) {
    // DEBT-NEW-7 H2 修复:在 backward 期间标记 in_backward=true，
    // 调度器的 inAutogradScope guard 借此识别反向传播路径（其 matmul 输入
    // 通常 requires_grad=false,如 x.T @ grad），跳过 c3 单 kernel 注入。
    // RAII 模式：函数出口自动清除 flag，即使中途抛异常。
    // 注：ct::detail 命名空间仅在 C3 启用时可用(由 CtorchScheduler.h 提供)
#ifndef CT_DISABLE_C3
    bool prev_in_backward = ct::detail::g_in_backward();
    ct::detail::set_in_backward(true);
    // [Dev 2026-08-12 修法 C-0.5 配套] 同步设置 ct::c3 内部 thread_local 标志,
    //   C3HotPathManager.h 在 C3 内部用 in_backward() 检测 forward/backward,
    //   不能直接 include CtorchScheduler.h (循环 include).
    bool prev_in_backward_local = ct::c3::in_backward();
    ct::c3::set_in_backward_local(true);
    struct FlagGuard {
        bool prev;
        bool prev_local;
        ~FlagGuard() {
            ct::detail::set_in_backward(prev);
            ct::c3::set_in_backward_local(prev_local);
        }
    } guard{prev_in_backward, prev_in_backward_local};
#endif

    bool original_enable_grad = AutoGrad::EnableGrad;
    AutoGrad::EnableGrad = false;

    GradBucket &bucket = GradBucket::getInstance();
    bucket.clear();

    const std::vector<size_t>& root_shape = root->getResultShape();
    auto root_result = root->getResult();
    DeviceType root_device = root_result ? root_result->device() : DeviceType::kCPU;
    
    if (root_shape.empty() || (root_shape.size() == 1 && root_shape[0] == 1)) {
        Tensor grad_tensor(1.0f, root_device);
        GradPack primary = {root, std::vector({grad_tensor}), -1};
        bucket.add(std::vector({std::move(primary)}));
    } else {
        Tensor cpu_tensor(ShapeTag{}, root_shape, DType::kFloat, DeviceType::kCPU, false);
        float* p = cpu_tensor.data_write<float>();
        std::fill(p, p + cpu_tensor.numel(), 1.0f);
        Tensor grad_tensor = cpu_tensor.to(root_device);
        GradPack primary = {root, std::vector({grad_tensor}), -1};
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - grad_tensor dim: " + std::to_string(grad_tensor.dim()));
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding primary grad pack");
        bucket.add(std::vector({std::move(primary)}));
    }

    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding root to ready nodes");
    addReadyNode(root);

    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Processing ready nodes");
    while (true) {
        std::shared_ptr<Node> node = tryPopReadyNode();
        if (!node) {
            if (!bucket.empty()) {
                CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Grad bucket not empty but no ready nodes");
            }
            break;
        }

        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Processing node");

        std::vector<Tensor> grads;
        if (!bucket.tryGetGrad(node, grads)) {
            CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - No grad found for node, skipping");
            continue;
        }

        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Processing node backward");

        // ===== C3 backward fusion 接线 (5d follow-up 完成, 2026-08-11) =====
        // 尝试用 C3 JIT 编译的反向 kernel 计算梯度;未命中/形状不匹配/禁用时回退 eager。
        // 安全护栏: 仅当 ① grad 恰好 1 个(聚合梯度) ② 有输入 ③ 上游节点数==输入数
        // ④ C3 返回梯度数==输入数 且 每个梯度 shape 与对应输入 shape 一致, 才接受。
        // 任一不满足 → result 保持空 → 走 eager node->backward(grads)。
        std::vector<GradPack> result;
#ifndef CT_DISABLE_C3
        if (grads.size() == 1 && !node->getInputs().empty()) {
            const auto& fwd_inputs = node->getInputs();
            auto upstream = node->getUpStreamNodes();
            if (upstream.size() == fwd_inputs.size()) {
                // 记录反向节点序列（累积频次触发反向融合异步编译）。
                // 内部仅登记 supportsNodeType 的 element-wise 节点；多输入节点安全跳过。
                // 必须在 tryExecuteBackward 之前调用，使序列频次在 execute 前已累计。
                ct::c3::C3BackwardCapture::getInstance().recordBackwardNode(
                    typeid(*node).name(),
                    grads[0].sizes(),
                    fwd_inputs[0].sizes(),
                    fwd_inputs);
                auto c3_result = ct::c3::C3BackwardCapture::getInstance().tryExecuteBackward(
                    node.get(), grads[0], fwd_inputs);
                if (c3_result.has_value() && c3_result->size() == fwd_inputs.size()) {
                    bool shape_ok = true;
                    for (size_t i = 0; i < c3_result->size(); ++i) {
                        if ((*c3_result)[i].sizes() != fwd_inputs[i].sizes()) {
                            shape_ok = false;
                            break;
                        }
                    }
                    if (shape_ok) {
                        result.reserve(c3_result->size());
                        for (size_t i = 0; i < c3_result->size(); ++i) {
                            result.push_back(GradPack{
                                upstream[i], {(*c3_result)[i]}, static_cast<int>(i)});
                        }
#ifdef CT_DEBUG
                        std::cerr << "[DBG-C3BW-WIRE] node=" << typeid(*node).name()
                                  << " C3-HIT n_grads=" << c3_result->size() << std::endl;
                        std::cerr.flush();
#endif
                    }
#ifdef CT_DEBUG
                    else {
                        std::cerr << "[DBG-C3BW-WIRE] node=" << typeid(*node).name()
                                  << " C3-SHAPE-MISMATCH fallback eager" << std::endl;
                        std::cerr.flush();
                    }
#endif
                }
            }
        }
#endif
        if (result.empty()) {
            result = node->backward(grads);
        }

        if (!result.empty()) {
            CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding " + std::to_string(result.size()) + " grad packs");

            std::vector<std::shared_ptr<Node>> ready_nodes;
            for (const auto &pack : result) {
                if (pack._targetNode && pack._targetNode->decrease()) {
                    ready_nodes.push_back(pack._targetNode);
                }
            }

            bucket.add(std::move(result));

            for (const auto &node : ready_nodes) {
                CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding upstream node to ready queue");
                addReadyNode(node);
            }
        }

        if (!retainGraph) {
            bucket.remove(node);
        }
    }

    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Done processing ready nodes");
    
    if (retainGraph) {
        std::unordered_set<Node *> restored;
        root->restoreRecursive(restored);
    } else {
        std::unordered_set<Node *> cleared;
        root->clearRecursive(cleared);
        root.reset();
    }

    AutoGrad::EnableGrad = original_enable_grad;
}