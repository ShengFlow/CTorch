/**
 * @file AutoDiff.cpp
 * @brief Ctorch 自动微分系统实现
 * @author GhostFace
 * @date 2025/12/21
 */

#include "../include/AutoDiff.h"
#include "../include/Tensor.h"
#include <cmath>

// ======================= 计算图节点实现 =======================

/**
 * @brief Node构造函数
 * @param id 张量ID
 * @param t 张量指针
 * @param req_grad 是否需要梯度
 * @param leaf 是否为叶子节点
 */
AutoDiff::Node::Node(size_t id, std::unique_ptr<Tensor> t, bool req_grad, bool leaf)
    : tensor_id(id), tensor(std::move(t)), requires_grad(req_grad), is_leaf(leaf) {
    operation = op::Add; // 默认值
}

/**
 * @brief 安全清理梯度的方法
 * @details 清理梯度张量的存储，避免内存泄漏
 */
void AutoDiff::Node::clear_grad_safely() {
    std::ostringstream oss;
    oss << ">>> Node::clear_grad_safely - 开始, 节点ID: " << tensor_id;
    std::string msg = oss.str();
    Ctorch_Error::trace(ErrorPlatform::kCPU, msg);
    if (grad) {
        grad->clear_storage();
    }
    Ctorch_Error::trace(ErrorPlatform::kCPU, "<<< Node::clear_grad_safely - 完成");
}

// ======================= 辅助方法 =======================

/**
 * @brief 根据ID获取节点
 * @param id 节点ID
 * @return 节点指针，如果未找到返回nullptr
 */
AutoDiff::Node *AutoDiff::get_node_by_id(size_t id) {
    std::lock_guard<std::mutex> lock(records_mutex);
    auto it = id_to_node.find(id);
    if (it != id_to_node.end()) {
        return it->second.get();
    }
    Ctorch_Error::trace(ErrorPlatform::kCPU, "警告: 找不到节点 " + std::to_string(id));
    return nullptr;
}

// ======================= 调试方法 =======================

/**
 * @brief 打印自动微分系统的当前状态
 * @param context 调试上下文
 */
void AutoDiff::debug_print_state(const std::string &context) {
    std::lock_guard<std::mutex> lock(records_mutex);
    std::ostringstream oss;
    oss << "=== AutoDiff状态 [" << context << "] ===" << std::endl;
    oss << "计算图节点 (" << id_to_node.size() << "): ";
    for (const auto &pair : id_to_node) {
        if (pair.second) {
            oss << pair.first << " ";
        }
    }
    oss << std::endl;
    oss << "待处理记录 (" << pending_records.size() << "): ";
    for (const auto &pair : pending_records) {
        oss << pair.first << "(committed=" << pair.second.committed << ") ";
    }
    oss << std::endl;
    oss << "=================================" << std::endl;
    std::string msg = oss.str();
    Ctorch_Error::trace(ErrorPlatform::kCPU, msg);
}

// ======================= get_grad函数 =======================
Tensor AutoDiff::get_grad(const Tensor *t) {
    if (!t || t->id() == 0) {
        Ctorch_Error::trace(ErrorPlatform::kCPU, ">>> get_grad: 无效输入");
        return Tensor();
    }

    std::ostringstream oss;
    oss << ">>> get_grad - 开始, 目标ID: " << t->id();
    std::string msg = oss.str();
    Ctorch_Error::trace(ErrorPlatform::kCPU, msg);

    // 保存输入张量的信息，用于创建默认零张量
    std::vector<size_t> input_shape = t->shape();
    DType input_dtype               = t->dtype();
    DeviceType input_device         = t->device();

    // 第一步：在锁内获取必要信息（不创建新Tensor）
    bool has_grad = false;
    std::vector<size_t> grad_shape;
    DType grad_dtype;
    DeviceType grad_device;
    float grad_value = 0.0f; // 存储梯度值用于调试

    {
        std::lock_guard<std::mutex> lock(records_mutex);
        auto it = id_to_node.find(t->id());
        if (it != id_to_node.end() && it->second && it->second->requires_grad && it->second->grad) {
            has_grad    = true;
            grad_shape  = it->second->grad->sizes();
            grad_dtype  = it->second->grad->dtype();
            grad_device = it->second->grad->device();

            // 获取梯度值用于调试
            if (grad_dtype == DType::kFloat) {
                const float *grad_data = it->second->grad->data<float>();
                if (grad_data) {
                    grad_value = grad_data[0]; // 对于标量，取第一个元素
                }
            }
            std::ostringstream osss;
            osss << ">>> 找到梯度，形状: [";
            for (auto s : grad_shape)
                osss << s << " ";
            osss << "], 值: " << grad_value;
            std::string msgs = osss.str();
            Ctorch_Error::trace(ErrorPlatform::kCPU, msgs);
        }
    } // 释放锁

    // 第二步：在锁外创建结果Tensor
    if (has_grad) {
        Ctorch_Error::trace(ErrorPlatform::kCPU, ">>> 创建梯度副本");
        Tensor result(ShapeTag{}, grad_shape, grad_dtype, grad_device);

        // 第三步：重新加锁复制数据
        {
            std::lock_guard<std::mutex> lock(records_mutex);
            auto it = id_to_node.find(t->id());
            if (it != id_to_node.end() && it->second && it->second->grad) {
                Ctorch_Error::trace(ErrorPlatform::kCPU, ">>> 复制梯度数据");
                // 直接复制数据，避免调用clone()
                switch (grad_dtype) {
                case DType::kFloat: {
                    const float *src = it->second->grad->data<float>();
                    float *dst       = result.data<float>();
                    size_t count     = result.numel();
                    for (size_t i = 0; i < count; ++i) {
                        dst[i] = src[i];
                    }
                    std::ostringstream osss;
                    osss << ">>> 复制了 " << count << " 个float值，第一个值: " << dst[0];
                    std::string msgs = osss.str();
                    Ctorch_Error::trace(ErrorPlatform::kCPU, msgs);
                    break;
                }
                case DType::kDouble: {
                    const double *src = it->second->grad->data<double>();
                    double *dst       = result.data<double>();
                    for (size_t i = 0; i < result.numel(); ++i) {
                        dst[i] = src[i];
                    }
                    break;
                }
                case DType::kInt: {
                    const int32_t *src = it->second->grad->data<int32_t>();
                    int32_t *dst       = result.data<int32_t>();
                    for (size_t i = 0; i < result.numel(); ++i) {
                        dst[i] = src[i];
                    }
                    break;
                }
                case DType::kLong: {
                    const int64_t *src = it->second->grad->data<int64_t>();
                    int64_t *dst       = result.data<int64_t>();
                    for (size_t i = 0; i < result.numel(); ++i) {
                        dst[i] = src[i];
                    }
                    break;
                }
                default:
                    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                                 "get_grad中不支持的dtype");
                }
            }
        }
        std::ostringstream osss;
        osss << ">>> 最终梯度副本形状: ";
        for (size_t s : result.sizes())
            osss << s << " ";
        osss << std::endl;
        osss << "<<< get_grad - 完成" << std::endl;

        std::string msgs = osss.str();
        Ctorch_Error::trace(ErrorPlatform::kCPU, msgs);
        return result;
    }
    // 未找到梯度时，输出Error级别提示并返回零张量
    Ctorch_Error::trace(ErrorPlatform::kCPU,
                        "[ERROR] [2025-12-21 18:05:10.882 1766311510882] [ERROR_CODE:0x5060400] "
                        "[PLATFORM:GENERAL] [TYPE:TENSOR_STATE] Tensor梯度未找到，返回零张量");
    // 创建与输入张量相同形状、数据类型和设备的零张量
    Tensor result(ShapeTag{}, input_shape, input_dtype, input_device);
    // 初始化所有元素为0
    switch (input_dtype) {
    case DType::kFloat: {
        float *dst   = result.data<float>();
        size_t count = result.numel();
        for (size_t i = 0; i < count; ++i) {
            dst[i] = 0.0f;
        }
        break;
    }
    case DType::kDouble: {
        double *dst  = result.data<double>();
        size_t count = result.numel();
        for (size_t i = 0; i < count; ++i) {
            dst[i] = 0.0;
        }
        break;
    }
    case DType::kInt: {
        int32_t *dst = result.data<int32_t>();
        size_t count = result.numel();
        for (size_t i = 0; i < count; ++i) {
            dst[i] = 0;
        }
        break;
    }
    case DType::kLong: {
        int64_t *dst = result.data<int64_t>();
        size_t count = result.numel();
        for (size_t i = 0; i < count; ++i) {
            dst[i] = 0;
        }
        break;
    }
    default:
        Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                     "get_grad中不支持的dtype");
    }
    return result;
}

// ======================= make_leaf函数 =======================
void AutoDiff::make_leaf(Tensor &t, bool requires_grad) {
    size_t id = t.id();
    if (id == 0) {
        Ctorch_Error::trace(ErrorPlatform::kCPU, "!!! 错误: 尝试注册ID为0的张量");
        return;
    }

    std::ostringstream osss;
    osss << ">>> AutoDiff::make_leaf - 开始, ID: " << id;

    std::string msgs = osss.str();
    Ctorch_Error::trace(ErrorPlatform::kCPU, msgs);
    debug_print_state("make_leaf开始前");
    {
        std::lock_guard<std::mutex> lock(records_mutex);
        if (id_to_node.find(id) != id_to_node.end()) {
            std::ostringstream oss;
            oss << ">>> 节点 " << id << " 已存在，更新requires_grad状态" << std::endl;
            std::string msg = oss.str();
            Ctorch_Error::trace(ErrorPlatform::kCPU, msg);
            // 更新节点的requires_grad状态
            id_to_node[id]->requires_grad = requires_grad;
            if (requires_grad && !id_to_node[id]->grad) {
                id_to_node[id]->grad =
                    std::make_unique<Tensor>(ShapeTag{}, t.sizes(), t.dtype(), t.device());
                id_to_node[id]->grad->zero();
            }
            return;
        }
    }

    // 直接使用原始张量的引用，而不是创建一个新的张量对象
    // 这样可以确保节点引用的是原始张量，而不是一个新创建的张量
    // 直接使用原始张量的引用，而不是创建一个新的张量对象
    // 这样可以确保节点引用的是原始张量，而不是一个新创建的张量
    // 创建一个指向原始张量的指针，而不是复制它
    auto tensor_ptr = std::make_unique<Tensor>(t);
    // 手动设置张量ID为原始张量的ID，以保持ID一致
    tensor_ptr->set_id(id);
    // 创建节点，使用新创建的张量对象
    auto node = std::make_unique<Node>(id, std::move(tensor_ptr), requires_grad, true);

    {
        std::lock_guard<std::mutex> lock(records_mutex);
        id_to_node[id] = std::move(node);
    }

    std::ostringstream oss;
    oss << "<<< AutoDiff::make_leaf - 完成, ID: " << id;
    std::string msg = oss.str();
    Ctorch_Error::trace(ErrorPlatform::kCPU, msg);
    debug_print_state("make_leaf完成后");
}

// ======================= 析构函数 =======================
AutoDiff::~AutoDiff() {
    Ctorch_Error::trace(ErrorPlatform::kCPU, ">>> AutoDiff 析构");
    // 避免在析构时进行复杂操作
    // 直接清空，不持有锁
    id_to_node.clear();
    pending_records.clear();
}

// ======================= defer_record函数 =======================
void AutoDiff::defer_record(size_t output_id, op operation, const std::vector<Tensor *> &inputs, int op_param_i) {
    std::ostringstream oss;
    oss << ">>> 进入 defer_record, output_id: " << output_id << ", 操作类型: " << static_cast<int>(operation);
    std::string msg = oss.str();
    Ctorch_Error::trace(ErrorPlatform::kCPU, msg);
    if (output_id == 0) {
        Ctorch_Error::log(ErrorLevel::WARN, ErrorPlatform::kCPU, ErrorType::TENSOR_STATE,
                          "警告: output_id 为0");
        return;
    }

    // 创建记录对象（不持有锁）
    PendingRecord record;
    record.operation = operation;
    record.op_param_i = op_param_i;

    std::vector<size_t> input_ids;
    std::vector<std::vector<size_t>> input_shapes;

    // 收集输入信息（不持有锁）
    for (Tensor *input : inputs) {
        if (input && input->id() != 0) {
            input_ids.push_back(input->id());
            input_shapes.push_back(input->shape());
            std::ostringstream osss;
            osss << ">>> 处理输入: " << input->id();
            std::string msgs = osss.str();
            Ctorch_Error::trace(ErrorPlatform::kCPU, msgs);

            // 检查是否已注册，避免不必要的锁
            bool needs_registration = false;
            {
                std::lock_guard<std::mutex> lock(records_mutex);
                needs_registration = (id_to_node.find(input->id()) == id_to_node.end());
            }

            if (needs_registration) {
                std::ostringstream ost;
                ost << ">>> 注册叶子节点: " << input->id() << std::endl;
                std::string msgt = ost.str();
                Ctorch_Error::trace(ErrorPlatform::kCPU, msgt);

                make_leaf(*input, input->requires_grad());
            }
        }
    }

    // 现在持有锁并设置记录
    {
        std::lock_guard<std::mutex> lock(records_mutex);
        record.input_ids           = input_ids;
        record.input_shapes        = input_shapes;
        pending_records[output_id] = record;
    }

    Ctorch_Error::trace(ErrorPlatform::kCPU, "<<< 离开 defer_record");
}

// ======================= commit_record函数 =======================
void AutoDiff::commit_record(Tensor &output) {
    size_t output_id = output.id();
    Ctorch_Error::trace(ErrorPlatform::kCPU,
                        "AutoDiff::commit_record - 开始, ID: " + std::to_string(output_id));

    if (output_id == 0) {
        Ctorch_Error::trace(ErrorPlatform::kCPU, "错误: 输出ID为0");
        return;
    }

    debug_print_state("commit_record开始前");

    // 第一步：收集必要信息（持有锁，但时间很短）
    PendingRecord record_copy;
    std::vector<size_t> input_ids_copy;
    bool should_create_node = false;

    {
        std::lock_guard<std::mutex> lock(records_mutex);

        auto it = pending_records.find(output_id);
        if (it == pending_records.end()) {
            Ctorch_Error::trace(ErrorPlatform::kCPU,
                                "警告: 找不到待处理记录 " + std::to_string(output_id));
            return;
        }

        PendingRecord &record = it->second;
        if (record.committed) {
            Ctorch_Error::trace(ErrorPlatform::kCPU,
                                "记录 " + std::to_string(output_id) + " 已提交，跳过");
            return;
        }

        Ctorch_Error::trace(ErrorPlatform::kCPU, "开始提交记录 " + std::to_string(output_id));
        Ctorch_Error::trace(ErrorPlatform::kCPU,
                            "操作类型: " + std::to_string(static_cast<int>(record.operation)));
        std::ostringstream oss;
        oss << "输入IDs: ";
        for (auto id : record.input_ids)
            oss << id << " ";
        Ctorch_Error::trace(ErrorPlatform::kCPU, oss.str());

        // 验证输入节点
        bool valid = true;
        for (size_t input_id : record.input_ids) {
            if (id_to_node.find(input_id) == id_to_node.end()) {
                Ctorch_Error::trace(ErrorPlatform::kCPU,
                                    "错误: 输入节点 " + std::to_string(input_id) + " 不存在");
                valid = false;
                break;
            } else {
                Ctorch_Error::trace(ErrorPlatform::kCPU,
                                    "输入节点 " + std::to_string(input_id) + " 存在");
            }
        }

        if (!valid) {
            Ctorch_Error::trace(ErrorPlatform::kCPU,
                                "记录验证失败，清除记录 " + std::to_string(output_id));
            pending_records.erase(it);
            return;
        }

        // 复制必要信息，然后释放锁
        record_copy        = record;
        input_ids_copy     = record.input_ids;
        should_create_node = true;

        // 立即标记为已提交，避免重复处理
        record.committed = true;
    } // 释放锁

    // 第二步：在锁外创建节点（避免死锁）
    if (should_create_node) {
        Ctorch_Error::trace(ErrorPlatform::kCPU, "在锁外创建操作节点 " + std::to_string(output_id));

        // 确定梯度需求（在锁外检查）
        bool requires_grad = false;
        for (size_t input_id : input_ids_copy) {
            Node *input_node = get_node_by_id(input_id);
            if (input_node && input_node->requires_grad) {
                requires_grad = true;
                break;
            }
        }

        Ctorch_Error::trace(ErrorPlatform::kCPU, "节点 " + std::to_string(output_id) +
                                                     " 需要梯度: " + (requires_grad ? "1" : "0"));

        // 创建节点（不持有锁）
        // 使用原始张量的拷贝，确保ID一致
        auto tensor_ptr = std::make_unique<Tensor>(output);
        auto output_node = 
            std::make_unique<Node>(output_id, std::move(tensor_ptr), requires_grad, false);
        // 确保操作类型被正确设置
        output_node->operation = record_copy.operation;
        output_node->op_param_i = record_copy.op_param_i;
        // 添加调试信息
        std::ostringstream oss;
        oss << "设置节点 " << output_id << " 的操作类型为: " << static_cast<int>(record_copy.operation);
        Ctorch_Error::trace(ErrorPlatform::kCPU, oss.str());
        output_node->input_ids = input_ids_copy;

        // 延迟创建梯度存储
        if (requires_grad) {
            Ctorch_Error::trace(ErrorPlatform::kCPU,
                                "为节点 " + std::to_string(output_id) + " 延迟分配梯度存储");
            // 注意：不在构造函数中创建 grad，避免死锁
        }

        // 第三步：重新加锁，快速插入节点
        {
            std::lock_guard<std::mutex> lock(records_mutex);
            id_to_node[output_id] = std::move(output_node);

            // 清理已提交的记录
            auto it = pending_records.find(output_id);
            if (it != pending_records.end() && it->second.committed) {
                pending_records.erase(it);
            }
        }

        Ctorch_Error::trace(ErrorPlatform::kCPU, "记录提交完成 " + std::to_string(output_id));
        debug_print_state("commit_record完成后");
    }
}

// ======================= zero_grad函数 =======================
void AutoDiff::zero_grad(Tensor& t) {
    if (t.id() == 0)
        return;
    std::lock_guard<std::mutex> lock(records_mutex);
    auto it = id_to_node.find(t.id());
    if (it != id_to_node.end() && it->second && it->second->grad) {
        it->second->grad->zero();
    }
}

// ======================= update_requires_grad函数 =======================
void AutoDiff::update_requires_grad(Tensor &t, bool requires_grad) {
    size_t id = t.id();
    if (id == 0)
        return;

    Ctorch_Error::trace(ErrorPlatform::kCPU, "更新梯度需求: " + std::to_string(id) + " -> " +
                                                 (requires_grad ? "1" : "0"));

    // 使用更短的锁作用域
    {
        std::lock_guard<std::mutex> lock(records_mutex);
        auto it = id_to_node.find(id);
        if (it != id_to_node.end() && it->second) {
            it->second->requires_grad = requires_grad;
            if (requires_grad) {
                if (!it->second->grad) {
                    it->second->grad =
                        std::make_unique<Tensor>(ShapeTag{}, t.sizes(), t.dtype(), t.device());
                    it->second->grad->zero();
                }
            }
        } else {
            // 如果节点不存在，可能需要创建它
            Ctorch_Error::trace(ErrorPlatform::kCPU,
                                "节点不存在，可能需要创建: " + std::to_string(id));
        }
    }

    Ctorch_Error::trace(ErrorPlatform::kCPU, "更新梯度需求完成");
}

// ======================= set_retain_graph函数 =======================
void AutoDiff::set_retain_graph(bool retain) { retain_graph = retain; }

// ======================= backward函数（版本1） =======================
void AutoDiff::backward(Tensor &root) {
    // 创建值为1的Tensor，但不使用AutoDiff上下文
    AutoDiffContext::Guard guard(nullptr); // 临时禁用AutoDiff上下文
    Tensor grad_output(1.0f);
    backward(root, grad_output);
}

// ======================= 私有辅助函数：检查和调整梯度形状 =======================
/**
 * @brief 检查并调整梯度张量形状，确保与目标形状匹配
 * @param grad 输入梯度张量
 * @param target_shape 目标形状
 * @return 调整后的梯度张量
 */
Tensor AutoDiff::check_and_adjust_grad_shape(const Tensor &grad,
                                             const std::vector<size_t> &target_shape) {
    // 如果形状已经匹配，直接返回
    if (grad.sizes() == target_shape) {
        return grad;
    }

    std::ostringstream oss;
    oss << "调整梯度形状: ";
    for (size_t s : grad.sizes())
        oss << s << " ";
    oss << "-> ";
    for (size_t s : target_shape)
        oss << s << " ";
    Ctorch_Error::trace(ErrorPlatform::kCPU, oss.str());

    // 处理标量情况
    if (grad.sizes().empty()) {
        // 标量梯度扩展到目标形状
        Tensor result(ShapeTag{}, target_shape, grad.dtype(), grad.device());

        // 根据数据类型处理
        switch (grad.dtype()) {
        case DType::kFloat: {
            float val    = grad.item<float>();
            float *data  = result.data<float>();
            size_t numel = result.numel();
            for (size_t i = 0; i < numel; ++i) {
                data[i] = val;
            }
            break;
        }
        case DType::kDouble: {
            double val   = grad.item<double>();
            double *data = result.data<double>();
            size_t numel = result.numel();
            for (size_t i = 0; i < numel; ++i) {
                data[i] = val;
            }
            break;
        }
        case DType::kInt: {
            int32_t val   = grad.item<int32_t>();
            int32_t *data = result.data<int32_t>();
            size_t numel  = result.numel();
            for (size_t i = 0; i < numel; ++i) {
                data[i] = val;
            }
            break;
        }
        case DType::kLong: {
            int64_t val   = grad.item<int64_t>();
            int64_t *data = result.data<int64_t>();
            size_t numel  = result.numel();
            for (size_t i = 0; i < numel; ++i) {
                data[i] = val;
            }
            break;
        }
        default:
            Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                         "check_and_adjust_grad_shape中不支持的dtype");
        }

        return result;
    }

    size_t grad_rank   = grad.sizes().size();
    size_t target_rank = target_shape.size();

    // 情况1：grad的秩小于target_shape的秩，需要在前面添加维度
    if (grad_rank < target_rank) {
        // 创建新的梯度张量，在前面添加维度
        std::vector<size_t> new_grad_shape(target_rank, 1);
        for (size_t i = 0; i < grad_rank; ++i) {
            new_grad_shape[target_rank - grad_rank + i] = grad.sizes()[i];
        }

        // 创建临时张量，形状为new_grad_shape
        Tensor temp_grad(ShapeTag{}, new_grad_shape, grad.dtype(), grad.device());

        // 根据数据类型复制数据
        size_t grad_numel = grad.numel();
        size_t temp_numel = temp_grad.numel();

        switch (grad.dtype()) {
        case DType::kFloat: {
            float *temp_data       = temp_grad.data<float>();
            const float *grad_data = grad.data<float>();
            for (size_t i = 0; i < temp_numel; ++i) {
                temp_data[i] = grad_data[i % grad_numel];
            }
            break;
        }
        case DType::kDouble: {
            double *temp_data       = temp_grad.data<double>();
            const double *grad_data = grad.data<double>();
            for (size_t i = 0; i < temp_numel; ++i) {
                temp_data[i] = grad_data[i % grad_numel];
            }
            break;
        }
        case DType::kInt: {
            int32_t *temp_data       = temp_grad.data<int32_t>();
            const int32_t *grad_data = grad.data<int32_t>();
            for (size_t i = 0; i < temp_numel; ++i) {
                temp_data[i] = grad_data[i % grad_numel];
            }
            break;
        }
        case DType::kLong: {
            int64_t *temp_data       = temp_grad.data<int64_t>();
            const int64_t *grad_data = grad.data<int64_t>();
            for (size_t i = 0; i < temp_numel; ++i) {
                temp_data[i] = grad_data[i % grad_numel];
            }
            break;
        }
        default:
            Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                         "check_and_adjust_grad_shape中不支持的dtype");
        }

        // 递归调用，处理新的形状
        return check_and_adjust_grad_shape(temp_grad, target_shape);
    }

    // 情况2：grad的秩大于等于target_shape的秩
    bool can_broadcast = true;

    // 从末尾开始比较形状
    for (size_t i = 1; i <= target_rank; ++i) {
        size_t grad_idx   = grad_rank - i;
        size_t target_idx = target_rank - i;

        size_t grad_dim   = grad.sizes()[grad_idx];
        size_t target_dim = target_shape[target_idx];

        if (grad_dim != target_dim && grad_dim != 1 && target_dim != 1) {
            can_broadcast = false;
            break;
        }
    }

    if (can_broadcast) {
        // 创建结果张量
        Tensor result(ShapeTag{}, target_shape, grad.dtype(), grad.device());
        result.zero();

        // 实现广播逻辑：对于每个目标元素，找到对应的grad元素并累加
        size_t result_numel = result.numel();

        // 计算每个维度的步幅
        std::vector<size_t> grad_strides   = grad.strides();
        std::vector<size_t> result_strides = result.strides();

        // 根据数据类型处理
        switch (grad.dtype()) {
        case DType::kFloat: {
            float *result_data     = result.data<float>();
            const float *grad_data = grad.data<float>();

            // 遍历结果张量的每个元素
            for (size_t i = 0; i < result_numel; ++i) {
                // 计算结果张量在各维度的索引
                std::vector<size_t> result_indices(target_rank, 0);
                size_t temp = i;
                for (int j = target_rank - 1; j >= 0; --j) {
                    result_indices[j] = temp / result_strides[j];
                    temp %= result_strides[j];
                }

                // 计算对应的grad索引
                std::vector<size_t> grad_indices(grad_rank, 0);

                // 从末尾开始匹配索引
                for (size_t j = 1; j <= target_rank; ++j) {
                    size_t grad_idx   = grad_rank - j;
                    size_t result_idx = target_rank - j;

                    // 如果grad维度是1，则使用0索引
                    if (grad.sizes()[grad_idx] == 1) {
                        grad_indices[grad_idx] = 0;
                    } else {
                        grad_indices[grad_idx] = result_indices[result_idx];
                    }
                }

                // 计算grad的线性索引
                size_t grad_index = 0;
                for (size_t j = 0; j < grad_rank; ++j) {
                    grad_index += grad_indices[j] * grad_strides[j];
                }

                // 累加梯度
                result_data[i] += grad_data[grad_index];
            }
            break;
        }
        case DType::kDouble: {
            double *result_data     = result.data<double>();
            const double *grad_data = grad.data<double>();

            // 遍历结果张量的每个元素
            for (size_t i = 0; i < result_numel; ++i) {
                // 计算结果张量在各维度的索引
                std::vector<size_t> result_indices(target_rank, 0);
                size_t temp = i;
                for (int j = target_rank - 1; j >= 0; --j) {
                    result_indices[j] = temp / result_strides[j];
                    temp %= result_strides[j];
                }

                // 计算对应的grad索引
                std::vector<size_t> grad_indices(grad_rank, 0);

                // 从末尾开始匹配索引
                for (size_t j = 1; j <= target_rank; ++j) {
                    size_t grad_idx   = grad_rank - j;
                    size_t result_idx = target_rank - j;

                    // 如果grad维度是1，则使用0索引
                    if (grad.sizes()[grad_idx] == 1) {
                        grad_indices[grad_idx] = 0;
                    } else {
                        grad_indices[grad_idx] = result_indices[result_idx];
                    }
                }

                // 计算grad的线性索引
                size_t grad_index = 0;
                for (size_t j = 0; j < grad_rank; ++j) {
                    grad_index += grad_indices[j] * grad_strides[j];
                }

                // 累加梯度
                result_data[i] += grad_data[grad_index];
            }
            break;
        }
        case DType::kInt: {
            int32_t *result_data     = result.data<int32_t>();
            const int32_t *grad_data = grad.data<int32_t>();

            // 遍历结果张量的每个元素
            for (size_t i = 0; i < result_numel; ++i) {
                // 计算结果张量在各维度的索引
                std::vector<size_t> result_indices(target_rank, 0);
                size_t temp = i;
                for (int j = target_rank - 1; j >= 0; --j) {
                    result_indices[j] = temp / result_strides[j];
                    temp %= result_strides[j];
                }

                // 计算对应的grad索引
                std::vector<size_t> grad_indices(grad_rank, 0);

                // 从末尾开始匹配索引
                for (size_t j = 1; j <= target_rank; ++j) {
                    size_t grad_idx   = grad_rank - j;
                    size_t result_idx = target_rank - j;

                    // 如果grad维度是1，则使用0索引
                    if (grad.sizes()[grad_idx] == 1) {
                        grad_indices[grad_idx] = 0;
                    } else {
                        grad_indices[grad_idx] = result_indices[result_idx];
                    }
                }

                // 计算grad的线性索引
                size_t grad_index = 0;
                for (size_t j = 0; j < grad_rank; ++j) {
                    grad_index += grad_indices[j] * grad_strides[j];
                }

                // 累加梯度
                result_data[i] += grad_data[grad_index];
            }
            break;
        }
        case DType::kLong: {
            int64_t *result_data     = result.data<int64_t>();
            const int64_t *grad_data = grad.data<int64_t>();

            // 遍历结果张量的每个元素
            for (size_t i = 0; i < result_numel; ++i) {
                // 计算结果张量在各维度的索引
                std::vector<size_t> result_indices(target_rank, 0);
                size_t temp = i;
                for (int j = target_rank - 1; j >= 0; --j) {
                    result_indices[j] = temp / result_strides[j];
                    temp %= result_strides[j];
                }

                // 计算对应的grad索引
                std::vector<size_t> grad_indices(grad_rank, 0);

                // 从末尾开始匹配索引
                for (size_t j = 1; j <= target_rank; ++j) {
                    size_t grad_idx   = grad_rank - j;
                    size_t result_idx = target_rank - j;

                    // 如果grad维度是1，则使用0索引
                    if (grad.sizes()[grad_idx] == 1) {
                        grad_indices[grad_idx] = 0;
                    } else {
                        grad_indices[grad_idx] = result_indices[result_idx];
                    }
                }

                // 计算grad的线性索引
                size_t grad_index = 0;
                for (size_t j = 0; j < grad_rank; ++j) {
                    grad_index += grad_indices[j] * grad_strides[j];
                }

                // 累加梯度
                result_data[i] += grad_data[grad_index];
            }
            break;
        }
        default:
            Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                         "check_and_adjust_grad_shape中不支持的dtype");
        }

        return result;
    }

    // 情况3：处理广播后的梯度求和
    // 例如：target_shape = [3], grad_shape = [2, 3]，则对grad的第一个维度求和
    bool can_reduce = true;

    // 检查是否可以通过求和得到目标形状
    for (size_t i = 1; i <= std::min(grad_rank, target_rank); ++i) {
        size_t grad_idx   = grad_rank - i;
        size_t target_idx = target_rank - i;

        size_t grad_dim   = grad.sizes()[grad_idx];
        size_t target_dim = target_shape[target_idx];

        if (grad_dim != target_dim && target_dim != 1) {
            can_reduce = false;
            break;
        }
    }

    if (can_reduce) {
        // 创建结果张量
        Tensor result(ShapeTag{}, target_shape, grad.dtype(), grad.device());
        result.zero();

        size_t result_numel = result.numel();
        size_t grad_numel   = grad.numel();

        // 根据数据类型处理
        switch (grad.dtype()) {
        case DType::kFloat: {
            float *result_data     = result.data<float>();
            const float *grad_data = grad.data<float>();

            // 简化处理：直接对所有元素求和
            for (size_t i = 0; i < grad_numel; ++i) {
                result_data[i % result_numel] += grad_data[i];
            }
            break;
        }
        case DType::kDouble: {
            double *result_data     = result.data<double>();
            const double *grad_data = grad.data<double>();

            // 简化处理：直接对所有元素求和
            for (size_t i = 0; i < grad_numel; ++i) {
                result_data[i % result_numel] += grad_data[i];
            }
            break;
        }
        case DType::kInt: {
            int32_t *result_data     = result.data<int32_t>();
            const int32_t *grad_data = grad.data<int32_t>();

            // 简化处理：直接对所有元素求和
            for (size_t i = 0; i < grad_numel; ++i) {
                result_data[i % result_numel] += grad_data[i];
            }
            break;
        }
        case DType::kLong: {
            int64_t *result_data     = result.data<int64_t>();
            const int64_t *grad_data = grad.data<int64_t>();

            // 简化处理：直接对所有元素求和
            for (size_t i = 0; i < grad_numel; ++i) {
                result_data[i % result_numel] += grad_data[i];
            }
            break;
        }
        default:
            Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                         "check_and_adjust_grad_shape中不支持的dtype");
        }

        return result;
    }

    // 情况4：特殊处理梯度形状
    // 例如：当grad的形状与预期不符时，尝试通过求和调整
    // 对于所有操作，尝试更灵活的形状调整
    Tensor result(ShapeTag{}, target_shape, grad.dtype(), grad.device());
    result.zero();

    size_t result_numel = result.numel();
    size_t grad_numel   = grad.numel();

    // 根据数据类型处理
    switch (grad.dtype()) {
    case DType::kFloat: {
        float *result_data     = result.data<float>();
        const float *grad_data = grad.data<float>();

        // 计算平均梯度
        for (size_t i = 0; i < result_numel; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < grad_numel; ++j) {
                if (j % result_numel == i) {
                    sum += grad_data[j];
                }
            }
            result_data[i] = sum;
        }
        break;
    }
    case DType::kDouble: {
        double *result_data     = result.data<double>();
        const double *grad_data = grad.data<double>();

        // 计算平均梯度
        for (size_t i = 0; i < result_numel; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < grad_numel; ++j) {
                if (j % result_numel == i) {
                    sum += grad_data[j];
                }
            }
            result_data[i] = sum;
        }
        break;
    }
    case DType::kInt: {
        int32_t *result_data     = result.data<int32_t>();
        const int32_t *grad_data = grad.data<int32_t>();

        // 计算平均梯度
        for (size_t i = 0; i < result_numel; ++i) {
            int32_t sum = 0;
            for (size_t j = 0; j < grad_numel; ++j) {
                if (j % result_numel == i) {
                    sum += grad_data[j];
                }
            }
            result_data[i] = sum;
        }
        break;
    }
    case DType::kLong: {
        int64_t *result_data     = result.data<int64_t>();
        const int64_t *grad_data = grad.data<int64_t>();

        // 计算平均梯度
        for (size_t i = 0; i < result_numel; ++i) {
            int64_t sum = 0;
            for (size_t j = 0; j < grad_numel; ++j) {
                if (j % result_numel == i) {
                    sum += grad_data[j];
                }
            }
            result_data[i] = sum;
        }
        break;
    }
    default:
        Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                     "check_and_adjust_grad_shape中不支持的dtype");
    }

    return result;
}

// ======================= 私有辅助函数：DFS拓扑排序 =======================
/**
 * @brief 执行DFS后序遍历，生成拓扑序列
 * @param node 当前节点
 * @param visited 已访问节点集合
 * @param result 拓扑序列结果
 */
void AutoDiff::dfs_topological_sort(Node *node, std::unordered_set<Node *> &visited,
                                    std::vector<Node *> &result) {
    if (!node)
        return;
    if (visited.find(node) != visited.end())
        return;

    visited.insert(node);

    // 先访问所有输入节点（依赖节点）
    for (size_t input_id : node->input_ids) {
        auto input_it = id_to_node.find(input_id);
        if (input_it != id_to_node.end() && input_it->second) {
            dfs_topological_sort(input_it->second.get(), visited, result);
        }
    }

    // 后序遍历：当前节点处理完所有依赖后，加入结果
    result.push_back(node);
}

// ======================= backward函数（版本2） =======================
void AutoDiff::backward(Tensor &root, Tensor grad_output) {
    Ctorch_Error::trace(ErrorPlatform::kAutoDiff,
                        std::string("======================================="));
    Ctorch_Error::trace(ErrorPlatform::kAutoDiff,
                        std::string("进入 backward 函数，root ID: ") + std::to_string(root.id()));

    if (root.id() == 0) {
        Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::TENSOR_STATE,
                                     "反向传播的无效根张量 (ID为0)");
        return;
    }

    // 阶段1：准备阶段（不持有锁）
    Ctorch_Error::trace(ErrorPlatform::kAutoDiff, std::string("阶段1: 准备阶段"));

    // 阶段2：验证阶段和设置根节点梯度
    Ctorch_Error::trace(ErrorPlatform::kAutoDiff, std::string("阶段2: 验证和设置根节点梯度"));
    Node *root_node         = nullptr;
    bool root_requires_grad = false;
    std::vector<size_t> root_shape;

    {
        std::lock_guard<std::mutex> lock(records_mutex);
        auto it = id_to_node.find(root.id());
        if (it == id_to_node.end() || !it->second) {
            Ctorch_Error::trace(ErrorPlatform::kCPU,
                                "错误: 找不到根节点 " + std::to_string(root.id()));
            return;
        }
        root_node          = it->second.get();
        root_requires_grad = root_node->requires_grad;
        root_shape         = root.sizes();

        if (!root_requires_grad) {
            Ctorch_Error::trace(ErrorPlatform::kCPU, "根节点不需要梯度，跳过反向传播");
            return;
        }

        // 设置根节点梯度，确保形状匹配
        if (!root_node->grad) {
            root_node->grad =
                std::make_unique<Tensor>(ShapeTag{}, root_shape, root.dtype(), root.device());
        }

        // 检查并调整梯度输出形状，确保与根节点形状匹配
        Tensor adjusted_grad_output = check_and_adjust_grad_shape(grad_output, root_shape);
        *root_node->grad            = adjusted_grad_output;
    }

    Ctorch_Error::trace(ErrorPlatform::kCPU, "根节点需要梯度，开始反向传播");
    Ctorch_Error::trace(ErrorPlatform::kCPU,
                        "根节点形状: " + std::to_string(root_shape.size()) + "D");
    Ctorch_Error::trace(ErrorPlatform::kCPU,
                        "梯度输出形状: " + std::to_string(grad_output.sizes().size()) + "D");

    // 阶段3：反向传播（实现实际的梯度计算）
    Ctorch_Error::trace(ErrorPlatform::kAutoDiff, std::string("阶段3: 反向传播计算"));

    std::vector<Node *> topological_order;
    {
        std::lock_guard<std::mutex> lock(records_mutex);

        // 生成拓扑序列
        std::unordered_set<Node *> visited;
        std::vector<Node *> forward_order;
        dfs_topological_sort(root_node, visited, forward_order);

        // 反转得到逆拓扑序列（反向传播顺序）
        topological_order = forward_order;
        std::reverse(topological_order.begin(), topological_order.end());

        Ctorch_Error::trace(ErrorPlatform::kCPU,
                            "拓扑排序完成，节点数量: " + std::to_string(topological_order.size()));
    }

    // 处理每个节点，传播梯度到输入节点
    {
        std::lock_guard<std::mutex> lock(records_mutex);
        for (Node *node : topological_order) {
            // 打印节点信息
            Ctorch_Error::trace(ErrorPlatform::kCPU,
                                "处理节点: " + std::to_string(node->tensor_id) +
                                    ", requires_grad: " + (node->requires_grad ? "1" : "0"));
            Ctorch_Error::trace(ErrorPlatform::kCPU,
                                "操作类型: " + std::to_string(static_cast<int>(node->operation)));
            if (node->grad) {
                Ctorch_Error::trace(ErrorPlatform::kCPU,
                                    "梯度形状: " + std::to_string(node->grad->sizes().size()) +
                                        "D");
                if (!node->grad->sizes().empty()) {
                    std::string shape_str = "";
                    for (size_t s : node->grad->sizes()) {
                        shape_str += std::to_string(s) + " ";
                    }
                    Ctorch_Error::trace(ErrorPlatform::kCPU, "梯度维度: " + shape_str);
                }
            }

            // 跳过不需要梯度的节点
            if (!node->requires_grad) {
                Ctorch_Error::trace(ErrorPlatform::kCPU, "跳过不需要梯度的节点");
                continue;
            }

            // 计算梯度并传播到输入节点
            if (node->grad) {
                // 对于张量梯度，直接使用梯度张量进行计算
                Tensor &node_grad = *node->grad;

                for (size_t input_id : node->input_ids) {
                    auto input_it = id_to_node.find(input_id);
                    if (input_it != id_to_node.end() && input_it->second &&
                        input_it->second->requires_grad) {
                        auto &input_node = *input_it->second;

                        // 打印输入节点信息
                        Ctorch_Error::trace(
                            ErrorPlatform::kCPU,
                            "处理输入节点: " + std::to_string(input_id) +
                                ", requires_grad: " + (input_node.requires_grad ? "1" : "0"));

                        // 初始化输入节点的梯度，确保形状匹配
                        if (!input_node.grad) {
                            input_node.grad = std::make_unique<Tensor>(
                                ShapeTag{}, input_node.tensor->sizes(), input_node.tensor->dtype(),
                                input_node.tensor->device());
                            input_node.grad->zero();
                            Ctorch_Error::trace(ErrorPlatform::kCPU, "初始化输入节点梯度");
                        }

                        Tensor &input_grad = *input_node.grad;

                        // 根据操作类型计算梯度
                        Tensor input_grad_val;

                        // 处理不同操作类型
                        if (node->operation == op::Add) {
                            // ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
                            // 对于张量加法，梯度直接传播
                            // 但需要确保梯度形状与输入张量形状匹配
                            input_grad_val =
                                check_and_adjust_grad_shape(node_grad, input_node.tensor->sizes());
                        } else if (node->operation == op::Sub) {
                            // 第一个输入是a，第二个是b，∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
                            if (input_id == node->input_ids[0]) {
                                // 对第一个输入的梯度是正的
                                input_grad_val = check_and_adjust_grad_shape(
                                    node_grad, input_node.tensor->sizes());
                            } else {
                                // 对第二个输入的梯度是负的
                                Tensor adjusted_grad = check_and_adjust_grad_shape(
                                    node_grad, input_node.tensor->sizes());
                                input_grad_val = -adjusted_grad;
                            }
                        } else if (node->operation == op::Mul) {
                            // 找到另一个输入张量
                            if (node->input_ids.size() > 1) {
                                size_t other_input_id = (input_id == node->input_ids[0])
                                                            ? node->input_ids[1]
                                                            : node->input_ids[0];
                                auto other_it         = id_to_node.find(other_input_id);
                                if (other_it != id_to_node.end() && other_it->second) {
                                    // 对于张量乘法，梯度是另一个输入张量与当前节点梯度的乘积
                                    Tensor &other_tensor = *other_it->second->tensor;
                                    // 检查是否是同一个张量相乘
                                    if (input_id == other_input_id) {
                                        // 对于同一个张量相乘，梯度是 2 * other_tensor * node_grad
                                        // 但由于会遍历两次输入节点，这里只计算一次
                                        input_grad_val = other_tensor * node_grad;
                                    } else {
                                        // 对于不同张量相乘，梯度是 other_tensor * node_grad
                                        input_grad_val = other_tensor * node_grad;
                                    }
                                }
                            } else {
                                // 对于标量乘法，梯度是标量值与当前节点梯度的乘积
                                // 注意：这里我们假设标量值已经被应用到了结果中，所以梯度计算比较简单
                                input_grad_val = node_grad;
                            }
                        } else if (node->operation == op::Div) {
                            // 第一个输入是a，第二个是b，∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/(b^2)
                            if (input_id == node->input_ids[0]) {
                                // ∂(a/b)/∂a = 1/b
                                size_t b_id = node->input_ids[1];
                                auto b_it   = id_to_node.find(b_id);
                                if (b_it != id_to_node.end() && b_it->second) {
                                    Tensor &b_tensor = *b_it->second->tensor;
                                    // 检查b_tensor是否包含零值
                                    bool has_zero = false;
                                    size_t numel  = b_tensor.numel();
                                    if (b_tensor.dtype() == DType::kFloat) {
                                        const float *b_data = b_tensor.data<float>();
                                        for (size_t i = 0; i < numel; ++i) {
                                            if (std::abs(b_data[i]) <
                                                std::numeric_limits<float>::epsilon()) {
                                                has_zero = true;
                                                break;
                                            }
                                        }
                                    }
                                    if (has_zero) {
                                        Ctorch_Error::throwException(
                                            ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE,
                                            "除零错误：Div操作的梯度计算中除数为零");
                                    }
                                    input_grad_val = node_grad / b_tensor;
                                }
                            } else {
                                // ∂(a/b)/∂b = -a/(b^2)
                                size_t a_id = node->input_ids[0];
                                auto a_it   = id_to_node.find(a_id);
                                auto b_it   = id_to_node.find(input_id);
                                if (a_it != id_to_node.end() && a_it->second &&
                                    b_it != id_to_node.end() && b_it->second) {
                                    Tensor &a_tensor = *a_it->second->tensor;
                                    Tensor &b_tensor = *b_it->second->tensor;
                                    // 检查b_tensor是否包含零值
                                    bool has_zero = false;
                                    size_t numel  = b_tensor.numel();
                                    if (b_tensor.dtype() == DType::kFloat) {
                                        const float *b_data = b_tensor.data<float>();
                                        for (size_t i = 0; i < numel; ++i) {
                                            if (std::abs(b_data[i]) <
                                                std::numeric_limits<float>::epsilon()) {
                                                has_zero = true;
                                                break;
                                            }
                                        }
                                    }
                                    if (has_zero) {
                                        Ctorch_Error::throwException(
                                            ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE,
                                            "除零错误：Div操作的梯度计算中除数为零");
                                    }
                                    input_grad_val = -a_tensor * node_grad / (b_tensor * b_tensor);
                                }
                            }
                        } else if (node->operation == op::Neg) {
                            // ∂(-a)/∂a = -1
                            input_grad_val = -node_grad;
                        } else if (node->operation == op::ReLU) {
                            // ∂ReLU(x)/∂x = 1 if x > 0 else 0
                            // 对于张量ReLU，梯度是node_grad乘以x>0的掩码
                            Tensor &x_tensor = *input_node.tensor;

                            // 实现ReLU梯度计算：直接在梯度上乘以mask
                            // 这里创建一个与x_tensor形状相同的张量，x>0时为1，否则为0
                            Tensor mask(ShapeTag{}, x_tensor.sizes(), x_tensor.dtype(),
                                        x_tensor.device());
                            size_t numel = x_tensor.numel();

                            // 根据数据类型处理
                            switch (x_tensor.dtype()) {
                            case DType::kFloat: {
                                float *mask_data    = mask.data<float>();
                                const float *x_data = x_tensor.data<float>();
                                for (size_t i = 0; i < numel; ++i) {
                                    mask_data[i] = (x_data[i] > 0.0f) ? 1.0f : 0.0f;
                                }
                                break;
                            }
                            case DType::kDouble: {
                                double *mask_data    = mask.data<double>();
                                const double *x_data = x_tensor.data<double>();
                                for (size_t i = 0; i < numel; ++i) {
                                    mask_data[i] = (x_data[i] > 0.0) ? 1.0 : 0.0;
                                }
                                break;
                            }
                            case DType::kInt: {
                                int32_t *mask_data    = mask.data<int32_t>();
                                const int32_t *x_data = x_tensor.data<int32_t>();
                                for (size_t i = 0; i < numel; ++i) {
                                    mask_data[i] = (x_data[i] > 0) ? 1 : 0;
                                }
                                break;
                            }
                            case DType::kLong: {
                                int64_t *mask_data    = mask.data<int64_t>();
                                const int64_t *x_data = x_tensor.data<int64_t>();
                                for (size_t i = 0; i < numel; ++i) {
                                    mask_data[i] = (x_data[i] > 0) ? 1 : 0;
                                }
                                break;
                            }
                            default:
                                Ctorch_Error::throwException(ErrorPlatform::kCPU,
                                                             ErrorType::DATATYPE,
                                                             "ReLU梯度计算中不支持的dtype");
                            }

                            input_grad_val = node_grad * mask;
                        } else if (node->operation == op::Sigmoid) {
                            // ∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x)) * grad
                            Tensor &x_tensor = *input_node.tensor;

                            // 计算 sigmoid(x)
                            Tensor sigmoid_x = x_tensor.sigmoid();

                            // 计算 1 - sigmoid(x)
                            Tensor one_minus_sigmoid_x = Tensor(1.0f) - sigmoid_x;

                            // 计算梯度
                            input_grad_val = sigmoid_x * one_minus_sigmoid_x * node_grad;
                        } else if (node->operation == op::Tanh) {
                            // ∂tanh(x)/∂x = (1 - tanh(x)^2) * grad
                            Tensor &x_tensor = *input_node.tensor;

                            // 计算 tanh(x)
                            Tensor tanh_x = x_tensor.tanh();

                            // 计算 tanh(x)^2
                            Tensor tanh_x_squared = tanh_x * tanh_x;

                            // 计算 1 - tanh(x)^2
                            Tensor one_minus_tanh_x_squared = Tensor(1.0f) - tanh_x_squared;

                            // 计算梯度
                            input_grad_val = one_minus_tanh_x_squared * node_grad;
                        } else if (node->operation == op::Sin) {
                            // ∂sin(x)/∂x = cos(x) * grad
                            Tensor &x_tensor = *input_node.tensor;

                            // 计算 cos(x)
                            Tensor cos_x = x_tensor.cos();

                            // 计算梯度
                            input_grad_val = cos_x * node_grad;
                        } else if (node->operation == op::Cos) {
                            // ∂cos(x)/∂x = -sin(x) * grad
                            Tensor &x_tensor = *input_node.tensor;

                            // 计算 sin(x)
                            Tensor sin_x = x_tensor.sin();

                            // 计算梯度
                            input_grad_val = -sin_x * node_grad;
                        } else if (node->operation == op::Softmax) {
                            // 支持 dim 的 softmax 反向：dL/dx = y * (g - sum(g*y, axis=dim))
                            // 这里使用 node->tensor（softmax 输出 y）而不是重新计算 softmax(x)
                            // node->op_param_i 保存 dim（由 Tensor::softmax(dim) 记录）
                            int dim = node->op_param_i;
                            const Tensor &y_tensor = *node->tensor;   // softmax 输出
                            const Tensor &g_tensor = node_grad;       // dL/dy

                            std::vector<size_t> shape = y_tensor.sizes();
                            if (y_tensor.dtype() != DType::kFloat) {
                                Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                                             "Softmax backward: 目前仅支持 float");
                            }
                            if (shape.size() == 1) {
                                // 1D: dim 只允许 0 或 -1
                                if (dim == -1) dim = 0;
                                if (dim != 0) {
                                    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                                                 "Softmax backward: 1D 张量仅支持 dim=0/-1");
                                }
                                Tensor dx(ShapeTag{}, shape, y_tensor.dtype(), y_tensor.device());
                                size_t n = y_tensor.numel();
                                const float* y = y_tensor.data<float>();
                                const float* g = g_tensor.data<float>();
                                float* out = dx.data<float>();

                                float dot = 0.0f;
                                for (size_t i = 0; i < n; ++i) dot += g[i] * y[i];
                                for (size_t i = 0; i < n; ++i) out[i] = y[i] * (g[i] - dot);
                                input_grad_val = dx;
                            } else if (shape.size() == 2) {
                                // 2D: 支持 dim=0 或 dim=1 或 -1(默认最后一维=1)
                                if (dim == -1) dim = 1;
                                if (dim != 0 && dim != 1) {
                                    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                                                 "Softmax backward: 2D 张量仅支持 dim=0/1/-1");
                                }
                                size_t rows = shape[0];
                                size_t cols = shape[1];
                                Tensor dx(ShapeTag{}, shape, y_tensor.dtype(), y_tensor.device());
                                const float* y = y_tensor.data<float>();
                                const float* g = g_tensor.data<float>();
                                float* out = dx.data<float>();

                                if (dim == 1) {
                                    // 按行
                                    for (size_t i = 0; i < rows; ++i) {
                                        float dot = 0.0f;
                                        for (size_t j = 0; j < cols; ++j) {
                                            dot += g[i * cols + j] * y[i * cols + j];
                                        }
                                        for (size_t j = 0; j < cols; ++j) {
                                            size_t idx = i * cols + j;
                                            out[idx] = y[idx] * (g[idx] - dot);
                                        }
                                    }
                                } else {
                                    // 按列
                                    for (size_t j = 0; j < cols; ++j) {
                                        float dot = 0.0f;
                                        for (size_t i = 0; i < rows; ++i) {
                                            dot += g[i * cols + j] * y[i * cols + j];
                                        }
                                        for (size_t i = 0; i < rows; ++i) {
                                            size_t idx = i * cols + j;
                                            out[idx] = y[idx] * (g[idx] - dot);
                                        }
                                    }
                                }
                                input_grad_val = dx;
                            } else {
                                Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                                             "Softmax backward: 暂仅支持 1D/2D 张量");
                            }
                        } else if (node->operation == op::MSE) {
                            // ∂MSE(pred, target)/∂pred = 2 * (pred - target) / n * grad
                            // 找到另一个输入张量（目标张量）
                            size_t target_id = (input_id == node->input_ids[0])
                                                   ? node->input_ids[1]
                                                   : node->input_ids[0];
                            auto target_it   = id_to_node.find(target_id);
                            if (target_it != id_to_node.end() && target_it->second) {
                                Tensor &pred_tensor   = *input_node.tensor;
                                Tensor &target_tensor = *target_it->second->tensor;

                                // 计算 pred - target
                                Tensor pred_minus_target = pred_tensor - target_tensor;

                                // 计算 2 * (pred - target)
                                Tensor two_times_diff = pred_minus_target * 2.0f;

                                // 计算元素数量
                                size_t n = pred_tensor.numel();

                                // 计算 2 * (pred - target) / n
                                Tensor grad_contrib = two_times_diff / static_cast<float>(n);

                                // 计算梯度
                                input_grad_val = grad_contrib * node_grad;
                            }
                        } else if (node->operation == op::CE) {
                            // ∂CrossEntropy(pred, target)/∂pred = (softmax(pred) - target) * grad
                            // 找到另一个输入张量（目标张量）
                            size_t target_id = (input_id == node->input_ids[0])
                                                   ? node->input_ids[1]
                                                   : node->input_ids[0];
                            auto target_it   = id_to_node.find(target_id);
                            if (target_it != id_to_node.end() && target_it->second) {
                                Tensor &pred_tensor   = *input_node.tensor;
                                Tensor &target_tensor = *target_it->second->tensor;

                                std::vector<size_t> pred_shape = pred_tensor.sizes();
                                if (pred_tensor.dtype() != DType::kFloat) {
                                    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                                                 "CrossEntropy backward: 目前仅支持 float");
                                }

                                // 只处理 logits 为 1D/2D 的常见分类场景：
                                // - 1D: [num_classes]（单样本）
                                // - 2D: [batch_size, num_classes]
                                if (pred_shape.size() == 2) {
                                    size_t batch_size = pred_shape[0];
                                    size_t num_classes = pred_shape[1];
                                    Tensor softmax_pred(ShapeTag{}, pred_shape, pred_tensor.dtype(), pred_tensor.device());
                                    
                                    const float* pred_data = pred_tensor.data<float>();
                                    float* softmax_data = softmax_pred.data<float>();
                                    
                                    // 对每一行（每个样本）计算 softmax
                                    for (size_t i = 0; i < batch_size; ++i) {
                                        // 找到该行的最大值（数值稳定性）
                                        float max_val = pred_data[i * num_classes];
                                        for (size_t j = 1; j < num_classes; ++j) {
                                            if (pred_data[i * num_classes + j] > max_val) {
                                                max_val = pred_data[i * num_classes + j];
                                            }
                                        }
                                        
                                        // 计算 exp 和
                                        float exp_sum = 0.0f;
                                        for (size_t j = 0; j < num_classes; ++j) {
                                            exp_sum += std::exp(pred_data[i * num_classes + j] - max_val);
                                        }
                                        
                                        // 计算 softmax
                                        for (size_t j = 0; j < num_classes; ++j) {
                                            softmax_data[i * num_classes + j] = 
                                                std::exp(pred_data[i * num_classes + j] - max_val) / exp_sum;
                                        }
                                    }

                                    // 计算 softmax(pred) - target
                                    Tensor softmax_minus_target = softmax_pred - target_tensor;

                                    // 计算梯度：∂CE/∂pred = (softmax(pred) - target) * grad
                                    // grad 是标量（CE loss 是标量），需要广播
                                    float grad_val = node_grad.item<float>();
                                    input_grad_val = softmax_minus_target * grad_val;
                                } else if (pred_shape.size() == 1) {
                                    // 1D logits：单样本
                                    size_t num_classes = pred_shape[0];
                                    Tensor softmax_pred(ShapeTag{}, pred_shape, pred_tensor.dtype(), pred_tensor.device());
                                    const float* pred_data = pred_tensor.data<float>();
                                    float* softmax_data = softmax_pred.data<float>();

                                    float max_val = pred_data[0];
                                    for (size_t j = 1; j < num_classes; ++j) {
                                        if (pred_data[j] > max_val) max_val = pred_data[j];
                                    }
                                    float exp_sum = 0.0f;
                                    for (size_t j = 0; j < num_classes; ++j) {
                                        exp_sum += std::exp(pred_data[j] - max_val);
                                    }
                                    for (size_t j = 0; j < num_classes; ++j) {
                                        softmax_data[j] = std::exp(pred_data[j] - max_val) / exp_sum;
                                    }

                                    Tensor softmax_minus_target = softmax_pred - target_tensor;
                                    float grad_val = node_grad.item<float>();
                                    input_grad_val = softmax_minus_target * grad_val;
                                } else {
                                    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                                                 "CrossEntropy backward: 暂仅支持 1D/2D logits");
                                }
                            }
                        } else if (node->operation == op::MAE) {
                            // ∂MAE(pred, target)/∂pred = sign(pred - target) / n * grad
                            // 找到另一个输入张量（目标张量）
                            size_t target_id = (input_id == node->input_ids[0])
                                                   ? node->input_ids[1]
                                                   : node->input_ids[0];
                            auto target_it   = id_to_node.find(target_id);
                            if (target_it != id_to_node.end() && target_it->second) {
                                Tensor &pred_tensor   = *input_node.tensor;
                                Tensor &target_tensor = *target_it->second->tensor;

                                // 计算 pred - target
                                Tensor pred_minus_target = pred_tensor - target_tensor;

                                // 计算 sign(pred - target)
                                Tensor sign_tensor(ShapeTag{}, pred_minus_target.sizes(),
                                                   pred_minus_target.dtype(),
                                                   pred_minus_target.device());
                                float *sign_data       = sign_tensor.data<float>();
                                const float *diff_data = pred_minus_target.data<float>();
                                size_t numel           = pred_minus_target.numel();

                                for (size_t i = 0; i < numel; ++i) {
                                    if (diff_data[i] > 0.0f) {
                                        sign_data[i] = 1.0f;
                                    } else if (diff_data[i] < 0.0f) {
                                        sign_data[i] = -1.0f;
                                    } else {
                                        sign_data[i] = 0.0f;
                                    }
                                }

                                // 计算元素数量
                                size_t n = pred_tensor.numel();

                                // 计算 sign(pred - target) / n
                                Tensor grad_contrib = sign_tensor / static_cast<float>(n);

                                // 计算梯度
                                input_grad_val = grad_contrib * node_grad;
                            }
                        } else if (node->operation == op::MatMul) {
                            // ∂(A*B)/∂A = grad * B^T, ∂(A*B)/∂B = A^T * grad
                            // 找到另一个输入张量
                            size_t other_input_id = (input_id == node->input_ids[0])
                                                        ? node->input_ids[1]
                                                        : node->input_ids[0];
                            auto other_it         = id_to_node.find(other_input_id);
                            if (other_it != id_to_node.end() && other_it->second) {
                                Tensor &other_tensor = *other_it->second->tensor;
                                if (input_id == node->input_ids[0]) {
                                    // 对第一个输入的梯度是 grad * B^T
                                    try {
                                        // 打印调试信息
                                        Ctorch_Error::trace(ErrorPlatform::kCPU,
                                                            "MatMul梯度计算：处理第一个输入");
                                        Ctorch_Error::trace(
                                            ErrorPlatform::kCPU,
                                            "输入形状: " +
                                                std::to_string(input_node.tensor->sizes()[0]) +
                                                "x" +
                                                std::to_string(input_node.tensor->sizes()[1]));
                                        Ctorch_Error::trace(
                                            ErrorPlatform::kCPU,
                                            "另一个输入形状: " +
                                                std::to_string(other_tensor.sizes()[0]) + "x" +
                                                std::to_string(other_tensor.sizes()[1]));
                                        Ctorch_Error::trace(
                                            ErrorPlatform::kCPU,
                                            "节点梯度形状: " +
                                                std::to_string(node_grad.sizes().size()) + "D");
                                        if (!node_grad.sizes().empty()) {
                                            Ctorch_Error::trace(
                                                ErrorPlatform::kCPU,
                                                "节点梯度维度: " +
                                                    std::to_string(node_grad.sizes()[0]) + "x" +
                                                    std::to_string(node_grad.sizes()[1]));
                                        }

                                        // 获取 node_grad 的形状和值
                                        bool is_scalar_grad = node_grad.sizes().empty();

                                        if (is_scalar_grad) {
                                            // 标量情况：grad * B^T 等同于 B^T * grad_value
                                            float grad_value = node_grad.item<float>();
                                            Ctorch_Error::trace(ErrorPlatform::kCPU,
                                                                "标量梯度值: " +
                                                                    std::to_string(grad_value));

                                            // 计算 B^T 并乘以 grad_value
                                            Tensor B_T     = other_tensor.t();
                                            input_grad_val = B_T * grad_value;
                                        } else {
                                            // 张量情况：grad * B^T
                                            if (node_grad.sizes().size() != 2) {
                                                throw std::runtime_error(
                                                    "MatMul gradient must be 2D tensor or scalar");
                                            }

                                            // 计算 B^T
                                            Tensor B_T = other_tensor.t();
                                            // 使用现有的matmul函数计算 grad * B^T
                                            input_grad_val = node_grad.matmul(B_T);
                                        }

                                        // 验证计算结果是否为非零张量
                                        bool result_is_zero = true;
                                        if (!input_grad_val.sizes().empty()) {
                                            size_t numel             = input_grad_val.numel();
                                            const float *result_data = input_grad_val.data<float>();
                                            for (size_t i = 0; i < numel; ++i) {
                                                if (std::abs(result_data[i]) > 1e-6) {
                                                    result_is_zero = false;
                                                    break;
                                                }
                                            }
                                        }
                                        Ctorch_Error::trace(ErrorPlatform::kCPU,
                                                            "result is zero: " +
                                                                std::to_string(result_is_zero));
                                    } catch (const std::exception &e) {
                                        // 如果矩阵乘法失败，详细记录错误并创建零梯度
                                        Ctorch_Error::trace(ErrorPlatform::kCPU,
                                                            "MatMul梯度计算失败，创建零梯度: " +
                                                                std::string(e.what()));
                                        std::vector<size_t> expected_shape =
                                            input_node.tensor->sizes();
                                        input_grad_val =
                                            Tensor(ShapeTag{}, expected_shape, node_grad.dtype(),
                                                   node_grad.device());
                                        input_grad_val.zero();
                                    }
                                } else {
                                    // 对第二个输入的梯度是 A^T * grad
                                    try {
                                        // 打印调试信息
                                        Ctorch_Error::trace(ErrorPlatform::kCPU,
                                                            "MatMul梯度计算：处理第二个输入");
                                        Ctorch_Error::trace(
                                            ErrorPlatform::kCPU,
                                            "输入形状: " +
                                                std::to_string(input_node.tensor->sizes()[0]) +
                                                "x" +
                                                std::to_string(input_node.tensor->sizes()[1]));
                                        Ctorch_Error::trace(
                                            ErrorPlatform::kCPU,
                                            "另一个输入形状: " +
                                                std::to_string(other_tensor.sizes()[0]) + "x" +
                                                std::to_string(other_tensor.sizes()[1]));
                                        Ctorch_Error::trace(
                                            ErrorPlatform::kCPU,
                                            "节点梯度形状: " +
                                                std::to_string(node_grad.sizes().size()) + "D");
                                        if (!node_grad.sizes().empty()) {
                                            Ctorch_Error::trace(
                                                ErrorPlatform::kCPU,
                                                "节点梯度维度: " +
                                                    std::to_string(node_grad.sizes()[0]) + "x" +
                                                    std::to_string(node_grad.sizes()[1]));
                                        }

                                        // 获取 node_grad 的形状和值
                                        bool is_scalar_grad = node_grad.sizes().empty();

                                        if (is_scalar_grad) {
                                            // 标量情况：A^T * grad 等同于 A^T * grad_value
                                            float grad_value = node_grad.item<float>();
                                            Ctorch_Error::trace(ErrorPlatform::kCPU,
                                                                "标量梯度值: " +
                                                                    std::to_string(grad_value));

                                            // 计算 A^T 并乘以 grad_value
                                            Tensor A_T     = other_tensor.t();
                                            input_grad_val = A_T * grad_value;
                                        } else {
                                            // 张量情况：A^T * grad
                                            if (node_grad.sizes().size() != 2) {
                                                throw std::runtime_error(
                                                    "MatMul gradient must be 2D tensor or scalar");
                                            }

                                            // 验证矩阵乘法维度是否匹配
                                            size_t A_rows    = other_tensor.sizes()[0];
                                            size_t grad_rows = node_grad.sizes()[0];
                                            if (A_rows != grad_rows) {
                                                throw std::runtime_error(
                                                    "MatMul gradient dimension mismatch: A_rows " +
                                                    std::to_string(A_rows) + " != grad_rows " +
                                                    std::to_string(grad_rows));
                                            }

                                            // 验证node_grad是否为非零张量
                                            bool is_zero           = true;
                                            size_t grad_numel      = node_grad.numel();
                                            const float *grad_data = node_grad.data<float>();
                                            for (size_t i = 0; i < grad_numel; ++i) {
                                                if (std::abs(grad_data[i]) > 1e-6) {
                                                    is_zero = false;
                                                    break;
                                                }
                                            }
                                            Ctorch_Error::trace(ErrorPlatform::kCPU,
                                                                "node_grad is zero: " +
                                                                    std::to_string(is_zero));

                                            // 计算 A^T
                                            Tensor A_T = other_tensor.t();
                                            // 使用现有的matmul函数计算 A^T * grad
                                            input_grad_val = A_T.matmul(node_grad);
                                        }

                                        // 验证计算结果是否为非零张量
                                        bool result_is_zero = true;
                                        if (!input_grad_val.sizes().empty()) {
                                            size_t numel             = input_grad_val.numel();
                                            const float *result_data = input_grad_val.data<float>();
                                            for (size_t i = 0; i < numel; ++i) {
                                                if (std::abs(result_data[i]) > 1e-6) {
                                                    result_is_zero = false;
                                                    break;
                                                }
                                            }
                                        }
                                        Ctorch_Error::trace(ErrorPlatform::kCPU,
                                                            "result is zero: " +
                                                                std::to_string(result_is_zero));
                                    } catch (const std::exception &e) {
                                        // 如果矩阵乘法失败，详细记录错误并创建零梯度
                                        Ctorch_Error::trace(ErrorPlatform::kCPU,
                                                            "MatMul梯度计算失败，创建零梯度: " +
                                                                std::string(e.what()));
                                        std::vector<size_t> expected_shape =
                                            input_node.tensor->sizes();
                                        input_grad_val =
                                            Tensor(ShapeTag{}, expected_shape, node_grad.dtype(),
                                                   node_grad.device());
                                        input_grad_val.zero();
                                    }
                                }
                            } else {
                                // 如果找不到另一个输入张量，创建一个零梯度
                                Ctorch_Error::trace(ErrorPlatform::kCPU,
                                                    "MatMul梯度计算失败：找不到另一个输入张量");
                                std::vector<size_t> expected_shape = input_node.tensor->sizes();
                                input_grad_val = Tensor(ShapeTag{}, expected_shape,
                                                        node_grad.dtype(), node_grad.device());
                                input_grad_val.zero();
                            }
                        } else if (node->operation == op::Sum) {
                            // ∂sum(x)/∂x = 1 for all elements
                            // 梯度是与输入张量形状相同的张量，每个元素的值等于输出梯度的值
                            // 当输出是标量时，输入梯度应该是全1的张量
                            input_grad_val =
                                check_and_adjust_grad_shape(node_grad, input_node.tensor->sizes());
                        } else {
                            Ctorch_Error::trace(
                                ErrorPlatform::kCPU,
                                "不支持的操作类型: " +
                                    std::to_string(static_cast<int>(node->operation)));
                            continue;
                        }

                        // 检查并调整梯度形状，确保与输入节点形状匹配
                        Tensor adjusted_grad_val =
                            check_and_adjust_grad_shape(input_grad_val, input_node.tensor->sizes());

                        // 累加梯度 - 直接修改input_grad张量
                        // 实现原地累加，避免创建新张量
                        size_t numel = input_grad.numel();
                        switch (input_grad.dtype()) {
                        case DType::kFloat: {
                            float *grad_data = input_grad.data<float>();
                            const float *val_data = adjusted_grad_val.data<float>();
                            for (size_t i = 0; i < numel; ++i) {
                                grad_data[i] += val_data[i];
                            }
                            break;
                        }
                        case DType::kDouble: {
                            double *grad_data = input_grad.data<double>();
                            const double *val_data = adjusted_grad_val.data<double>();
                            for (size_t i = 0; i < numel; ++i) {
                                grad_data[i] += val_data[i];
                            }
                            break;
                        }
                        case DType::kInt: {
                            int32_t *grad_data = input_grad.data<int32_t>();
                            const int32_t *val_data = adjusted_grad_val.data<int32_t>();
                            for (size_t i = 0; i < numel; ++i) {
                                grad_data[i] += val_data[i];
                            }
                            break;
                        }
                        case DType::kLong: {
                            int64_t *grad_data = input_grad.data<int64_t>();
                            const int64_t *val_data = adjusted_grad_val.data<int64_t>();
                            for (size_t i = 0; i < numel; ++i) {
                                grad_data[i] += val_data[i];
                            }
                            break;
                        }
                        default:
                            Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                                         "不支持的dtype类型");
                        }

                        Ctorch_Error::trace(ErrorPlatform::kCPU,
                                            "节点 " + std::to_string(node->tensor_id) + " 向节点 " +
                                                std::to_string(input_id) + " 传播梯度");
                        Ctorch_Error::trace(
                            ErrorPlatform::kCPU,
                            "梯度形状: " + std::to_string(input_grad_val.sizes().size()) + "D");
                    }
                }
            }
        }
    }

    Ctorch_Error::trace(ErrorPlatform::kAutoDiff, std::string("反向传播完成"));
    Ctorch_Error::trace(ErrorPlatform::kAutoDiff,
                        std::string("========================================"));
}

void AutoDiff::clear_graph() {
    std::lock_guard lock(records_mutex);
    id_to_node.clear();
    pending_records.clear();
    retain_graph = false;
}

// ======================= 辅助函数 =======================
Tensor create_empty_tensor() {
    return Tensor(ShapeTag{}, std::vector<size_t>{}, DType::kFloat, DeviceType::kCPU);
}
