/**
* @file AutoDiff.cpp
* @brief Ctorch 自动微分系统实现
* @author GhostFace
* @date 2025/12/21
*/

#include "AutoDiff.h"
#include "Tensor.h"

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
AutoDiff::Node* AutoDiff::get_node_by_id(size_t id) {
    std::lock_guard<std::mutex> lock(records_mutex);
    auto it = id_to_node.find(id);
    if (it != id_to_node.end()) {
        return it->second.get();
    }
    std::cout << ">>> 警告: 找不到节点 " << id << std::endl;
    return nullptr;
}

// ======================= 调试方法 =======================

/**
 * @brief 打印自动微分系统的当前状态
 * @param context 调试上下文
 */
void AutoDiff::debug_print_state(const std::string& context) {
    std::lock_guard<std::mutex> lock(records_mutex);
    std::ostringstream oss;
    oss << "=== AutoDiff状态 [" << context << "] ===" << std::endl;
    oss << "计算图节点 (" << id_to_node.size() << "): ";
    for (const auto& pair : id_to_node) {
        if (pair.second) {
            oss << pair.first << " ";
        }
    }
    oss << std::endl;
    oss << "待处理记录 (" << pending_records.size() << "): ";
    for (const auto& pair : pending_records) {
        oss << pair.first << "(committed=" << pair.second.committed << ") ";
    }
    oss << std::endl;
    oss << "=================================" << std::endl;
    std::string msg = oss.str();
    Ctorch_Error::trace(ErrorPlatform::kCPU, msg);
}

// ======================= get_grad函数 =======================
Tensor AutoDiff::get_grad(const Tensor* t) {
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
    DType input_dtype = t->dtype();
    DeviceType input_device = t->device();

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
            has_grad = true;
            grad_shape = it->second->grad->sizes();
            grad_dtype = it->second->grad->dtype();
            grad_device = it->second->grad->device();

            // 获取梯度值用于调试
            if (grad_dtype == DType::kFloat && !grad_shape.empty()) {
                const float *grad_data = it->second->grad->data<float>();
                if (grad_data) {
                    grad_value = grad_data[0]; // 对于标量，取第一个元素
                }
            }
            std::ostringstream osss;
            osss << ">>> 找到梯度，形状: [";
            for (auto s: grad_shape) osss << s << " ";
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
                        float *dst = result.data<float>();
                        size_t count = result.numel();
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
                        double *dst = result.data<double>();
                        for (size_t i = 0; i < result.numel(); ++i) {
                            dst[i] = src[i];
                        }
                        break;
                    }
                    case DType::kInt: {
                        const int32_t *src = it->second->grad->data<int32_t>();
                        int32_t *dst = result.data<int32_t>();
                        for (size_t i = 0; i < result.numel(); ++i) {
                            dst[i] = src[i];
                        }
                        break;
                    }
                    case DType::kLong: {
                        const int64_t *src = it->second->grad->data<int64_t>();
                        int64_t *dst = result.data<int64_t>();
                        for (size_t i = 0; i < result.numel(); ++i) {
                            dst[i] = src[i];
                        }
                        break;
                    }
                    default:
                        Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE, "get_grad中不支持的dtype");
                }
            }
        }
        std::ostringstream osss;
        osss << ">>> 最终梯度副本形状: ";
        for (size_t s : result.sizes()) osss << s << " ";
        osss << std::endl;
        osss << "<<< get_grad - 完成" << std::endl;

        std::string msgs = osss.str();
        Ctorch_Error::trace(ErrorPlatform::kCPU, msgs);
        return result;
    }
    // 未找到梯度时，输出Error级别提示并返回零张量
    Ctorch_Error::trace(ErrorPlatform::kCPU, "[ERROR] [2025-12-21 18:05:10.882 1766311510882] [ERROR_CODE:0x5060400] [PLATFORM:GENERAL] [TYPE:TENSOR_STATE] Tensor梯度未找到，返回零张量");
    // 创建与输入张量相同形状、数据类型和设备的零张量
    Tensor result(ShapeTag{}, input_shape, input_dtype, input_device);
    // 初始化所有元素为0
    switch (input_dtype) {
        case DType::kFloat: {
            float *dst = result.data<float>();
            size_t count = result.numel();
            for (size_t i = 0; i < count; ++i) {
                dst[i] = 0.0f;
            }
            break;
        }
        case DType::kDouble: {
            double *dst = result.data<double>();
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
            Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE, "get_grad中不支持的dtype");
    }
    return result;
}

// ======================= make_leaf函数 =======================
void AutoDiff::make_leaf(Tensor& t, bool requires_grad) {
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
            oss << ">>> 节点 " << id << " 已存在，跳过创建" << std::endl;
            std::string msg = oss.str();
            Ctorch_Error::trace(ErrorPlatform::kCPU, msg);
            return;
        }
    }

    // 创建节点时不持有锁
    auto tensor_ptr = std::make_unique<Tensor>(t.clone());
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
void AutoDiff::defer_record(size_t output_id, op operation, const std::vector<Tensor*>& inputs) {
    std::ostringstream oss;
    oss << ">>> 进入 defer_record, output_id: " << output_id;
    std::string msg = oss.str();
    Ctorch_Error::trace(ErrorPlatform::kCPU, msg);
    if (output_id == 0) {
        Ctorch_Error::log(ErrorLevel::WARN, ErrorPlatform::kCPU, ErrorType::TENSOR_STATE, "警告: output_id 为0");
        return;
    }

    // 创建记录对象（不持有锁）
    PendingRecord record;
    record.operation = operation;

    std::vector<size_t> input_ids;
    std::vector<std::vector<size_t>> input_shapes;

    // 收集输入信息（不持有锁）
    for (Tensor* input : inputs) {
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
        record.input_ids = input_ids;
        record.input_shapes = input_shapes;
        pending_records[output_id] = record;
    }

    Ctorch_Error::trace(ErrorPlatform::kCPU, "<<< 离开 defer_record");
}

// ======================= commit_record函数 =======================
void AutoDiff::commit_record(Tensor& output) {
    size_t output_id = output.id();
    std::cout << ">>> AutoDiff::commit_record - 开始, ID: " << output_id << std::endl;

    if (output_id == 0) {
        std::cout << "!!! 错误: 输出ID为0" << std::endl;
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
            std::cout << "!!! 警告: 找不到待处理记录 " << output_id << std::endl;
            return;
        }

        PendingRecord& record = it->second;
        if (record.committed) {
            std::cout << ">>> 记录 " << output_id << " 已提交，跳过" << std::endl;
            return;
        }

        std::cout << ">>> 开始提交记录 " << output_id << std::endl;
        std::cout << ">>> 操作类型: " << static_cast<int>(record.operation) << std::endl;
        std::cout << ">>> 输入IDs: ";
        for (auto id : record.input_ids) std::cout << id << " ";
        std::cout << std::endl;

        // 验证输入节点
        bool valid = true;
        for (size_t input_id : record.input_ids) {
            if (id_to_node.find(input_id) == id_to_node.end()) {
                std::cout << "!!! 错误: 输入节点 " << input_id << " 不存在" << std::endl;
                valid = false;
                break;
            } else {
                std::cout << ">>> 输入节点 " << input_id << " 存在" << std::endl;
            }
        }

        if (!valid) {
            std::cout << "!!! 记录验证失败，清除记录 " << output_id << std::endl;
            pending_records.erase(it);
            return;
        }

        // 复制必要信息，然后释放锁
        record_copy = record;
        input_ids_copy = record.input_ids;
        should_create_node = true;

        // 立即标记为已提交，避免重复处理
        record.committed = true;
    } // 释放锁

    // 第二步：在锁外创建节点（避免死锁）
    if (should_create_node) {
        std::cout << ">>> 在锁外创建操作节点 " << output_id << std::endl;

        // 确定梯度需求（在锁外检查）
        bool requires_grad = false;
        for (size_t input_id : input_ids_copy) {
            Node* input_node = get_node_by_id(input_id);
            if (input_node && input_node->requires_grad) {
                requires_grad = true;
                break;
            }
        }

        std::cout << ">>> 节点 " << output_id << " 需要梯度: " << requires_grad << std::endl;

        // 创建节点（不持有锁）
        auto tensor_ptr = std::make_unique<Tensor>(output.clone());
        auto output_node = std::make_unique<Node>(output_id, std::move(tensor_ptr), requires_grad, false);
        output_node->operation = record_copy.operation;
        output_node->input_ids = input_ids_copy;

        // 延迟创建梯度存储
        if (requires_grad) {
            std::cout << ">>> 为节点 " << output_id << " 延迟分配梯度存储" << std::endl;
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

        std::cout << "<<< 记录提交完成 " << output_id << std::endl;
        debug_print_state("commit_record完成后");
    }
}

// ======================= update_requires_grad函数 =======================
void AutoDiff::update_requires_grad(Tensor& t, bool requires_grad) {
    size_t id = t.id();
    if (id == 0) return;

    std::cout << ">>> 更新梯度需求: " << id << " -> " << requires_grad << std::endl;

    // 使用更短的锁作用域
    {
        std::lock_guard<std::mutex> lock(records_mutex);
        auto it = id_to_node.find(id);
        if (it != id_to_node.end() && it->second) {
            it->second->requires_grad = requires_grad;
            if (requires_grad) {
                if (!it->second->grad) {
                    it->second->grad = std::make_unique<Tensor>(ShapeTag{}, t.sizes(), t.dtype(), t.device());
                    it->second->grad->zero();
                }
            }
        } else {
            // 如果节点不存在，可能需要创建它
            std::cout << ">>> 节点不存在，可能需要创建: " << id << std::endl;
        }
    }

    std::cout << "<<< 更新梯度需求完成" << std::endl;
}

// ======================= set_retain_graph函数 =======================
void AutoDiff::set_retain_graph(bool retain) {
    retain_graph = retain;
}

// ======================= backward函数（版本1） =======================
void AutoDiff::backward(Tensor& root) {
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
Tensor AutoDiff::check_and_adjust_grad_shape(const Tensor& grad, const std::vector<size_t>& target_shape) {
    // 如果形状已经匹配，直接返回
    if (grad.sizes() == target_shape) {
        return grad;
    }
    
    std::cout << ">>> 调整梯度形状: ";
    for (size_t s : grad.sizes()) std::cout << s << " ";
    std::cout << "-> ";
    for (size_t s : target_shape) std::cout << s << " ";
    std::cout << std::endl;
    
    // 处理标量情况
    if (grad.sizes().empty()) {
        // 标量梯度扩展到目标形状
        Tensor result(ShapeTag{}, target_shape, grad.dtype(), grad.device());
        float val = grad.item<float>();
        
        // 使用循环填充张量
        size_t numel = result.numel();
        float* data = result.data<float>();
        for (size_t i = 0; i < numel; ++i) {
            data[i] = val;
        }
        
        return result;
    }
    
    size_t grad_rank = grad.sizes().size();
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
        
        // 将grad数据复制到temp_grad
        float* temp_data = temp_grad.data<float>();
        const float* grad_data = grad.data<float>();
        size_t grad_numel = grad.numel();
        size_t temp_numel = temp_grad.numel();
        
        for (size_t i = 0; i < temp_numel; ++i) {
            temp_data[i] = grad_data[i % grad_numel];
        }
        
        // 递归调用，处理新的形状
        return check_and_adjust_grad_shape(temp_grad, target_shape);
    }
    
    // 情况2：grad的秩大于等于target_shape的秩
    bool can_broadcast = true;
    
    // 从末尾开始比较形状
    for (size_t i = 1; i <= target_rank; ++i) {
        size_t grad_idx = grad_rank - i;
        size_t target_idx = target_rank - i;
        
        size_t grad_dim = grad.sizes()[grad_idx];
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
        float* result_data = result.data<float>();
        const float* grad_data = grad.data<float>();
        
        // 计算每个维度的步幅
        std::vector<size_t> grad_strides = grad.strides();
        std::vector<size_t> result_strides = result.strides();
        
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
                size_t grad_idx = grad_rank - j;
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
        
        return result;
    }
    
    // 情况3：处理广播后的梯度求和
    // 例如：target_shape = [3], grad_shape = [2, 3]，则对grad的第一个维度求和
    bool can_reduce = true;
    
    // 检查是否可以通过求和得到目标形状
    for (size_t i = 1; i <= std::min(grad_rank, target_rank); ++i) {
        size_t grad_idx = grad_rank - i;
        size_t target_idx = target_rank - i;
        
        size_t grad_dim = grad.sizes()[grad_idx];
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
        
        // 实现简单的求和逻辑
        float* result_data = result.data<float>();
        const float* grad_data = grad.data<float>();
        
        size_t result_numel = result.numel();
        size_t grad_numel = grad.numel();
        
        // 简化处理：直接对所有元素求和
        for (size_t i = 0; i < grad_numel; ++i) {
            result_data[i % result_numel] += grad_data[i];
        }
        
        return result;
    }
    
    // 其他情况：如果形状不匹配且无法广播，返回零张量
    std::cout << ">>> 警告: 无法调整梯度形状，返回零张量" << std::endl;
    Tensor result(ShapeTag{}, target_shape, grad.dtype(), grad.device());
    result.zero();
    return result;
}

// ======================= 私有辅助函数：DFS拓扑排序 =======================
/**
 * @brief 执行DFS后序遍历，生成拓扑序列
 * @param node 当前节点
 * @param visited 已访问节点集合
 * @param result 拓扑序列结果
 */
void AutoDiff::dfs_topological_sort(Node* node, std::unordered_set<Node*>& visited, std::vector<Node*>& result) {
    if (!node) return;
    if (visited.find(node) != visited.end()) return;
    
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
void AutoDiff::backward(Tensor& root, Tensor grad_output) {
    std::cout << ">>> =========================================" << std::endl;
    std::cout << ">>> 进入 backward 函数，root ID: " << root.id() << std::endl;

    if (root.id() == 0) {
        Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::TENSOR_STATE, "反向传播的无效根张量 (ID为0)");
        return;
    }

    // 阶段1：准备阶段（不持有锁）
    std::cout << ">>> 阶段1: 准备阶段" << std::endl;

    // 阶段2：验证阶段和设置根节点梯度
    std::cout << ">>> 阶段2: 验证和设置根节点梯度" << std::endl;
    Node* root_node = nullptr;
    bool root_requires_grad = false;
    std::vector<size_t> root_shape;

    {
        std::lock_guard<std::mutex> lock(records_mutex);
        auto it = id_to_node.find(root.id());
        if (it == id_to_node.end() || !it->second) {
            std::cout << "!!! 错误: 找不到根节点 " << root.id() << std::endl;
            return;
        }
        root_node = it->second.get();
        root_requires_grad = root_node->requires_grad;
        root_shape = root.sizes();

        if (!root_requires_grad) {
            std::cout << ">>> 根节点不需要梯度，跳过反向传播" << std::endl;
            return;
        }

        // 设置根节点梯度，确保形状匹配
        if (!root_node->grad) {
            root_node->grad = std::make_unique<Tensor>(ShapeTag{}, root_shape, root.dtype(), root.device());
        }
        
        // 检查并调整梯度输出形状，确保与根节点形状匹配
        Tensor adjusted_grad_output = check_and_adjust_grad_shape(grad_output, root_shape);
        *root_node->grad = adjusted_grad_output;
    }

    std::cout << ">>> 根节点需要梯度，开始反向传播" << std::endl;
    std::cout << ">>> 根节点形状: " << root_shape.size() << "D" << std::endl;
    std::cout << ">>> 梯度输出形状: " << grad_output.sizes().size() << "D" << std::endl;

    // 阶段3：反向传播（实现实际的梯度计算）
    std::cout << ">>> 阶段3: 反向传播计算" << std::endl;
    
    std::vector<Node*> topological_order;
    {
        std::lock_guard<std::mutex> lock(records_mutex);
        
        // 生成拓扑序列
        std::unordered_set<Node*> visited;
        std::vector<Node*> forward_order;
        dfs_topological_sort(root_node, visited, forward_order);
        
        // 反转得到逆拓扑序列（反向传播顺序）
        topological_order = forward_order;
        std::reverse(topological_order.begin(), topological_order.end());
        
        std::cout << ">>> 拓扑排序完成，节点数量: " << topological_order.size() << std::endl;
    }
    
    // 处理每个节点，传播梯度到输入节点
    {
        std::lock_guard<std::mutex> lock(records_mutex);
        for (Node* node : topological_order) {
            // 跳过不需要梯度的节点
            if (!node->requires_grad) {
                continue;
            }
            
            // 计算梯度并传播到输入节点
            if (node->grad) {
                // 对于张量梯度，直接使用梯度张量进行计算
                Tensor& node_grad = *node->grad;
                
                for (size_t input_id : node->input_ids) {
                    auto input_it = id_to_node.find(input_id);
                    if (input_it != id_to_node.end() && input_it->second && input_it->second->requires_grad) {
                        auto& input_node = *input_it->second;
                        
                        // 初始化输入节点的梯度，确保形状匹配
                        if (!input_node.grad) {
                            input_node.grad = std::make_unique<Tensor>(ShapeTag{}, input_node.tensor->sizes(), input_node.tensor->dtype(), input_node.tensor->device());
                            input_node.grad->zero();
                        }
                        
                        Tensor& input_grad = *input_node.grad;
                        
                        // 根据操作类型计算梯度
                        Tensor input_grad_val;
                        
                        // 处理不同操作类型
                        if (node->operation == op::Add) {
                            // ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
                            // 对于张量加法，梯度直接传播
                            input_grad_val = node_grad;
                        } else if (node->operation == op::Sub) {
                            // 第一个输入是a，第二个是b，∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
                            if (input_id == node->input_ids[0]) {
                                // 对第一个输入的梯度是正的
                                input_grad_val = node_grad;
                            } else {
                                // 对第二个输入的梯度是负的
                                input_grad_val = -node_grad;
                            }
                        } else if (node->operation == op::Mul) {
                            // 找到另一个输入张量
                            size_t other_input_id = (input_id == node->input_ids[0]) ? node->input_ids[1] : node->input_ids[0];
                            auto other_it = id_to_node.find(other_input_id);
                            if (other_it != id_to_node.end() && other_it->second) {
                                // 对于张量乘法，梯度是另一个输入张量与当前节点梯度的乘积
                                Tensor& other_tensor = *other_it->second->tensor;
                                input_grad_val = other_tensor * node_grad;
                            }
                        } else if (node->operation == op::Div) {
                            // 第一个输入是a，第二个是b，∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/(b^2)
                            if (input_id == node->input_ids[0]) {
                                // ∂(a/b)/∂a = 1/b
                                size_t b_id = node->input_ids[1];
                                auto b_it = id_to_node.find(b_id);
                                if (b_it != id_to_node.end() && b_it->second) {
                                    Tensor& b_tensor = *b_it->second->tensor;
                                    input_grad_val = node_grad / b_tensor;
                                }
                            } else {
                                // ∂(a/b)/∂b = -a/(b^2)
                                size_t a_id = node->input_ids[0];
                                auto a_it = id_to_node.find(a_id);
                                auto b_it = id_to_node.find(input_id);
                                if (a_it != id_to_node.end() && a_it->second && b_it != id_to_node.end() && b_it->second) {
                                    Tensor& a_tensor = *a_it->second->tensor;
                                    Tensor& b_tensor = *b_it->second->tensor;
                                    input_grad_val = -a_tensor * node_grad / (b_tensor * b_tensor);
                                }
                            }
                        } else if (node->operation == op::Neg) {
                            // ∂(-a)/∂a = -1
                            input_grad_val = -node_grad;
                        } else if (node->operation == op::ReLU) {
                            // ∂ReLU(x)/∂x = 1 if x > 0 else 0
                            // 对于张量ReLU，梯度是node_grad乘以x>0的掩码
                            Tensor& x_tensor = *input_node.tensor;
                            
                            // 实现ReLU梯度计算：直接在梯度上乘以mask
                            // 这里简化处理，直接创建一个与x_tensor形状相同的张量，x>0时为1，否则为0
                            Tensor mask(ShapeTag{}, x_tensor.sizes(), x_tensor.dtype(), x_tensor.device());
                            float* mask_data = mask.data<float>();
                            const float* x_data = x_tensor.data<float>();
                            size_t numel = x_tensor.numel();
                            
                            for (size_t i = 0; i < numel; ++i) {
                                mask_data[i] = (x_data[i] > 0.0f) ? 1.0f : 0.0f;
                            }
                            
                            input_grad_val = node_grad * mask;
                        } else if (node->operation == op::Sigmoid) {
                            // ∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x)) * grad
                            Tensor& x_tensor = *input_node.tensor;
                            
                            // 计算 sigmoid(x)
                            Tensor sigmoid_x = x_tensor.sigmoid();
                            
                            // 计算 1 - sigmoid(x)
                            Tensor one_minus_sigmoid_x = Tensor(1.0f) - sigmoid_x;
                            
                            // 计算梯度
                            input_grad_val = sigmoid_x * one_minus_sigmoid_x * node_grad;
                        } else if (node->operation == op::Tanh) {
                            // ∂tanh(x)/∂x = (1 - tanh(x)^2) * grad
                            Tensor& x_tensor = *input_node.tensor;
                            
                            // 计算 tanh(x)
                            Tensor tanh_x = x_tensor.tanh();
                            
                            // 计算 tanh(x)^2
                            Tensor tanh_x_squared = tanh_x * tanh_x;
                            
                            // 计算 1 - tanh(x)^2
                            Tensor one_minus_tanh_x_squared = Tensor(1.0f) - tanh_x_squared;
                            
                            // 计算梯度
                            input_grad_val = one_minus_tanh_x_squared * node_grad;
                        } else if (node->operation == op::Softmax) {
                            // ∂softmax(x)/∂x = softmax(x) * (grad - sum(softmax(x) * grad))
                            Tensor& x_tensor = *input_node.tensor;
                            
                            // 计算 softmax(x)
                            Tensor softmax_x = x_tensor.softmax(0);
                            
                            // 计算 softmax(x) * grad
                            Tensor softmax_x_times_grad = softmax_x * node_grad;
                            
                            // 计算 sum(softmax(x) * grad)
                            Tensor sum_softmax_x_times_grad = softmax_x_times_grad.sum();
                            
                            // 计算 grad - sum(softmax(x) * grad)
                            Tensor grad_minus_sum = node_grad - sum_softmax_x_times_grad;
                            
                            // 计算梯度
                            input_grad_val = softmax_x * grad_minus_sum;
                        } else if (node->operation == op::MSE) {
                            // ∂MSE(pred, target)/∂pred = 2 * (pred - target) / n * grad
                            // 找到另一个输入张量（目标张量）
                            size_t target_id = (input_id == node->input_ids[0]) ? node->input_ids[1] : node->input_ids[0];
                            auto target_it = id_to_node.find(target_id);
                            if (target_it != id_to_node.end() && target_it->second) {
                                Tensor& pred_tensor = *input_node.tensor;
                                Tensor& target_tensor = *target_it->second->tensor;
                                
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
                            size_t target_id = (input_id == node->input_ids[0]) ? node->input_ids[1] : node->input_ids[0];
                            auto target_it = id_to_node.find(target_id);
                            if (target_it != id_to_node.end() && target_it->second) {
                                Tensor& pred_tensor = *input_node.tensor;
                                Tensor& target_tensor = *target_it->second->tensor;
                                
                                // 计算 softmax(pred)
                                Tensor softmax_pred = pred_tensor.softmax(0);
                                
                                // 计算 softmax(pred) - target
                                Tensor softmax_minus_target = softmax_pred - target_tensor;
                                
                                // 计算梯度
                                input_grad_val = softmax_minus_target * node_grad;
                            }
                        } else if (node->operation == op::MAE) {
                            // ∂MAE(pred, target)/∂pred = sign(pred - target) / n * grad
                            // 找到另一个输入张量（目标张量）
                            size_t target_id = (input_id == node->input_ids[0]) ? node->input_ids[1] : node->input_ids[0];
                            auto target_it = id_to_node.find(target_id);
                            if (target_it != id_to_node.end() && target_it->second) {
                                Tensor& pred_tensor = *input_node.tensor;
                                Tensor& target_tensor = *target_it->second->tensor;
                                
                                // 计算 pred - target
                                Tensor pred_minus_target = pred_tensor - target_tensor;
                                
                                // 计算 sign(pred - target)
                                Tensor sign_tensor(ShapeTag{}, pred_minus_target.sizes(), pred_minus_target.dtype(), pred_minus_target.device());
                                float* sign_data = sign_tensor.data<float>();
                                const float* diff_data = pred_minus_target.data<float>();
                                size_t numel = pred_minus_target.numel();
                                
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
                        } else {
                            std::cout << "!!! 不支持的操作类型: " << static_cast<int>(node->operation) << std::endl;
                            continue;
                        }
                        
                        // 检查并调整梯度形状，确保与输入节点形状匹配
                        Tensor adjusted_grad_val = check_and_adjust_grad_shape(input_grad_val, input_node.tensor->sizes());
                        
                        // 累加梯度 - 使用Tensor的operator+和赋值操作符
                        input_grad = input_grad + adjusted_grad_val;
                        
                        std::cout << ">>> 节点 " << node->tensor_id << " 向节点 " << input_id << " 传播梯度" << std::endl;
                        std::cout << ">>> 梯度形状: " << input_grad_val.sizes().size() << "D" << std::endl;
                    }
                }
            }
        }
    }

    std::cout << ">>> 反向传播完成" << std::endl;
    std::cout << "<<< =========================================" << std::endl;
}

// ======================= 辅助函数 =======================
Tensor create_empty_tensor() {
    return Tensor(ShapeTag{}, std::vector<size_t>{}, DType::kFloat, DeviceType::kCPU);
}