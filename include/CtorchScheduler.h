/**
 * @file CtorchScheduler.h
 * @brief Ctorch 框架的核心调度器类
 * @details 采用单例模式实现，负责管理所有 kernel 映射关系，根据算子类型和设备类型，
 * 自动查找并调用对应的 kernel，实现 kernel 的统一调度。
 * @author GhostFace
 * @date 2025/12/20
 */
#ifndef CTORCH_SCHEDULER_H
#define CTORCH_SCHEDULER_H
#include "CtorchError.h"
#include "Tensor.h"
#include "AutoGrad.h"
#include "AutoGrad/Nodes/AddNode.h"
#include "AutoGrad/Nodes/SubNode.h"
#include "AutoGrad/Nodes/MulNode.h"
#include "AutoGrad/Nodes/DivNode.h"
#include "AutoGrad/Nodes/NegNode.h"
#include "AutoGrad/Nodes/SinNode.h"
#include "AutoGrad/Nodes/CosNode.h"
#include "AutoGrad/Nodes/TanhNode.h"
#include "AutoGrad/Nodes/SigmoidNode.h"
#include "AutoGrad/Nodes/ReLUNode.h"
#include "AutoGrad/Nodes/MatMulNode.h"
#include "AutoGrad/Nodes/CrossEntropyNode.h"
#include "AutoGrad/Nodes/SoftmaxNode.h"
#include "./../src/kernels/kernels.h"

class CtorchScheduler{
private:
    CtorchScheduler() = default;
    // 禁止拷贝构造：防止通过“实例拷贝”创建新对象
    CtorchScheduler(const CtorchScheduler&);
    // 禁止赋值重载：防止通过“赋值”创建新对象
    CtorchScheduler& operator=(const CtorchScheduler&) = delete;
    std::mutex mutex_;
    bool if_first = true;
    // kernel映射表：OpType → DeviceType → BinaryKernelFunc（双输入算子）
    std::unordered_map<op, std::unordered_map<DeviceType, BinaryKernelFunc>> binary_kernel_map_;
    // 单输入算子映射表
    std::unordered_map<op, std::unordered_map<DeviceType, UnaryKernelFunc>> unary_kernel_map_;

    // 私有方法：初始化kernel映射表（注册所有kernel）
    void initKernelMap() {
        // ================= 双输入算子注册 =================
        binary_kernel_map_[op::Add][DeviceType::kCPU] = Add_BASIC_kernel;
        binary_kernel_map_[op::Sub][DeviceType::kCPU] = Sub_BASIC_kernel;
        binary_kernel_map_[op::Mul][DeviceType::kCPU] = Mul_BASIC_kernel;
        binary_kernel_map_[op::Div][DeviceType::kCPU] = Div_BASIC_kernel;
        binary_kernel_map_[op::MatMul][DeviceType::kCPU] = MatMul_BASIC_kernel;
        binary_kernel_map_[op::MatMul][DeviceType::kAMX] = MatMul_AMX_kernel;
        binary_kernel_map_[op::Dot][DeviceType::kCPU] = Dot_BASIC_kernel;
        binary_kernel_map_[op::MSE][DeviceType::kCPU] = MSE_BASIC_kernel;
        binary_kernel_map_[op::CE][DeviceType::kCPU] = CrossEntropy_BASIC_kernel;
        binary_kernel_map_[op::MAE][DeviceType::kCPU] = MAE_BASIC_kernel;
        
        // ================= 单输入算子注册 =================
        unary_kernel_map_[op::Neg][DeviceType::kCPU] = Neg_BASIC_kernel;
        unary_kernel_map_[op::ReLU][DeviceType::kCPU] = ReLU_BASIC_kernel;
        unary_kernel_map_[op::Cos][DeviceType::kCPU] = Cos_BASIC_kernel;
        unary_kernel_map_[op::Sin][DeviceType::kCPU] = Sin_BASIC_kernel;
        unary_kernel_map_[op::Tanh][DeviceType::kCPU] = Tanh_BASIC_kernel;
        unary_kernel_map_[op::Sigmoid][DeviceType::kCPU] = Sigmoid_BASIC_kernel;
        unary_kernel_map_[op::Softmax][DeviceType::kCPU] = Softmax_BASIC_kernel;
        
        // LReLU算子 - 仅注册映射关系，不绑定具体函数
        unary_kernel_map_[op::LReLU];
    }
public:
    static CtorchScheduler& getInstance() {
        static CtorchScheduler instance_;
        std::lock_guard<std::mutex> lock(instance_.mutex_);
        if (instance_.if_first) {
            printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "[%s %" PRIu64 "] Ctorch Scheduler Started\n", getFormattedTimeMs().c_str(), getTimestampMs());
            instance_.if_first = false;
        }
        return instance_;
    }


    // 辅助函数1：检测设备是否可用（简化版，后续可扩展
     static bool isDeviceAvailable(DeviceType dev_type) {
        switch (dev_type) {
            case DeviceType::kCPU: return true; // CPU必可用
            case DeviceType::kCUDA: return false; // 后续实现后改为true
            case DeviceType::kMPS: return false;
            case DeviceType::kAMX: return true; // Apple Silicon Accelerate可用
            default: return false;
        }
    }

    // 辅助函数：获取输入张量设备
    static DeviceType getTargetDevice(const Tensor& a, const Tensor& b) {
        if (a.device() != b.device()) {
            CtorchError::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::DEVICE_COMPAT,"Ctorch_Scheduler: Tensor不在同一平台");
        }
        return a.device();
    }

// 公共接口实现：dispatch（双输入算子）
    Tensor dispatch(const Tensor& a, const Tensor& b, op op_type) {
        // 1. 参数合法性校验（dtype一致）
        if (a.dtype() != b.dtype()) {
            CtorchError::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::DATATYPE,"Ctorch_Scheduler: Tensor类型不一致");
        }
        // 对于加法、乘法、减法、除法、交叉熵和矩阵乘法操作，允许形状不同（支持广播、不同标签格式和矩阵乘法维度要求）
        if (op_type != op::Add && op_type != op::Mul && op_type != op::Sub && op_type != op::Div && op_type != op::CE && op_type != op::MatMul && a.sizes() != b.sizes()) {
            CtorchError::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::DIMENSION,"Ctorch_Scheduler: Tensor形状不一致");
        }

        // 获取调度器实例，初始化kernel映射表（仅首次调用）
        CtorchScheduler &instance = getInstance();
        {
            std::lock_guard<std::mutex> lock(instance.mutex_);
            static bool kernel_map_inited = false;
            if (!kernel_map_inited) {
                instance.initKernelMap();
                kernel_map_inited = true;
            }
        }

        // 确定目标设备，查找可用kernel
        DeviceType target_dev = getTargetDevice(a, b);
        BinaryKernelFunc target_kernel = nullptr;

        {
            std::lock_guard<std::mutex> lock(instance.mutex_);
            // 从映射表中查找对应kernel
            auto op_it = instance.binary_kernel_map_.find(op_type);
            if (op_it != instance.binary_kernel_map_.end()) {
                DeviceType search_dev = target_dev;
                
                // 对于 MatMul，如果 target_dev 是 CPU 但 AMX 可用，则使用 AMX
                if (op_type == op::MatMul && target_dev == DeviceType::kCPU && isDeviceAvailable(DeviceType::kAMX)) {
                    auto amx_it = op_it->second.find(DeviceType::kAMX);
                    if (amx_it != op_it->second.end()) {
                        search_dev = DeviceType::kAMX;
                    }
                }
                
                auto dev_it = op_it->second.find(search_dev);
                if (dev_it != op_it->second.end() && isDeviceAvailable(search_dev)) {
                    target_kernel = dev_it->second;
                }
            }
        }

        // 未找到kernel则抛异常
        if (target_kernel == nullptr) {
            CtorchError::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的Kernel");
        }
        // 调用kernel，执行计算并返回结果
        Tensor result = target_kernel(a, b);
        
        // 记录操作到计算图（使用AutoGrad）
        result.requires_grad(true);
        auto result_ptr = std::make_shared<Tensor>(result);
        // 根据op_type注册对应的节点
        if (AutoGrad::EnableGrad) {

            std::weak_ptr<Tensor> result_weak = result_ptr;
            switch (op_type) {
            case op::Add:
                AutoGrad::registerNode<AddNode>(a, b, result_weak);
                break;
            case op::Sub:
                AutoGrad::registerNode<SubNode>(a, b, result_weak);
                break;
            case op::Mul:
                AutoGrad::registerNode<MulNode>(a, b, result_weak);
                break;
            case op::Div:
                AutoGrad::registerNode<DivNode>(a, b, result_weak);
                break;
            case op::MatMul:
                AutoGrad::registerNode<MatMulNode>(a, b, result_weak);
                break;
            case op::CE:
                AutoGrad::registerNode<CrossEntropyNode>(a, b, result_weak);
                break;
            default:
                break;
            }
            if (result_ptr->getRelatedNode()) {
                result.setRelatedNode(result_ptr->getRelatedNode());
            }
        }

        return result;
    }
    
    // 公共接口实现：dispatch（单输入算子）
    Tensor dispatch(const Tensor& a, op op_type) {
        // 获取调度器实例，初始化kernel映射表（仅首次调用）
        CtorchScheduler &instance = getInstance();
        {
            std::lock_guard<std::mutex> lock(instance.mutex_);
            static bool kernel_map_inited = false;
            if (!kernel_map_inited) {
                instance.initKernelMap();
                kernel_map_inited = true;
            }
        }

        // 确定目标设备，查找可用kernel
        DeviceType target_dev = a.device();
        UnaryKernelFunc target_kernel = nullptr;

        {
            std::lock_guard<std::mutex> lock(instance.mutex_);
            // 从映射表中查找对应kernel
            auto op_it = instance.unary_kernel_map_.find(op_type);
            if (op_it != instance.unary_kernel_map_.end()) {
                auto dev_it = op_it->second.find(target_dev);
                if (dev_it != op_it->second.end() && isDeviceAvailable(target_dev)) {
                    target_kernel = dev_it->second;
                }
            }
        }

        // 未找到kernel则抛异常
        if (target_kernel == nullptr) {
            CtorchError::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的Kernel");
        }
        // 调用kernel，执行计算并返回结果
        Tensor result = target_kernel(a);
        
        // 记录操作到计算图（使用AutoGrad）
        if (a.requires_grad()) {
            result.requires_grad(true);
            // 根据op_type注册对应的节点
            auto result_ptr = std::make_shared<Tensor>(result);
            std::weak_ptr<Tensor> result_weak = result_ptr;
            std::vector<Tensor> inputs = {a};
            switch (op_type) {
                case op::Neg:
                    AutoGrad::registerNode<NegNode>(inputs, result_weak);
                    break;
                case op::ReLU:
                    AutoGrad::registerNode<ReLUNode>(inputs, result_weak);
                    break;
                case op::Cos:
                    AutoGrad::registerNode<CosNode>(inputs, result_weak);
                    break;
                case op::Sin:
                    AutoGrad::registerNode<SinNode>(inputs, result_weak);
                    break;
                case op::Tanh:
                    AutoGrad::registerNode<TanhNode>(inputs, result_weak);
                    break;
                case op::Sigmoid:
                    AutoGrad::registerNode<SigmoidNode>(inputs, result_weak);
                    break;
                case op::Softmax:
                    AutoGrad::registerNode<SoftmaxNode>(inputs, result_weak);
                    break;
                default:
                    break;
            }
            
            // 将新创建的result_ptr中的_node属性复制到原始的result张量中
            if (result_ptr->getRelatedNode()) {
                result.setRelatedNode(result_ptr->getRelatedNode());
            }
        }
        
        return result;
    }

};
#endif //CTORCH_SCHEDULER_H
