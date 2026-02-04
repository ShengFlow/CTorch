/**
 * @file Ctorch_Scheduler.h
 * @brief Ctorch 框架的核心调度器类
 * @details 采用单例模式实现，负责管理所有 kernel 映射关系，根据算子类型和设备类型，
 * 自动查找并调用对应的 kernel，实现 kernel 的统一调度。
 * @author GhostFace
 * @date 2025/12/20
 */
#ifndef CTORCH_SCHEDULER_H
#define CTORCH_SCHEDULER_H
#include "Ctorch_Error.h"
#include "Tensor.h"
#include "kernels/kernels.h"

class Ctorch_Scheduler{
private:
    Ctorch_Scheduler() = default;
    // 禁止拷贝构造：防止通过“实例拷贝”创建新对象
    Ctorch_Scheduler(const Ctorch_Scheduler&);
    // 禁止赋值重载：防止通过“赋值”创建新对象
    Ctorch_Scheduler& operator=(const Ctorch_Scheduler&) = delete;
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
        binary_kernel_map_[op::Dot][DeviceType::kCPU] = Dot_BASIC_kernel;
        
        // ================= 单输入算子注册 =================
        unary_kernel_map_[op::Neg][DeviceType::kCPU] = Neg_BASIC_kernel;
        unary_kernel_map_[op::ReLU][DeviceType::kCPU] = ReLU_BASIC_kernel;
        unary_kernel_map_[op::Cos][DeviceType::kCPU] = Cos_BASIC_kernel;
        unary_kernel_map_[op::Sin][DeviceType::kCPU] = Sin_BASIC_kernel;
        
        // Tanh算子 - 仅注册映射关系，不绑定具体函数
        unary_kernel_map_[op::Tanh];
        
        // Sigmoid算子 - 仅注册映射关系，不绑定具体函数
        unary_kernel_map_[op::Sigmoid];
        
        // Softmax算子 - 仅注册映射关系，不绑定具体函数
        unary_kernel_map_[op::Softmax];
        
        // LReLU算子 - 仅注册映射关系，不绑定具体函数
        unary_kernel_map_[op::LReLU];
    }
public:
    static Ctorch_Scheduler& getInstance() {
        static Ctorch_Scheduler instance_;
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
            case DeviceType::kAMX: return false;
            default: return false;
        }
    }

    // 辅助函数：获取输入张量设备
    static DeviceType getTargetDevice(const Tensor& a, const Tensor& b) {
        if (a.device() != b.device()) {
            Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::DEVICE_COMPAT,"Ctorch_Scheduler: Tensor不在同一平台");
        }
        return a.device();
    }

// 公共接口实现：dispatch（双输入算子）
    Tensor dispatch(const Tensor& a, const Tensor& b, op op_type) {
        // 1. 参数合法性校验（形状、dtype一致）
        if (a.sizes() != b.sizes()) {
            Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::DIMENSION,"Ctorch_Scheduler: Tensor形状不一致");
        }
        if (a.dtype() != b.dtype()) {
            Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::DATATYPE,"Ctorch_Scheduler: Tensor类型不一致");
        }

        // 获取调度器实例，初始化kernel映射表（仅首次调用）
        Ctorch_Scheduler &instance = getInstance();
        std::lock_guard<std::mutex> lock(instance.mutex_);
        static bool kernel_map_inited = false;
        if (!kernel_map_inited) {
            instance.initKernelMap();
            kernel_map_inited = true;
        }

        // 确定目标设备，查找可用kernel
        DeviceType target_dev = getTargetDevice(a, b);
        BinaryKernelFunc target_kernel = nullptr;

        // 从映射表中查找对应kernel
        auto op_it = instance.binary_kernel_map_.find(op_type);
        if (op_it != instance.binary_kernel_map_.end()) {
            auto dev_it = op_it->second.find(target_dev);
            if (dev_it != op_it->second.end() && isDeviceAvailable(target_dev)) {
                target_kernel = dev_it->second;
            }
        }

        // 未找到kernel则抛异常
        if (target_kernel == nullptr) {
            Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的Kernel");
        }
        // 调用kernel，执行计算并返回结果
        return target_kernel(a, b);
    }
    
    // 公共接口实现：dispatch（单输入算子）
    Tensor dispatch(const Tensor& a, op op_type) {
        // 获取调度器实例，初始化kernel映射表（仅首次调用）
        Ctorch_Scheduler &instance = getInstance();
        std::lock_guard<std::mutex> lock(instance.mutex_);
        static bool kernel_map_inited = false;
        if (!kernel_map_inited) {
            instance.initKernelMap();
            kernel_map_inited = true;
        }

        // 确定目标设备，查找可用kernel
        DeviceType target_dev = a.device();
        UnaryKernelFunc target_kernel = nullptr;

        // 从映射表中查找对应kernel
        auto op_it = instance.unary_kernel_map_.find(op_type);
        if (op_it != instance.unary_kernel_map_.end()) {
            auto dev_it = op_it->second.find(target_dev);
            if (dev_it != op_it->second.end() && isDeviceAvailable(target_dev)) {
                target_kernel = dev_it->second;
            }
        }

        // 未找到kernel则抛异常
        if (target_kernel == nullptr) {
            Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的Kernel");
        }
        // 调用kernel，执行计算并返回结果
        return target_kernel(a);
    }

};
#endif //CTORCH_SCHEDULER_H
