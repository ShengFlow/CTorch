module;                                                                   
#include <mutex>                                                         
#include <unordered_map>                                                 
import Ctools;                                                          
import Tensor;                                                          
import Ctorch_Error;                                                    

export module Ctorch_Scheduler;                                         

export typedef Tensor (*BinaryKernelFunc)(const Tensor& a, const Tensor& b);
export typedef Tensor (*UnaryKernelFunc)(const Tensor& a);

export Tensor Add_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor Sub_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor Mul_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor Div_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor MatMul_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor Dot_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor MSE_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor CrossEntropy_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor MAE_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor Neg_BASIC_kernel(const Tensor& a);
export Tensor ReLU_BASIC_kernel(const Tensor& a);
export Tensor Cos_BASIC_kernel(const Tensor& a);
export Tensor Sin_BASIC_kernel(const Tensor& a);
export Tensor Tanh_BASIC_kernel(const Tensor& a);
export Tensor Sigmoid_BASIC_kernel(const Tensor& a);
export Tensor Softmax_BASIC_kernel(const Tensor& a);

export class Ctorch_Scheduler {
private:
    Ctorch_Scheduler() = default;
    Ctorch_Scheduler(const Ctorch_Scheduler&);
    Ctorch_Scheduler& operator=(const Ctorch_Scheduler&) = delete;
    std::mutex mutex_;
    bool if_first = true;
    std::unordered_map<op, std::unordered_map<DeviceType, BinaryKernelFunc>> binary_kernel_map_;
    std::unordered_map<op, std::unordered_map<DeviceType, UnaryKernelFunc>> unary_kernel_map_;

    void initKernelMap() {
        binary_kernel_map_[op::Add][DeviceType::kCPU] = Add_BASIC_kernel;
        binary_kernel_map_[op::Sub][DeviceType::kCPU] = Sub_BASIC_kernel;
        binary_kernel_map_[op::Mul][DeviceType::kCPU] = Mul_BASIC_kernel;
        binary_kernel_map_[op::Div][DeviceType::kCPU] = Div_BASIC_kernel;
        binary_kernel_map_[op::MatMul][DeviceType::kCPU] = MatMul_BASIC_kernel;
        binary_kernel_map_[op::Dot][DeviceType::kCPU] = Dot_BASIC_kernel;
        binary_kernel_map_[op::MSE][DeviceType::kCPU] = MSE_BASIC_kernel;
        binary_kernel_map_[op::CE][DeviceType::kCPU] = CrossEntropy_BASIC_kernel;
        binary_kernel_map_[op::MAE][DeviceType::kCPU] = MAE_BASIC_kernel;
        
        unary_kernel_map_[op::Neg][DeviceType::kCPU] = Neg_BASIC_kernel;
        unary_kernel_map_[op::ReLU][DeviceType::kCPU] = ReLU_BASIC_kernel;
        unary_kernel_map_[op::Cos][DeviceType::kCPU] = Cos_BASIC_kernel;
        unary_kernel_map_[op::Sin][DeviceType::kCPU] = Sin_BASIC_kernel;
        unary_kernel_map_[op::Tanh][DeviceType::kCPU] = Tanh_BASIC_kernel;
        unary_kernel_map_[op::Sigmoid][DeviceType::kCPU] = Sigmoid_BASIC_kernel;
        unary_kernel_map_[op::Softmax][DeviceType::kCPU] = Softmax_BASIC_kernel;
        
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

    static bool isDeviceAvailable(DeviceType dev_type) {
        switch (dev_type) {
            case DeviceType::kCPU: return true;
            case DeviceType::kCUDA: return false;
            case DeviceType::kMPS: return false;
            case DeviceType::kAMX: return false;
            default: return false;
        }
    }

    static DeviceType getTargetDevice(const Tensor& a, const Tensor& b) {
        if (a.device() != b.device()) {
            Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT, "Ctorch_Scheduler: Tensor不在同一平台");
        }
        return a.device();
    }

    Tensor dispatch(const Tensor& a, const Tensor& b, op op_type) {
        if (a.sizes() != b.sizes()) {
            Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "Ctorch_Scheduler: Tensor形状不一致");
        }
        if (a.dtype() != b.dtype()) {
            Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "Ctorch_Scheduler: Tensor类型不一致");
        }

        Ctorch_Scheduler &instance = getInstance();
        std::lock_guard<std::mutex> lock(instance.mutex_);
        static bool kernel_map_inited = false;
        if (!kernel_map_inited) {
            instance.initKernelMap();
            kernel_map_inited = true;
        }

        DeviceType target_dev = getTargetDevice(a, b);
        BinaryKernelFunc target_kernel = nullptr;

        auto op_it = instance.binary_kernel_map_.find(op_type);
        if (op_it != instance.binary_kernel_map_.end()) {
            auto dev_it = op_it->second.find(target_dev);
            if (dev_it != op_it->second.end() && isDeviceAvailable(target_dev)) {
                target_kernel = dev_it->second;
            }
        }

        if (target_kernel == nullptr) {
            Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API, "Ctorch_Scheduler: 没有可用的Kernel");
        }
        return target_kernel(a, b);
    }
    
    Tensor dispatch(const Tensor& a, op op_type) {
        Ctorch_Scheduler &instance = getInstance();
        std::lock_guard<std::mutex> lock(instance.mutex_);
        static bool kernel_map_inited = false;
        if (!kernel_map_inited) {
            instance.initKernelMap();
            kernel_map_inited = true;
        }

        DeviceType target_dev = a.device();
        UnaryKernelFunc target_kernel = nullptr;

        auto op_it = instance.unary_kernel_map_.find(op_type);
        if (op_it != instance.unary_kernel_map_.end()) {
            auto dev_it = op_it->second.find(target_dev);
            if (dev_it != op_it->second.end() && isDeviceAvailable(target_dev)) {
                target_kernel = dev_it->second;
            }
        }

        if (target_kernel == nullptr) {
            Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API, "Ctorch_Scheduler: 没有可用的Kernel");
        }
        return target_kernel(a);
    }
};