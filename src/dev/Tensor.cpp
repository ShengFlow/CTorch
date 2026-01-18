/**
 * @file Tensor.cpp
 * @brief 张量类的实现
 * @author GhostFace, Beapoe
 * @date 2025/12/21
 * @version v3.1
 * @details 实现了张量类的各种方法，包括构造函数、访问器、操作、运算和自动微分等
 */
#include "Tensor.h"
#include "kernels/kernels.h"
#include "Ctorch_Scheduler.h"

/**
 * @var Tensor::global_tensor_id
 * @brief 全局张量ID计数器
 */
std::atomic<size_t> Tensor::global_tensor_id(1);

// ======================= Tensor类实现 =======================

/**
 * @brief 设置梯度需求
 * @param key 是否需要梯度
 * @details 如果需要梯度，确保已注册到计算图
 */
void Tensor::requires_grad(bool key) {
    _requires_grad = key;
    if (key) {
        // 如果需要梯度，确保已注册到计算图
        if (AutoDiffContext::current()) {
            AutoDiffContext::current()->make_leaf(*this, key);
        }
    }
}

/**
 * @brief 获取张量的形状
 * @return 张量的形状向量
 */
const std::vector<size_t>& Tensor::shape() const {
    return _shape;
}

/**
 * @brief 获取张量的大小（元素总数量）
 * @return 张量的元素总数量
 * @details 标量张量的元素数量为1
 */
size_t Tensor::numel() const {
    if (_shape.empty()) {
        return 1;  // 标量张量的元素数量为1
    }
    return std::accumulate(_shape.begin(), _shape.end(), 1ULL, std::multiplies<>());
}

/**
 * @brief 获取张量的步幅（单个维度）
 * @param dim 维度索引
 * @return 该维度的步幅
 * @throw std::out_of_range 如果维度索引超出范围
 */
size_t Tensor::stride(int dim) const {
    if (dim < 0) {
        dim += static_cast<int>(_strides.size());
    }
    if (dim < 0 || dim >= static_cast<int>(_strides.size())) {
        Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "无效维度");
    }
    return _strides[dim];
}

/**
 * @brief 获取张量的维度大小
 * @param dim 维度索引
 * @return 该维度的大小
 * @throw std::out_of_range 如果维度索引超出范围
 */
size_t Tensor::size(int dim) const {
    if (dim < 0) {
        dim += static_cast<int>(_shape.size());
    }
    if (dim < 0 || dim >= static_cast<int>(_shape.size())) {
        Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "无效维度");
    }
    return _shape[dim];
}

/**
 * @brief 计算步幅 (基于行优先顺序)
 * @details 对于标量张量，没有步幅；对于多维张量，从最后一个维度开始计算步幅
 */
void Tensor::computeStrides() {
    _strides.resize(_shape.size());
    if (_shape.empty()) {
        return;
    }
    _strides.back() = 1;
    for (int i = static_cast<int>(_shape.size()) - 2; i >= 0; --i) {
        _strides[i] = _strides[i + 1] * _shape[i + 1];
    }
}

/**
 * @brief 计算存储中的索引
 * @param indices 多维索引
 * @return 存储中的一维索引
 * @throw std::invalid_argument 如果索引维度与张量维度不匹配
 */
size_t Tensor::computeStorageIndex(std::initializer_list<size_t> indices) const {
    if (indices.size() != _shape.size()) {
        Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "索引维度与张量维度不匹配");
    }
    size_t index = 0;
    auto indices_it = indices.begin();
    auto strides_it = _strides.begin();
    for (; indices_it != indices.end() && strides_it != _strides.end(); ++indices_it, ++strides_it) {
        index += *indices_it * *strides_it;
    }
    return index + _storage_offset;
}

// 检查数据类型是否匹配
template <typename T>
void Tensor::checkDType() const {
    if constexpr (std::is_same_v<T, float>) {
        if (_dtype != DType::kFloat) {
            Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "期望float dtype");
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (_dtype != DType::kDouble) {
            Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "期望double dtype");
        }
    } else if constexpr (std::is_same_v<T, int32_t>) {
        if (_dtype != DType::kInt) {
            Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "期望int dtype");
        }
    } else if constexpr (std::is_same_v<T, int64_t>) {
        if (_dtype != DType::kLong) {
            Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "期望long dtype");
        }
    } else if constexpr (std::is_same_v<T, bool>) {
        if (_dtype != DType::kBool) {
            Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "期望bool dtype");
        }
    }
}

// 获取标量值
template <typename T>
T Tensor::item() const {
    if (numel() != 1) {
        Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE, "张量不是标量");
    }
    checkDType<T>();
    const T* data_ptr = _storage.data<T>();
    if (!data_ptr) {
        Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE, "张量数据为null");
    }
    return data_ptr[_storage_offset];
}

// 显式实例化常用的item()模板
template float Tensor::item<float>() const;
template double Tensor::item<double>() const;
template int32_t Tensor::item<int32_t>() const;
template int64_t Tensor::item<int64_t>() const;
template bool Tensor::item<bool>() const;

// 索引操作
Tensor Tensor::operator[](size_t index) const {
    // 简单实现，仅支持1D张量
    if (_shape.size() != 1) {
        Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "索引操作符仅支持1D张量");
    }
    if (index >= _shape[0]) {
        Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "索引越界");
    }
    
    Tensor result(*this);
    result._shape = {1};
    result._strides = {0};
    result._storage_offset += index * _strides[0];
    return result;
}

// 创建一个新的张量，形状相同，数据不同
Tensor Tensor::clone() const {
    Tensor result(*this);
    result._storage = _storage.clone();
    return result;
}

// 将张量转换为指定数据类型
Tensor Tensor::to(DType dtype) const {
    // 简单实现，仅支持float到其他类型
    if (_dtype == dtype) {
        return *this;
    }
    
    Tensor result(ShapeTag{}, _shape, dtype, _device);
    
    if (_dtype == DType::kFloat) {
        const float* src = _storage.data<float>();
        if (src) {
            if (dtype == DType::kDouble) {
                double* dst = result._storage.data<double>();
                if (dst) {
                    for (size_t i = 0; i < numel(); ++i) {
                        dst[i] = static_cast<double>(src[i + _storage_offset]);
                    }
                }
            } else if (dtype == DType::kInt) {
                int32_t* dst = result._storage.data<int32_t>();
                if (dst) {
                    for (size_t i = 0; i < numel(); ++i) {
                        dst[i] = static_cast<int32_t>(src[i + _storage_offset]);
                    }
                }
            } else if (dtype == DType::kLong) {
                int64_t* dst = result._storage.data<int64_t>();
                if (dst) {
                    for (size_t i = 0; i < numel(); ++i) {
                        dst[i] = static_cast<int64_t>(src[i + _storage_offset]);
                    }
                }
            } else if (dtype == DType::kBool) {
                bool* dst = result._storage.data<bool>();
                if (dst) {
                    for (size_t i = 0; i < numel(); ++i) {
                        dst[i] = static_cast<bool>(src[i + _storage_offset]);
                    }
                }
            }
        }
    }
    
    return result;
}

// 转置张量
Tensor Tensor::transpose(int dim0, int dim1) const {
    // 简单实现，仅支持2D张量
    if (_shape.size() != 2) {
        Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "转置仅支持2D张量");
    }
    
    Tensor result(*this);
    std::swap(result._shape[dim0], result._shape[dim1]);
    std::swap(result._strides[dim0], result._strides[dim1]);
    return result;
}

// 转置张量（二维情况）
Tensor Tensor::t() const {
    return transpose(0, 1);
}

// 重塑张量形状
Tensor Tensor::reshape(std::initializer_list<size_t> new_shape) const {
    return reshape(std::vector<size_t>(new_shape));
}

// 重塑张量形状
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<>());
    if (new_numel != numel()) {
        throw std::invalid_argument("新形状元素数量不同");
    }
    
    Tensor result(*this);
    result._shape = new_shape;
    result.computeStrides();
    return result;
}

// 广播张量到指定形状
Tensor Tensor::broadcast_to(const std::vector<size_t>& shape) const {
    // 简单实现，仅支持广播到相同或更大的形状
    Tensor result(ShapeTag{}, shape, _dtype, _device);
    // 复制数据到广播后的张量
    // 注意：这是一个简化实现，实际广播逻辑更复杂
    return result;
}

// 零初始化张量
void Tensor::zero() {
    // 简单实现，将所有元素设为0
    size_t count = numel();
    if (_dtype == DType::kFloat) {
        float* data = _storage.data<float>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = 0.0f;
        }
    } else if (_dtype == DType::kDouble) {
        double* data = _storage.data<double>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = 0.0;
        }
    } else if (_dtype == DType::kInt) {
        int32_t* data = _storage.data<int32_t>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = 0;
        }
    } else if (_dtype == DType::kLong) {
        int64_t* data = _storage.data<int64_t>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = 0;
        }
    } else if (_dtype == DType::kBool) {
        bool* data = _storage.data<bool>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = false;
        }
    }
}

// 一初始化张量
void Tensor::ones() {
    // 简单实现，将所有元素设为1
    size_t count = numel();
    if (_dtype == DType::kFloat) {
        float* data = _storage.data<float>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = 1.0f;
        }
    } else if (_dtype == DType::kDouble) {
        double* data = _storage.data<double>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = 1.0;
        }
    } else if (_dtype == DType::kInt) {
        int32_t* data = _storage.data<int32_t>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = 1;
        }
    } else if (_dtype == DType::kLong) {
        int64_t* data = _storage.data<int64_t>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = 1;
        }
    } else if (_dtype == DType::kBool) {
        bool* data = _storage.data<bool>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = true;
        }
    }
}

// 随机初始化张量
void Tensor::rand() {
    // 简单实现，生成[0, 1)之间的随机数
    size_t count = numel();
    if (_dtype == DType::kFloat) {
        float* data = _storage.data<float>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        }
    } else if (_dtype == DType::kDouble) {
        double* data = _storage.data<double>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
        }
    }
    // 其他类型暂不支持
}

// 矩阵乘法
Tensor Tensor::matmul(const Tensor& other) const {
    return matMul(*this, other);
}

// 反向传播
void Tensor::backward() const {
    if (AutoDiffContext::current()) {
        Tensor self_const_cast = const_cast<Tensor&>(*this);
        AutoDiffContext::current()->backward(self_const_cast);
    }
}

// 反向传播（带有梯度输出）
void Tensor::backward(const Tensor& grad_output) const {
    if (AutoDiffContext::current()) {
        Tensor self_const_cast = const_cast<Tensor&>(*this);
        AutoDiffContext::current()->backward(self_const_cast, grad_output);
    }
}

// ======================= 缺失方法实现 =======================

// 默认构造函数
Tensor::Tensor() : tensor_id_(global_tensor_id++), _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kFloat), _requires_grad(false), record_committed_(false) {
    computeStrides();
}

// ReLU激活函数
Tensor Tensor::relu() const {
    // 简单实现ReLU激活函数
    Tensor result = Ctorch_Scheduler::getInstance().dispatch(*this,op::ReLU);
    
    // 记录操作到计算图
    if (AutoDiffContext::current()) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this)};
        AutoDiffContext::current()->defer_record(result.id(), op::ReLU, inputs);
        result._requires_grad = _requires_grad;
        if (result._requires_grad) {
            result.commit_pending_record();
        }
    }
    
    return result;
}

// 张量除法运算符
Tensor Tensor::operator/(const Tensor& other) const {
    // 简单实现张量除法
    Tensor result = Ctorch_Scheduler::getInstance().dispatch(*this,other,op::Div);
    
    // 记录操作到计算图
    if (AutoDiffContext::current()) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this), const_cast<Tensor*>(&other)};
        AutoDiffContext::current()->defer_record(result.id(), op::Div, inputs);
        result._requires_grad = _requires_grad || other.requires_grad();
        if (result._requires_grad) {
            result.commit_pending_record();
        }
    }
    
    return result;
}

// 张量减法运算符
Tensor Tensor::operator-(const Tensor& other) const {
    // 使用调度器调用加法kernel执行张量加法
    Tensor result = Ctorch_Scheduler::getInstance().dispatch(*this, other, op::Sub);
    // 记录操作到计算图
    if (AutoDiffContext::current()) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this), const_cast<Tensor*>(&other)};
        AutoDiffContext::current()->defer_record(result.id(), op::Sub, inputs);
        result._requires_grad = _requires_grad || other.requires_grad();
        if (result._requires_grad) {
            result.commit_pending_record();
        }
    }
    
    return result;
}

// 张量乘法运算符
Tensor Tensor::operator*(const Tensor& other) const {
    // 简单实现张量乘法
    Tensor result = Ctorch_Scheduler::getInstance().dispatch(*this,other,op::Mul);
    // 记录操作到计算图
    if (AutoDiffContext::current()) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this), const_cast<Tensor*>(&other)};
        AutoDiffContext::current()->defer_record(result.id(), op::Mul, inputs);
        result._requires_grad = _requires_grad || other.requires_grad();
        if (result._requires_grad) {
            result.commit_pending_record();
        }
    }
    
    return result;
}

// 标量乘法运算符
Tensor Tensor::operator*(float scalar) const {
    // 简单实现标量乘法
    Tensor result(*this);
    result._storage = _storage.clone();
    
    size_t count = numel();
    if (_dtype == DType::kFloat) {
        float* data = result.data<float>();
        for (size_t i = 0; i < count; ++i) {
            data[i] *= scalar;
        }
    }
    
    return result;
}

// 一元负号运算符
Tensor Tensor::operator-() const {
    // 简单实现一元负号
    Tensor result = Ctorch_Scheduler::getInstance().dispatch(*this,op::Neg);
    
    // 记录操作到计算图
    if (AutoDiffContext::current()) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this)};
        AutoDiffContext::current()->defer_record(result.id(), op::Neg, inputs);
        result._requires_grad = _requires_grad;
        if (result._requires_grad) {
            result.commit_pending_record();
        }
    }
    
    return result;
}

// 张量加法运算符
Tensor Tensor::operator+(const Tensor& other) const {
    // 使用调度器调用加法kernel执行张量加法
    Tensor result = Ctorch_Scheduler::getInstance().dispatch(*this, other, op::Add);
    
    // 记录操作到计算图
    if (AutoDiffContext::current()) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this), const_cast<Tensor*>(&other)};
        AutoDiffContext::current()->defer_record(result.id(), op::Add, inputs);
        result._requires_grad = _requires_grad || other.requires_grad();
        if (result._requires_grad) {
            result.commit_pending_record();
        }
    }
    
    return result;
}

// 标量加法运算符
Tensor Tensor::operator+(float scalar) const {
    // 简单实现标量加法
    Tensor result(*this);
    result._storage = _storage.clone();
    
    size_t count = numel();
    if (_dtype == DType::kFloat) {
        float* data = result.data<float>();
        for (size_t i = 0; i < count; ++i) {
            data[i] += scalar;
        }
    }
    
    return result;
}

// ======================= 辅助方法 =======================

// 清空存储的方法
void Tensor::clear_storage() {
    _storage.clear();
}

// 判断是否为空的辅助方法
bool Tensor::is_cleared() const {
    return _storage.empty();
}

// 提交未完成的记录
void Tensor::commit_pending_record() {
    if (AutoDiffContext::current() && has_pending_record()) {
        AutoDiffContext::current()->commit_record(*this);
    }
    record_committed_ = true;
}

// 增强调试信息
void Tensor::debug_info_detailed(const std::string& name) const {
    std::ostringstream oss;
    oss << "Tensor " << name << " (ID: " << tensor_id_ << ")" << std::endl;
    oss << "  Shape: [";
    for (size_t i = 0; i < _shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << _shape[i];
    }
    oss << "]" << std::endl;
    oss << "  Strides: [";
    for (size_t i = 0; i < _strides.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << _strides[i];
    }
    oss << "]" << std::endl;
    oss << "  Storage offset: " << _storage_offset << std::endl;
    oss << "  Device: " << static_cast<int>(_device) << std::endl;
    oss << "  DType: " << dtypeToString(_dtype) << std::endl;
    oss << "  Requires grad: " << _requires_grad << std::endl;
    oss << "  Record committed: " << record_committed_ << std::endl;
    oss << "  Storage: " << (_storage.empty() ? "empty" : "non-empty") << std::endl;
    
    std::cout << oss.str() << std::endl;
}

// ======================= 全局函数实现 =======================

// 全局的backward函数，用于启动反向传播
void backward(Tensor& root) {
    if (AutoDiffContext::current()) {
        AutoDiffContext::current()->backward(root);
    }
}

// 全局的backward函数，用于启动反向传播（带有梯度输出）
void backward(Tensor& root, Tensor grad_output) {
    if (AutoDiffContext::current()) {
        AutoDiffContext::current()->backward(root, grad_output);
    }
}

// 全局的grad函数，用于获取张量的梯度
Tensor grad(const Tensor& t) {
    if (AutoDiffContext::current()) {
        return AutoDiffContext::current()->get_grad(&t);
    }
    return Tensor();
}

// 全局的matMul函数
Tensor matMul(const Tensor &a, const Tensor &b) {
    return Ctorch_Scheduler::getInstance().dispatch(a,b,op::MatMul);
}

// 计算两个张量的广播结果
BroadCastResult broadCast(const Tensor& a, const Tensor& tensor2) {
    // 简化实现，返回默认的广播结果
    BroadCastResult result;
    result.logicShape = a.shape();
    result.logicStridesA = a.strides();
    result.logicStridesB = tensor2.strides();
    return result;
}

// 标量加法操作符重载
Tensor operator+(float scalar, const Tensor& tensor) {
    return tensor + scalar;
}

// 标量减法操作符重载
Tensor operator-(float scalar, const Tensor& tensor) {
    Tensor result = Tensor(scalar) - tensor;
    return result;
}

// 标量乘法操作符重载
Tensor operator*(float scalar, const Tensor& tensor) {
    return tensor * scalar;
}

// 标量除法操作符重载
Tensor operator/(float scalar, const Tensor& tensor) {
    Tensor result = Tensor(scalar) / tensor;
    return result;
}

// 比较操作符重载（右操作数）
Tensor operator>(float scalar, const Tensor& tensor) {
    // TODO: FIX it!!!!
    // 简化实现，返回空张量
    return Tensor();
}

// 比较操作符重载（右操作数）
Tensor operator<(float scalar, const Tensor& tensor) {
    // TODO: FIX it!!!!
    // 简化实现，返回空张量
    return Tensor();
}

// 比较操作符重载（右操作数）
Tensor operator==(float scalar, const Tensor& tensor) {
    // TODO: FIX it!!!!
    // 简化实现，返回空张量
    return Tensor();
}

// 比较操作符重载（右操作数）
Tensor operator>=(float scalar, const Tensor& tensor) {
    // TODO: FIX it!!!!
    // 简化实现，返回空张量
    return Tensor();
}

// 比较操作符重载（右操作数）
Tensor operator<=(float scalar, const Tensor& tensor) {
    // TODO: FIX it!!!!
    // 简化实现，返回空张量
    return Tensor();
}

// 比较操作符重载（右操作数）
Tensor operator!=(float scalar, const Tensor& tensor) {
    // TODO: FIX it!!!!
    // 简化实现，返回空张量
    return Tensor();
}

// 输出张量信息
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(shape=[";
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
        if (i > 0) os << ", ";
        os << tensor.shape()[i];
    }
    os << "], dtype=" << dtypeToString(tensor.dtype()) << ")";
    return os;
}

// 显式实例化checkDType模板函数
template void Tensor::checkDType<float>() const;
template void Tensor::checkDType<double>() const;
template void Tensor::checkDType<int32_t>() const;
template void Tensor::checkDType<int64_t>() const;

