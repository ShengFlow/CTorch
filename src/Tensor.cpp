/**
 * @file Tensor.cpp
 * @brief 张量类的实现
 * @author GhostFace, Beapoe
 * @date 2025/12/21
 * @version v3.1
 * @details 实现了张量类的各种方法，包括构造函数、访问器、操作、运算和自动微分等
 */
#include "../include/Tensor.h"
#include "kernels/kernels.h"
#include "../include/AutoGrad.h"
#include "../include/AutoGrad/Nodes/GradAccumulator.h"
#include "../include/DeviceAllocator.h"
#include "../include/CtorchScheduler.h"

#include <random>
#include <cmath>
#include <cstring>

/**
 * @var Tensor::global_tensor_id
 * @brief 全局张量ID计数器
 */
std::atomic<size_t> Tensor::global_tensor_id(1);

// ======================= Tensor类实现 =======================

std::shared_ptr<Node> Tensor::createGradAccumulator(const std::shared_ptr<Tensor> &self) {
    return std::make_shared<GradAccumulator>(self);
}

/**
 * @brief 设置梯度需求
 * @param key 是否需要梯度
 * @details 如果需要梯度，确保已注册到计算图
 */
void Tensor::requires_grad(bool key) {
    _autograd_meta._requires_grad = key;
    if (key && !_autograd_meta._node) {
        auto grad_accumulator = std::make_shared<GradAccumulator>(_autograd_meta._self);
        _autograd_meta._node  = grad_accumulator;
    }
}

/**
 * @brief 获取张量的形状
 * @return 张量的形状向量
 */
const std::vector<size_t> &Tensor::shape() const { return _shape; }

/**
 * @brief 获取张量的大小（元素总数量）
 * @return 张量的元素总数量
 * @details 标量张量的元素数量为1
 */
size_t Tensor::numel() const {
    if (_shape.empty()) {
        return 1; // 标量张量的元素数量为1
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
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "无效维度");
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
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "无效维度");
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
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                    "索引维度与张量维度不匹配");
    }
    size_t index    = 0;
    auto indices_it = indices.begin();
    auto strides_it = _strides.begin();
    for (; indices_it != indices.end() && strides_it != _strides.end();
         ++indices_it, ++strides_it) {
        index += *indices_it * *strides_it;
    }
    return index + _storage_offset;
}

// 检查数据类型是否匹配
template <typename T> void Tensor::checkDType() const {
    if constexpr (std::is_same_v<T, float>) {
        if (_dtype != DType::kFloat) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                        "期望float dtype");
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (_dtype != DType::kDouble) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                        "期望double dtype");
        }
    } else if constexpr (std::is_same_v<T, int32_t>) {
        if (_dtype != DType::kInt) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                        "期望int dtype");
        }
    } else if constexpr (std::is_same_v<T, int64_t>) {
        if (_dtype != DType::kLong) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                        "期望long dtype");
        }
    } else if constexpr (std::is_same_v<T, bool>) {
        if (_dtype != DType::kBool) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                        "期望bool dtype");
        }
    }
}

// 获取标量值
template <typename T> T Tensor::item() const {
    if (numel() != 1) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE,
                                    "张量不是标量");
    }

    // data_read() 在 MPS 路径会自动同步，无需在这里重复 flush。
    const T *data_ptr = data_read<T>();
    if (!data_ptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE,
                                    "张量数据为null");
    }
    return data_ptr[0];
}

// 显式实例化常用的item()模板
template float Tensor::item<float>() const;
template double Tensor::item<double>() const;
template int32_t Tensor::item<int32_t>() const;
template int64_t Tensor::item<int64_t>() const;
template bool Tensor::item<bool>() const;

void Tensor::check_inplace_safe_(const char *op_name) const {
    if (requires_grad()) {
        std::string msg = std::string(op_name) +
                          "(): in-place 操作不支持 requires_grad=true 的张量，" +
                          "请先 detach() 或 set_requires_grad(false)";
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE, msg);
    }
}

// 索引操作
Tensor Tensor::operator[](size_t index) const {
    // 简单实现，仅支持1D张量
    if (_shape.size() != 1) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                    "索引操作符仅支持1D张量");
    }
    if (index >= _shape[0]) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "索引越界");
    }

    Tensor result(*this);
    result._shape   = {1};
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

Tensor Tensor::to(DType dtype) const {
    if (_dtype == dtype) {
        return *this;
    }

    Tensor result(ShapeTag{}, _shape, dtype, _device);
    size_t n = numel();

    auto convert = [&](auto src_data) {
        if (!src_data)
            return;
        if (dtype == DType::kFloat) {
            float *dst = result._storage.data<float>();
            if (dst) {
                for (size_t i = 0; i < n; ++i) {
                    dst[i] = static_cast<float>(src_data[i + _storage_offset]);
                }
            }
        } else if (dtype == DType::kDouble) {
            double *dst = result._storage.data<double>();
            if (dst) {
                for (size_t i = 0; i < n; ++i) {
                    dst[i] = static_cast<double>(src_data[i + _storage_offset]);
                }
            }
        } else if (dtype == DType::kInt) {
            int32_t *dst = result._storage.data<int32_t>();
            if (dst) {
                for (size_t i = 0; i < n; ++i) {
                    dst[i] = static_cast<int32_t>(src_data[i + _storage_offset]);
                }
            }
        } else if (dtype == DType::kLong) {
            int64_t *dst = result._storage.data<int64_t>();
            if (dst) {
                for (size_t i = 0; i < n; ++i) {
                    dst[i] = static_cast<int64_t>(src_data[i + _storage_offset]);
                }
            }
        } else if (dtype == DType::kBool) {
            bool *dst = result._storage.data<bool>();
            if (dst) {
                for (size_t i = 0; i < n; ++i) {
                    dst[i] = static_cast<bool>(src_data[i + _storage_offset]);
                }
            }
        }
    };

    switch (_dtype) {
    case DType::kFloat:
        convert(_storage.data<float>());
        break;
    case DType::kDouble:
        convert(_storage.data<double>());
        break;
    case DType::kInt:
        convert(_storage.data<int32_t>());
        break;
    case DType::kLong:
        convert(_storage.data<int64_t>());
        break;
    case DType::kBool:
        convert(_storage.data<bool>());
        break;
    default:
        break;
    }

    return result;
}

// 转置张量
Tensor Tensor::transpose(int dim0, int dim1) const {
    // 检查维度索引是否有效
    if (dim0 < 0 || dim0 >= static_cast<int>(_shape.size()) || dim1 < 0 ||
        dim1 >= static_cast<int>(_shape.size())) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                    "转置维度索引超出范围");
    }

    Tensor result(*this);
    std::swap(result._shape[dim0], result._shape[dim1]);
    std::swap(result._strides[dim0], result._strides[dim1]);
    return result;
}

// 转置张量（二维情况）
Tensor Tensor::t() const { return transpose(0, 1); }

// 重塑张量形状
Tensor Tensor::reshape(std::initializer_list<size_t> new_shape) const {
    return reshape(std::vector<size_t>(new_shape));
}

// 重塑张量形状
Tensor Tensor::reshape(const std::vector<size_t> &new_shape) const {
    size_t new_numel =
        std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<>());
    if (new_numel != numel()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                    "新形状元素数量不同");
    }

    Tensor result(*this);
    result._shape = new_shape;
    result.computeStrides();
    return result;
}

/**
 * @brief 广播张量到指定形状
 * @details 实现标准的NumPy风格广播规则，支持完整的广播逻辑
 * @param shape 目标形状
 * @return 广播后的张量
 * @throw CtorchError 如果广播目标形状为空
 * @throw CtorchError 如果广播形状不兼容
 */
Tensor Tensor::broadcast_to(const std::vector<size_t> &shape) const {
    // 实现标准的NumPy风格广播规则

    // 步骤1：检查输入形状是否有效
    if (shape.empty()) {
        // 目标形状为空，返回标量张量
        return *this;
    }

    // 步骤2：计算广播后的形状和当前张量的扩展形状
    std::vector<size_t> current_shape = _shape;
    std::vector<size_t> target_shape  = shape;

    // 补全维度，确保两个张量的维度数相同
    while (current_shape.size() < target_shape.size()) {
        current_shape.insert(current_shape.begin(), 1);
    }
    while (target_shape.size() < current_shape.size()) {
        target_shape.insert(target_shape.begin(), 1);
    }

    // 步骤3：检查广播是否可行
    for (size_t i = 0; i < current_shape.size(); ++i) {
        if (current_shape[i] != target_shape[i] && current_shape[i] != 1) {
            std::ostringstream oss;
            oss << "广播形状不兼容: 当前形状 [";
            for (size_t j = 0; j < current_shape.size(); ++j) {
                if (j > 0)
                    oss << ", ";
                oss << current_shape[j];
            }
            oss << "], 目标形状 [";
            for (size_t j = 0; j < target_shape.size(); ++j) {
                if (j > 0)
                    oss << ", ";
                oss << target_shape[j];
            }
            oss << "], 在维度 " << i << " 不兼容 (当前: " << current_shape[i]
                << ", 目标: " << target_shape[i] << ")";
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, oss.str());
        }
    }

    Tensor result(ShapeTag{}, shape, _dtype, _device);

    size_t src_numel = numel();
    size_t dst_numel = result.numel();

    if (src_numel == 1) {
        size_t elem_size    = dtypeSize(_dtype);
        const char *src_ptr = _storage.data<char>() + _storage_offset * elem_size;
        char *dst_ptr       = result._storage.data<char>();
        for (size_t i = 0; i < dst_numel; ++i) {
            std::memcpy(dst_ptr + i * elem_size, src_ptr, elem_size);
        }
    } else {
        std::vector<size_t> src_strides(current_shape.size());
        src_strides.back() = 1;
        for (int i = static_cast<int>(current_shape.size()) - 2; i >= 0; --i) {
            src_strides[i] = src_strides[i + 1] * current_shape[i + 1];
        }

        std::vector<size_t> dst_strides(target_shape.size());
        dst_strides.back() = 1;
        for (int i = static_cast<int>(target_shape.size()) - 2; i >= 0; --i) {
            dst_strides[i] = dst_strides[i + 1] * target_shape[i + 1];
        }

        size_t elem_size     = dtypeSize(_dtype);
        const char *src_base = _storage.data<char>() + _storage_offset * elem_size;
        char *dst_base       = result._storage.data<char>();

        for (size_t i = 0; i < dst_numel; ++i) {
            std::vector<size_t> dst_indices(target_shape.size());
            size_t temp = i;
            // 从最后一维开始分解索引
            for (int j = static_cast<int>(target_shape.size()) - 1; j >= 0; --j) {
                dst_indices[j] = temp % target_shape[j];
                temp /= target_shape[j];
            }

            size_t src_idx = 0;
            for (size_t j = 0; j < current_shape.size(); ++j) {
                size_t idx = (current_shape[j] == 1) ? 0 : dst_indices[j];
                src_idx += idx * src_strides[j];
            }

            if (src_idx >= src_numel) {
                CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                            "广播时源索引越界");
            }

            std::memcpy(dst_base + i * elem_size, src_base + src_idx * elem_size, elem_size);
        }
    }

    return result;
}

// 零初始化张量
void Tensor::zero() {
    // MPS kernel 目前仅支持 float32；非 float 类型或 CPU 路径直接在 host 侧 memset
    if (_device == DeviceType::kMPS && _dtype == DType::kFloat) {
        Zero_MPS_kernel(*this);
        return;
    }
    size_t count     = numel();
    size_t elem_size = dtypeSize(_dtype);
    void *ptr        = nullptr;
    switch (_dtype) {
    case DType::kFloat:
        ptr = _storage.data<float>();
        break;
    case DType::kDouble:
        ptr = _storage.data<double>();
        break;
    case DType::kInt:
        ptr = _storage.data<int32_t>();
        break;
    case DType::kLong:
        ptr = _storage.data<int64_t>();
        break;
    case DType::kBool:
        ptr = _storage.data<bool>();
        break;
    default:
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                    "zero(): 未知数据类型");
    }
    if (ptr) {
        std::memset(static_cast<char *>(ptr) + _storage_offset * elem_size, 0, count * elem_size);
    }
}

// 一初始化张量
void Tensor::ones() {
    // 简单实现，将所有元素设为1
    size_t count = numel();
    if (_dtype == DType::kFloat) {
        float *data = _storage.data<float>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = 1.0f;
        }
    } else if (_dtype == DType::kDouble) {
        double *data = _storage.data<double>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = 1.0;
        }
    } else if (_dtype == DType::kInt) {
        int32_t *data = _storage.data<int32_t>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = 1;
        }
    } else if (_dtype == DType::kLong) {
        int64_t *data = _storage.data<int64_t>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = 1;
        }
    } else if (_dtype == DType::kBool) {
        bool *data = _storage.data<bool>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = true;
        }
    }
}

/**
 * @brief 随机初始化张量
 * @details 使用C++11线程安全的随机数生成器，生成[0, 1)之间的随机数
 * @note 仅支持float和double类型的张量
 */
void Tensor::rand() {
    // 使用C++11线程安全的随机数生成器
    size_t count = numel();

    // 为每个线程创建独立的随机数生成器
    thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution_float(0.0f, 1.0f);
    std::uniform_real_distribution<double> distribution_double(0.0, 1.0);

    if (_dtype == DType::kFloat) {
        float *data = _storage.data<float>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = distribution_float(generator);
        }
    } else if (_dtype == DType::kDouble) {
        double *data = _storage.data<double>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = distribution_double(generator);
        }
    }
    // 其他类型暂不支持
}

// 矩阵乘法
Tensor Tensor::matmul(const Tensor &other) const {
    // 使用调度器执行矩阵乘法
    Tensor result = AutoGrad::dispatch<op::MatMul>(*this, other);

    return result;
}

// ======================= 缺失方法实现 =======================

// 默认构造函数
Tensor::Tensor()
    : tensor_id_(global_tensor_id++), _storage_offset(0), _device(DeviceType::kCPU),
      _dtype(DType::kFloat) {
    _autograd_meta._self = std::shared_ptr<Tensor>(this, [](Tensor *) {});
    computeStrides();
}

// 检查存储偏移是否有效
bool Tensor::check_storage_offset() const { return _storage_offset < _storage.size(); }

// ReLU激活函数
Tensor Tensor::relu() const {
    // 简单实现ReLU激活函数
    Tensor result = AutoGrad::dispatch<op::ReLU>(*this);

    return result;
}

Tensor &Tensor::relu_() {
    check_inplace_safe_("relu_");
    CtorchScheduler::getInstance().dispatch_inplace(*this, op::ReLU);
    return *this;
}

// Leaky ReLU激活函数
Tensor Tensor::leaky_relu(float negative_slope) const {
    // 当前 LReLU kernel 固定 negative_slope = 0.01f；传入其他值会静默使用默认值
    (void)negative_slope;
    Tensor result = AutoGrad::dispatch<op::LReLU>(*this);

    return result;
}

Tensor &Tensor::leaky_relu_(float negative_slope) {
    check_inplace_safe_("leaky_relu_");
    // 当前 LReLU kernel 固定 negative_slope = 0.01f；传入其他值会静默使用默认值
    (void)negative_slope;
    CtorchScheduler::getInstance().dispatch_inplace(*this, op::LReLU);
    return *this;
}

Tensor Tensor::dot(const Tensor &other) const {
    Tensor result = AutoGrad::dispatch<op::Dot>(*this, other);

    return result;
}

Tensor Tensor::cos() const {
    Tensor result = AutoGrad::dispatch<op::Cos>(*this);

    return result;
}

Tensor &Tensor::cos_() {
    check_inplace_safe_("cos_");
    CtorchScheduler::getInstance().dispatch_inplace(*this, op::Cos);
    return *this;
}

Tensor Tensor::sin() const {
    Tensor result = AutoGrad::dispatch<op::Sin>(*this);

    return result;
}

Tensor &Tensor::sin_() {
    check_inplace_safe_("sin_");
    CtorchScheduler::getInstance().dispatch_inplace(*this, op::Sin);
    return *this;
}

Tensor &Tensor::neg_() {
    check_inplace_safe_("neg_");
    CtorchScheduler::getInstance().dispatch_inplace(*this, op::Neg);
    return *this;
}

// 求和操作
Tensor Tensor::sum() const {
    Tensor result(ShapeTag{}, {1}, _dtype, _device);

    if (_dtype == DType::kFloat) {
        if (_device == DeviceType::kMPS) {
            MPS_flush_wait(true);
        }
        const float *data = _storage.data<float>();
        float sum         = 0.0f;
        for (size_t i = 0; i < numel(); ++i) {
            sum += data[i + _storage_offset];
        }
        float *result_data = result.data<float>();
        if (result_data) {
            *result_data = sum;
        }
    }

    return result;
}

// ======================= 运算符模板辅助函数 =======================

// 二元 Tensor-Tensor 运算模板 (+, -, *, /)
template <op OpType> static Tensor binaryOpImpl(const Tensor &a, const Tensor &b) {
    if (a.dtype() != b.dtype()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                    "张量数据类型不匹配");
    }
    if (a.device() != b.device()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
                                    "张量设备类型不匹配");
    }
    return AutoGrad::dispatch<OpType>(a, b);
}

// 标量运算模板 (Tensor op float)
template <typename OpFunc>
static Tensor scalarOpImpl(const Tensor &self, float scalar, OpFunc &&op_func) {
    Tensor result(self);
    result.storage() = self.storage().clone();
    size_t count     = self.numel();
    switch (self.dtype()) {
    case DType::kFloat: {
        float *data = result.data<float>();
        for (size_t i = 0; i < count; ++i)
            op_func(data[i], scalar);
        break;
    }
    case DType::kDouble: {
        double *data = result.data<double>();
        for (size_t i = 0; i < count; ++i)
            op_func(data[i], static_cast<double>(scalar));
        break;
    }
    case DType::kInt: {
        int32_t *data = result.data<int32_t>();
        for (size_t i = 0; i < count; ++i)
            op_func(data[i], static_cast<int32_t>(scalar));
        break;
    }
    case DType::kLong: {
        int64_t *data = result.data<int64_t>();
        for (size_t i = 0; i < count; ++i)
            op_func(data[i], static_cast<int64_t>(scalar));
        break;
    }
    default:
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                    "标量运算不支持的dtype");
    }
    return result;
}

// 比较运算模板: Tensor vs 标量
template <typename CmpFunc>
static Tensor cmpScalarOpImpl(const Tensor &self, float scalar, CmpFunc &&cmp) {
    Tensor result(ShapeTag{}, self.shape(), DType::kFloat, self.device());
    if (self.dtype() == DType::kFloat) {
        size_t count       = self.numel();
        const float *data  = self.data_read<float>();
        float *result_data = result.data_write<float>();
        for (size_t i = 0; i < count; ++i)
            result_data[i] = cmp(data[i], scalar) ? 1.0f : 0.0f;
    }
    return result;
}

// 比较运算模板: Tensor vs Tensor
template <typename CmpFunc>
static Tensor cmpTensorOpImpl(const Tensor &self, const Tensor &other, CmpFunc &&cmp) {
    Tensor result(ShapeTag{}, self.shape(), DType::kFloat, self.device());
    if (self.shape() != other.shape()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                    "张量形状不匹配");
    }
    if (self.dtype() != other.dtype()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                    "张量数据类型不匹配");
    }
    if (self.dtype() == DType::kFloat) {
        size_t count            = self.numel();
        const float *data       = self.data_read<float>();
        const float *other_data = other.data_read<float>();
        float *result_data      = result.data_write<float>();
        for (size_t i = 0; i < count; ++i)
            result_data[i] = cmp(data[i], other_data[i]) ? 1.0f : 0.0f;
    }
    return result;
}

// 比较运算模板: 标量 vs Tensor (自由函数用)
template <typename CmpFunc>
static Tensor cmpScalarTensorOpImpl(float scalar, const Tensor &tensor, CmpFunc &&cmp) {
    Tensor result(ShapeTag{}, tensor.shape(), DType::kFloat, tensor.device());
    if (tensor.dtype() == DType::kFloat) {
        size_t count       = tensor.numel();
        const float *data  = tensor.data_read<float>();
        float *result_data = result.data_write<float>();
        for (size_t i = 0; i < count; ++i)
            result_data[i] = cmp(scalar, data[i]) ? 1.0f : 0.0f;
    }
    return result;
}

// ======================= 二元 Tensor-Tensor 运算符 (+, -, *, /) =======================

Tensor Tensor::operator/(const Tensor &other) const { return binaryOpImpl<op::Div>(*this, other); }
Tensor Tensor::operator-(const Tensor &other) const { return binaryOpImpl<op::Sub>(*this, other); }
Tensor Tensor::operator*(const Tensor &other) const { return binaryOpImpl<op::Mul>(*this, other); }

// ======================= 标量运算符 (Tensor op float) =======================

Tensor Tensor::operator*(float scalar) const {
    Tensor scalar_tensor(scalar, _device);
    scalar_tensor = scalar_tensor.to(_dtype);
    return binaryOpImpl<op::Mul>(*this, scalar_tensor);
}

// 一元负号运算符
Tensor Tensor::operator-() const { return AutoGrad::dispatch<op::Neg>(*this); }

// 张量加法运算符
Tensor Tensor::operator+(const Tensor &other) const { return binaryOpImpl<op::Add>(*this, other); }

Tensor Tensor::operator+(float scalar) const {
    Tensor scalar_tensor(scalar, _device);
    scalar_tensor = scalar_tensor.to(_dtype);
    return binaryOpImpl<op::Add>(*this, scalar_tensor);
}

Tensor Tensor::operator-(float scalar) const {
    Tensor scalar_tensor(scalar, _device);
    scalar_tensor = scalar_tensor.to(_dtype);
    return binaryOpImpl<op::Sub>(*this, scalar_tensor);
}

Tensor Tensor::operator/(float scalar) const {
    if (std::abs(scalar) < std::numeric_limits<float>::epsilon()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE,
                                    "除零错误：标量除法中除数为零");
    }
    return scalarOpImpl(*this, scalar, [](auto &val, auto s) { val /= s; });
}

// ======================= 比较操作符实现 =======================

// Tensor vs 标量
Tensor Tensor::operator>(float scalar) const {
    return cmpScalarOpImpl(*this, scalar, std::greater<>{});
}
Tensor Tensor::operator<(float scalar) const {
    return cmpScalarOpImpl(*this, scalar, std::less<>{});
}
Tensor Tensor::operator==(float scalar) const {
    return cmpScalarOpImpl(*this, scalar, std::equal_to<>{});
}
Tensor Tensor::operator>=(float scalar) const {
    return cmpScalarOpImpl(*this, scalar, std::greater_equal<>{});
}
Tensor Tensor::operator<=(float scalar) const {
    return cmpScalarOpImpl(*this, scalar, std::less_equal<>{});
}
Tensor Tensor::operator!=(float scalar) const {
    return cmpScalarOpImpl(*this, scalar, std::not_equal_to<>{});
}

// Tensor vs Tensor
Tensor Tensor::operator>(const Tensor &other) const {
    return cmpTensorOpImpl(*this, other, std::greater<>{});
}
Tensor Tensor::operator<(const Tensor &other) const {
    return cmpTensorOpImpl(*this, other, std::less<>{});
}
Tensor Tensor::operator==(const Tensor &other) const {
    return cmpTensorOpImpl(*this, other, std::equal_to<>{});
}
Tensor Tensor::operator>=(const Tensor &other) const {
    return cmpTensorOpImpl(*this, other, std::greater_equal<>{});
}
Tensor Tensor::operator<=(const Tensor &other) const {
    return cmpTensorOpImpl(*this, other, std::less_equal<>{});
}
Tensor Tensor::operator!=(const Tensor &other) const {
    return cmpTensorOpImpl(*this, other, std::not_equal_to<>{});
}

// ======================= 辅助方法 =======================

// 检查索引是否在边界内
bool Tensor::check_index_bounds(const std::vector<size_t> &indices) const {
    if (indices.size() != _shape.size()) {
        return false;
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= _shape[i]) {
            return false;
        }
    }
    return true;
}

// ======================= 全局函数实现 =======================

// 全局的matMul函数
Tensor matMul(const Tensor &a, const Tensor &b) { return AutoGrad::dispatch<op::MatMul>(a, b); }

// 计算两个张量的广播结果
BroadCastResult broadCast(const Tensor &a, const Tensor &tensor2) {
    // 简化实现，返回默认的广播结果
    BroadCastResult result;
    result.logicShape    = a.shape();
    result.logicStridesA = a.strides();
    result.logicStridesB = tensor2.strides();
    return result;
}

/**
 * @brief 标量加法操作符重载（右操作数）
 * @details 实现标量与张量的加法操作
 * @param scalar 标量值
 * @param tensor 张量
 * @return 加法结果张量
 */
Tensor operator+(float scalar, const Tensor &tensor) { return tensor + scalar; }

/**
 * @brief 标量减法操作符重载（右操作数）
 * @details 实现标量与张量的减法操作
 * @param scalar 标量值
 * @param tensor 张量
 * @return 减法结果张量
 */
Tensor operator-(float scalar, const Tensor &tensor) {
    Tensor scalar_tensor(scalar, tensor.device());
    scalar_tensor = scalar_tensor.to(tensor.dtype());
    return scalar_tensor - tensor;
}

/**
 * @brief 标量乘法操作符重载（右操作数）
 * @details 实现标量与张量的乘法操作
 * @param scalar 标量值
 * @param tensor 张量
 * @return 乘法结果张量
 */
Tensor operator*(float scalar, const Tensor &tensor) { return tensor * scalar; }

/**
 * @brief 标量除法操作符重载（右操作数）
 * @details 实现标量与张量的除法操作
 * @param scalar 标量值
 * @param tensor 张量
 * @return 除法结果张量
 */
Tensor operator/(float scalar, const Tensor &tensor) {
    Tensor scalar_tensor(scalar, tensor.device());
    scalar_tensor = scalar_tensor.to(tensor.dtype());
    return scalar_tensor / tensor;
}

// ======================= 比较操作符重载（标量 vs 张量，右操作数） =======================

Tensor operator>(float scalar, const Tensor &tensor) {
    return cmpScalarTensorOpImpl(scalar, tensor, std::greater<>{});
}
Tensor operator<(float scalar, const Tensor &tensor) {
    return cmpScalarTensorOpImpl(scalar, tensor, std::less<>{});
}
Tensor operator==(float scalar, const Tensor &tensor) {
    return cmpScalarTensorOpImpl(scalar, tensor, std::equal_to<>{});
}
Tensor operator>=(float scalar, const Tensor &tensor) {
    return cmpScalarTensorOpImpl(scalar, tensor, std::greater_equal<>{});
}
Tensor operator<=(float scalar, const Tensor &tensor) {
    return cmpScalarTensorOpImpl(scalar, tensor, std::less_equal<>{});
}
Tensor operator!=(float scalar, const Tensor &tensor) {
    return cmpScalarTensorOpImpl(scalar, tensor, std::not_equal_to<>{});
}

// 输出张量信息
std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    os << "Tensor(shape=[";
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
        if (i > 0)
            os << ", ";
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

// Tanh激活函数
Tensor Tensor::tanh() const {
    Tensor result = AutoGrad::dispatch<op::Tanh>(*this);

    return result;
}

Tensor &Tensor::tanh_() {
    check_inplace_safe_("tanh_");
    CtorchScheduler::getInstance().dispatch_inplace(*this, op::Tanh);
    return *this;
}

// Sigmoid激活函数
Tensor Tensor::sigmoid() const {
    Tensor result = AutoGrad::dispatch<op::Sigmoid>(*this);

    return result;
}

Tensor &Tensor::sigmoid_() {
    check_inplace_safe_("sigmoid_");
    CtorchScheduler::getInstance().dispatch_inplace(*this, op::Sigmoid);
    return *this;
}

// GELU激活函数
Tensor Tensor::gelu() const {
    Tensor result = AutoGrad::dispatch<op::GELU>(*this);

    return result;
}

Tensor &Tensor::gelu_() {
    check_inplace_safe_("gelu_");
    CtorchScheduler::getInstance().dispatch_inplace(*this, op::GELU);
    return *this;
}

// Softmax激活函数
Tensor Tensor::softmax(int dim) const { return AutoGrad::dispatch_softmax(*this, dim); }

// Max操作
Tensor Tensor::max() const {
    Tensor result(ShapeTag{}, {}, _dtype, _device);

    if (_dtype == DType::kFloat) {
        const float *data = _storage.data<float>();
        float max_val     = data[0];
        for (size_t i = 1; i < numel(); ++i) {
            if (data[i + _storage_offset] > max_val) {
                max_val = data[i + _storage_offset];
            }
        }
        result._storage    = Storage(1, _dtype, _device);
        float *result_data = result._storage.data<float>();
        if (result_data) {
            *result_data = max_val;
        }
    }

    return result;
}

// Min操作
Tensor Tensor::min() const {
    Tensor result(ShapeTag{}, {}, _dtype, _device);

    if (_dtype == DType::kFloat) {
        const float *data = _storage.data<float>();
        float min_val     = data[0];
        for (size_t i = 1; i < numel(); ++i) {
            if (data[i + _storage_offset] < min_val) {
                min_val = data[i + _storage_offset];
            }
        }
        result._storage    = Storage(1, _dtype, _device);
        float *result_data = result._storage.data<float>();
        if (result_data) {
            *result_data = min_val;
        }
    }

    return result;
}

// Square操作
Tensor Tensor::square() const {
    Tensor result = *this * *this;

    return result;
}

// MSE损失函数
Tensor Tensor::mse_loss(const Tensor &target) const {
    Tensor result = AutoGrad::dispatch<op::MSE>(*this, target);

    return result;
}

// CrossEntropy损失函数
Tensor Tensor::cross_entropy(const Tensor &target) const {
    Tensor result = AutoGrad::dispatch<op::CE>(*this, target);

    return result;
}

// MAE损失函数
Tensor Tensor::mae_loss(const Tensor &target) const {
    Tensor result = AutoGrad::dispatch<op::MAE>(*this, target);

    return result;
}

std::shared_ptr<Node> Tensor::getRelatedNode() { return _autograd_meta._node; }
std::shared_ptr<Node> Tensor::getRelatedNode() const { return _autograd_meta._node; }
void Tensor::setRelatedNode(std::shared_ptr<Node> ptr) const {
    _autograd_meta._node = std::move(ptr);
}
void Tensor::setGrad(std::shared_ptr<Tensor> grad) { _autograd_meta._grad = std::move(grad); }
// 求张量的和
Tensor Tensor::sum(int dim, bool keepdim) const {
    Tensor result;

    if (dim < 0) {
        dim += _shape.size();
    }
    if (dim < 0 || dim >= static_cast<int>(_shape.size())) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                    "sum: 维度超出范围");
    }

    std::vector<size_t> output_shape;
    for (size_t i = 0; i < _shape.size(); ++i) {
        if (i == static_cast<size_t>(dim)) {
            if (keepdim) {
                output_shape.push_back(1);
            }
        } else {
            output_shape.push_back(_shape[i]);
        }
    }

    if (output_shape.empty()) {
        output_shape = {1};
    }

    result = Tensor(ShapeTag{}, output_shape, _dtype, _device);

    if (_device == DeviceType::kMPS) {
        MPS_flush_wait(true);
    }

    size_t dim_size   = _shape[dim];
    size_t stride_dim = _strides[dim];

    size_t pre_dim_elements = 1;
    for (int i = 0; i < dim; ++i) {
        pre_dim_elements *= _shape[i];
    }

    size_t post_dim_elements = 1;
    for (size_t i = dim + 1; i < _shape.size(); ++i) {
        post_dim_elements *= _shape[i];
    }

    size_t pre_dim_stride = stride_dim * dim_size;

    switch (_dtype) {
    case DType::kFloat: {
        const float *data  = this->data<float>();
        float *result_data = result.data<float>();
        for (size_t i = 0; i < pre_dim_elements; ++i) {
            for (size_t k = 0; k < post_dim_elements; ++k) {
                float sum = 0.0f;
                for (size_t j = 0; j < dim_size; ++j) {
                    size_t index = i * pre_dim_stride + j * stride_dim + k;
                    sum += data[index];
                }
                result_data[i * post_dim_elements + k] = sum;
            }
        }
        break;
    }
    case DType::kDouble: {
        const double *data  = this->data<double>();
        double *result_data = result.data<double>();
        for (size_t i = 0; i < pre_dim_elements; ++i) {
            for (size_t k = 0; k < post_dim_elements; ++k) {
                double sum = 0.0;
                for (size_t j = 0; j < dim_size; ++j) {
                    size_t index = i * pre_dim_stride + j * stride_dim + k;
                    sum += data[index];
                }
                result_data[i * post_dim_elements + k] = sum;
            }
        }
        break;
    }
    default:
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                    "sum: 不支持的数据类型");
    }

    return result;
}

Tensor Tensor::sum(const std::vector<int> &dims, bool keepdim) const {
    std::vector<int> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int>());

    Tensor result = *this;
    for (int dim : sorted_dims) {
        result = result.sum(dim, keepdim);
    }
    return result;
}

Tensor Tensor::log() const { return AutoGrad::dispatch<op::Log>(*this); }

Tensor &Tensor::log_() {
    check_inplace_safe_("log_");
    CtorchScheduler::getInstance().dispatch_inplace(*this, op::Log);
    return *this;
}

Tensor Tensor::exp() const { return AutoGrad::dispatch<op::Exp>(*this); }

Tensor &Tensor::exp_() {
    check_inplace_safe_("exp_");
    CtorchScheduler::getInstance().dispatch_inplace(*this, op::Exp);
    return *this;
}

Tensor Tensor::abs() const { return AutoGrad::dispatch<op::Abs>(*this); }

Tensor &Tensor::abs_() {
    check_inplace_safe_("abs_");
    CtorchScheduler::getInstance().dispatch_inplace(*this, op::Abs);
    return *this;
}

Tensor Tensor::min(const Tensor &other) const { return AutoGrad::dispatch<op::Min>(*this, other); }

Tensor Tensor::max(const Tensor &other) const { return AutoGrad::dispatch<op::Max>(*this, other); }

Tensor Tensor::mean() const {
    Tensor result(ShapeTag{}, {}, _dtype, _device);

    if (_dtype == DType::kFloat) {
        const float *data = _storage.data<float>();
        float sum         = 0.0f;
        for (size_t i = 0; i < numel(); ++i) {
            sum += data[i + _storage_offset];
        }
        result._storage    = Storage(1, _dtype, _device);
        float *result_data = result._storage.data<float>();
        if (result_data) {
            *result_data = sum / static_cast<float>(numel());
        }
    }

    return result;
}

Tensor Tensor::clamp(float min_val, float max_val) const {
    Tensor result(*this);
    result._storage = _storage.clone();

    if (_dtype == DType::kFloat) {
        float *data = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) {
            float val = data[i];
            if (val < min_val)
                val = min_val;
            if (val > max_val)
                val = max_val;
            data[i] = val;
        }
    }

    return result;
}

Tensor Tensor::detach() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    result._storage                      = _storage.clone();
    result._autograd_meta._requires_grad = false;
    result._autograd_meta._node.reset();

    return result;
}

Tensor Tensor::to(DeviceType target_device) const {
    if (_device == target_device) {
        return *this;
    }

    Tensor result(ShapeTag{}, _shape, _dtype, target_device);

    size_t bytes = numel() * dtypeSize(_dtype);

    // Storage::data<T>() 对 T=char 会触发 dtype 检查失败，因此按实际 dtype 取指针
    const char *src = nullptr;
    char *dst       = nullptr;
    switch (_dtype) {
    case DType::kFloat:
        src = reinterpret_cast<const char *>(_storage.data<float>());
        dst = reinterpret_cast<char *>(result._storage.data<float>());
        break;
    case DType::kDouble:
        src = reinterpret_cast<const char *>(_storage.data<double>());
        dst = reinterpret_cast<char *>(result._storage.data<double>());
        break;
    case DType::kInt:
        src = reinterpret_cast<const char *>(_storage.data<int32_t>());
        dst = reinterpret_cast<char *>(result._storage.data<int32_t>());
        break;
    case DType::kLong:
        src = reinterpret_cast<const char *>(_storage.data<int64_t>());
        dst = reinterpret_cast<char *>(result._storage.data<int64_t>());
        break;
    case DType::kBool:
        src = reinterpret_cast<const char *>(_storage.data<bool>());
        dst = reinterpret_cast<char *>(result._storage.data<bool>());
        break;
    default:
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                    "to(device): 未知数据类型");
    }

    if (!src || !dst) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
                                    "to(device): 无法获取源或目标存储指针");
    }

    if (_device == DeviceType::kCPU || target_device == DeviceType::kCPU) {
        std::memcpy(dst, src, bytes);
    } else {
        DeviceAllocator *allocator = AllocatorManager::getInstance().getAllocator(target_device);
        if (allocator) {
            allocator->memcpy(dst, src, bytes, target_device, _device);
        } else {
            std::memcpy(dst, src, bytes);
        }
    }

    result._autograd_meta._requires_grad = _autograd_meta._requires_grad;
    return result;
}

Tensor::~Tensor() {
    _autograd_meta._self.reset();
    detach_autograd();
}
