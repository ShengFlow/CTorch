/**
 * @file Tensor.cpp
 * @brief 张量类的实现
 * @author GhostFace, Beapoe
 * @date 2025/12/21
 * @version v3.1
 * @details 实现了张量类的各种方法，包括构造函数、访问器、操作、运算和自动微分等
 */
#include "../include/Tensor.h"
#include "../include/Ctorch_Scheduler.h"
#include "kernels/kernels.h"
#include <cmath>
#include <random>

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
      // 先尝试make_leaf
      AutoDiffContext::current()->make_leaf(*this, key);
      // 然后强制更新requires_grad状态，确保节点在当前上下文中正确注册
      AutoDiffContext::current()->update_requires_grad(*this, key);
    }
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
  return std::accumulate(_shape.begin(), _shape.end(), 1ULL,
                         std::multiplies<>());
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
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "无效维度");
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
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "无效维度");
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
size_t
Tensor::computeStorageIndex(std::initializer_list<size_t> indices) const {
  if (indices.size() != _shape.size()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "索引维度与张量维度不匹配");
  }
  size_t index = 0;
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
      Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                   "期望float dtype");
    }
  } else if constexpr (std::is_same_v<T, double>) {
    if (_dtype != DType::kDouble) {
      Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                   "期望double dtype");
    }
  } else if constexpr (std::is_same_v<T, int32_t>) {
    if (_dtype != DType::kInt) {
      Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                   "期望int dtype");
    }
  } else if constexpr (std::is_same_v<T, int64_t>) {
    if (_dtype != DType::kLong) {
      Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                   "期望long dtype");
    }
  } else if constexpr (std::is_same_v<T, bool>) {
    if (_dtype != DType::kBool) {
      Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                   "期望bool dtype");
    }
  }
}

// 获取标量值
template <typename T> T Tensor::item() const {
  if (numel() != 1) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::TENSOR_STATE, "张量不是标量");
  }
  checkDType<T>();
  const T *data_ptr = _storage.data<T>();
  if (!data_ptr) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::TENSOR_STATE, "张量数据为null");
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
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "索引操作符仅支持1D张量");
  }
  if (index >= _shape[0]) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "索引越界");
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
    const float *src = _storage.data<float>();
    if (src) {
      if (dtype == DType::kDouble) {
        double *dst = result._storage.data<double>();
        if (dst) {
          for (size_t i = 0; i < numel(); ++i) {
            dst[i] = static_cast<double>(src[i + _storage_offset]);
          }
        }
      } else if (dtype == DType::kInt) {
        int32_t *dst = result._storage.data<int32_t>();
        if (dst) {
          for (size_t i = 0; i < numel(); ++i) {
            dst[i] = static_cast<int32_t>(src[i + _storage_offset]);
          }
        }
      } else if (dtype == DType::kLong) {
        int64_t *dst = result._storage.data<int64_t>();
        if (dst) {
          for (size_t i = 0; i < numel(); ++i) {
            dst[i] = static_cast<int64_t>(src[i + _storage_offset]);
          }
        }
      } else if (dtype == DType::kBool) {
        bool *dst = result._storage.data<bool>();
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
  // 检查维度索引是否有效
  if (dim0 < 0 || dim0 >= static_cast<int>(_shape.size()) || dim1 < 0 ||
      dim1 >= static_cast<int>(_shape.size())) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
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
  size_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL,
                                     std::multiplies<>());
  if (new_numel != numel()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
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
 * @throw Ctorch_Error 如果广播目标形状为空
 * @throw Ctorch_Error 如果广播形状不兼容
 */
Tensor Tensor::broadcast_to(const std::vector<size_t> &shape) const {
  // 实现标准的NumPy风格广播规则

  // 步骤1：检查输入形状是否有效
  if (shape.empty()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "广播目标形状不能为空");
  }

  // 步骤2：计算广播后的形状和当前张量的扩展形状
  std::vector<size_t> current_shape = _shape;
  std::vector<size_t> target_shape = shape;

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
      Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                   ErrorType::DIMENSION, "广播形状不兼容");
    }
  }

  // 步骤4：创建结果张量
  Tensor result(ShapeTag{}, shape, _dtype, _device);

  // 步骤5：执行广播（复制数据）
  if (_dtype == DType::kFloat) {
    const float *src_data = data<float>();
    float *dst_data = result.data<float>();

    // 检查数据指针是否有效
    if (!src_data || !dst_data) {
      Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
                                   "张量数据指针无效");
    }

    // 计算当前张量的元素数量
    size_t src_numel = numel();

    // 计算广播后的元素数量
    size_t dst_numel = result.numel();

    // 对于标量广播，直接复制到所有位置
    if (src_numel == 1) {
      float value = src_data[0];
      for (size_t i = 0; i < dst_numel; ++i) {
        dst_data[i] = value;
      }
    } else {
      // 实现完整的NumPy风格广播逻辑
      // 计算当前张量的步幅
      std::vector<size_t> src_strides(current_shape.size());
      src_strides.back() = 1;
      for (int i = static_cast<int>(current_shape.size()) - 2; i >= 0; --i) {
        src_strides[i] = src_strides[i + 1] * current_shape[i + 1];
      }

      // 计算目标张量的步幅
      std::vector<size_t> dst_strides(target_shape.size());
      dst_strides.back() = 1;
      for (int i = static_cast<int>(target_shape.size()) - 2; i >= 0; --i) {
        dst_strides[i] = dst_strides[i + 1] * target_shape[i + 1];
      }

      // 对于广播后的每个元素，计算原始张量中的对应索引
      for (size_t i = 0; i < dst_numel; ++i) {
        // 计算目标张量中元素i的多维索引
        std::vector<size_t> dst_indices(target_shape.size());
        size_t temp = i;
        for (int j = static_cast<int>(target_shape.size()) - 1; j >= 0; --j) {
          dst_indices[j] = temp / dst_strides[j];
          temp %= dst_strides[j];
        }

        // 计算原始张量中的对应索引
        size_t src_idx = 0;
        for (size_t j = 0; j < current_shape.size(); ++j) {
          // 如果当前维度大小为1，则使用0索引（广播）
          size_t idx = (current_shape[j] == 1) ? 0 : dst_indices[j];
          src_idx += idx * src_strides[j];
        }

        // 检查源索引是否在有效范围内
        if (src_idx >= src_numel) {
          Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                       ErrorType::DIMENSION,
                                       "广播时源索引越界");
        }

        // 复制数据
        dst_data[i] = src_data[src_idx];
      }
    }
  }

  return result;
}

// 零初始化张量
void Tensor::zero() {
  // 简单实现，将所有元素设为0
  size_t count = numel();
  if (_dtype == DType::kFloat) {
    float *data = _storage.data<float>();
    for (size_t i = 0; i < count; ++i) {
      data[i] = 0.0f;
    }
  } else if (_dtype == DType::kDouble) {
    double *data = _storage.data<double>();
    for (size_t i = 0; i < count; ++i) {
      data[i] = 0.0;
    }
  } else if (_dtype == DType::kInt) {
    int32_t *data = _storage.data<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      data[i] = 0;
    }
  } else if (_dtype == DType::kLong) {
    int64_t *data = _storage.data<int64_t>();
    for (size_t i = 0; i < count; ++i) {
      data[i] = 0;
    }
  } else if (_dtype == DType::kBool) {
    bool *data = _storage.data<bool>();
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
  Tensor result =
      Ctorch_Scheduler::getInstance().dispatch(*this, other, op::MatMul);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this),
                                    const_cast<Tensor *>(&other)};
    AutoDiffContext::current()->defer_record(result.id(), op::MatMul, inputs);
    result._requires_grad = _requires_grad || other.requires_grad();
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

// 反向传播
void Tensor::backward() const {
  // 错误检查1：确保自动微分上下文存在
  if (!AutoDiffContext::current()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::TENSOR_STATE,
                                 "反向传播需要自动微分上下文");
  }

  // 错误检查2：确保张量需要梯度
  if (!_requires_grad) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::TENSOR_STATE,
                                 "张量不需要梯度，无法执行反向传播");
  }

  // 错误检查3：确保张量数据有效
  if (is_cleared() || numel() == 0) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::TENSOR_STATE,
                                 "张量数据无效，无法执行反向传播");
  }

  // 执行反向传播
  Tensor &self_ref = const_cast<Tensor &>(*this);
  AutoDiffContext::current()->backward(self_ref);
}

// 反向传播（带有梯度输出）
void Tensor::backward(const Tensor &grad_output) const {
  // 错误检查1：确保自动微分上下文存在
  if (!AutoDiffContext::current()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::TENSOR_STATE,
                                 "反向传播需要自动微分上下文");
  }

  // 错误检查2：确保张量需要梯度
  if (!_requires_grad) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::TENSOR_STATE,
                                 "张量不需要梯度，无法执行反向传播");
  }

  // 错误检查3：确保张量数据有效
  if (is_cleared() || numel() == 0) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::TENSOR_STATE,
                                 "张量数据无效，无法执行反向传播");
  }

  // 错误检查4：确保梯度输出形状与张量形状匹配
  if (grad_output.shape() != _shape) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "梯度输出形状与张量形状不匹配");
  }

  // 错误检查5：确保梯度输出数据类型与张量数据类型匹配
  if (grad_output.dtype() != _dtype) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "梯度输出数据类型与张量数据类型不匹配");
  }

  // 执行反向传播
  Tensor &self_ref = const_cast<Tensor &>(*this);
  AutoDiffContext::current()->backward(self_ref, grad_output);
}

// ======================= 缺失方法实现 =======================

// 默认构造函数
Tensor::Tensor()
    : tensor_id_(global_tensor_id++), record_committed_(false),
      _requires_grad(false), _storage_offset(0), _device(DeviceType::kCPU),
      _dtype(DType::kFloat) {
  computeStrides();
}

// 检查存储偏移是否有效
bool Tensor::check_storage_offset() const {
  return _storage_offset < _storage.size();
}

// ReLU激活函数
Tensor Tensor::relu() const {
  // 简单实现ReLU激活函数
  Tensor result = Ctorch_Scheduler::getInstance().dispatch(*this, op::ReLU);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this)};
    AutoDiffContext::current()->defer_record(result.id(), op::ReLU, inputs);
    result._requires_grad = _requires_grad;
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

Tensor Tensor::dot(const Tensor &other) const {
  Tensor result =
      Ctorch_Scheduler::getInstance().dispatch(*this, other, op::Dot);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this)};
    AutoDiffContext::current()->defer_record(result.id(), op::Dot, inputs);
    result._requires_grad = _requires_grad;
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

Tensor Tensor::cos() const {
  Tensor result = Ctorch_Scheduler::getInstance().dispatch(*this, op::Cos);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this)};
    AutoDiffContext::current()->defer_record(result.id(), op::Cos, inputs);
    result._requires_grad = _requires_grad;
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

Tensor Tensor::sin() const {
  Tensor result = Ctorch_Scheduler::getInstance().dispatch(*this, op::Sin);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this)};
    AutoDiffContext::current()->defer_record(result.id(), op::Sin, inputs);
    result._requires_grad = _requires_grad;
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

// 求和操作
Tensor Tensor::sum() const {
  // 简单实现求和操作
  Tensor result(ShapeTag{}, {}, _dtype, _device);

  if (_dtype == DType::kFloat) {
    const float *data = _storage.data<float>();
    float sum = 0.0f;
    for (size_t i = 0; i < numel(); ++i) {
      sum += data[i + _storage_offset];
    }
    result._storage = Storage(1, _dtype, _device);
    float *result_data = result._storage.data<float>();
    if (result_data) {
      *result_data = sum;
    }
  }

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this)};
    AutoDiffContext::current()->defer_record(result.id(), op::Sum, inputs);
    result._requires_grad = _requires_grad;
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

// 张量除法运算符
Tensor Tensor::operator/(const Tensor &other) const {
  // 检查数据类型是否匹配
  if (_dtype != other.dtype()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "张量数据类型不匹配");
  }

  // 检查设备类型是否匹配
  if (_device != other.device()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::DEVICE_COMPAT,
                                 "张量设备类型不匹配");
  }

  // 简单实现张量除法
  Tensor result =
      Ctorch_Scheduler::getInstance().dispatch(*this, other, op::Div);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this),
                                    const_cast<Tensor *>(&other)};
    AutoDiffContext::current()->defer_record(result.id(), op::Div, inputs);
    result._requires_grad = _requires_grad || other.requires_grad();
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

// 张量减法运算符
Tensor Tensor::operator-(const Tensor &other) const {
  // 检查数据类型是否匹配
  if (_dtype != other.dtype()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "张量数据类型不匹配");
  }

  // 检查设备类型是否匹配
  if (_device != other.device()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::DEVICE_COMPAT,
                                 "张量设备类型不匹配");
  }

  // 使用调度器调用加法kernel执行张量加法
  Tensor result =
      Ctorch_Scheduler::getInstance().dispatch(*this, other, op::Sub);
  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this),
                                    const_cast<Tensor *>(&other)};
    AutoDiffContext::current()->defer_record(result.id(), op::Sub, inputs);
    result._requires_grad = _requires_grad || other.requires_grad();
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

// 张量乘法运算符
Tensor Tensor::operator*(const Tensor &other) const {
  // 检查数据类型是否匹配
  if (_dtype != other.dtype()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "张量数据类型不匹配");
  }

  // 检查设备类型是否匹配
  if (_device != other.device()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::DEVICE_COMPAT,
                                 "张量设备类型不匹配");
  }

  // 简单实现张量乘法
  Tensor result =
      Ctorch_Scheduler::getInstance().dispatch(*this, other, op::Mul);
  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this),
                                    const_cast<Tensor *>(&other)};
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
  // 根据数据类型处理
  switch (_dtype) {
  case DType::kFloat: {
    float *data = result.data<float>();
    for (size_t i = 0; i < count; ++i) {
      data[i] *= scalar;
    }
    break;
  }
  case DType::kDouble: {
    double *data = result.data<double>();
    for (size_t i = 0; i < count; ++i) {
      data[i] *= static_cast<double>(scalar);
    }
    break;
  }
  case DType::kInt: {
    int32_t *data = result.data<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      data[i] *= static_cast<int32_t>(scalar);
    }
    break;
  }
  case DType::kLong: {
    int64_t *data = result.data<int64_t>();
    for (size_t i = 0; i < count; ++i) {
      data[i] *= static_cast<int64_t>(scalar);
    }
    break;
  }
  default:
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "标量乘法不支持的dtype");
  }

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    // 对于标量操作，我们创建一个标量张量作为另一个输入
    Tensor scalar_tensor(scalar);
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this), &scalar_tensor};
    AutoDiffContext::current()->defer_record(result.id(), op::Mul, inputs);
    result._requires_grad = _requires_grad;
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  } else {
    result._requires_grad = _requires_grad;
  }

  return result;
}

// 一元负号运算符
Tensor Tensor::operator-() const {
  // 简单实现一元负号
  Tensor result = Ctorch_Scheduler::getInstance().dispatch(*this, op::Neg);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this)};
    AutoDiffContext::current()->defer_record(result.id(), op::Neg, inputs);
    result._requires_grad = _requires_grad;
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

// 张量加法运算符
Tensor Tensor::operator+(const Tensor &other) const {
  // 检查数据类型是否匹配
  if (_dtype != other.dtype()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "张量数据类型不匹配");
  }

  // 检查设备类型是否匹配
  if (_device != other.device()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::DEVICE_COMPAT,
                                 "张量设备类型不匹配");
  }

  // 使用调度器调用加法kernel执行张量加法
  Tensor result =
      Ctorch_Scheduler::getInstance().dispatch(*this, other, op::Add);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this),
                                    const_cast<Tensor *>(&other)};
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
  // 根据数据类型处理
  switch (_dtype) {
  case DType::kFloat: {
    float *data = result.data<float>();
    for (size_t i = 0; i < count; ++i) {
      data[i] += scalar;
    }
    break;
  }
  case DType::kDouble: {
    double *data = result.data<double>();
    for (size_t i = 0; i < count; ++i) {
      data[i] += static_cast<double>(scalar);
    }
    break;
  }
  case DType::kInt: {
    int32_t *data = result.data<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      data[i] += static_cast<int32_t>(scalar);
    }
    break;
  }
  case DType::kLong: {
    int64_t *data = result.data<int64_t>();
    for (size_t i = 0; i < count; ++i) {
      data[i] += static_cast<int64_t>(scalar);
    }
    break;
  }
  default:
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "标量加法不支持的dtype");
  }

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    // 对于标量操作，我们创建一个标量张量作为另一个输入
    Tensor scalar_tensor(scalar);
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this), &scalar_tensor};
    AutoDiffContext::current()->defer_record(result.id(), op::Add, inputs);
    result._requires_grad = _requires_grad;
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  } else {
    result._requires_grad = _requires_grad;
  }

  return result;
}

// 标量除法运算符
Tensor Tensor::operator/(float scalar) const {
  // 简单实现标量除法
  Tensor result(*this);
  result._storage = _storage.clone();

  // 检查除数是否为零
  if (std::abs(scalar) < std::numeric_limits<float>::epsilon()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL,
                                 ErrorType::TENSOR_STATE,
                                 "除零错误：标量除法中除数为零");
  }

  size_t count = numel();
  // 根据数据类型处理
  switch (_dtype) {
  case DType::kFloat: {
    float *data = result.data<float>();
    for (size_t i = 0; i < count; ++i) {
      data[i] /= scalar;
    }
    break;
  }
  case DType::kDouble: {
    double *data = result.data<double>();
    for (size_t i = 0; i < count; ++i) {
      data[i] /= static_cast<double>(scalar);
    }
    break;
  }
  case DType::kInt: {
    int32_t *data = result.data<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      data[i] /= static_cast<int32_t>(scalar);
    }
    break;
  }
  case DType::kLong: {
    int64_t *data = result.data<int64_t>();
    for (size_t i = 0; i < count; ++i) {
      data[i] /= static_cast<int64_t>(scalar);
    }
    break;
  }
  default:
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "标量除法不支持的dtype");
  }

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    // 对于标量操作，我们创建一个标量张量作为另一个输入
    Tensor scalar_tensor(scalar);
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this), &scalar_tensor};
    AutoDiffContext::current()->defer_record(result.id(), op::Div, inputs);
    result._requires_grad = _requires_grad;
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  } else {
    result._requires_grad = _requires_grad;
  }

  return result;
}

// ======================= 比较操作符实现 =======================

// 张量与标量之间的比较操作符

/**
 * @brief 张量大于标量比较操作符
 * @details 实现张量与标量的元素级大于比较
 * @param scalar 标量值
 * @return 比较结果张量，元素为布尔类型
 */
Tensor Tensor::operator>(float scalar) const {
  // 实现张量大于标量的元素级比较
  Tensor result(ShapeTag{}, _shape, DType::kBool, _device);

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *data = this->data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = data[i] > scalar;
    }
  }

  return result;
}

/**
 * @brief 张量小于标量比较操作符
 * @details 实现张量与标量的元素级小于比较
 * @param scalar 标量值
 * @return 比较结果张量，元素为布尔类型
 */
Tensor Tensor::operator<(float scalar) const {
  // 实现张量小于标量的元素级比较
  Tensor result(ShapeTag{}, _shape, DType::kBool, _device);

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *data = this->data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = data[i] < scalar;
    }
  }

  return result;
}

/**
 * @brief 张量等于标量比较操作符
 * @details 实现张量与标量的元素级等于比较
 * @param scalar 标量值
 * @return 比较结果张量，元素为布尔类型
 */
Tensor Tensor::operator==(float scalar) const {
  // 实现张量等于标量的元素级比较
  Tensor result(ShapeTag{}, _shape, DType::kBool, _device);

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *data = this->data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = data[i] == scalar;
    }
  }

  return result;
}

/**
 * @brief 张量大于等于标量比较操作符
 * @details 实现张量与标量的元素级大于等于比较
 * @param scalar 标量值
 * @return 比较结果张量，元素为布尔类型
 */
Tensor Tensor::operator>=(float scalar) const {
  // 实现张量大于等于标量的元素级比较
  Tensor result(ShapeTag{}, _shape, DType::kBool, _device);

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *data = this->data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = data[i] >= scalar;
    }
  }

  return result;
}

/**
 * @brief 张量小于等于标量比较操作符
 * @details 实现张量与标量的元素级小于等于比较
 * @param scalar 标量值
 * @return 比较结果张量，元素为布尔类型
 */
Tensor Tensor::operator<=(float scalar) const {
  // 实现张量小于等于标量的元素级比较
  Tensor result(ShapeTag{}, _shape, DType::kBool, _device);

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *data = this->data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = data[i] <= scalar;
    }
  }

  return result;
}

/**
 * @brief 张量不等于标量比较操作符
 * @details 实现张量与标量的元素级不等于比较
 * @param scalar 标量值
 * @return 比较结果张量，元素为布尔类型
 */
Tensor Tensor::operator!=(float scalar) const {
  // 实现张量不等于标量的元素级比较
  Tensor result(ShapeTag{}, _shape, DType::kBool, _device);

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *data = this->data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = data[i] != scalar;
    }
  }

  return result;
}

// 张量与张量之间的比较操作符

/**
 * @brief 张量大于张量比较操作符
 * @details 实现张量与张量的元素级大于比较
 * @param other 另一个张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor Tensor::operator>(const Tensor &other) const {
  // 实现张量大于张量的元素级比较
  Tensor result(ShapeTag{}, _shape, DType::kBool, _device);

  // 检查形状是否匹配
  if (_shape != other.shape()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "张量形状不匹配");
  }

  // 检查数据类型是否匹配
  if (_dtype != other.dtype()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "张量数据类型不匹配");
  }

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *data = this->data<float>();
    const float *other_data = other.data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = data[i] > other_data[i];
    }
  }

  return result;
}

/**
 * @brief 张量小于张量比较操作符
 * @details 实现张量与张量的元素级小于比较
 * @param other 另一个张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor Tensor::operator<(const Tensor &other) const {
  // 实现张量小于张量的元素级比较
  Tensor result(ShapeTag{}, _shape, DType::kBool, _device);

  // 检查形状是否匹配
  if (_shape != other.shape()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "张量形状不匹配");
  }

  // 检查数据类型是否匹配
  if (_dtype != other.dtype()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "张量数据类型不匹配");
  }

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *data = this->data<float>();
    const float *other_data = other.data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = data[i] < other_data[i];
    }
  }

  return result;
}

/**
 * @brief 张量等于张量比较操作符
 * @details 实现张量与张量的元素级等于比较
 * @param other 另一个张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor Tensor::operator==(const Tensor &other) const {
  // 实现张量等于张量的元素级比较
  Tensor result(ShapeTag{}, _shape, DType::kBool, _device);

  // 检查形状是否匹配
  if (_shape != other.shape()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "张量形状不匹配");
  }

  // 检查数据类型是否匹配
  if (_dtype != other.dtype()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "张量数据类型不匹配");
  }

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *data = this->data<float>();
    const float *other_data = other.data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = data[i] == other_data[i];
    }
  }

  return result;
}

/**
 * @brief 张量大于等于张量比较操作符
 * @details 实现张量与张量的元素级大于等于比较
 * @param other 另一个张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor Tensor::operator>=(const Tensor &other) const {
  // 实现张量大于等于张量的元素级比较
  Tensor result(ShapeTag{}, _shape, DType::kBool, _device);

  // 检查形状是否匹配
  if (_shape != other.shape()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "张量形状不匹配");
  }

  // 检查数据类型是否匹配
  if (_dtype != other.dtype()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "张量数据类型不匹配");
  }

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *data = this->data<float>();
    const float *other_data = other.data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = data[i] >= other_data[i];
    }
  }

  return result;
}

/**
 * @brief 张量小于等于张量比较操作符
 * @details 实现张量与张量的元素级小于等于比较
 * @param other 另一个张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor Tensor::operator<=(const Tensor &other) const {
  // 实现张量小于等于张量的元素级比较
  Tensor result(ShapeTag{}, _shape, DType::kBool, _device);

  // 检查形状是否匹配
  if (_shape != other.shape()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "张量形状不匹配");
  }

  // 检查数据类型是否匹配
  if (_dtype != other.dtype()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "张量数据类型不匹配");
  }

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *data = this->data<float>();
    const float *other_data = other.data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = data[i] <= other_data[i];
    }
  }

  return result;
}

/**
 * @brief 张量不等于张量比较操作符
 * @details 实现张量与张量的元素级不等于比较
 * @param other 另一个张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor Tensor::operator!=(const Tensor &other) const {
  // 实现张量不等于张量的元素级比较
  Tensor result(ShapeTag{}, _shape, DType::kBool, _device);

  // 检查形状是否匹配
  if (_shape != other.shape()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "张量形状不匹配");
  }

  // 检查数据类型是否匹配
  if (_dtype != other.dtype()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "张量数据类型不匹配");
  }

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *data = this->data<float>();
    const float *other_data = other.data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = data[i] != other_data[i];
    }
  }

  return result;
}

// ======================= 辅助方法 =======================

// 清空存储的方法
void Tensor::clear_storage() { _storage.clear(); }

// 判断是否为空的辅助方法
bool Tensor::is_cleared() const { return _storage.empty(); }

// 提交未完成的记录
void Tensor::commit_pending_record() {
  if (AutoDiffContext::current() && has_pending_record()) {
    AutoDiffContext::current()->commit_record(*this);
  }
  record_committed_ = true;
}

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

// 增强调试信息
void Tensor::debug_info_detailed(const std::string &name) const {
  std::ostringstream oss;
  oss << "Tensor " << name << " (ID: " << tensor_id_ << ")" << std::endl;
  oss << "  Shape: [";
  for (size_t i = 0; i < _shape.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << _shape[i];
  }
  oss << "]" << std::endl;
  oss << "  Strides: [";
  for (size_t i = 0; i < _strides.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << _strides[i];
  }
  oss << "]" << std::endl;
  oss << "  Storage offset: " << _storage_offset << std::endl;
  oss << "  Device: " << static_cast<int>(_device) << std::endl;
  oss << "  DType: " << dtypeToString(_dtype) << std::endl;
  oss << "  Requires grad: " << _requires_grad << std::endl;
  oss << "  Record committed: " << record_committed_ << std::endl;
  oss << "  Storage: " << (_storage.empty() ? "empty" : "non-empty")
      << std::endl;

  Ctorch_Error::trace(ErrorPlatform::kCPU, oss.str());
}

// ======================= 全局函数实现 =======================

// 全局的backward函数，用于启动反向传播
void backward(Tensor &root) {
  if (AutoDiffContext::current()) {
    AutoDiffContext::current()->backward(root);
  }
}

// 全局的backward函数，用于启动反向传播（带有梯度输出）
void backward(Tensor &root, Tensor grad_output) {
  if (AutoDiffContext::current()) {
    AutoDiffContext::current()->backward(root, grad_output);
  }
}

// 全局的grad函数，用于获取张量的梯度
Tensor grad(const Tensor &t) {
  if (AutoDiffContext::current()) {
    return AutoDiffContext::current()->get_grad(&t);
  }
  return Tensor();
}

// 全局的matMul函数
Tensor matMul(const Tensor &a, const Tensor &b) {
  return Ctorch_Scheduler::getInstance().dispatch(a, b, op::MatMul);
}

// 计算两个张量的广播结果
BroadCastResult broadCast(const Tensor &a, const Tensor &tensor2) {
  // 简化实现，返回默认的广播结果
  BroadCastResult result;
  result.logicShape = a.shape();
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
  Tensor result = Tensor(scalar) - tensor;
  return result;
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
  Tensor result = Tensor(scalar) / tensor;
  return result;
}

/**
 * @brief 比较操作符重载：标量大于张量（右操作数）
 * @details 实现标量与张量的元素级大于比较
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator>(float scalar, const Tensor &tensor) {
  // 实现标量大于张量的元素级比较
  Tensor result(ShapeTag{}, tensor.shape(), DType::kBool, tensor.device());

  size_t count = tensor.numel();
  if (tensor.dtype() == DType::kFloat) {
    const float *data = tensor.data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = scalar > data[i];
    }
  }

  return result;
}

/**
 * @brief 比较操作符重载：标量小于张量（右操作数）
 * @details 实现标量与张量的元素级小于比较
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator<(float scalar, const Tensor &tensor) {
  // 实现标量小于张量的元素级比较
  Tensor result(ShapeTag{}, tensor.shape(), DType::kBool, tensor.device());

  size_t count = tensor.numel();
  if (tensor.dtype() == DType::kFloat) {
    const float *data = tensor.data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = scalar < data[i];
    }
  }

  return result;
}

/**
 * @brief 比较操作符重载：标量等于张量（右操作数）
 * @details 实现标量与张量的元素级等于比较
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator==(float scalar, const Tensor &tensor) {
  // 实现标量等于张量的元素级比较
  Tensor result(ShapeTag{}, tensor.shape(), DType::kBool, tensor.device());

  size_t count = tensor.numel();
  if (tensor.dtype() == DType::kFloat) {
    const float *data = tensor.data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = scalar == data[i];
    }
  }

  return result;
}

/**
 * @brief 比较操作符重载：标量大于等于张量（右操作数）
 * @details 实现标量与张量的元素级大于等于比较
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator>=(float scalar, const Tensor &tensor) {
  // 实现标量大于等于张量的元素级比较
  Tensor result(ShapeTag{}, tensor.shape(), DType::kBool, tensor.device());

  size_t count = tensor.numel();
  if (tensor.dtype() == DType::kFloat) {
    const float *data = tensor.data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = scalar >= data[i];
    }
  }

  return result;
}

/**
 * @brief 比较操作符重载：标量小于等于张量（右操作数）
 * @details 实现标量与张量的元素级小于等于比较
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator<=(float scalar, const Tensor &tensor) {
  // 实现标量小于等于张量的元素级比较
  Tensor result(ShapeTag{}, tensor.shape(), DType::kBool, tensor.device());

  size_t count = tensor.numel();
  if (tensor.dtype() == DType::kFloat) {
    const float *data = tensor.data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = scalar <= data[i];
    }
  }

  return result;
}

/**
 * @brief 比较操作符重载：标量不等于张量（右操作数）
 * @details 实现标量与张量的元素级不等于比较
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator!=(float scalar, const Tensor &tensor) {
  // 实现标量不等于张量的元素级比较
  Tensor result(ShapeTag{}, tensor.shape(), DType::kBool, tensor.device());

  size_t count = tensor.numel();
  if (tensor.dtype() == DType::kFloat) {
    const float *data = tensor.data<float>();
    bool *result_data = result.data<bool>();
    for (size_t i = 0; i < count; ++i) {
      result_data[i] = scalar != data[i];
    }
  }

  return result;
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
  Tensor result = Ctorch_Scheduler::getInstance().dispatch(*this, op::Tanh);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this)};
    AutoDiffContext::current()->defer_record(result.id(), op::Tanh, inputs);
    result._requires_grad = _requires_grad;
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

// Sigmoid激活函数
Tensor Tensor::sigmoid() const {
  Tensor result = Ctorch_Scheduler::getInstance().dispatch(*this, op::Sigmoid);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this)};
    AutoDiffContext::current()->defer_record(result.id(), op::Sigmoid, inputs);
    result._requires_grad = _requires_grad;
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

// Softmax激活函数
Tensor Tensor::softmax(int dim) const {
  // 彻底实现 dim-softmax（目前支持 1D/2D；dim=-1 表示最后一维）
  std::vector<size_t> shape = this->sizes();
  if (shape.empty()) {
    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                 "Softmax: 不支持标量 softmax");
  }

  int rank = static_cast<int>(shape.size());
  if (dim < 0)
    dim += rank; // -1 -> last dim
  if (dim < 0 || dim >= rank) {
    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                 "Softmax: dim 越界");
  }

  if (this->device() != DeviceType::kCPU) {
    Ctorch_Error::throwException(DeviceTypeToErrorPlatform(this->device()),
                                 ErrorType::DEVICE_COMPAT,
                                 "Softmax: 当前仅实现 CPU");
  }

  Tensor result(ShapeTag{}, shape, this->dtype(), this->device());

  if (shape.size() == 1) {
    // 1D: 只有 dim=0 合法
    if (dim != 0) {
      Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                   "Softmax: 1D 张量仅支持 dim=0/-1");
    }
    size_t n = this->numel();
    switch (this->dtype()) {
    case DType::kFloat: {
      const float *in = this->data<float>();
      float *out = result.data<float>();
      float max_val = in[0];
      for (size_t i = 1; i < n; ++i)
        if (in[i] > max_val)
          max_val = in[i];
      float sum = 0.0f;
      for (size_t i = 0; i < n; ++i) {
        out[i] = std::exp(in[i] - max_val);
        sum += out[i];
      }
      for (size_t i = 0; i < n; ++i)
        out[i] /= sum;
      break;
    }
    case DType::kDouble: {
      const double *in = this->data<double>();
      double *out = result.data<double>();
      double max_val = in[0];
      for (size_t i = 1; i < n; ++i)
        if (in[i] > max_val)
          max_val = in[i];
      double sum = 0.0;
      for (size_t i = 0; i < n; ++i) {
        out[i] = std::exp(in[i] - max_val);
        sum += out[i];
      }
      for (size_t i = 0; i < n; ++i)
        out[i] /= sum;
      break;
    }
    default:
      Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                   "Softmax: 仅支持 float/double");
    }
  } else if (shape.size() == 2) {
    size_t rows = shape[0];
    size_t cols = shape[1];
    // 2D: 支持 dim=0 或 dim=1
    if (dim != 0 && dim != 1) {
      Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                   "Softmax: 2D 张量仅支持 dim=0/1/-1");
    }
    switch (this->dtype()) {
    case DType::kFloat: {
      const float *in = this->data<float>();
      float *out = result.data<float>();
      if (dim == 1) {
        // 按行 softmax
        for (size_t i = 0; i < rows; ++i) {
          float max_val = in[i * cols];
          for (size_t j = 1; j < cols; ++j) {
            float v = in[i * cols + j];
            if (v > max_val)
              max_val = v;
          }
          float sum = 0.0f;
          for (size_t j = 0; j < cols; ++j) {
            float e = std::exp(in[i * cols + j] - max_val);
            out[i * cols + j] = e;
            sum += e;
          }
          for (size_t j = 0; j < cols; ++j)
            out[i * cols + j] /= sum;
        }
      } else {
        // 按列 softmax
        for (size_t j = 0; j < cols; ++j) {
          float max_val = in[j];
          for (size_t i = 1; i < rows; ++i) {
            float v = in[i * cols + j];
            if (v > max_val)
              max_val = v;
          }
          float sum = 0.0f;
          for (size_t i = 0; i < rows; ++i) {
            float e = std::exp(in[i * cols + j] - max_val);
            out[i * cols + j] = e;
            sum += e;
          }
          for (size_t i = 0; i < rows; ++i)
            out[i * cols + j] /= sum;
        }
      }
      break;
    }
    case DType::kDouble: {
      const double *in = this->data<double>();
      double *out = result.data<double>();
      if (dim == 1) {
        for (size_t i = 0; i < rows; ++i) {
          double max_val = in[i * cols];
          for (size_t j = 1; j < cols; ++j) {
            double v = in[i * cols + j];
            if (v > max_val)
              max_val = v;
          }
          double sum = 0.0;
          for (size_t j = 0; j < cols; ++j) {
            double e = std::exp(in[i * cols + j] - max_val);
            out[i * cols + j] = e;
            sum += e;
          }
          for (size_t j = 0; j < cols; ++j)
            out[i * cols + j] /= sum;
        }
      } else {
        for (size_t j = 0; j < cols; ++j) {
          double max_val = in[j];
          for (size_t i = 1; i < rows; ++i) {
            double v = in[i * cols + j];
            if (v > max_val)
              max_val = v;
          }
          double sum = 0.0;
          for (size_t i = 0; i < rows; ++i) {
            double e = std::exp(in[i * cols + j] - max_val);
            out[i * cols + j] = e;
            sum += e;
          }
          for (size_t i = 0; i < rows; ++i)
            out[i * cols + j] /= sum;
        }
      }
      break;
    }
    default:
      Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                   "Softmax: 仅支持 float/double");
    }
  } else {
    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                 "Softmax: 暂仅支持 1D/2D");
  }

  // 记录操作到计算图（把 dim 写进 op_param_i）
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this)};
    AutoDiffContext::current()->defer_record(result.id(), op::Softmax, inputs,
                                             dim);
    result._requires_grad = _requires_grad;
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

// MSE损失函数
Tensor Tensor::mse_loss(const Tensor &target) const {
  Tensor result =
      Ctorch_Scheduler::getInstance().dispatch(*this, target, op::MSE);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this),
                                    const_cast<Tensor *>(&target)};
    AutoDiffContext::current()->defer_record(result.id(), op::MSE, inputs);
    result._requires_grad = _requires_grad || target.requires_grad();
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

// CrossEntropy损失函数
Tensor Tensor::cross_entropy(const Tensor &target) const {
  Tensor result =
      Ctorch_Scheduler::getInstance().dispatch(*this, target, op::CE);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this),
                                    const_cast<Tensor *>(&target)};
    AutoDiffContext::current()->defer_record(result.id(), op::CE, inputs);
    result._requires_grad = _requires_grad || target.requires_grad();
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

// MAE损失函数
Tensor Tensor::mae_loss(const Tensor &target) const {
  Tensor result =
      Ctorch_Scheduler::getInstance().dispatch(*this, target, op::MAE);

  // 记录操作到计算图
  if (AutoDiffContext::current()) {
    std::vector<Tensor *> inputs = {const_cast<Tensor *>(this),
                                    const_cast<Tensor *>(&target)};
    AutoDiffContext::current()->defer_record(result.id(), op::MAE, inputs);
    result._requires_grad = _requires_grad || target.requires_grad();
    if (result._requires_grad) {
      result.commit_pending_record();
    }
  }

  return result;
}

// 提取张量的子部分
Tensor Tensor::slice(int dim, size_t start, size_t end) const {
  if (dim < 0) {
    dim += static_cast<int>(_shape.size());
  }
  if (dim < 0 || dim >= static_cast<int>(_shape.size())) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "无效维度");
  }
  if (start >= end || end > _shape[dim]) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "无效的切片范围");
  }

  Tensor result(*this);
  result._shape[dim] = end - start;
  result._storage_offset += start * _strides[dim];
  result.computeStrides();
  return result;
}

// 将另一个张量的内容复制到当前张量
Tensor& Tensor::copy_(const Tensor &other) {
  if (_shape != other.shape()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                 "张量形状不匹配");
  }
  if (_dtype != other.dtype()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                                 "张量数据类型不匹配");
  }
  if (_device != other.device()) {
    Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
                                 "张量设备类型不匹配");
  }

  size_t count = numel();
  if (_dtype == DType::kFloat) {
    const float *src_data = other.data<float>();
    float *dst_data = data<float>();
    std::memcpy(dst_data, src_data, count * sizeof(float));
  } else if (_dtype == DType::kDouble) {
    const double *src_data = other.data<double>();
    double *dst_data = data<double>();
    std::memcpy(dst_data, src_data, count * sizeof(double));
  } else if (_dtype == DType::kInt) {
    const int32_t *src_data = other.data<int32_t>();
    int32_t *dst_data = data<int32_t>();
    std::memcpy(dst_data, src_data, count * sizeof(int32_t));
  } else if (_dtype == DType::kLong) {
    const int64_t *src_data = other.data<int64_t>();
    int64_t *dst_data = data<int64_t>();
    std::memcpy(dst_data, src_data, count * sizeof(int64_t));
  } else if (_dtype == DType::kBool) {
    const bool *src_data = other.data<bool>();
    bool *dst_data = data<bool>();
    std::memcpy(dst_data, src_data, count * sizeof(bool));
  }

  return *this;
}
