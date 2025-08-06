/*
* Tensor.cppm
* Created by Beapoe & GhostFace on 2025.7
* Main Classes: Storage & Tensor & Auto_diff
* Version : v1.7 (fixed on 2025.7.29 15:59)
* Log 1.3: 增加了注释及代码易读性
* Log 1.4: 增加了AutoGrad自动微分类
* Log 1.5: 增加了连续性检查，修复了变量命名，增加了对自动微分状态的输出，修复了移动时不移动自动微分状态的bug
* Log 1.6: 修复了广播操作并对所有二元操作进行广播处理，优化了矩阵乘法
* Log 1.7: 增加了标量运算
* Unfix : matMul
*/

module;

#include <memory>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <unordered_set>

module Tensor_dev;

// 辅助函数
int minx(int a,int b){
    int diff = b - a;
    return a + (diff & (diff >> 31));
}

// Storage
Storage::Storage() : _size(0), _dtype(DType::kFloat), _device(DeviceType::kCPU) {}

Storage::Storage(size_t size, DType dtype, DeviceType device): _size(size), _dtype(dtype), _device(device),_data(size > 0 ? std::shared_ptr<char[]>(new char[size * dtypeSize(dtype)]) : nullptr) {}

size_t Storage::size() const { return _size; }

DType Storage::dtype() const { return _dtype; }

DeviceType Storage::device() const { return _device; }

Storage Storage::clone() const {
    Storage new_storage(_size, _dtype, _device);
    if (_size > 0 && _data) {
        std::memcpy(new_storage._data.get(), _data.get(), _size * dtypeSize(_dtype));
    }
    return new_storage;
}

bool Storage::empty() const { return _size == 0 || !_data; }

void Storage::serialize(std::ofstream &os) const {
    if (!os) {
        throw std::runtime_error("序列化失败：输出流无效");
    }

    // 写入基本属性（size、dtype、device）
    os.write(reinterpret_cast<const char*>(&_size), sizeof(_size));
    os.write(reinterpret_cast<const char*>(&_dtype), sizeof(_dtype));
    os.write(reinterpret_cast<const char*>(&_device), sizeof(_device));

    // 写入原始数据（仅当有数据时）
    if (_size > 0 && _data) {
        size_t elem_size = dtypeSize(_dtype);
        os.write(_data.get(), _size * elem_size);  // 一次性写入所有数据，提升性能
    }
}

void Storage::deserialize(std::ifstream &is) {
    if (!is) {
        throw std::runtime_error("反序列化失败：输入流无效");
    }

    // 读取基本属性
    is.read(reinterpret_cast<char*>(&_size), sizeof(_size));
    is.read(reinterpret_cast<char*>(&_dtype), sizeof(_dtype));
    is.read(reinterpret_cast<char*>(&_device), sizeof(_device));

    // 读取原始数据（仅当有数据时）
    if (_size > 0) {
        size_t elem_size = dtypeSize(_dtype);
        size_t total_bytes = _size * elem_size;
        _data = std::make_shared<char[]>(total_bytes);  // 分配内存
        is.read(_data.get(), total_bytes);              // 一次性读取所有数据

        // 检查读取是否完整
        if (is.gcount() != static_cast<std::streamsize>(total_bytes)) {
            throw std::runtime_error("反序列化失败：数据不完整");
        }
    } else {
        _data = nullptr;  // 空数据
    }
}

// Tensor
void Tensor::computeStrides() {
    if (_shape.empty()) {
        _strides.clear();
        return;
    }
    _strides.resize(_shape.size());

    // 行优先布局: 最后一个维度步幅为1
    _strides[_shape.size() - 1] = 1;
    for (int i = static_cast<int>(_shape.size()) - 2; i >= 0; --i) {
        _strides[i] = _strides[i + 1] * _shape[i + 1];
    }
}

size_t Tensor::computeStorageIndex(std::initializer_list<size_t> indices) const {
    if (indices.size() != dim()) {
        throw std::runtime_error("Indices count mismatch");
    }

    if (dim() == 0) {
        return _storage_offset; // 标量情况
    }

    size_t index = _storage_offset;
    size_t i = 0;
    for (const auto& idx : indices) {
        if (idx >= _shape[i]) {
            throw std::out_of_range("Tensor index out of bounds");
        }
        index += idx * _strides[i];
        ++i;
    }
    return index;
}
Tensor Tensor::cos() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    switch (_dtype) {
    case DType::kFloat: {
        const auto* src = data<float>();
        float* dst = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = std::cos(src[i]);
        }
        break;
    }
    case DType::kDouble: {
        const double* src = data<double>();
        double* dst = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = std::cos(src[i]);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for cos");
    }

    // 记录操作到自动微分计算图
    AutoGrad* ctx = AutoGradContext::current();
    if (ctx) {
        ctx->record_op(
            {&result},
            op::Cos,
            {const_cast<Tensor*>(this)}
        );
    }
    return result;
}

Tensor Tensor::sin() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    switch (_dtype) {
    case DType::kFloat: {
        const float* src = data<float>();
        float* dst = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = std::sin(src[i]);
        }
        break;
    }
    case DType::kDouble: {
        const double* src = data<double>();
        double* dst = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = std::sin(src[i]);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for sin");
    }

    // 记录操作到自动微分计算图
    AutoGrad* ctx = AutoGradContext::current();
    if (ctx) {
        ctx->record_op(
            {&result},
            op::Sin,
            {const_cast<Tensor*>(this)}
        );
    }
    return result;
}

Tensor Tensor::relu() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    switch (_dtype) {
    case DType::kFloat: {
        const float* src = data<float>();
        float* dst = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
        }
        break;
    }
    case DType::kDouble: {
        const double* src = data<double>();
        double* dst = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = src[i] > 0.0 ? src[i] : 0.0;
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for ReLU");
    }

    // 记录操作到自动微分计算图
    AutoGrad* ctx = AutoGradContext::current();
    if (ctx) {
        ctx->record_op(
            {&result},
            op::ReLU,
            {const_cast<Tensor*>(this)}
        );
    }
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    switch (_dtype) {
    case DType::kFloat: {
        const float* src = data<float>();
        float* dst = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
        }
        break;
    }
    case DType::kDouble: {
        const double* src = data<double>();
        double* dst = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = 1.0 / (1.0 + std::exp(-src[i]));
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for sigmoid");
    }

    // 记录操作到自动微分计算图
    AutoGrad* ctx = AutoGradContext::current();
    if (ctx) {
        ctx->record_op(
            {&result},
            op::Sigmoid,
            {const_cast<Tensor*>(this)}
        );
    }
    return result;
}

Tensor Tensor::tanh() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    switch (_dtype) {
    case DType::kFloat: {
        const float* src = data<float>();
        float* dst = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) {
            float exp_2x = std::exp(2 * src[i]);
            dst[i] = (exp_2x - 1) / (exp_2x + 1);
        }
        break;
    }
    case DType::kDouble: {
        const double* src = data<double>();
        double* dst = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            double exp_2x = std::exp(2 * src[i]);
            dst[i] = (exp_2x - 1) / (exp_2x + 1);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for tanh");
    }

    // 记录操作到自动微分计算图
    AutoGrad* ctx = AutoGradContext::current();
    if (ctx) {
        ctx->record_op(
            {&result},
            op::Tanh,
            {const_cast<Tensor*>(this)}
        );
    }
    return result;
}

Tensor Tensor::softmax(int dim) const {
    int actual_dim = dim;
   if (actual_dim < 0) {
       actual_dim = this->dim() + actual_dim;  // 负值表示从后往前计数
   }

   if (actual_dim < 0 || actual_dim >= static_cast<int>(this->dim())) {
       throw std::runtime_error("Invalid dimension for softmax");
   }
   Tensor result(ShapeTag{}, _shape, _dtype, _device);
   switch (_dtype) {
   case DType::kFloat: {
       const float* src = data<float>();
       float* dst = result.data<float>();

       // 计算每个切片的softmax
       size_t slice_size = _shape[dim];
       size_t num_slices = numel() / slice_size;

       for (size_t s = 0; s < num_slices; ++s) {
           // 找到最大值防止数值溢出
           float max_val = src[s * slice_size];
           for (size_t i = 1; i < slice_size; ++i) {
               if (src[s * slice_size + i] > max_val) {
                   max_val = src[s * slice_size + i];
               }
           }

           // 计算指数和
           float exp_sum = 0.0f;
           for (size_t i = 0; i < slice_size; ++i) {
               float val = std::exp(src[s * slice_size + i] - max_val);
               dst[s * slice_size + i] = val;
               exp_sum += val;
           }

           // 归一化
           for (size_t i = 0; i < slice_size; ++i) {
               dst[s * slice_size + i] /= exp_sum;
           }
       }
       break;
   }
   case DType::kDouble: {
       const double* src = data<double>();
       double* dst = result.data<double>();

       // 计算每个切片的softmax
       size_t slice_size = _shape[dim];
       size_t num_slices = numel() / slice_size;

       for (size_t s = 0; s < num_slices; ++s) {
           // 找到最大值防止数值溢出
           double max_val = src[s * slice_size];
           for (size_t i = 1; i < slice_size; ++i) {
               if (src[s * slice_size + i] > max_val) {
                   max_val = src[s * slice_size + i];
               }
           }

           // 计算指数和
           double exp_sum = 0.0;
           for (size_t i = 0; i < slice_size; ++i) {
               double val = std::exp(src[s * slice_size + i] - max_val);
               dst[s * slice_size + i] = val;
               exp_sum += val;
           }

           // 归一化
           for (size_t i = 0; i < slice_size; ++i) {
               dst[s * slice_size + i] /= exp_sum;
           }
       }
       break;
   }
   default:
       throw std::runtime_error("Unsupported dtype for softmax");
   }

   // 记录操作到自动微分计算图
   AutoGrad* ctx = AutoGradContext::current();
   if (ctx) {
       std::vector<Tensor*> outputs_vec = {&result};
       std::vector<Tensor*> inputs_vec = {const_cast<Tensor*>(this)};
       ctx->record_op(outputs_vec, op::Softmax, inputs_vec);
   }
   return result;
}

 Tensor::Tensor() : _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kFloat){}

Tensor::Tensor(float value): _shape({}), _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kFloat) {
    computeStrides();
    _storage = Storage(1, _dtype, _device);
    *_storage.data<float>() = value;
}

Tensor::Tensor(std::initializer_list<float> values): _shape({values.size()}), _storage_offset(0),_device(DeviceType::kCPU), _dtype(DType::kFloat) {
    computeStrides();
    _storage = Storage(values.begin(), values.size(), _dtype, _device);
}

Tensor::Tensor(std::initializer_list<bool> values): _shape({values.size()}), _storage_offset(0),
        _device(DeviceType::kCPU), _dtype(DType::kBool) {
    computeStrides();
    _storage = Storage(values.size(), _dtype, _device);
    bool* data = _storage.data<bool>();
    size_t i = 0;
    for (bool val : values) {
        data[i++] = val;
    }
}

 Tensor::Tensor(ShapeTag, const std::vector<size_t> &shape, DType dtype, DeviceType device,
 bool zero_init): _shape(shape), _storage_offset(0), _device(device), _dtype(dtype) {
    computeStrides();
    _storage = Storage(numel(), _dtype, _device);
    if(zero_init) zero();
}

Tensor::Tensor(const Tensor& other) : _shape(other._shape), _strides(other._strides),
         _storage_offset(other._storage_offset),
         _device(other._device), _dtype(other._dtype),
         _storage(other._storage.clone()) {}

Tensor::Tensor(Tensor &&other) noexcept:
        _strides(std::move(other._strides)),
        _storage_offset(other._storage_offset),
        _device(other._device),
        _dtype(other._dtype), _storage(std::move(other._storage)),
        _shape(std::move(other._shape)) {
    other._storage_offset = 0;
    other._shape.clear();
    other._strides.clear();
}

const std::vector<size_t> &Tensor::shape() const { return _shape; }

const std::vector<size_t> Tensor::strides() const { return _strides; }

size_t Tensor::dim() const { return _shape.size(); }

size_t Tensor::numel() const {
    if (_shape.empty()) return 1; // 标量有1个元素
    return std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<>());
}

DType Tensor::dtype() const { return _dtype; }

DeviceType Tensor::device() const { return _device; }

bool Tensor::is_contiguous() const {
    if (_shape.empty()) return 1; // 标量有1个元素
    return std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<>());
}

bool Tensor::isGradRequired() const { return _requires_grad; }

void Tensor::requires_grad(bool key) { _requires_grad = key; }

Tensor Tensor::grad() const {
    AutoGrad* ctx = AutoGradContext::current();
    if (!ctx)
        throw std::runtime_error("No active AD context");

    // 使用公共接口获取梯度
    return ctx->getGrad(const_cast<Tensor*>(this));
}

void Tensor::setDtype(DType dtype) { _dtype = dtype; }

size_t Tensor::storageOffset() const { return _storage_offset; }

Tensor Tensor::clone() const {
    Tensor copy;
    copy._shape = _shape;
    copy._strides = _strides;
    copy._storage_offset = 0;
    copy._device = _device;
    copy._dtype = _dtype;
    copy._storage = _storage.clone(); // 深拷贝存储
    copy._requires_grad = _requires_grad;
    return copy;
}

Tensor Tensor::view(const std::vector<size_t> &new_shape) const {
    // 计算新形状的元素总数
    size_t new_numel = 1;
    for (auto dim : new_shape) {
        if (dim == 0) throw std::runtime_error("Zero dimension in shape");
        new_numel *= dim;
    }

    // 验证新形状的元素总数是否匹配
    if (new_numel != numel()) {
        throw std::runtime_error("Shape size mismatch in view()");
    }

    if (!is_contiguous()) {
        throw std::runtime_error("Cannot view non-contiguous tensor. Call clone() first.");
    }

    Tensor result(ShapeTag{}, new_shape, _dtype, _device);
    result._storage = _storage;  // 共享存储（使用shared_ptr实现共享所有权）
    result._storage_offset = _storage_offset;
    result.computeStrides();  // 根据新形状重新计算步幅
    result._requires_grad= _requires_grad;
    return result;
}

Tensor Tensor::sum(const std::vector<int> &dims, bool keepdim) const {
    // 处理全局求和（dims 为空）
    if (dims.empty()) {
        Tensor result(ShapeTag{}, {}, _dtype, _device);
        result._storage = Storage(1, _dtype, _device);
        switch (_dtype) {
        case DType::kFloat: {
            float* dst = result.data<float>();
            const float* src = data<float>();
            *dst = std::accumulate(src, src + numel(), 0.0f);
            break;
        }
        case DType::kDouble: {
            double* dst = result.data<double>();
            const double* src = data<double>();
            *dst = std::accumulate(src, src + numel(), 0.0);
            break;
        }
        case DType::kInt: {
            int32_t* dst = result.data<int32_t>();
            const int32_t* src = data<int32_t>();
            *dst = std::accumulate(src, src + numel(), 0);
            break;
        }
        case DType::kLong: {
            int64_t* dst = result.data<int64_t>();
            const int64_t* src = data<int64_t>();
            *dst = std::accumulate(src, src + numel(), 0LL);
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for sum");
        }

        // 记录自动微分操作

        return result;
    }
}

Tensor Tensor::sum(int dim, bool keepdim) const { return sum(std::vector<int>{dim}, false); }

Tensor Tensor::sum() const { return sum(std::vector<int>{}, false); }

Tensor Tensor::transpose() const {
    if (dim() < 2) {
        throw std::runtime_error("transpose requires at least 2 dimensions");
    }

    // 创建新的形状和步幅
    std::vector<size_t> new_shape = _shape;
    std::vector<size_t> new_strides = _strides;

    // 交换最后两个维度
    std::swap(new_shape[dim()-1], new_shape[dim()-2]);
    std::swap(new_strides[dim()-1], new_strides[dim()-2]);

    Tensor result = *this;
    result._shape = new_shape;
    result._strides = new_strides;
    return result;
}

void Tensor::fill(float value) {
    switch (_dtype) {
    case DType::kFloat: {
        float* ptr = data<float>();
        std::fill(ptr, ptr + numel(), value);
        break;
    }
    case DType::kDouble: {
        double* ptr = data<double>();
        std::fill(ptr, ptr + numel(), static_cast<double>(value));
        break;
    }
    case DType::kInt: {
        int32_t* ptr = data<int32_t>();
        std::fill(ptr, ptr + numel(), static_cast<int32_t>(value));
        break;
    }
    case DType::kLong: {
        int64_t* ptr = data<int64_t>();
        std::fill(ptr, ptr + numel(), static_cast<int64_t>(value));
        break;
    }
    case DType::kBool: {
        bool* ptr = data<bool>();
        std::fill(ptr, ptr + numel(), value != 0.0f);
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for fill()");
    }
}

void Tensor::zero() {
    switch (_dtype) {
    case DType::kFloat:   fill(0.0f); break;
    case DType::kDouble:  fill(0.0); break;
    case DType::kInt:     fill(0); break;
    case DType::kLong:    fill(0L); break;
    case DType::kBool:    fill(false); break;
    default: throw std::runtime_error("Unsupported dtype for zero()");
    }
}

void Tensor::ones() {
    switch (_dtype) {
    case DType::kFloat:   fill(1.0f); break;
    case DType::kDouble:  fill(1.0); break;
    case DType::kInt:     fill(1); break;
    case DType::kLong:    fill(1); break;
    case DType::kBool:    fill(true); break;
    default: throw std::runtime_error("Unsupported dtype for ones()");
    }
}

bool Tensor::empty() const { return numel() == 0; }

void Tensor::zeroGrad() const {
    if (!AutoGradContext::current()) throw std::runtime_error("No active AG context");
    AutoGradContext::current()->zeroGrad(const_cast<Tensor *>(this));
}

void Tensor::serialize(std::ofstream &os) const { _storage.serialize(os); }

void Tensor::deserialize(std::ifstream &is) { _storage.deserialize(is); }

std::string Tensor::toString() const {
    std::string str;
    _requires_grad ? str = "True" : str = "False";
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < _shape.size(); ++i) {
        oss << _shape[i];
        if (i < _shape.size() - 1) oss << ", ";
    }
    oss << "], dtype=" << dtypeToString(_dtype)
        << ", device=" << (_device == DeviceType::kCPU ? "cpu" : "gpu") << ", AutoDiff=" << str << ")\n";

    // 打印张量内容
    if (numel() == 0) {
        oss << "[]";
    } else {
        oss << "[";
        try {
            if (_dtype == DType::kFloat) {
                printRecursive<float>(oss, 0, std::vector<size_t>());
            } else if (_dtype == DType::kDouble) {
                printRecursive<double>(oss, 0, std::vector<size_t>());
            } else if (_dtype == DType::kInt) {
                printRecursive<int32_t>(oss, 0, std::vector<size_t>());
            } else if (_dtype == DType::kLong) {
                printRecursive<int64_t>(oss, 0, std::vector<size_t>());
            } else if (_dtype == DType::kBool) {
                printRecursive<bool>(oss, 0, std::vector<size_t>());
            }
        } catch (const std::exception& e) {
            oss << "Error: " << e.what();
        }
        oss << "]";
    }
    return oss.str();
}

void Tensor::print() const { std::cout << toString() << std::endl; }

Tensor Tensor::transpose_last_two() const { return this->transpose(); }

Tensor Tensor::operator>(float scalar) const {
    Tensor result(ShapeTag{}, _shape, DType::kBool, _device);
    switch (_dtype) {
    case DType::kFloat: {
        const float* src = data<float>();
        bool* dst = result.data<bool>();
        for (size_t i = 0; i < numel(); ++i) dst[i] = src[i] > scalar;
        break;
    }
    case DType::kDouble: {
        const double * src = data<double>();
        bool* dst = result.data<bool>();
        for (size_t i = 0; i < numel(); ++i) dst[i] = src[i] > scalar;
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for comparison");
    }
    return result;
}

Tensor Tensor::operator+(const Tensor &rhs) const {
    // 验证形状和类型匹配
   if (_dtype != rhs._dtype) throw std::runtime_error("DType mismatch in addition");
   if (_device != rhs._device) throw std::runtime_error("Device mismatch in addition");
   if (_shape == rhs._shape){
       // 创建结果张量时继承上下文
       Tensor result(ShapeTag{}, _shape, _dtype, _device);
       const size_t n = numel();

       // 根据数据类型分派加法操作
       switch (_dtype) {
       case DType::kFloat:
           elementwiseOp<float>(result, *this, rhs, [](float a, float b) { return a + b; });
           break;
       case DType::kDouble:
           elementwiseOp<double>(result, *this, rhs, [](double a, double b) { return a + b; });
           break;
       case DType::kInt:
           elementwiseOp<int32_t>(result, *this, rhs, [](int32_t a, int32_t b) { return a + b; });
           break;
       case DType::kLong:
           elementwiseOp<int64_t>(result, *this, rhs, [](int64_t a, int64_t b) { return a + b; });
           break;
       default:
           throw std::runtime_error("Unsupported dtype for addition");
       }
       AutoGrad* ctx = AutoGradContext::current();
       if (ctx) {
           std::vector<Tensor*> outputs_vec = {&result};
           std::vector<Tensor*> inputs_vec = {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)};
           ctx->record_op(outputs_vec, op::Add, inputs_vec);
       }
       return result;
   } else {
       // 形状不同时进行广播
       BroadCastResult bc = broadCast(*this, rhs);

       // 创建结果张量（使用广播后的形状）
       Tensor result(ShapeTag{}, bc.logicShape, _dtype, _device);

       // 根据数据类型分派操作
       switch (_dtype) {
       case DType::kFloat:
           broadcast_elementwise_op<float>(
               result, *this, rhs, bc,
               [](float a, float b) { return a + b; }
           );
           break;
       case DType::kDouble:
           broadcast_elementwise_op<double>(
               result, *this, rhs, bc,
               [](double a, double b) { return a + b; }
           );
           break;
       case DType::kLong:
           broadcast_elementwise_op<long>(
               result, *this, rhs, bc,
               [](long a, long b) { return a + b; }
           );
           break;
       case DType::kInt:
           broadcast_elementwise_op<long>(
               result, *this, rhs, bc,
               [](int a, int b) { return a + b; }
           );
           break;
       default: throw std::runtime_error("Unsupported DType");
       }
       AutoGrad* ctx = AutoGradContext::current();
       if (ctx) {
           ctx->record_op(
               {&result},  // 输出张量列表
               op::Add,    // 操作类型
               {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)} // 输入张量
           );
       }
       return result;
   }
}

Tensor Tensor::operator-(const Tensor &rhs) const {
    // 验证形状和类型匹配
   if (_dtype != rhs._dtype) throw std::runtime_error("DType mismatch in addition");
   if (_device != rhs._device) throw std::runtime_error("Device mismatch in addition");
   if (_shape == rhs._shape){
       // 创建结果张量时继承上下文
       Tensor result(ShapeTag{}, _shape, _dtype, _device);
       const size_t n = numel();

       // 根据数据类型分派加法操作
       switch (_dtype) {
       case DType::kFloat:
           elementwiseOp<float>(result, *this, rhs, [](float a, float b) { return a - b; });
           break;
       case DType::kDouble:
           elementwiseOp<double>(result, *this, rhs, [](double a, double b) { return a - b; });
           break;
       case DType::kInt:
           elementwiseOp<int32_t>(result, *this, rhs, [](int32_t a, int32_t b) { return a - b; });
           break;
       case DType::kLong:
           elementwiseOp<int64_t>(result, *this, rhs, [](int64_t a, int64_t b) { return a - b; });
           break;
       default:
           throw std::runtime_error("Unsupported dtype for addition");
       }
       AutoGrad* ctx = AutoGradContext::current();
       if (ctx) {
           std::vector<Tensor*> outputs_vec = {&result};
           std::vector<Tensor*> inputs_vec = {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)};
           ctx->record_op(outputs_vec, op::Sub, inputs_vec);
       }
       return result;
   } else {
       // 形状不同时进行广播
       BroadCastResult bc = broadCast(*this, rhs);

       // 创建结果张量（使用广播后的形状）

       Tensor result(ShapeTag{}, bc.logicShape, _dtype, _device);

       // 根据数据类型分派操作
       switch (_dtype) {
       case DType::kFloat:
           broadcast_elementwise_op<float>(
               result, *this, rhs, bc,
               [](float a, float b) { return a - b; }
           );
           break;
       case DType::kDouble:
           broadcast_elementwise_op<double>(
               result, *this, rhs, bc,
               [](double a, double b) { return a - b; }
           );
           break;
       case DType::kLong:
           broadcast_elementwise_op<long>(
               result, *this, rhs, bc,
               [](long a, long b) { return a - b; }
           );
           break;
       case DType::kInt:
           broadcast_elementwise_op<long>(
               result, *this, rhs, bc,
               [](int a, int b) { return a - b; }
           );
           break;
       default: throw std::runtime_error("Unsupported DType");
       }

       // 记录自动微分操作
       AutoGrad* ctx = AutoGradContext::current();
       if (ctx) {
           std::vector<Tensor*> outputs_vec = {&result};
           std::vector<Tensor*> inputs_vec = {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)};
           ctx->record_op(outputs_vec, op::Sub, inputs_vec);
       }

       return result;
   }
}

Tensor Tensor::operator-(float scalar) const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);

    switch (_dtype) {
    case DType::kFloat: {
        const float* src = data<float>();
        float* dst = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = src[i] - scalar;
        }
        break;
    }
    case DType::kDouble: {
        const double* src = data<double>();
        double* dst = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = src[i] - static_cast<double>(scalar);
        }
        break;
    }
    case DType::kInt: {
        const int32_t* src = data<int32_t>();
        int32_t* dst = result.data<int32_t>();
        int32_t s = static_cast<int32_t>(scalar);
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = src[i] - s;
        }
        break;
    }
    case DType::kLong: {
        const int64_t* src = data<int64_t>();
        int64_t* dst = result.data<int64_t>();
        int64_t s = static_cast<int64_t>(scalar);
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = src[i] - s;
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for subtraction with scalar");
    }

    // 自动微分记录
    AutoGrad* ctx = AutoGradContext::current();
    if (ctx) {
        ctx->record_op(
            {&result},
            op::Sub,
            {const_cast<Tensor*>(this)}
        );
    }

    return result;
}

Tensor Tensor::operator*(const Tensor &rhs) const {
    // 验证形状和类型匹配
   if (_dtype != rhs._dtype) throw std::runtime_error("DType mismatch in addition");
   if (_device != rhs._device) throw std::runtime_error("Device mismatch in addition");
   if (_shape == rhs._shape){
       // 创建结果张量时继承上下文
       Tensor result(ShapeTag{}, _shape, _dtype, _device);
       const size_t n = numel();

       // 根据数据类型分派加法操作
       switch (_dtype) {
       case DType::kFloat:
           elementwiseOp<float>(result, *this, rhs, [](float a, float b) { return a * b; });
           break;
       case DType::kDouble:
           elementwiseOp<double>(result, *this, rhs, [](double a, double b) { return a * b; });
           break;
       case DType::kInt:
           elementwiseOp<int32_t>(result, *this, rhs, [](int32_t a, int32_t b) { return a * b; });
           break;
       case DType::kLong:
           elementwiseOp<int64_t>(result, *this, rhs, [](int64_t a, int64_t b) { return a * b; });
           break;
       default:
           throw std::runtime_error("Unsupported dtype for addition");
       }
       AutoGrad* ctx = AutoGradContext::current();
       if (ctx) {
           std::vector<Tensor*> outputs_vec = {&result};
           std::vector<Tensor*> inputs_vec = {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)};
           ctx->record_op(outputs_vec, op::Mul, inputs_vec);
       }
       return result;
   } else {
       // 形状不同时进行广播
       BroadCastResult bc = broadCast(*this, rhs);

       // 创建结果张量（使用广播后的形状）
       Tensor result(ShapeTag{}, bc.logicShape, _dtype, _device);

       // 根据数据类型分派操作
       switch (_dtype) {
       case DType::kFloat:
           broadcast_elementwise_op<float>(
               result, *this, rhs, bc,
               [](float a, float b) { return a * b; }
           );
           break;
       case DType::kDouble:
           broadcast_elementwise_op<double>(
               result, *this, rhs, bc,
               [](double a, double b) { return a * b; }
           );
           break;
       case DType::kLong:
           broadcast_elementwise_op<long>(
               result, *this, rhs, bc,
               [](long a, long b) { return a * b; }
           );
           break;
       case DType::kInt:
           broadcast_elementwise_op<long>(
               result, *this, rhs, bc,
               [](int a, int b) { return a * b; }
           );
           break;
       default: throw std::runtime_error("Unsupported DType");
       }

       // 记录自动微分操作
       AutoGrad* ctx = AutoGradContext::current();
       if (ctx) {
           std::vector<Tensor*> outputs_vec = {&result};
           std::vector<Tensor*> inputs_vec = {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)};
           ctx->record_op(outputs_vec, op::Mul, inputs_vec);
       }

       return result;
   }
}

Tensor Tensor::operator/(const Tensor &rhs) const {
    // 验证形状和类型匹配
   if (_dtype != rhs._dtype) throw std::runtime_error("DType mismatch in addition");
   if (_device != rhs._device) throw std::runtime_error("Device mismatch in addition");
   if (_shape == rhs._shape){
       // 创建结果张量时继承上下文
       Tensor result(ShapeTag{}, _shape, _dtype, _device);
       const size_t n = numel();

       // 根据数据类型分派加法操作
       switch (_dtype) {
       case DType::kFloat:
           elementwiseOp<float>(result, *this, rhs, [](float a, float b) {
               if (b == 0.0f) throw std::runtime_error("Division by zero");
               return a / b;
           });
           break;
       case DType::kDouble:
           elementwiseOp<double>(result, *this, rhs, [](double a, double b) {
               if (b == 0.0) throw std::runtime_error("Division by zero");
               return a / b;
           });
           break;
       case DType::kInt:
           elementwiseOp<int32_t>(result, *this, rhs, [](int32_t a, int32_t b) {
               if (b == 0) throw std::runtime_error("Division by zero");
               return a / b;
           });
           break;
       case DType::kLong:
           elementwiseOp<int64_t>(result, *this, rhs, [](int64_t a, int64_t b) {
               if (b == 0) throw std::runtime_error("Division by zero");
               return a / b;
           });
           break;
       default:
           throw std::runtime_error("Unsupported dtype for addition");
       }
       AutoGrad* ctx = AutoGradContext::current();
       if (ctx) {
           std::vector<Tensor*> outputs_vec = {&result};
           std::vector<Tensor*> inputs_vec = {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)};
           ctx->record_op(outputs_vec, op::Div, inputs_vec);
       }
       return result;
   } else {
       // 形状不同时进行广播
       BroadCastResult bc = broadCast(*this, rhs);

       // 创建结果张量（使用广播后的形状）
       Tensor result(ShapeTag{}, bc.logicShape, _dtype, _device);

       // 根据数据类型分派操作
       switch (_dtype) {
       case DType::kFloat:
           broadcast_elementwise_op<float>(
               result, *this, rhs, bc,
               [](float a, float b) {
                   if (b == 0.0f) throw std::runtime_error("Division by zero");
                   return a / b; }
           );
           break;
       case DType::kDouble:
           broadcast_elementwise_op<double>(
               result, *this, rhs, bc,
               [](double a, double b) {
                   if (b == 0.0) throw std::runtime_error("Division by zero");
                   return a / b; }
           );
           break;
       case DType::kLong:
           broadcast_elementwise_op<long>(
               result, *this, rhs, bc,
               [](long a, long b) {
                   if (b == 0) throw std::runtime_error("Division by zero");
                   return a / b; }
           );
           break;
       case DType::kInt:
           broadcast_elementwise_op<long>(
               result, *this, rhs, bc,
               [](int a, int b) {
                   if (b == 0) throw std::runtime_error("Division by zero");
                   return a / b; }
           );
           break;
       default: throw std::runtime_error("Unsupported DType");
       }

       AutoGrad* ctx = AutoGradContext::current();
       if (ctx) {
           std::vector<Tensor*> outputs_vec = {&result};
           std::vector<Tensor*> inputs_vec = {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)};
           ctx->record_op(outputs_vec, op::Div, inputs_vec);
       }
       return result;
   }
}

Tensor Tensor::operator-() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    switch (_dtype) {
    case DType::kFloat: {
        float* ptr = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) ptr[i] = -ptr[i];
        break;
    }
    case DType::kDouble: {
        double * ptr = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) ptr[i] = -ptr[i];
        break;
    }
    case DType::kLong: {
        long* ptr = result.data<long>();
        for (size_t i = 0; i < numel(); ++i) ptr[i] = -ptr[i];
        break;
    }
    case DType::kInt: {
        int* ptr = result.data<int>();
        for (size_t i = 0; i < numel(); ++i) ptr[i] = -ptr[i];
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for subtraction");
    }
    return result;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor result = *this;
    for (size_t i{0};i<_storage.size();i++) result.data<double>()[i] = _storage.data<double>()[i]*scalar;
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result = *this;
    for (size_t i{0};i<_storage.size();i++) result.data<float>()[i] = _storage.data<float>()[i]*scalar;
    return result;
}

Tensor Tensor::operator*(int scalar) const {
    Tensor result = *this;
    for (size_t i{0};i<_storage.size();i++) result.data<int>()[i] = _storage.data<int>()[i]*scalar;
    return result;
}

Tensor Tensor::operator*(long scalar) const {
    Tensor result = *this;
    for (size_t i{0};i<_storage.size();i++) result.data<long>()[i] = _storage.data<long>()[i]*scalar;
    return result;
}

Tensor Tensor::operator/(double scalar) const {
    Tensor result = *this;
    for (size_t i{0};i<_storage.size();i++) result.data<double>()[i] = _storage.data<double>()[i]*(1/scalar);
    return result;
}

Tensor Tensor::operator/(float scalar) const {
    Tensor result = *this;
    for (size_t i{0};i<_storage.size();i++) result.data<float>()[i] = _storage.data<float>()[i]*(1/scalar);
    return result;
}

Tensor Tensor::operator/(int scalar) const {
    Tensor result = *this;
    for (size_t i{0};i<_storage.size();i++) result.data<int>()[i] = _storage.data<int>()[i]*(1/scalar);
    return result;
}

Tensor Tensor::operator/(long scalar) const {
    Tensor result = *this;
    for (size_t i{0};i<_storage.size();i++) result.data<long>()[i] = _storage.data<long>()[i]*(1/scalar);
    return result;
}

Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
        _shape = other._shape;
        _strides = other._strides;
        _storage_offset = other._storage_offset;
        _device = other._device;
        _dtype = other._dtype;
        _storage = other._storage.clone(); // 深拷贝存储
        _requires_grad = other._requires_grad;
    }
    return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
    if (this != &other) {
        _shape = std::move(other._shape);
        _strides = std::move(other._strides);
        _storage_offset = other._storage_offset;
        _device = other._device;
        _dtype = other._dtype;
        _storage = std::move(other._storage);
        _requires_grad = other._requires_grad;

        other._storage_offset = 0;
        other._shape.clear();
        other._strides.clear();
    }
    return *this;
}

bool Tensor::operator==(const Tensor &other) const {
    if (_shape != other._shape || _dtype != other._dtype) {
        return false;
    }

    const size_t n = numel();
    if (n == 0) return true; // 空张量相等

    switch (_dtype) {
    case DType::kFloat:
        return std::equal(data<float>(), data<float>() + n, other.data<float>());
    case DType::kDouble:
        return std::equal(data<double>(), data<double>() + n, other.data<double>());
    case DType::kInt:
        return std::equal(data<int32_t>(), data<int32_t>() + n, other.data<int32_t>());
    case DType::kLong:
        return std::equal(data<int64_t>(), data<int64_t>() + n, other.data<int64_t>());
    case DType::kBool:
        return std::equal(data<bool>(), data<bool>() + n, other.data<bool>());
    default:
        return false;
    }
}

void Tensor::backward(Tensor &root, Tensor grad_output) {
    if (!AutoGradContext::current())
        throw std::runtime_error("No active AD context");

    AutoGradContext::current()->backward(root, grad_output);
}

void Tensor::registerHook(Hook _fn) { _hooks.push_back(_fn); }

void Tensor::removeHook(size_t idx) { _hooks.erase(_hooks.begin() + idx); }

void Tensor::removeAllHooks() { _hooks.clear(); }

std::vector<Tensor::Hook> Tensor::hooks() const { return _hooks; }

Tensor::Hook Tensor::hook(size_t idx) const { return _hooks.at(idx); }

// MatMul
Tensor matMul(const Tensor &a, const Tensor &b) {
    // 严格检查维度：必须是2D矩阵
    if (a.dim() != 2 || b.dim() != 2) {
        throw std::runtime_error("matMul requires both tensors to be 2D matrices");
    }

    // 检查内层维度是否匹配
    size_t a_cols = a.shape()[1];
    size_t b_rows = b.shape()[0];
    if (a_cols != b_rows) {
        std::ostringstream oss;
        oss << "Matrix dimensions mismatch: " << a_cols << " != " << b_rows;
        throw std::runtime_error(oss.str());
    }

    // 创建结果张量 [M x N]
    size_t M = a.shape()[0];
    size_t N = b.shape()[1];
    Tensor result(ShapeTag{}, {M, N}, a.dtype(), a.device());
    result.zero(); // 初始化结果为零矩阵

    // 根据数据类型进行计算
    switch (a.dtype()) {
    case DType::kFloat: {
        const float* a_data = a.data<float>();
        const float* b_data = b.data<float>();
        float* r_data = result.data<float>();
        const int BLOCK = 512;
        // 三层分块循环
#pragma omp parallel for
        for (int i0 = 0; i0 < M; i0 += BLOCK) {
            int i_end = minx(i0 + BLOCK, M);  // 计算行边界
            for (int k0 = 0; k0 < a_cols; k0 += BLOCK) {
                int k_end = minx(k0 + BLOCK, a_cols);  // 计算中间维度边界
                for (int j0 = 0; j0 < N; j0 += BLOCK) {
                    int j_end = minx(j0 + BLOCK, N);  // 计算列边界
                    // 核心计算：只处理完整块内的元素
                    for (int i = i0; i < i_end; i++) {
                        for (int k = k0; k < k_end; k++) {
                            float a_val = a_data[i*a_cols + k];  // 一次加载A元素
                            // 内层循环：连续访问B和C
                            for (int j = j0; j < j_end; j++) {
                                r_data[i*N + j] += a_val * b_data[k*N + j];
                            }
                        }
                    }
                }
            }
        }
        break;
    }
    case DType::kDouble: {
        const double* a_data = a.data<double>();
        const double* b_data = b.data<double>();
        double* r_data = result.data<double>();

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < a_cols; ++k) {
                double a_val = a_data[i * a_cols + k];
                for (size_t j = 0; j < N; ++j) {
                    r_data[i * N + j] += a_val * b_data[k * N + j];
                }
            }
        }
        break;
    }
    case DType::kInt: {
        const int* a_data = a.data<int>();
        const int* b_data = b.data<int>();
        int* r_data = result.data<int>();

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < a_cols; ++k) {
                int a_val = a_data[i * a_cols + k];
                for (size_t j = 0; j < N; ++j) {
                    r_data[i * N + j] += a_val * b_data[k * N + j];
                }
            }
        }
        break;
    }
    case DType::kLong: {
        const long* a_data = a.data<long>();
        const long* b_data = b.data<long>();
        long* r_data = result.data<long>();

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < a_cols; ++k) {
                long a_val = a_data[i * a_cols + k];
                for (size_t j = 0; j < N; ++j) {
                    r_data[i * N + j] += a_val * b_data[k * N + j];
                }
            }
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for matMul");
    }

    return result;
}

Tensor matMulNative(const Tensor &a,const Tensor &b){
   // 检查维度
   if (a.dim() < 2 || b.dim() < 2) {
       throw std::runtime_error("Both tensors must be at least 2D for matrix multiplication");
   }

   // 检查内层维度是否匹配
   size_t a_cols = a.shape()[a.dim() - 1];
   size_t b_rows = b.shape()[b.dim() - 2];

   if (a_cols != b_rows) {
       std::ostringstream oss;
       oss << "Matrix dimensions mismatch: " << a_cols << " != " << b_rows;
       throw std::runtime_error(oss.str());
   }

   // 处理批量矩阵乘法
   BroadCastResult logic = broadCast(a, b);
   const std::vector<size_t>& batch_dims = logic.logicShape;
   size_t batch_size = std::accumulate(batch_dims.begin(), batch_dims.end() - 2, 1, std::multiplies<size_t>());
   size_t M = a.shape()[a.dim() - 2];
   size_t K = a.shape()[a.dim() - 1];
   size_t N = b.shape()[b.dim() - 1];

   // 创建结果张量
   std::vector<size_t> result_shape = batch_dims;
   result_shape[result_shape.size() - 2] = M;
   result_shape[result_shape.size() - 1] = N;

   ShapeTag tag;
   Tensor result(tag, result_shape, a.dtype(), a.device());
   result.zero();

   // 根据数据类型进行计算
   switch (a.dtype()) {
   case DType::kFloat: {
       const float* a_data = a.data<float>();
       const float* b_data = b.data<float>();
       float* r_data = result.data<float>();

       // 批量矩阵乘法
       for (size_t batch = 0; batch < batch_size; ++batch) {
           for (size_t i = 0; i < M; ++i) {
               for (size_t k = 0; k < K; ++k) {
                   float a_val = a_data[batch * M * K + i * K + k];
                   for (size_t j = 0; j < N; ++j) {
                       r_data[batch * M * N + i * N + j] +=
                           a_val * b_data[batch * K * N + k * N + j];
                   }
               }
           }
       }
       break;
   }
   case DType::kDouble: {
       const double * a_data = a.data<double>();
       const double* b_data = b.data<double>();
       double* r_data = result.data<double>();

       // 批量矩阵乘法
       for (size_t batch = 0; batch < batch_size; ++batch) {
           for (size_t i = 0; i < M; ++i) {
               for (size_t k = 0; k < K; ++k) {
                   float a_val = a_data[batch * M * K + i * K + k];
                   for (size_t j = 0; j < N; ++j) {
                       r_data[batch * M * N + i * N + j] +=
                           a_val * b_data[batch * K * N + k * N + j];
                   }
               }
           }
       }
       break;
   }
   case DType::kInt: {
       const int* a_data = a.data<int>();
       const int* b_data = b.data<int>();
       int* r_data = result.data<int>();

       // 批量矩阵乘法
       for (size_t batch = 0; batch < batch_size; ++batch) {
           for (size_t i = 0; i < M; ++i) {
               for (size_t k = 0; k < K; ++k) {
                   float a_val = a_data[batch * M * K + i * K + k];
                   for (size_t j = 0; j < N; ++j) {
                       r_data[batch * M * N + i * N + j] +=
                           a_val * b_data[batch * K * N + k * N + j];
                   }
               }
           }
       }
       break;
   }
   case DType::kLong: {
       const long * a_data = a.data<long>();
       const long * b_data = b.data<long>();
       long* r_data = result.data<long>();

       // 批量矩阵乘法
       for (size_t batch = 0; batch < batch_size; ++batch) {
           for (size_t i = 0; i < M; ++i) {
               for (size_t k = 0; k < K; ++k) {
                   float a_val = a_data[batch * M * K + i * K + k];
                   for (size_t j = 0; j < N; ++j) {
                       r_data[batch * M * N + i * N + j] +=
                           a_val * b_data[batch * K * N + k * N + j];
                   }
               }
           }
       }
       break;
   }
   default:
       throw std::runtime_error("Unsupported dtype for matMul");
   }

   return result;
}

Tensor matMulBlocked(const Tensor &a,const Tensor &b){
   // 严格检查维度：必须是2D矩阵
   if (a.dim() != 2 || b.dim() != 2) {
       throw std::runtime_error("matMul requires both tensors to be 2D matrices");
   }

   // 检查内层维度是否匹配
   size_t a_cols = a.shape()[1];
   size_t b_rows = b.shape()[0];
   if (a_cols != b_rows) {
       std::ostringstream oss;
       oss << "Matrix dimensions mismatch: " << a_cols << " != " << b_rows;
       throw std::runtime_error(oss.str());
   }

   // 创建结果张量 [M x N]
   size_t M = a.shape()[0];
   size_t N = b.shape()[1];
   Tensor result(ShapeTag{}, {M, N}, a.dtype(), a.device());
   result.zero(); // 初始化结果为零矩阵

   // 根据数据类型进行计算
   switch (a.dtype()) {
   case DType::kFloat: {
       const float* a_data = a.data<float>();
       const float* b_data = b.data<float>();
       float* r_data = result.data<float>();
       const int BLOCK = 512;
       // 三层分块循环
       for (int i0 = 0; i0 < M; i0 += BLOCK) {
           int i_end = minx(i0 + BLOCK, M);  // 计算行边界
           for (int k0 = 0; k0 < a_cols; k0 += BLOCK) {
               int k_end = minx(k0 + BLOCK, a_cols);  // 计算中间维度边界
               for (int j0 = 0; j0 < N; j0 += BLOCK) {
                   int j_end = minx(j0 + BLOCK, N);  // 计算列边界
                   // 核心计算：只处理完整块内的元素
                   for (int i = i0; i < i_end; i++) {
                       for (int k = k0; k < k_end; k++) {
                           float a_val = a_data[i*a_cols + k];  // 一次加载A元素
                           // 内层循环：连续访问B和C
                           for (int j = j0; j < j_end; j++) {
                               r_data[i*N + j] += a_val * b_data[k*N + j];
                           }
                       }
                   }
               }
           }
       }
       break;
   }
   case DType::kDouble: {
       const double* a_data = a.data<double>();
       const double* b_data = b.data<double>();
       double* r_data = result.data<double>();

       for (size_t i = 0; i < M; ++i) {
           for (size_t k = 0; k < a_cols; ++k) {
               double a_val = a_data[i * a_cols + k];
               for (size_t j = 0; j < N; ++j) {
                   r_data[i * N + j] += a_val * b_data[k * N + j];
               }
           }
       }
       break;
   }
   case DType::kInt: {
       const int* a_data = a.data<int>();
       const int* b_data = b.data<int>();
       int* r_data = result.data<int>();

       for (size_t i = 0; i < M; ++i) {
           for (size_t k = 0; k < a_cols; ++k) {
               int a_val = a_data[i * a_cols + k];
               for (size_t j = 0; j < N; ++j) {
                   r_data[i * N + j] += a_val * b_data[k * N + j];
               }
           }
       }
       break;
   }
   case DType::kLong: {
       const long* a_data = a.data<long>();
       const long* b_data = b.data<long>();
       long* r_data = result.data<long>();

       for (size_t i = 0; i < M; ++i) {
           for (size_t k = 0; k < a_cols; ++k) {
               long a_val = a_data[i * a_cols + k];
               for (size_t j = 0; j < N; ++j) {
                   r_data[i * N + j] += a_val * b_data[k * N + j];
               }
           }
       }
       break;
   }
   default:
       throw std::runtime_error("Unsupported dtype for matMul");
   }

   return result;
}

Tensor matMulAMX(const Tensor &a, const Tensor &b){
   // 严格检查维度：必须是2D矩阵
   if (a.dim() != 2 || b.dim() != 2) {
       throw std::runtime_error("matMul requires both tensors to be 2D matrices");
   }

   // 检查内层维度是否匹配
   size_t a_cols = a.shape()[1];
   size_t b_rows = b.shape()[0];
   if (a_cols != b_rows) {
       std::ostringstream oss;
       oss << "Matrix dimensions mismatch: " << a_cols << " != " << b_rows;
       throw std::runtime_error(oss.str());
   }

   // 创建结果张量 [M x N]
   size_t M = a.shape()[0];
   size_t N = b.shape()[1];
   size_t K = a_cols; // 公共维度
   Tensor result(ShapeTag{}, {M, N}, a.dtype(), a.device());
   result.zero(); // 初始化结果为零矩阵

   // 根据数据类型进行计算
   switch (a.dtype()) {
   case DType::kFloat: {
       const float* a_data = a.data<float>();
       const float* b_data = b.data<float>();
       float* r_data = result.data<float>();

#ifdef __APPLE__
       // 使用Apple的AMX加速BLAS库
       cblas_sgemm(CblasRowMajor,   // 行主序存储
                   CblasNoTrans,   // 不转置A
                   CblasNoTrans,   // 不转置B
                   M,              // A的行数
                   N,              // B的列数
                   K,              // 公共维度
                   1.0f,           // alpha系数
                   a_data,         // A数据指针
                   K,              // A的列步幅（lda）
                   b_data,         // B数据指针
                   N,              // B的列步幅（ldb）
                   0.0f,           // beta系数
                   r_data,         // 结果数据指针
                   N);             // 结果的列步幅（ldc）
#endif
       break;
   }
   case DType::kDouble: {
       const double* a_data = a.data<double>();
       const double* b_data = b.data<double>();
       double* r_data = result.data<double>();

       // 双精度使用标准实现
       for (size_t i = 0; i < M; ++i) {
           for (size_t k = 0; k < K; ++k) {
               double a_val = a_data[i * K + k];
               for (size_t j = 0; j < N; ++j) {
                   r_data[i * N + j] += a_val * b_data[k * N + j];
               }
           }
       }
       break;
   }
   case DType::kInt: {
       const int* a_data = a.data<int>();
       const int* b_data = b.data<int>();
       int* r_data = result.data<int>();

       for (size_t i = 0; i < M; ++i) {
           for (size_t k = 0; k < K; ++k) {
               int a_val = a_data[i * K + k];
               for (size_t j = 0; j < N; ++j) {
                   r_data[i * N + j] += a_val * b_data[k * N + j];
               }
           }
       }
       break;
   }
   case DType::kLong: {
       const long* a_data = a.data<long>();
       const long* b_data = b.data<long>();
       long* r_data = result.data<long>();

       for (size_t i = 0; i < M; ++i) {
           for (size_t k = 0; k < K; ++k) {
               long a_val = a_data[i * K + k];
               for (size_t j = 0; j < N; ++j) {
                   r_data[i * N + j] += a_val * b_data[k * N + j];
               }
           }
       }
       break;
   }
   default:
       throw std::runtime_error("Unsupported dtype for matMul");
   }

   return result;
}

Tensor matMulRecursive(const Tensor &a, const Tensor &b){
   // 获取矩阵维度
   size_t M = a.shape()[0];
   size_t K = a.shape()[1];
   size_t N = b.shape()[1];

   // 基准情况：小矩阵使用分块优化
   if (M <= SMALL_SIZE || N <= SMALL_SIZE || K <= SMALL_SIZE) {
       return matMulRecursive(a, b);
   }
   // 中等矩阵使用AMX加速
   else if (M <= MEDIUM_SIZE && N <= MEDIUM_SIZE && K <= MEDIUM_SIZE) {
       return matMulAMX(a, b);
   }

   // 大矩阵使用Strassen算法递归
   // 计算填充后的新维度（偶数）
   size_t new_M = (M % 2 == 0) ? M : M + 1;
   size_t new_K = (K % 2 == 0) ? K : K + 1;
   size_t new_N = (N % 2 == 0) ? N : N + 1;

   // 创建填充后的矩阵并初始化为0
   Tensor A_pad(ShapeTag{}, {new_M, new_K}, a.dtype(), a.device());
   Tensor B_pad(ShapeTag{}, {new_K, new_N}, b.dtype(), b.device());
   A_pad.zero();
   B_pad.zero();

   // 将原始数据复制到填充矩阵的左上角
   for (size_t i = 0; i < M; i++) {
       for (size_t j = 0; j < K; j++) {
           A_pad({i, j}) = a({i, j});
       }
   }
   for (size_t i = 0; i < K; i++) {
       for (size_t j = 0; j < N; j++) {
           B_pad({i, j}) = b({i, j});
       }
   }

   // 计算子矩阵大小
   size_t half_M = new_M / 2;
   size_t half_K = new_K / 2;
   size_t half_N = new_N / 2;

   // 提取子矩阵（通过拷贝）
   auto slice = [](const Tensor& t, size_t start_i, size_t end_i, size_t start_j, size_t end_j) {
       Tensor result(ShapeTag{}, {end_i - start_i, end_j - start_j}, t.dtype(), t.device());
       for (size_t i = start_i; i < end_i; i++) {
           for (size_t j = start_j; j < end_j; j++) {
               switch (t.dtype()) {
               case DType::kFloat:
                   result({i - start_i, j - start_j}) = t({i, j});
                   break;
               case DType::kDouble:
                   result({i - start_i, j - start_j}) = t({i, j});
                   break;
               case DType::kInt:
                   result({i - start_i, j - start_j}) = t({i, j});
                   break;
               case DType::kLong:
                   result({i - start_i, j - start_j}) = t({i, j});
                   break;
               default: break;
               }
           }
       }
       return result;
   };

   // 划分矩阵
   Tensor A11 = slice(A_pad, 0, half_M, 0, half_K);
   Tensor A12 = slice(A_pad, 0, half_M, half_K, new_K);
   Tensor A21 = slice(A_pad, half_M, new_M, 0, half_K);
   Tensor A22 = slice(A_pad, half_M, new_M, half_K, new_K);

   Tensor B11 = slice(B_pad, 0, half_K, 0, half_N);
   Tensor B12 = slice(B_pad, 0, half_K, half_N, new_N);
   Tensor B21 = slice(B_pad, half_K, new_K, 0, half_N);
   Tensor B22 = slice(B_pad, half_K, new_K, half_N, new_N);

   // 计算Strassen的7个乘法
   Tensor M1 = matMulRecursive(A11 + A22, B11 + B22);
   Tensor M2 = matMulRecursive(A21 + A22, B11);
   Tensor M3 = matMulRecursive(A11, B12 - B22);
   Tensor M4 = matMulRecursive(A22, B21 - B11);
   Tensor M5 = matMulRecursive(A11 + A12, B22);
   Tensor M6 = matMulRecursive(A21 - A11, B11 + B12);
   Tensor M7 = matMulRecursive(A12 - A22, B21 + B22);

   // 计算结果子矩阵
   Tensor C11 = M1 + M4 - M5 + M7;
   Tensor C12 = M3 + M5;
   Tensor C21 = M2 + M4;
   Tensor C22 = M1 - M2 + M3 + M6;

   // 组合结果矩阵
   Tensor C_pad(ShapeTag{}, {new_M, new_N}, a.dtype(), a.device());
   auto assign_block = [](Tensor& dest, const Tensor& src, size_t start_i, size_t start_j) {
       for (size_t i = 0; i < src.shape()[0]; i++) {
           for (size_t j = 0; j < src.shape()[1]; j++) {
               switch (src.dtype()) {
               case DType::kFloat:
                   dest({start_i + i, start_j + j}) = src({i, j});
                   break;
               case DType::kDouble:
                   dest({start_i + i, start_j + j}) = src({i, j});
                   break;
               case DType::kInt:
                   dest({start_i + i, start_j + j}) = src({i, j});
                   break;
               case DType::kLong:
                   dest({start_i + i, start_j + j}) = src({i, j});
                   break;
               default: break;
               }
           }
       }
   };

   assign_block(C_pad, C11, 0, 0);
   assign_block(C_pad, C12, 0, half_N);
   assign_block(C_pad, C21, half_M, 0);
   assign_block(C_pad, C22, half_M, half_N);

   // 裁剪回原始大小
   Tensor C = slice(C_pad, 0, M, 0, N);
   return C;
}

Tensor matMulTest(const Tensor &a, const Tensor &b){
   // 检查维度
   if (a.dim() != 2 || b.dim() != 2) {
       throw std::runtime_error("matMul requires both tensors to be 2D matrices");
   }

   // 检查内层维度
   size_t a_cols = a.shape()[1];
   size_t b_rows = b.shape()[0];
   if (a_cols != b_rows) {
       std::ostringstream oss;
       oss << "Matrix dimensions mismatch: " << a_cols << " != " << b_rows;
       throw std::runtime_error(oss.str());
   }

   // 根据矩阵大小选择算法
   size_t M = a.shape()[0];
   size_t K = a.shape()[1];
   size_t N = b.shape()[1];

   if (M <= SMALL_SIZE && N <= SMALL_SIZE && K <= SMALL_SIZE) {
       return matMulBlocked(a, b);  // 小矩阵使用分块优化
   }
   else if (M <= MEDIUM_SIZE && N <= MEDIUM_SIZE && K <= MEDIUM_SIZE) {
       return matMulAMX(a, b);     // 中等矩阵使用AMX加速
   }
   else {
       return matMulRecursive(a, b); // 大矩阵使用Strassen递归
   }
}

// AutoGradContext
AutoGrad *&AutoGradContext::current() {
    thread_local AutoGrad* ctx = nullptr;
    return ctx;
}

AutoGradContext::Guard::Guard(AutoGrad *ctx):prev_ctx(current()) {
    current() = ctx;  // 现在可以正确赋值
}

 AutoGradContext::Guard::~Guard() { current() = prev_ctx; }

AutoGrad::Node::Node(Tensor t, bool req_grad, bool leaf)
    : tensor(std::move(t)), requires_grad(req_grad), is_leaf(leaf) {
    if (requires_grad) {
        // 初始化梯度为相同形状的零张量
        grad = Tensor(ShapeTag{}, tensor.shape(), tensor.dtype(), tensor.device());
        grad.zero();
    }
}

void AutoGrad::backward_add(Node *node) {
    Tensor& grad_out = node->grad;
    for (Node* input_node : node->inputs) {
        if (!input_node->requires_grad) continue;

        // 处理广播：将梯度reduce到输入形状
        Tensor grad_input = reduce_to_match(grad_out, input_node->tensor.shape());
        input_node->grad = input_node->grad + grad_input;
    }
}

void AutoGrad::backward_sub(Node *node) {
    Tensor& grad_out = node->grad;
    if (node->inputs.size() != 2) {
        throw std::runtime_error("Subtraction requires exactly 2 inputs");
    }

    Node* a = node->inputs[0];
    Node* b = node->inputs[1];

    if (a->requires_grad) {
        Tensor grad_a = reduce_to_match(grad_out, a->tensor.shape());
        a->grad = a->grad + grad_a;
    }

    if (b->requires_grad) {
        Tensor grad_b = reduce_to_match(grad_out, b->tensor.shape());
        b->grad = b->grad - grad_b;
    }
}

void AutoGrad::backward_mul(Node *node) {
    Tensor& grad_out = node->grad;
    if (node->inputs.size() != 2) {
        throw std::runtime_error("Multiplication requires exactly 2 inputs");
    }

    Node* a = node->inputs[0];
    Node* b = node->inputs[1];

    if (a->requires_grad) {
        Tensor grad_a = grad_out * b->tensor;
        // 处理广播梯度
        grad_a = reduce_to_match(grad_a, a->tensor.shape());
        a->grad = a->grad + grad_a;
    }

    if (b->requires_grad) {
        Tensor grad_b = grad_out * a->tensor;
        // 处理广播梯度
        grad_b = reduce_to_match(grad_b, b->tensor.shape());
        b->grad = b->grad + grad_b;
    }
}

void AutoGrad::backward_div(Node *node) {
    Tensor& grad_out = node->grad;
    if (node->inputs.size() != 2) {
        throw std::runtime_error("Division requires exactly 2 inputs");
    }

    Node* a = node->inputs[0];
    Node* b = node->inputs[1];

    if (a->requires_grad) {
        Tensor grad_a = grad_out / b->tensor;
        grad_a = reduce_to_match(grad_a, a->tensor.shape());
        a->grad = a->grad + grad_a;
    }

    if (b->requires_grad) {
        Tensor grad_b = grad_out * (-a->tensor) / (b->tensor * b->tensor);
        grad_b = reduce_to_match(grad_b, b->tensor.shape());
        b->grad = b->grad + grad_b;
    }
}

void AutoGrad::backward_matmul(Node *node) {
    Tensor& grad_out = node->grad;
    if (node->inputs.size() != 2) {
        throw std::runtime_error("Matrix multiplication requires exactly 2 inputs");
    }

    Node* a = node->inputs[0];
    Node* b = node->inputs[1];

    if (a->requires_grad) {
        // dL/dA = dL/dC * B^T
        Tensor b_t = b->tensor.transpose_last_two();
        Tensor grad_a = matMul(grad_out,b_t);
        grad_a = reduce_to_match(grad_a, a->tensor.shape());
        a->grad = a->grad + grad_a;
    }

    if (b->requires_grad) {
        // dL/dB = A^T * dL/dC
        Tensor a_t = a->tensor.transpose_last_two();
        Tensor grad_b = matMul(a_t,grad_out);
        grad_b = reduce_to_match(grad_b, b->tensor.shape());
        b->grad = b->grad + grad_b;
    }
}

void AutoGrad::backward_dot(Node *node) {
    Tensor& grad_out = node->grad;
    if (node->inputs.size() != 2) {
        throw std::runtime_error("Dot product requires exactly 2 inputs");
    }

    Node* a = node->inputs[0];
    Node* b = node->inputs[1];

    if (a->requires_grad) {
        // dL/da = dL/dout * b
        Tensor grad_a = grad_out * b->tensor;
        a->grad = a->grad + grad_a;
    }

    if (b->requires_grad) {
        // dL/db = dL/dout * a
        Tensor grad_b = grad_out * a->tensor;
        b->grad = b->grad + grad_b;
    }
}

void AutoGrad::backward_cos(Node *node) {
    Tensor& grad_out = node->grad;
    Node* input_node = node->inputs[0];

    if (input_node->requires_grad) {
        // ∂cos(x)/∂x = -sin(x)
        Tensor sin_x = input_node->tensor.sin();
        Tensor grad = grad_out * (-sin_x);
        input_node->grad = input_node->grad + grad;
    }
}

void AutoGrad::backward_sin(Node *node) {
    Tensor& grad_out = node->grad;
    Node* input_node = node->inputs[0];

    if (input_node->requires_grad) {
        // ∂sin(x)/∂x = cos(x)
        Tensor cos_x = input_node->tensor.cos();
        Tensor grad = grad_out * cos_x;
        input_node->grad = input_node->grad + grad;
    }
}

void AutoGrad::backward_relu(Node *node) {
    Tensor& grad_out = node->grad;
    Node* input_node = node->inputs[0];

    if (input_node->requires_grad) {
        // ∂ReLU(x)/∂x = {1 if x > 0, else 0}
        Tensor mask = input_node->tensor > 0.0f;
        Tensor grad = grad_out * mask;
        input_node->grad = input_node->grad + grad;
    }
}

void AutoGrad::backward_sigmoid(Node *node) {
    Tensor& grad_out = node->grad;
    Node* input_node = node->inputs[0];

    if (input_node->requires_grad) {
        // ∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x))
        Tensor sig_x = node->tensor;
        Tensor grad = grad_out * sig_x * (1.0f - sig_x);
        input_node->grad = input_node->grad + grad;
    }
}

void AutoGrad::backward_tanh(Node *node) {
    Tensor& grad_out = node->grad;
    Node* input_node = node->inputs[0];

    if (input_node->requires_grad) {
        // ∂tanh(x)/∂x = 1 - tanh²(x)
        Tensor tanh_x = node->tensor;
        Tensor grad = grad_out * (1.0f - tanh_x * tanh_x);
        input_node->grad = input_node->grad + grad;
    }
}

void AutoGrad::backward_softmax(Node *node) {
    Tensor& grad_out = node->grad;
    Node* input_node = node->inputs[0];

    if (input_node->requires_grad) {
        // 简化实现：更高效的方式是使用雅可比矩阵
        Tensor s = node->tensor;
        Tensor grad = grad_out;

        // 计算 (grad_out * s) 的和
        Tensor sum_term = (grad_out * s).sum(-1, true);

        // ∂L/∂x = s * (grad_out - sum_term)
        Tensor grad_input = s * (grad_out - sum_term);

        input_node->grad = input_node->grad + grad_input;
    }
}

void AutoGrad::backward_sum(Node *node) {
    Tensor& grad_out = node->grad;
    Node* input_node = node->inputs[0];

    if (input_node->requires_grad) {
        // 将梯度广播回原始形状
        Tensor expanded_grad;

        // 如果原始张量是标量，直接使用梯度
        if (input_node->tensor.shape().empty()) {
            expanded_grad = grad_out;
        }
        // 如果梯度形状与输入形状匹配
        else if (grad_out.shape() == input_node->tensor.shape()) {
            expanded_grad = grad_out;
        }
        // 需要广播的情况
        else {
            // 创建与输入相同形状的张量
            expanded_grad = Tensor(ShapeTag{}, input_node->tensor.shape(),
                                   grad_out.dtype(), grad_out.device());

            // 用梯度值填充整个张量
            switch (grad_out.dtype()) {
            case DType::kFloat: {
                float value = grad_out.item<float>();
                expanded_grad.fill(value);
                break;
            }
            case DType::kDouble: {
                double value = grad_out.item<double>();
                expanded_grad.fill(value);
                break;
            }
            case DType::kInt: {
                int32_t value = grad_out.item<int32_t>();
                expanded_grad.fill(static_cast<float>(value));
                break;
            }
            case DType::kLong: {
                int64_t value = grad_out.item<int64_t>();
                expanded_grad.fill(static_cast<float>(value));
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype in sum backward");
            }
        }

        input_node->grad = input_node->grad + expanded_grad;
    }
}

Tensor AutoGrad::reduce_to_match(Tensor grad, const std::vector<size_t> &target_shape) {
    if (grad.shape() == target_shape) {
        return grad;
    }

    // 计算需要求和的维度
    std::vector<int> reduce_dims;
    std::vector<size_t> grad_shape = grad.shape();
    std::vector<size_t> target = target_shape;

    // 从后往前对齐维度
    int g_idx = static_cast<int>(grad_shape.size()) - 1;
    int t_idx = static_cast<int>(target.size()) - 1;

    while (g_idx >= 0 || t_idx >= 0) {
        size_t g_dim = (g_idx >= 0) ? grad_shape[g_idx] : 1;
        size_t t_dim = (t_idx >= 0) ? target[t_idx] : 1;

        if (g_dim != t_dim) {
            if (t_dim == 1) {
                // 梯度在这个维度上需要求和
                reduce_dims.push_back(g_idx);
            } else if (g_dim == 1) {
                // 不需要操作，维度大小1会自动扩展
            } else {
                throw std::runtime_error("Cannot reduce gradient to target shape");
            }
        }

        if (g_idx >= 0) g_idx--;
        if (t_idx >= 0) t_idx--;
    }

    // 在需要求和的维度上求和
    if (!reduce_dims.empty()) {
        Tensor reduced = grad.sum(reduce_dims, true);

        // 移除多余的维度（如果需要）
        std::vector<size_t> reduced_shape = reduced.shape();
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < reduced_shape.size(); i++) {
            if (std::find(reduce_dims.begin(), reduce_dims.end(), i) == reduce_dims.end()) {
                new_shape.push_back(reduced_shape[i]);
            }
        }

        if (new_shape != target_shape) {
            throw std::runtime_error("Failed to reduce gradient to target shape");
        }

        return reduced;
    }

    return grad;
}

Tensor AutoGrad::getGrad(Tensor *t) {
    auto it = tensor_to_node.find(t);
    if (it != tensor_to_node.end() && it->second) {
        return it->second->grad;
    }
    return {}; // 返回空张量
}

void AutoGrad::zeroGrad(Tensor *t) {
    auto it = tensor_to_node.find(t);
    if (it != tensor_to_node.end() && it->second) {
        it->second->grad = Tensor();
    }
}

void AutoGrad::set_retain_graph(bool retain) { retain_graph = retain; }

AutoGrad::Node *AutoGrad::get_node(Tensor *t) {
    auto it = tensor_to_node.find(t);
    return it != tensor_to_node.end() ? it->second : nullptr;
}

void AutoGrad::make_leaf(Tensor &t, bool requires_grad) {
    if (tensor_to_node.find(&t) != tensor_to_node.end()) {
        return; // 已注册，跳过
    }
    auto node = std::make_unique<Node>(t.clone(), requires_grad);
    tensor_to_node[&t] = node.get();
    nodes.push_back(std::move(node));
}

void AutoGrad::record_op(const std::vector<Tensor *> &outputs, op operation,
                         const std::vector<Tensor *> &inputs) {
    std::vector<Tensor*> inputs_copy = inputs;
    // 为每个输出创建节点
    for (Tensor* out : outputs) {
        if (tensor_to_node.find(out) == tensor_to_node.end()) {
            auto new_node = std::make_unique<Node>(out->clone(), false, false);
            tensor_to_node[out] = new_node.get();
            nodes.push_back(std::move(new_node));
        }

        Node* node = tensor_to_node[out];
        node->operation = operation;
        node->inputs.clear();

        for (Tensor* input : inputs_copy) {
            if (tensor_to_node.find(input) == tensor_to_node.end()) {
                make_leaf(*input, input->isGradRequired());
            }
            node->inputs.push_back(tensor_to_node[input]);
        }
    }
}

void AutoGrad::backward(Tensor &root, Tensor grad_output) {
    if (tensor_to_node.find(&root) == tensor_to_node.end()) {
           throw std::runtime_error("Tensor not in computation graph");
       }

       Node* root_node = tensor_to_node[&root];
       if (!root_node->requires_grad) {
           throw std::runtime_error("Root tensor doesn't require gradient");
       }

       // 设置初始梯度
       if (grad_output.empty()) {
           // 默认标量输出梯度为1
           if (root.numel() == 1) {
               root_node->grad.fill(1.0f);
           } else {
               throw std::runtime_error("Grad output must be specified for non-scalar tensors");
           }
       } else {
           // 检查梯度形状是否匹配
           if (grad_output.shape() != root.shape()) {
               throw std::runtime_error("Grad output shape mismatch");
           }
           root_node->grad = grad_output;
       }

       // 拓扑排序（深度优先）
       std::vector<Node*> order;
       std::unordered_set<Node*> visited;
       std::function<void(Node*)> dfs = [&](Node* node) {
           if (visited.find(node) != visited.end()) return;
           visited.insert(node);

           for (Node* input : node->inputs) {
               dfs(input);
           }
           order.push_back(node);
       };
       dfs(root_node);

       // 反向遍历计算梯度
       std::reverse(order.begin(), order.end());
       for (Node* node : order) {
           // 跳过叶子节点（梯度已存储）
           if (node->is_leaf) continue;

           // 根据操作类型计算梯度
           switch (node->operation) {
           case op::Add:
               backward_add(node);
               break;
           case op::Sub:
               backward_sub(node);
               break;
           case op::Mul:
               backward_mul(node);
               break;
           case op::Div:
               backward_div(node);
               break;
           case op::MatMul:
               backward_matmul(node);
               break;
           case op::Dot:
               backward_dot(node);
               break;
           case op::Cos:
               backward_cos(node);
               break;
           case op::Sin:
               backward_sin(node);
               break;
           case op::ReLU:
               backward_relu(node);
               break;
           case op::Sigmoid:
               backward_sigmoid(node);
               break;
           case op::Tanh:
               backward_tanh(node);
               break;
           case op::Softmax:
               backward_softmax(node);
               break;
           case op::Sum:
               backward_sum(node);
               break;
           default:
               throw std::runtime_error("Unsupported operation in backward");
           }

           // 如果不是保留计算图，释放中间梯度
           if (!retain_graph && !node->is_leaf) {
               node->grad = Tensor(); // 释放梯度内存
           }
       }

       // 如果不保留计算图，清除所有节点
       if (!retain_graph) {
           clearGraph();
       }
}

void AutoGrad::clearGraph() {
    tensor_to_node.clear();
    nodes.clear();
}

BroadCastResult broadCast(const Tensor& a, const Tensor& b) {
    const std::vector<size_t>& shapeA = a.shape();
    const std::vector<size_t>& shapeB = b.shape();

    // 确定最大维度
    size_t max_dims = std::max(shapeA.size(), shapeB.size());
    std::vector<size_t> logicShape(max_dims);
    std::vector<size_t> logicStridesA(max_dims, 0);
    std::vector<size_t> logicStridesB(max_dims, 0);

    // 从后往前对齐维度
    for (int i = max_dims - 1, idxA = shapeA.size() - 1, idxB = shapeB.size() - 1;
         i >= 0; i--, idxA--, idxB--) {
        size_t dimA = (idxA >= 0) ? shapeA[idxA] : 1;
        size_t dimB = (idxB >= 0) ? shapeB[idxB] : 1;

        if (dimA == dimB) {
            logicShape[i] = dimA;
            logicStridesA[i] = (idxA >= 0) ? a.strides()[idxA] : 0;
            logicStridesB[i] = (idxB >= 0) ? b.strides()[idxB] : 0;
        } else if (dimA == 1) {
            logicShape[i] = dimB;
            logicStridesA[i] = 0; // 广播维度步幅为0
            logicStridesB[i] = (idxB >= 0) ? b.strides()[idxB] : 0;
        } else if (dimB == 1) {
            logicShape[i] = dimA;
            logicStridesA[i] = (idxA >= 0) ? a.strides()[idxA] : 0;
            logicStridesB[i] = 0; // 广播维度步幅为0
        } else {
            throw std::runtime_error("The shape of Tensor provided is incompatible for broadcasting");
        }
         }

    return {logicShape, logicStridesA, logicStridesB};
}

Tensor operator-(float scalar, const Tensor& tensor) {
    Tensor result(ShapeTag{}, tensor.shape(), tensor.dtype(), tensor.device());

    switch (tensor.dtype()) {
    case DType::kFloat: {
        const float* src = tensor.data<float>();
        float* dst = result.data<float>();
        for (size_t i = 0; i < tensor.numel(); ++i) {
            dst[i] = scalar - src[i];
        }
        break;
    }
    case DType::kDouble: {
        const double* src = tensor.data<double>();
        double* dst = result.data<double>();
        for (size_t i = 0; i < tensor.numel(); ++i) {
            dst[i] = static_cast<double>(scalar) - src[i];
        }
        break;
    }
    case DType::kInt: {
        const int32_t* src = tensor.data<int32_t>();
        int32_t* dst = result.data<int32_t>();
        int32_t s = static_cast<int32_t>(scalar);
        for (size_t i = 0; i < tensor.numel(); ++i) {
            dst[i] = s - src[i];
        }
        break;
    }
    case DType::kLong: {
        const int64_t* src = tensor.data<int64_t>();
        int64_t* dst = result.data<int64_t>();
        int64_t s = static_cast<int64_t>(scalar);
        for (size_t i = 0; i < tensor.numel(); ++i) {
            dst[i] = s - src[i];
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for scalar - tensor");
    }

    // 自动微分记录
    AutoGrad* ctx = AutoGradContext::current();
    if (ctx) {
        ctx->record_op(
            {&result},
            op::Sub,
            {const_cast<Tensor*>(&tensor)}
        );
    }

    return result;
}