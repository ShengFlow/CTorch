module Tensor;

// 自动微分类操作符枚举
enum class op {

    // 基本运算
    Add,    // 加
    Sub,    // 减
    Mul,    // 乘
    Div,    // 除
    MatMul, // 矩阵乘法
    Dot,    // 点乘

    // 卷积操作
    Conv, // 卷积
    Pool, // 池化

    // 激活函数
    ReLU, // 线性整流函数
    Tanh, // 双曲正切函数
    Sigmoid,
    Softmax,

    // 激活函数变种
    LReLU, // 渗漏线性整流函数
    PReLU, // 参数化线性整流函数

    // 损失函数
    MSE, // 均方误差
    MAE, // 平均绝对误差
    CE,  // 交叉熵损失
    BCE, // 二元交叉熵损失
};

constexpr const char *dtypeToString(Dtype dtype) const {
    switch (dtype) {
    case DType::kFloat:
        return "float32";
    case DType::kDouble:
        return "float64";
    case DType::kInt:
        return "int32";
    case DType::kLong:
        return "int64";
    case DType::kBool:
        return "bool";
    default:
        return "unknown";
    }
}

constexpr size_t dtypeSize(DType dtype) {
    switch (dtype) {
    case DType::kFloat:
        return sizeof(float);
    case DType::kDouble:
        return sizeof(double);
    case DType::kInt:
        return sizeof(int32_t);
    case DType::kLong:
        return sizeof(int64_t);
    case DType::kBool:
        return sizeof(bool);
    default:
        throw std::runtime_error("Unsupported dtype");
    }
}

void Storage::checkDType() const {
    if ((std::is_same_v<T, float> && _dtype != DType::kFloat) ||
        (std::is_same_v<T, double> && _dtype != DType::kDouble) ||
        (std::is_same_v<T, int32_t> && _dtype != DType::kInt) ||
        (std::is_same_v<T, int64_t> && _dtype != DType::kLong) ||
        (std::is_same_v<T, bool> && _dtype != DType::kBool)) {
        std::cerr << "Storage data type mismatch: T=" << typeid(T).name()
                  << ", dtype=" << dtypeToString(_dtype) << std::endl;
        throw std::runtime_error("Storage data type mismatch");
    }
}

Storage::Storage(size_t size, DType dtype, DeviceType device = DeviceType::kCPU)
    : _size(size), _dtype(dtype), _device(device),
      _data(size > 0 ? std::shared_ptr<char[]>(new char[size * dtypeSize(dtype)]) : nullptr) {}

Storage::Storage() : _size(0), _dtype(DType::kFloat), _device(DeviceType::kCPU) {}

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
    size_t i     = 0;
    for (const auto &idx : indices) {
        if (idx >= _shape[i]) {
            throw std::out_of_range("Tensor index out of bounds");
        }
        index += idx * _strides[i];
        ++i;
    }
    return index;
}

Tensor::Tensor() : _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kFloat) {
    computeStrides();
    _storage = Storage(numel(), _dtype, _device);
}

Tensor::Tensor(float value)
    : _shape({}), _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kFloat) {
    computeStrides();
    _storage                = Storage(1, _dtype, _device);
    *_storage.data<float>() = value;
}

Tensor::Tensor(std::initializer_list<bool> values)
    : _shape({values.size()}), _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kBool) {
    computeStrides();
    _storage   = Storage(values.size(), _dtype, _device);
    bool *data = _storage.data<bool>();
    size_t i   = 0;
    for (bool val : values) {
        data[i++] = val;
    }
}

Tensor::Tensor(ShapeTag, const std::vector<size_t> &shape, DType dtype = DType::kFloat,
               DeviceType device = DeviceType::kCPU, bool zero_init = true)
    : _shape(shape), _storage_offset(0), _device(device), _dtype(dtype) {
    computeStrides();
    _storage = Storage(numel(), _dtype, _device);
    if (zero_init)
        zero();
}

Tensor::Tensor(const Tensor &other)
    : _shape(other._shape), _strides(other._strides), _storage_offset(other._storage_offset),
      _device(other._device), _dtype(other._dtype), _storage(other._storage.clone()) {}

Tensor::Tensor(Tensor &&other) noexcept
    : _shape(std::move(other._shape)), _strides(std::move(other._strides)),
      _storage_offset(other._storage_offset), _device(other._device), _dtype(other._dtype),
      _storage(std::move(other._storage)) {
    other._storage_offset = 0;
    other._shape.clear();
    other._strides.clear();
}

size_t Tensor::dim() const { return _shape.size(); }

const std::vector<size_t> &Tensor::sizes() const { return _shape; }

size_t Tensor::numel() const {
    if (_shape.empty())
        return 1; // 标量有1个元素
    return std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<>());
}

DType Tensor::dtype() const { return _dtype; }

DeviceType Tensor::device() const { return _device; }

Tensor Tensor::clone() const {
    Tensor copy;
    copy._shape          = _shape;
    copy._strides        = _strides;
    copy._storage_offset = 0;
    copy._device         = _device;
    copy._dtype          = _dtype;
    copy._storage        = _storage.clone(); // 深拷贝存储
    copy._requires_grad  = _requires_grad;
    return copy;
}

Tensor Tensor::view(const std::vector<size_t> &new_shape) const {
    // 计算新形状的元素总数
    size_t new_numel = 1;
    for (auto dim : new_shape) {
        if (dim == 0)
            throw std::runtime_error("Zero dimension in shape");
        new_numel *= dim;
    }

    // 验证新形状的元素总数是否匹配
    if (new_numel != numel()) {
        throw std::runtime_error("Shape size mismatch in view()");
    }

    Tensor result(ShapeTag{}, new_shape, _dtype, _device);
    result._storage        = _storage; // 共享存储（使用shared_ptr实现共享所有权）
    result._storage_offset = _storage_offset;
    result._strides        = _strides; // 保持原步幅
    result._requires_grad  = _requires_grad;
    return result;
}

Tensor Tensor::transpose() const {
    if (dim() < 2) {
        throw std::runtime_error("transpose requires at least 2 dimensions");
    }

    // 创建新的形状和步幅
    std::vector<size_t> new_shape   = _shape;
    std::vector<size_t> new_strides = _strides;

    // 交换最后两个维度
    std::swap(new_shape[dim() - 1], new_shape[dim() - 2]);
    std::swap(new_strides[dim() - 1], new_strides[dim() - 2]);

    Tensor result   = *this;
    result._shape   = new_shape;
    result._strides = new_strides;
    return result;
}

const Tensor::broadCast(Tensor &a, Tensor &b) const {
    Tensor *large = &(a._shape.size() > b._shape.size() ? a : b);
    Tensor *min   = &(a._shape.size() < b._shape.size() ? a : b);
    for (size_t i{large->_shape.size() - 1}; i >= 0 && (a._shape[i] && b._shape[i]); i--) {
        if (a._shape[i] != b._shape[i] && (a._shape[i] != 1 || b._shape[i] != 1))
            throw std::runtime_error("The shape of Tensor provided is incompatible.");
    }

    std::vector<size_t> logicShape(large->_shape.size(), 1);
    std::vector<size_t> logicStrides(large->_shape.size(), 0);

    for (size_t i{large->_shape.size() - 1}; i >= 0; i--) {
        if ((*large)._shape[i] && (*min)._shape[i] && (*min)._shape[i] != 1) {
            logicShape[i]   = (*min)._shape[i];
            logicStrides[i] = (*min)._strides[i];
        }
        logicShape[i] = (*large)._shape[i];
    }
    return {logicShape, logicStrides};
}

Tensor Tensor::operator+(const Tensor &rhs) const {
    // 验证形状和类型匹配
    if (_shape != rhs._shape)
        throw std::runtime_error("Shape mismatch in addition");
    if (_dtype != rhs._dtype)
        throw std::runtime_error("DType mismatch in addition");
    if (_device != rhs._device)
        throw std::runtime_error("Device mismatch in addition");

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
    return result;
}

Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
        _shape          = other._shape;
        _strides        = other._strides;
        _storage_offset = other._storage_offset;
        _device         = other._device;
        _dtype          = other._dtype;
        _storage        = other._storage.clone(); // 深拷贝存储
        _requires_grad  = other._requires_grad;
    }
    return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
    if (this != &other) {
        _shape          = std::move(other._shape);
        _strides        = std::move(other._strides);
        _storage_offset = other._storage_offset;
        _device         = other._device;
        _dtype          = other._dtype;
        _storage        = std::move(other._storage);
        _requires_grad  = other._requires_grad;

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
    if (n == 0)
        return true; // 空张量相等

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

std::string Tensor::toString() const {
    std::string str;
    _requires_grad ? str = "True" : str = "False";
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < _shape.size(); ++i) {
        oss << _shape[i];
        if (i < _shape.size() - 1)
            oss << ", ";
    }
    oss << "], dtype=" << dtypeToString(_dtype)
        << ", device=" << (_device == DeviceType::kCPU ? "cpu" : "gpu") << ", AutoDiff=" << str
        << ")\n";

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
        } catch (const std::exception &e) {
            oss << "Error: " << e.what();
        }
        oss << "]";
    }
    return oss.str();
}

void Tensor::print() const { std::cout << toString() << std::endl; }

bool is_contiguous() const {
    size_t stride = 1;
    for (int i = dim() - 1; i >= 0; --i) {
        if (_strides[i] != stride)
            return false;
        stride *= _shape[i];
    }
    return true;
}

void Tensor::zero() { fill(0); }

void Tensor::ones() {
    switch (_dtype) {
    case DType::kFloat:
        fill<float>(1.0f);
        break;
    case DType::kDouble:
        fill<double>(1.0);
        break;
    case DType::kInt:
        fill<int32_t>(1);
        break;
    case DType::kLong:
        fill<int64_t>(1);
        break;
    case DType::kBool:
        fill<bool>(true);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for ones()");
    }
}

bool Tensor::isAuto_diff() { return _requires_grad; }

void Tensor::grad() {}