/*
 * Tensor.h
 * Created by Beapoe & GhostFace on 2025.7
 * Main Classes: Storage & Tensor & Auto_diff
 * Version : v1.5 (fixed on 2025.7.22 12:18)
 * Log 1.3: 增加了注释及代码易读性
 * Log 1.4: 增加了AutoDiff自动微分类
 * Log 1.5:
 * 增加了连续性检查，修复了变量命名，增加了对自动微分状态的输出，修复了移动时不移动自动微分状态的bug
 */

// includes
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <sstream>
#include <functional>
#include <cstring>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <unordered_set>
// ======================= 类型定义和枚举 =======================

// 设备类型 - 定义张量存储的位置
enum class DeviceType {
    kCPU,  //< 主存储器 (RAM)
    kCUDA, //< NVIDIA GPU (暂未实现)
    kMPS,  //< Apple Silicon (暂未实现)
};

// 数据类型 - 定义张量元素的类型
enum class DType {
    kFloat,  //< 32位浮点数 (torch.float32)
    kDouble, //< 64位浮点数 (torch.float64)
    kInt,    //< 32位整数 (torch.int32)
    kLong,   //< 64位整数 (torch.int64)
    kBool,   //< 布尔值 (torch.bool)
};

// 自动微分类操作符枚举
enum class op {

    // 基本运算
    Add,    // 加
    Sub,    // 减
    Mul,    // 乘
    Div,    // 除
    MatMul, // 矩阵乘法
    Dot,    // 点乘
    Cos,
    Sin,

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

// 广播变形数据结构体
struct LogicData {
    std::vector<size_t>
        logicShape; // 由于广播之后shape和strides都发生了改变，所以广播之后的计算应该依赖于logicShape和logicStrides
    std::vector<size_t> logicStrides;
};

// ======================= 辅助函数 =======================

// 将数据类型转换为字符串表示
constexpr const char *dtypeToString(DType dtype) {
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

// 获取数据类型的字节大小
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

// ======================= 存储类 (Storage) =======================

// 存储类 - 管理张量的原始内存
/* class Storage
 *
 * 成员函数：
 * 1.Storage(size_t size, DType dtype, DeviceType device = DeviceType::kCPU)
 * 2.Storage(const T* data, size_t size, DType dtype, DeviceType device = DeviceType::kCPU)
 * 3.size_t size()
 * 4.DType dtype()
 * 5.DeviceType device()
 * 6.Storage clone()
 * 7.bool empty()
 * 8.void checkDType()
 * 9.默认解析函数Storage() : _size(0), _dtype(DType::kFloat), _device(DeviceType::kCPU)
 * 10.默认析构函数~Storage()
 *
 * 成员变量：
 * 1.size_t _size{};
 * 2.DType _dtype;
 * 3.DeviceType _device;
 * 4.std::shared_ptr<char[]> _data;
 *
 * 运算符重载：
 * Storage& operator=(Storage&&) = default;         // 移动赋值
 * Storage& operator=(const Storage&) = default;    // 拷贝赋值
 */
class AutoDiff; // 前置声明

class Storage {
  private:
    size_t
        _size{}; // 存储的元素数量，此处使用C++11的新特性花括号初始化，避免类型转换，实际上等同于size_t
                 // _size = 0;
    DType _dtype;       // 数据类型 用于枚举
    DeviceType _device; // 设备类型 用于枚举
    std::shared_ptr<char[]>
        _data; // 原始内存指针（使用shared_ptr实现共享所有权）避免出现手动delete的问题和delete数组和默认方法不匹配
    // 此处定义为char[]能够最大限度的节省内存并支持存储任意类型的数据
    // 使用shared_ptr能够共享对内存的所有权，使得同等的tensor可以共用一块内存，减少不必要的内存占用
    // 在需要深拷贝时，提供了一个clone函数，可以调用

    // 检查模板类型是否与存储类型匹配
    template <typename T>

    // 在如下的checkDType函数中，std::is_same_v的用法为is_same_v<type,type>，返回true/false，用以判断两个类型是否相同
    // 此函数用来强制类型检查，避免不必要的内存问题

    void checkDType() const {
        if ((std::is_same<T, float>::value && _dtype != DType::kFloat) ||
            (std::is_same<T, double>::value && _dtype != DType::kDouble) ||
            (std::is_same<T, int32_t>::value && _dtype != DType::kInt) ||
            (std::is_same<T, int64_t>::value && _dtype != DType::kLong) ||
            (std::is_same<T, bool>::value && _dtype != DType::kBool)) {
            std::cerr << "Storage data type mismatch: T=" << typeid(T).name()
                      << ", dtype=" << dtypeToString(_dtype) << std::endl;
            throw std::runtime_error("Storage data type mismatch");
        }
    }


public:
    // 构造函数：分配未初始化的内存
    Storage(size_t size, DType dtype, DeviceType device = DeviceType::kCPU)
        : _size(size), _dtype(dtype), _device(device),
          _data(size > 0 ? std::shared_ptr<char[]>(new char[size * dtypeSize(dtype)]) : nullptr) {}
    // 如果初始化列表中_size为0，那么初始化为nullptr

    // 构造函数：从现有数据复制
    template <typename T>
    Storage(const T *data, size_t size, DType dtype, DeviceType device = DeviceType::kCPU)
        : Storage(size, dtype, device) {
        if (size > 0 && _data.get()) {
            std::memcpy(_data.get(), data, size * dtypeSize(dtype));
        }
    }
    // 注意，此处实际上将从现有数据复制的操作委托给了第一个构造函数，然后进行memcpy的操作进行复制

    // 默认拷贝构造函数和拷贝赋值运算符（使用shared_ptr，所以是浅拷贝）
    // 此处同上，减少内存开销
    Storage(const Storage &)            = default;
    Storage &operator=(const Storage &) = default;

    // 默认移动构造函数和移动赋值运算符
    Storage(Storage &&)            = default;
    Storage &operator=(Storage &&) = default;

    Storage() : _size(0), _dtype(DType::kFloat), _device(DeviceType::kCPU) {}

    ~Storage() = default;

    // 获取原始数据的类型化指针
    template <typename T> T *data() {
        if (_size == 0 || !_data)
            return nullptr;
        checkDType<T>();
        return reinterpret_cast<T *>(_data.get());
    }

    // 获取常量原始数据的类型化指针
    template <typename T> const T *data() const {
        if (_size == 0 || !_data)
            return nullptr;
        checkDType<T>();
        return reinterpret_cast<const T *>(_data.get());
    }

    // 获取存储中的元素数量
    size_t size() const { return _size; }

    // 获取数据类型
    DType dtype() const { return _dtype; }

    // 获取设备类型
    DeviceType device() const { return _device; }

    // 创建存储的深拷贝
    Storage clone() const {
        Storage new_storage(_size, _dtype, _device);
        if (_size > 0 && _data) {
            std::memcpy(new_storage._data.get(), _data.get(), _size * dtypeSize(_dtype));
        }
        return new_storage;
    }

    // 检查存储是否为空
    bool empty() const { return _size == 0 || !_data; }
};

template<typename T>
DType cppType2Dtype(){
    if constexpr (std::is_same_v<T, float>) {
        return DType::kFloat;
    } else if constexpr (std::is_same_v<T, double>) {
        return DType::kDouble;
    } else if constexpr (std::is_same_v<T, int32_t> || (std::is_integral_v<T> && sizeof(T) == 4)) {
        return DType::kInt;
    } else if constexpr (std::is_same_v<T, int64_t> || (std::is_integral_v<T> && sizeof(T) == 8)) {
        return DType::kLong;
    } else if constexpr (std::is_same_v<T, bool>) {
        return DType::kBool;
    } else {
        throw std::runtime_error("Unsupported data type in initializer_list");
    }
}

// ======================= 张量类 (Tensor) =======================
struct ShapeTag {}; // 此处结构体为了使编译器区分构造函数

class Tensor {
  private:
    bool _requires_grad = false; // 是否参与自动微分计算，默认不参与
    // 张量的维度大小
    std::vector<size_t> _strides;     // 每个维度的步幅
    size_t _storage_offset;           // 存储中的起始偏移量
    DeviceType _device;               // 张量所在的设备
    DType _dtype;                     // 张量元素的数据类型
    Storage _storage;                 // 存储张量数据的对象
    AutoDiff *autograd_ctx = nullptr; // 自动微分上下文指针

    // ======================= 内部辅助函数 =======================

    // 计算步幅 (基于行优先顺序)
    void computeStrides() {
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

    // 计算存储中的索引
    size_t computeStorageIndex(std::initializer_list<size_t> indices) const {
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

    // 检查数据类型是否匹配
    template <typename T> void checkDType() const {
        if ((std::is_same_v<T, float> && _dtype != DType::kFloat) ||
            (std::is_same_v<T, double> && _dtype != DType::kDouble) ||
            (std::is_same_v<T, int32_t> && _dtype != DType::kInt) ||
            (std::is_same_v<T, int64_t> && _dtype != DType::kLong) ||
            (std::is_same_v<T, bool> && _dtype != DType::kBool)) {
            throw std::runtime_error("Tensor data type mismatch");
        }
    }

    // 通用逐元素操作
    template <typename T, typename Op>
    void elementwiseOp(Tensor &result, const Tensor &a, const Tensor &b, Op op) const {
        const size_t n  = a.numel();
        T *out          = result.data<T>();
        const T *a_data = a.data<T>();
        const T *b_data = b.data<T>();

        for (size_t i = 0; i < n; ++i) {
            out[i] = op(a_data[i], b_data[i]);
        }
    }

    // 递归打印张量内容（改进版）
    template <typename T>
    void printRecursive(std::ostream &os, size_t dim, std::vector<size_t> indices) const {
        if (dim == this->dim()) {
            // 到达最后一个维度，打印元素
            size_t index = 0;
            for (size_t i = 0; i < indices.size(); ++i) {
                index += indices[i] * _strides[i];
            }
            index += _storage_offset;

            if constexpr (std::is_same_v<T, bool>) {
                os << (_storage.data<T>()[index] ? "true" : "false");
            } else if constexpr (std::is_floating_point_v<T>) {
                os << std::fixed << std::setprecision(2) << _storage.data<T>()[index];
            } else {
                os << _storage.data<T>()[index];
            }
            return;
        }

        // 添加换行和缩进
        if (dim > 0) {
            os << "\n";
            for (size_t i = 0; i < dim; ++i)
                os << "  ";
        }
        os << "[";

        const size_t max_display   = 3; // 每维度最大显示元素数
        const size_t display_count = std::min(_shape[dim], max_display);
        const bool truncated       = _shape[dim] > max_display;

        for (size_t i = 0; i < display_count; ++i) {
            indices.push_back(i);
            printRecursive<T>(os, dim + 1, indices);
            indices.pop_back();

            if (i < display_count - 1) {
                os << ", ";
            }
        }

        if (truncated) {
            os << ", ..."; // 添加截断指示
        }

        os << "]";
    }

  protected:
    std::vector<size_t> _shape;

  public:
    // ======================= 构造和析构 =======================

    // 默认构造函数：创建空张量
    Tensor() : _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kFloat) {
        computeStrides();
        _storage = Storage(numel(), _dtype, _device);
    }

    // 标量构造函数
    Tensor(float value)
        : _shape({}), _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kFloat) {
        computeStrides();
        _storage                = Storage(1, _dtype, _device);
        *_storage.data<float>() = value;
    }

    // 构造函数：从初始值列表创建1D张量
    template <typename T>
    Tensor(std::initializer_list<T> values,std::vector<size_t> shape)
        : _storage_offset(0), _device(DeviceType::kCPU), _dtype(cppType2Dtype<T>()),
          _shape(std::move(shape)) {
        computeStrides();
        _storage = Storage(values.begin(), values.size(), _dtype, _device);
    }

    // TODO: 考虑把下面这个构造合并到上面

    // 添加布尔张量构造函数
    Tensor(std::initializer_list<bool> values)
        : _shape({values.size()}), _storage_offset(0), _device(DeviceType::kCPU),
          _dtype(DType::kBool) {
        computeStrides();
        _storage   = Storage(values.size(), _dtype, _device);
        bool *data = _storage.data<bool>();
        size_t i   = 0;
        for (bool val : values) {
            data[i++] = val;
        }
    }

    // 构造函数：指定形状和数据类型（使用 ShapeTag 避免歧义）
    Tensor(ShapeTag, const std::vector<size_t> &shape, DType dtype = DType::kFloat,
           DeviceType device = DeviceType::kCPU, bool zero_init = true)
        : _shape(shape), _storage_offset(0), _device(device), _dtype(dtype) {
        computeStrides();
        _storage = Storage(numel(), _dtype, _device);
        if (zero_init)
            zero();
    }

    // 拷贝构造函数：创建深拷贝
    Tensor(const Tensor &other)
        : _shape(other._shape), _strides(other._strides), _storage_offset(other._storage_offset),
          _device(other._device), _dtype(other._dtype), _storage(other._storage.clone()) {
    } // 深拷贝存储

    // 移动构造函数
    Tensor(Tensor &&other) noexcept
        : _shape(std::move(other._shape)), _strides(std::move(other._strides)),
          _storage_offset(other._storage_offset), _device(other._device), _dtype(other._dtype),
          _storage(std::move(other._storage)) {
        other._storage_offset = 0;
        other._shape.clear();
        other._strides.clear();
    }

    ~Tensor() = default;
    std::vector<size_t> shape() const { return _shape; }
    std::vector<size_t> strides() const { return _strides; }

    // ======================= 基本属性 =======================

    // 获取张量的维度数
    size_t dim() const { return _shape.size(); }

    // 获取张量的形状
    const std::vector<size_t> &sizes() const { return _shape; }

    // 获取张量中元素的总数
    size_t numel() const {
        if (_shape.empty())
            return 1; // 标量有1个元素
        return std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<>());
    }

    // 获取张量的数据类型
    DType dtype() const { return _dtype; }

    // 获取张量所在的设备
    DeviceType device() const { return _device; }

    // ======================= 索引和访问 =======================

    // 1D张量的索引访问
    template <typename T = float> T &operator[](size_t index) {
        checkDType<T>();
        if (dim() != 1)
            throw std::runtime_error("Requires 1D tensor");
        if (index >= _shape[0])
            throw std::out_of_range("Tensor index out of bounds");
        return _storage.data<T>()[_storage_offset + index];
    }

    // 1D张量的常量索引访问
    template <typename T = float> const T &operator[](size_t index) const {
        checkDType<T>();
        if (dim() != 1)
            throw std::runtime_error("Requires 1D tensor");
        if (index >= _shape[0])
            throw std::out_of_range("Tensor index out of bounds");
        return _storage.data<T>()[_storage_offset + index];
    }

    // 多维张量的索引访问
    template <typename T = float> T &operator()(std::initializer_list<size_t> indices) {
        return _storage.data<T>()[computeStorageIndex(indices)];
    }

    // 多维张量的常量索引访问
    template <typename T = float> const T &operator()(std::initializer_list<size_t> indices) const {
        return _storage.data<T>()[computeStorageIndex(indices)];
    }

    // 标量访问（0维张量）
    template <typename T = float> T &item() {
        if (dim() != 0)
            throw std::runtime_error("item() only works on 0-dimensional tensors");
        return *_storage.data<T>();
    }

    // 常量标量访问
    template <typename T = float> const T &item() const {
        if (dim() != 0)
            throw std::runtime_error("item() only works on 0-dimensional tensors");
        return *_storage.data<T>();
    }

    // 获取原始数据的类型化指针
    template <typename T = float> T *data() {
        checkDType<T>();
        if (_storage.empty())
            return nullptr;
        return _storage.data<T>() + _storage_offset;
    }

    // 获取常量原始数据的类型化指针
    template <typename T = float> const T *data() const {
        checkDType<T>();
        if (_storage.empty())
            return nullptr;
        return _storage.data<T>() + _storage_offset;
    }

    // ======================= 张量操作 =======================

    // 创建张量的深拷贝
    Tensor clone() const {
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

    // 改变张量的形状 (不改变内存布局)
    Tensor view(const std::vector<size_t> &new_shape) const {
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

        if (!is_contiguous()) {
            throw std::runtime_error("Cannot view non-contiguous tensor. Call clone() first.");
        }

        Tensor result(ShapeTag{}, new_shape, _dtype, _device);
        result._storage        = _storage; // 共享存储（使用shared_ptr实现共享所有权）
        result._storage_offset = _storage_offset;
        result._strides        = _strides; // 保持原步幅
        result._requires_grad  = _requires_grad;
        return result;
    }

    Tensor sum(const std::vector<int> &dims, bool keepdim = false) const {
        // 只支持全维度求和
        Tensor result(ShapeTag{}, {}, _dtype, _device);
        result._storage = Storage(1, _dtype, _device);

        switch (_dtype) {
        case DType::kFloat: {
            float *dst       = result.data<float>();
            const float *src = data<float>();
            *dst             = std::accumulate(src, src + numel(), 0.0f);
            break;
        }
        case DType::kDouble: {
            double *dst       = result.data<double>();
            const double *src = data<double>();
            *dst              = std::accumulate(src, src + numel(), 0.0);
            break;
        }
        case DType::kInt: {
            int *dst       = result.data<int>();
            const int *src = data<int>();
            *dst           = std::accumulate(src, src + numel(), 0);
            break;
        }
        case DType::kLong: {
            long *dst       = result.data<long>();
            const long *src = data<long>();
            *dst            = std::accumulate(src, src + numel(), 0.0L);
            break;
        }
        }
        return result;
    }

    Tensor sum(int dim, bool keepdim = false) const { return sum(std::vector<int>{dim}, keepdim); }

    // 转置最后两个维度
    Tensor transpose() const {
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

    // ======================= 运算符重载 =======================

    Tensor transpose_last_two() const { return this->transpose(); }

    // 大于运算符，用于ad类
    Tensor operator>(float scalar) const;

    // 张量加法 (逐元素)
    Tensor operator+(const Tensor &rhs) const;

    // 逐元素减法
    Tensor operator-(const Tensor &rhs) const;

    // 逐元素除法
    Tensor operator/(const Tensor &rhs) const;
    // 负号
    Tensor operator-() const;

    // 乘法
    Tensor operator*(const Tensor &rhs) const;

    // 成员函数：Tensor - float
    Tensor operator-(float scalar) const;

    // 成员函数：float - Tensor（友元，需放在类外）
    friend Tensor operator-(float scalar, const Tensor &tensor);

    // 逐元素余弦
    Tensor cos() const;

    // 逐元素正弦
    Tensor sin() const;

    // ReLU激活函数
    Tensor relu() const;

    // Sigmoid激活函数
    Tensor sigmoid() const;

    // Tanh激活函数
    Tensor tanh() const;

    // Softmax激活函数
    Tensor softmax(int dim = -1) const;

    // 张量赋值运算符（深拷贝）
    Tensor &operator=(const Tensor &other) {
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

    // 张量移动赋值运算符
    Tensor &operator=(Tensor &&other) noexcept {
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

    // 张量相等比较
    bool operator==(const Tensor &other) const {
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

    // ======================= 输出和调试 =======================

    // 将张量转换为字符串表示
    std::string toString() const {
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

    // 打印张量信息
    void print() const { std::cout << toString() << std::endl; }

    // 检查张量是否连续
    bool is_contiguous() const {
        size_t stride = 1;
        for (int i = dim() - 1; i >= 0; --i) {
            if (_strides[i] != stride)
                return false;
            stride *= _shape[i];
        }
        return true;
    }

    // 用指定值填充整个张量
    template <typename T> void fill(T value) {
        checkDType<T>();
        const size_t n = numel();
        T *ptr         = data<T>();
        for (size_t i = 0; i < n; ++i) {
            ptr[i] = value;
        }
    }

    // 将张量所有元素设置为0
    void zero() { fill(0); }

    // 将张量所有元素设置为1
    void ones() {
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

    // 是否设置自动微分
    bool isAuto_diff() { return _requires_grad; }

    void requires_grad(bool key) { _requires_grad = key; }
    void set_autograd_ctx(AutoDiff *ctx);

    bool empty() const { return numel() == 0; }

    Tensor grad() const;

    void setDtype(const DType dtype);
};
Tensor matMul(Tensor &a, Tensor &b);
// ======================= 自动微分类 (Auto_Diff) ===================
// ======================= AutoDiff类反向传播实现 =======================

// ======================= 自动微分类 (AutoDiff) =======================
class AutoDiff {
  private:
    // 计算图节点定义
    struct Node {
        Tensor tensor;              // 存储的Tensor值
        Tensor grad;                // 梯度值
        std::vector<Node *> inputs; // 输入节点指针
        op operation;               // 操作类型
        bool requires_grad;         // 是否需要梯度
        bool is_leaf;               // 是否为叶子节点
        size_t retain_count = 0;    // 保留计数（用于高阶导数）

        Node(Tensor t, bool req_grad, bool leaf = true)
            : tensor(std::move(t)), requires_grad(req_grad), is_leaf(leaf) {
            if (requires_grad) {
                // 初始化梯度为相同形状的零张量
                grad = Tensor(ShapeTag{}, tensor.sizes(), tensor.dtype(), tensor.device());
                grad.zero();
            }
        }
    };

    std::unordered_map<Tensor *, Node *> tensor_to_node; // Tensor到节点的映射
    std::vector<std::unique_ptr<Node>> nodes;            // 节点存储
    bool retain_graph = false;                           // 是否保留计算图

  public:
    // 创建叶子节点
    void make_leaf(Tensor &t, bool requires_grad) {
        if (tensor_to_node.find(&t) != tensor_to_node.end()) {
            return; // 已注册，跳过
        }
        auto node          = std::make_unique<Node>(t.clone(), requires_grad);
        tensor_to_node[&t] = node.get();
        nodes.push_back(std::move(node));
    }

    // 记录操作
    void record_op(Tensor &result, op operation, std::initializer_list<Tensor *> inputs) {
        std::vector<Node *> input_nodes;
        bool requires_grad = false;

        // 收集输入节点并检查是否需要梯度
        for (Tensor *input : inputs) {
            if (tensor_to_node.find(input) == tensor_to_node.end()) {
                make_leaf(*input, input->isAuto_diff());
            }
            Node *node = tensor_to_node[input];
            input_nodes.push_back(node);
            if (node->requires_grad)
                requires_grad = true;
        }

        // 创建新节点
        auto new_node       = std::make_unique<Node>(result.clone(), requires_grad, false);
        new_node->inputs    = input_nodes;
        new_node->operation = operation;

        tensor_to_node[&result] = new_node.get();
        nodes.push_back(std::move(new_node));
    }

    // 设置是否保留计算图
    void set_retain_graph(bool retain) { retain_graph = retain; }

    // 获取节点
    Node *get_node(Tensor *t) {
        auto it = tensor_to_node.find(t);
        return it != tensor_to_node.end() ? it->second : nullptr;
    }

    // 反向传播
    void backward(Tensor &root, Tensor grad_output = Tensor()) {
        if (tensor_to_node.find(&root) == tensor_to_node.end()) {
            throw std::runtime_error("Tensor not in computation graph");
        }

        Node *root_node = tensor_to_node[&root];
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
            if (grad_output.sizes() != root.sizes()) {
                throw std::runtime_error("Grad output shape mismatch");
            }
            root_node->grad = grad_output;
        }

        // 拓扑排序（深度优先）
        std::vector<Node *> order;
        std::unordered_set<Node *> visited;
        std::function<void(Node *)> dfs = [&](Node *node) {
            if (visited.find(node) != visited.end())
                return;
            visited.insert(node);

            for (Node *input : node->inputs) {
                dfs(input);
            }
            order.push_back(node);
        };
        dfs(root_node);

        // 反向遍历计算梯度
        std::reverse(order.begin(), order.end());
        for (Node *node : order) {
            // 跳过叶子节点（梯度已存储）
            if (node->is_leaf)
                continue;

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
            clear_graph();
        }
    }

    // 清除计算图
    void clear_graph() {
        tensor_to_node.clear();
        nodes.clear();
    }

  private:
    // ======================= 反向传播函数 =======================

    // 加法反向传播
    void backward_add(Node *node) {
        Tensor &grad_out = node->grad;
        for (size_t i = 0; i < node->inputs.size(); ++i) {
            Node *input_node = node->inputs[i];
            if (!input_node->requires_grad)
                continue;

            // 加法梯度直接传递（需要处理广播）
            Tensor grad_input = grad_out;

            // 如果输入形状不匹配，需要求和降维
            if (input_node->tensor.sizes() != grad_out.sizes()) {
                // 计算需要求和的维度
                std::vector<int> reduce_dims;
                std::vector<size_t> grad_shape  = grad_out.sizes();
                std::vector<size_t> input_shape = input_node->tensor.sizes();

                // 从后往前对齐维度
                int g_idx = grad_shape.size() - 1;
                int i_idx = input_shape.size() - 1;

                while (g_idx >= 0 || i_idx >= 0) {
                    size_t g_dim = (g_idx >= 0) ? grad_shape[g_idx] : 1;
                    size_t i_dim = (i_idx >= 0) ? input_shape[i_idx] : 1;

                    if (g_dim != i_dim && i_dim == 1) {
                        reduce_dims.push_back(g_idx);
                    }

                    g_idx--;
                    i_idx--;
                }

                // 在额外维度上求和
                if (g_idx >= 0) {
                    for (int j = 0; j <= g_idx; j++) {
                        reduce_dims.push_back(j);
                    }
                }

                // 执行求和
                if (!reduce_dims.empty()) {
                    grad_input = grad_out.sum(reduce_dims);
                }
            }

            input_node->grad = input_node->grad + grad_input;
        }
    }

    // 减法反向传播
    void backward_sub(Node *node) {
        Tensor &grad_out = node->grad;
        if (node->inputs.size() != 2) {
            throw std::runtime_error("Subtraction requires exactly 2 inputs");
        }

        Node *a = node->inputs[0];
        Node *b = node->inputs[1];

        if (a->requires_grad) {
            a->grad = a->grad + grad_out;
        }

        if (b->requires_grad) {
            b->grad = b->grad - grad_out; // 注意负号
        }
    }

    // 乘法反向传播
    void backward_mul(Node *node) {
        Tensor &grad_out = node->grad;
        if (node->inputs.size() != 2) {
            throw std::runtime_error("Multiplication requires exactly 2 inputs");
        }

        Node *a = node->inputs[0];
        Node *b = node->inputs[1];

        if (a->requires_grad) {
            Tensor grad_a = grad_out * b->tensor;
            // 处理广播梯度
            grad_a  = reduce_to_match(grad_a, a->tensor.sizes());
            a->grad = a->grad + grad_a;
        }

        if (b->requires_grad) {
            Tensor grad_b = grad_out * a->tensor;
            // 处理广播梯度
            grad_b  = reduce_to_match(grad_b, b->tensor.sizes());
            b->grad = b->grad + grad_b;
        }
    }

    // 除法反向传播
    void backward_div(Node *node) {
        Tensor &grad_out = node->grad;
        if (node->inputs.size() != 2) {
            throw std::runtime_error("Division requires exactly 2 inputs");
        }

        Node *a = node->inputs[0];
        Node *b = node->inputs[1];

        if (a->requires_grad) {
            Tensor grad_a = grad_out / b->tensor;
            grad_a        = reduce_to_match(grad_a, a->tensor.sizes());
            a->grad       = a->grad + grad_a;
        }

        if (b->requires_grad) {
            Tensor grad_b = grad_out * (-a->tensor) / (b->tensor * b->tensor);
            grad_b        = reduce_to_match(grad_b, b->tensor.sizes());
            b->grad       = b->grad + grad_b;
        }
    }

    // 矩阵乘法反向传播
    void backward_matmul(Node *node) {
        Tensor &grad_out = node->grad;
        if (node->inputs.size() != 2) {
            throw std::runtime_error("Matrix multiplication requires exactly 2 inputs");
        }

        Node *a = node->inputs[0];
        Node *b = node->inputs[1];

        if (a->requires_grad) {
            // dL/dA = dL/dC * B^T
            Tensor b_t    = b->tensor.transpose_last_two();
            Tensor grad_a = matMul(grad_out, b_t);
            grad_a        = reduce_to_match(grad_a, a->tensor.sizes());
            a->grad       = a->grad + grad_a;
        }

        if (b->requires_grad) {
            // dL/dB = A^T * dL/dC
            Tensor a_t    = a->tensor.transpose_last_two();
            Tensor grad_b = matMul(a_t, grad_out);
            grad_b        = reduce_to_match(grad_b, b->tensor.sizes());
            b->grad       = b->grad + grad_b;
        }
    }

    // 点积反向传播
    static void backward_dot(Node *node) {
        Tensor &grad_out = node->grad;
        if (node->inputs.size() != 2) {
            throw std::runtime_error("Dot product requires exactly 2 inputs");
        }

        Node *a = node->inputs[0];
        Node *b = node->inputs[1];

        if (a->requires_grad) {
            // dL/da = dL/dout * b
            Tensor grad_a = grad_out * b->tensor;
            a->grad       = a->grad + grad_a;
        }

        if (b->requires_grad) {
            // dL/db = dL/dout * a
            Tensor grad_b = grad_out * a->tensor;
            b->grad       = b->grad + grad_b;
        }
    }

    // 余弦函数反向传播
    static void backward_cos(Node *node) {
        Tensor &grad_out = node->grad;
        Node *input_node = node->inputs[0];

        if (input_node->requires_grad) {
            // ∂cos(x)/∂x = -sin(x)
            Tensor sin_x     = input_node->tensor.sin();
            Tensor grad      = grad_out * (-sin_x);
            input_node->grad = input_node->grad + grad;
        }
    }

    // 正弦函数反向传播
    static void backward_sin(Node *node) {
        Tensor &grad_out = node->grad;
        Node *input_node = node->inputs[0];

        if (input_node->requires_grad) {
            // ∂sin(x)/∂x = cos(x)
            Tensor cos_x     = input_node->tensor.cos();
            Tensor grad      = grad_out * cos_x;
            input_node->grad = input_node->grad + grad;
        }
    }

    // ReLU反向传播
    static void backward_relu(Node *node) {
        Tensor &grad_out = node->grad;
        Node *input_node = node->inputs[0];

        if (input_node->requires_grad) {
            // ∂ReLU(x)/∂x = {1 if x > 0, else 0}
            Tensor mask      = input_node->tensor > 0.0f;
            Tensor grad      = grad_out * mask;
            input_node->grad = input_node->grad + grad;
        }
    }

    // Sigmoid反向传播
    static void backward_sigmoid(Node *node) {
        Tensor &grad_out = node->grad;
        Node *input_node = node->inputs[0];

        if (input_node->requires_grad) {
            // ∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x))
            Tensor sig_x     = node->tensor;
            Tensor grad      = grad_out * sig_x * (1.0f - sig_x);
            input_node->grad = input_node->grad + grad;
        }
    }

    // Tanh反向传播
    static void backward_tanh(Node *node) {
        Tensor &grad_out = node->grad;
        Node *input_node = node->inputs[0];

        if (input_node->requires_grad) {
            // ∂tanh(x)/∂x = 1 - tanh²(x)
            Tensor tanh_x    = node->tensor;
            Tensor grad      = grad_out * (1.0f - tanh_x * tanh_x);
            input_node->grad = input_node->grad + grad;
        }
    }

    // Softmax反向传播
    static void backward_softmax(Node *node) {
        Tensor &grad_out = node->grad;
        Node *input_node = node->inputs[0];

        if (input_node->requires_grad) {
            // 简化实现：更高效的方式是使用雅可比矩阵
            Tensor s    = node->tensor;
            Tensor grad = grad_out;

            // 计算 (grad_out * s) 的和
            Tensor sum_term = (grad_out * s).sum(-1, true);

            // ∂L/∂x = s * (grad_out - sum_term)
            Tensor grad_input = s * (grad_out - sum_term);

            input_node->grad = input_node->grad + grad_input;
        }
    }

    // ======================= 辅助函数 =======================

    // 将梯度减少到目标形状（处理广播）
    static Tensor reduce_to_match(Tensor grad, const std::vector<size_t> &target_shape) {
        if (grad.sizes() == target_shape) {
            return grad;
        }

        // 计算需要求和的维度
        std::vector<int> reduce_dims;
        std::vector<size_t> grad_shape = grad.sizes();

        // 从后往前对齐维度
        int g_idx = grad_shape.size() - 1;
        int t_idx = target_shape.size() - 1;

        while (g_idx >= 0 || t_idx >= 0) {
            size_t g_dim = (g_idx >= 0) ? grad_shape[g_idx] : 1;
            size_t t_dim = (t_idx >= 0) ? target_shape[t_idx] : 1;

            if (g_dim != t_dim && t_dim == 1) {
                reduce_dims.push_back(g_idx);
            }

            if (g_idx >= 0)
                g_idx--;
            if (t_idx >= 0)
                t_idx--;
        }

        // 在额外维度上求和
        if (!reduce_dims.empty()) {
            return grad.sum(reduce_dims);
        }

        return grad;
    }
};

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
    if (autograd_ctx) {
        autograd_ctx->record_op(result, op::Add,
                                {const_cast<Tensor *>(this), const_cast<Tensor *>(&rhs)});
    }
    return result;
}

Tensor Tensor::operator/(const Tensor &rhs) const {
    // 验证形状和类型匹配
    if (_shape != rhs._shape)
        throw std::runtime_error("Shape mismatch in division");
    if (_dtype != rhs._dtype)
        throw std::runtime_error("DType mismatch in division");

    Tensor result(ShapeTag{}, _shape, _dtype, _device);

    switch (_dtype) {
    case DType::kFloat:
        elementwiseOp<float>(result, *this, rhs, [](float a, float b) {
            if (b == 0.0f)
                throw std::runtime_error("Division by zero");
            return a / b;
        });
        break;
    case DType::kDouble:
        elementwiseOp<double>(result, *this, rhs, [](double a, double b) {
            if (b == 0.0)
                throw std::runtime_error("Division by zero");
            return a / b;
        });
        break;
    case DType::kInt:
        elementwiseOp<int32_t>(result, *this, rhs, [](int32_t a, int32_t b) {
            if (b == 0)
                throw std::runtime_error("Division by zero");
            return a / b;
        });
        break;
    case DType::kLong:
        elementwiseOp<int64_t>(result, *this, rhs, [](int64_t a, int64_t b) {
            if (b == 0)
                throw std::runtime_error("Division by zero");
            return a / b;
        });
        break;
    default:
        throw std::runtime_error("Unsupported dtype for division");
    }

    // 记录操作到自动微分计算图
    if (autograd_ctx) {
        autograd_ctx->record_op(result, op::Div,
                                {const_cast<Tensor *>(this), const_cast<Tensor *>(&rhs)});
    }

    return result;
}

Tensor Tensor::operator-(const Tensor &rhs) const {
    // 验证形状和类型匹配
    if (_shape != rhs._shape)
        throw std::runtime_error("Shape mismatch in subtraction");
    if (_dtype != rhs._dtype)
        throw std::runtime_error("DType mismatch in subtraction");

    Tensor result(ShapeTag{}, _shape, _dtype, _device);

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
        throw std::runtime_error("Unsupported dtype for subtraction");
    }

    // 记录操作到自动微分计算图
    if (autograd_ctx) {
        autograd_ctx->record_op(result, op::Sub,
                                {const_cast<Tensor *>(this), const_cast<Tensor *>(&rhs)});
    }

    return result;
}
void Tensor::setDtype(const DType dtype) { _dtype = dtype; }

LogicData broadCast(Tensor &a, Tensor &b) {
    Tensor *large = &(a.shape().size() > b.shape().size() ? a : b);
    Tensor *min   = &(a.shape().size() < b.shape().size() ? a : b);
    for (size_t i{large->shape().size() - 1}; i >= 0 && (a.shape()[i] && b.shape()[i]); i--) {
        if (a.shape()[i] != b.shape()[i] && (a.shape()[i] != 1 || b.shape()[i] != 1))
            throw std::runtime_error("The shape of Tensor provided is incompatible.");
    }

    std::vector<size_t> logicShape(large->shape().size(), 1);
    std::vector<size_t> logicStrides(large->shape().size(), 0);

    for (size_t i{large->shape().size() - 1}; i >= 0; i--) {
        if ((*large).shape()[i] && (*min).shape()[i] && (*min).shape()[i] != 1) {
            logicShape[i]   = (*min).shape()[i];
            logicStrides[i] = (*min).strides()[i];
        }
        logicShape[i] = (*large).shape()[i];
    }
    return {logicShape, logicStrides};
}

Tensor matMul(Tensor &a, Tensor &b) {
    Tensor *min = (a.shape().size() > b.shape().size() ? &a : &b);
    Tensor *max = (a.shape().size() < b.shape().size() ? &a : &b);
    if ((*min).shape().size() > 2 || (*max).shape().size() < 2)
        throw std::runtime_error("Tensors provided are not matrix");
    Tensor result;
    if (!(a.shape() == b.shape())) {
        LogicData logics;
        logics = broadCast(a, b);
        ShapeTag tag;
        result = Tensor(tag, std::vector<size_t>({(*max).shape()[0], logics.logicShape[1]}));
        if ((*max).dtype() != (*min).dtype())
            throw std::runtime_error("DType dosen't match");
        for (size_t row{0}; row < (*max).shape()[0]; row++) {
            int product   = 0;
            size_t column = 0;
            for (; column < logics.logicShape[1]; column++) {
                switch ((*max).dtype()) {
                case DType::kFloat:
                    product += (*max).data<float>()[row * (*max).strides()[1]] *
                               (*min).data<float>()[column * logics.logicStrides[0]];
                    break;
                case DType::kDouble:
                    product += (*max).data<double>()[row * (*max).strides()[1]] *
                               (*min).data<double>()[column * logics.logicStrides[0]];
                    break;
                case DType::kInt:
                    product += (*max).data<int>()[row * (*max).strides()[1]] *
                               (*min).data<int>()[column * logics.logicStrides[0]];
                    break;
                case DType::kLong:
                    product += (*max).data<long>()[row * (*max).strides()[1]] *
                               (*min).data<long>()[column * logics.logicStrides[0]];
                    break;
                case DType::kBool:
                    throw std::runtime_error("Boolean type is not supported for multiplication");
                default:
                    throw std::runtime_error("Unsupported data type for multiplication");
                }
            }
            result({row, column}) = product;
        }
    } else {
        ShapeTag tag;
        result = Tensor(tag, std::vector<size_t>({a.shape()[0], a.shape()[1]}));
        for (size_t row{0}; row < a.shape()[0]; row++) {
            int product   = 0;
            size_t column = 0;
            for (; column < a.shape()[1]; column++) {
                switch (a.dtype()) {
                case DType::kFloat:
                    product += a.data<float>()[row * a.strides()[1]] *
                               a.data<float>()[column * b.strides()[0]];
                    break;
                case DType::kDouble:
                    product += a.data<double>()[row * a.strides()[1]] *
                               a.data<double>()[column * b.strides()[0]];
                    break;
                case DType::kInt:
                    product += a.data<int>()[row * a.strides()[1]] *
                               a.data<int>()[column * b.strides()[0]];
                    break;
                case DType::kLong:
                    product += a.data<long>()[row * a.strides()[1]] *
                               a.data<long>()[column * b.strides()[0]];
                    break;
                case DType::kBool:
                    throw std::runtime_error("Boolean type is not supported for multiplication");
                default:
                    throw std::runtime_error("Unsupported data type for multiplication");
                    break;
                }
            }
            result({row, column}) = product;
        }
    }

    return result;
}

Tensor Tensor::operator*(const Tensor &rhs) const {
    // 验证形状和类型匹配
    if (_shape != rhs._shape)
        throw std::runtime_error("Shape mismatch in multiplication");
    if (_dtype != rhs._dtype)
        throw std::runtime_error("DType mismatch in multiplication");

    Tensor result(ShapeTag{}, _shape, _dtype, _device);

    switch (_dtype) {
    case DType::kFloat:
        elementwiseOp<float>(result, *this, rhs, [](float a, float b) { return a * b; });
        break;
        // 其他数据类型类似...
    }

    // 记录操作到自动微分计算图
    if (autograd_ctx) {
        autograd_ctx->record_op(result, op::Mul,
                                {const_cast<Tensor *>(this), const_cast<Tensor *>(&rhs)});
    }

    return result;
}

inline Tensor operator*(const float &factor,const Tensor a) {
    Tensor result = a;
    for (size_t i{0};i<a.numel();i++) result.data<float>()[i] = factor * result.data<float>()[i];
    result.setDtype(cppType2Dtype<float>());

    // TODO: 补全自动微分相关
    return result;
}

inline Tensor operator*(const Tensor a,const float &factor) {
    Tensor result = a;
    for (size_t i{0};i<a.numel();i++) result.data<float>()[i] = factor * result.data<float>()[i];
    result.setDtype(cppType2Dtype<float>());

    // TODO: 补全自动微分相关
    return result;
}

inline Tensor operator*(const double &factor,const Tensor a) {
    Tensor result = a;
    for (size_t i{0};i<a.numel();i++) result.data<double>()[i] = factor * result.data<double>()[i];
    result.setDtype(cppType2Dtype<double>());

    // TODO: 补全自动微分相关
    return result;
}

inline Tensor operator*(const Tensor a,const double &factor) {
    Tensor result = a;
    for (size_t i{0};i<a.numel();i++) result.data<double>()[i] = factor * result.data<double>()[i];
    result.setDtype(cppType2Dtype<double>());

    // TODO: 补全自动微分相关
    return result;
}

inline Tensor operator*(const int &factor,const Tensor a) {
    Tensor result = a;
    for (size_t i{0};i<a.numel();i++) result.data<int>()[i] = factor * result.data<int>()[i];
    result.setDtype(cppType2Dtype<int>());

    // TODO: 补全自动微分相关
    return result;
}

inline Tensor operator*(const Tensor a,const int &factor) {
    Tensor result = a;
    for (size_t i{0};i<a.numel();i++) result.data<int>()[i] = factor * result.data<int>()[i];
    result.setDtype(cppType2Dtype<int>());

    // TODO: 补全自动微分相关
    return result;
}

inline Tensor operator*(const long &factor,const Tensor a) {
    Tensor result = a;
    for (size_t i{0};i<a.numel();i++) result.data<long>()[i] = factor * result.data<long>()[i];
    result.setDtype(cppType2Dtype<long>());

    // TODO: 补全自动微分相关
    return result;
}

inline Tensor operator*(const Tensor a,const long &factor) {
    Tensor result = a;
    for (size_t i{0};i<a.numel();i++) result.data<long>()[i] = factor * result.data<long>()[i];
    result.setDtype(cppType2Dtype<long>());

    // TODO: 补全自动微分相关
    return result;
}

void Tensor::set_autograd_ctx(AutoDiff *ctx) {
    autograd_ctx = ctx;
    if (ctx && _requires_grad) {
        ctx->make_leaf(*this, true);
    }
}

Tensor Tensor::softmax(int dim) const {
    int actual_dim = dim;
    if (actual_dim < 0) {
        actual_dim = this->dim() + actual_dim; // 负值表示从后往前计数
    }

    if (actual_dim < 0 || actual_dim >= static_cast<int>(this->dim())) {
        throw std::runtime_error("Invalid dimension for softmax");
    }

    Tensor result(ShapeTag{}, _shape, _dtype, _device);

    switch (_dtype) {
    case DType::kFloat: {
        const float *src = data<float>();
        float *dst       = result.data<float>();

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
                float val               = std::exp(src[s * slice_size + i] - max_val);
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
        const double *src = data<double>();
        double *dst       = result.data<double>();

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
                double val              = std::exp(src[s * slice_size + i] - max_val);
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
    if (autograd_ctx) {
        autograd_ctx->record_op(result, op::Softmax, {const_cast<Tensor *>(this)});
    }

    return result;
}

Tensor Tensor::tanh() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);

    switch (_dtype) {
    case DType::kFloat: {
        const float *src = data<float>();
        float *dst       = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) {
            float exp_2x = std::exp(2 * src[i]);
            dst[i]       = (exp_2x - 1) / (exp_2x + 1);
        }
        break;
    }
    case DType::kDouble: {
        const double *src = data<double>();
        double *dst       = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            double exp_2x = std::exp(2 * src[i]);
            dst[i]        = (exp_2x - 1) / (exp_2x + 1);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for tanh");
    }

    // 记录操作到自动微分计算图
    if (autograd_ctx) {
        autograd_ctx->record_op(result, op::Tanh, {const_cast<Tensor *>(this)});
    }

    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);

    switch (_dtype) {
    case DType::kFloat: {
        const float *src = data<float>();
        float *dst       = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
        }
        break;
    }
    case DType::kDouble: {
        const double *src = data<double>();
        double *dst       = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = 1.0 / (1.0 + std::exp(-src[i]));
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for sigmoid");
    }

    // 记录操作到自动微分计算图
    if (autograd_ctx) {
        autograd_ctx->record_op(result, op::Sigmoid, {const_cast<Tensor *>(this)});
    }

    return result;
}

Tensor Tensor::relu() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);

    switch (_dtype) {
    case DType::kFloat: {
        const float *src = data<float>();
        float *dst       = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
        }
        break;
    }
    case DType::kDouble: {
        const double *src = data<double>();
        double *dst       = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = src[i] > 0.0 ? src[i] : 0.0;
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for ReLU");
    }

    // 记录操作到自动微分计算图
    if (autograd_ctx) {
        autograd_ctx->record_op(result, op::ReLU, {const_cast<Tensor *>(this)});
    }

    return result;
}

Tensor Tensor::sin() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);

    switch (_dtype) {
    case DType::kFloat: {
        const float *src = data<float>();
        float *dst       = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = std::sin(src[i]);
        }
        break;
    }
    case DType::kDouble: {
        const double *src = data<double>();
        double *dst       = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = std::sin(src[i]);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for sin");
    }

    // 记录操作到自动微分计算图
    if (autograd_ctx) {
        autograd_ctx->record_op(result, op::Sin, {const_cast<Tensor *>(this)});
    }

    return result;
}

Tensor Tensor::cos() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);

    switch (_dtype) {
    case DType::kFloat: {
        const float *src = data<float>();
        float *dst       = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = std::cos(src[i]);
        }
        break;
    }
    case DType::kDouble: {
        const double *src = data<double>();
        double *dst       = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            dst[i] = std::cos(src[i]);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for cos");
    }

    // 记录操作到自动微分计算图
    if (autograd_ctx) {
        autograd_ctx->record_op(result, op::Cos, {const_cast<Tensor *>(this)});
    }

    return result;
}

Tensor Tensor::grad() const {
    if (!autograd_ctx)
        throw std::runtime_error("No autograd context");
    auto *node = autograd_ctx->get_node(const_cast<Tensor *>(this));
    if (!node || node->grad.empty())
        throw std::runtime_error("No gradient available");
    return node->grad;
}

Tensor Tensor::operator-() const {
    Tensor result = this->clone();
    switch (_dtype) {
    case DType::kFloat: {
        float *ptr = result.data<float>();
        for (size_t i = 0; i < numel(); ++i)
            ptr[i] = -ptr[i];
        break;
    }
    case DType::kDouble: {
        double *ptr = result.data<double>();
        for (size_t i = 0; i < numel(); ++i)
            ptr[i] = -ptr[i];
        break;
    }
    case DType::kLong: {
        long *ptr = result.data<long>();
        for (size_t i = 0; i < numel(); ++i)
            ptr[i] = -ptr[i];
        break;
    }
    case DType::kInt: {
        int *ptr = result.data<int>();
        for (size_t i = 0; i < numel(); ++i)
            ptr[i] = -ptr[i];
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for subtraction");
    }
    return result;
}

Tensor Tensor::operator>(float scalar) const {
    Tensor result(ShapeTag{}, _shape, DType::kBool, _device);
    switch (_dtype) {
    case DType::kFloat: {
        const float *src = data<float>();
        bool *dst        = result.data<bool>();
        for (size_t i = 0; i < numel(); ++i)
            dst[i] = src[i] > scalar;
        break;
    }
    case DType::kDouble: {
        const double *src = data<double>();
        bool *dst         = result.data<bool>();
        for (size_t i = 0; i < numel(); ++i)
            dst[i] = src[i] > scalar;
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for comparison");
    }
    return result;
}
Tensor Tensor::operator-(float scalar) const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    switch (_dtype) {
    case DType::kFloat: {
        const float *src = data<float>();
        float *dst       = result.data<float>();
        for (size_t i = 0; i < numel(); ++i)
            dst[i] = src[i] - scalar;
        break;
    }
    case DType::kDouble: {
        const double *src = data<double>();
        double *dst       = result.data<double>();
        for (size_t i = 0; i < numel(); ++i)
            dst[i] = src[i] - scalar;
        break;
    }
    case DType::kInt: {
        const int32_t *src = data<int32_t>();
        int32_t *dst       = result.data<int32_t>();
        for (size_t i = 0; i < numel(); ++i)
            dst[i] = src[i] - static_cast<int32_t>(scalar);
        break;
    }
    case DType::kLong: {
        const int64_t *src = data<int64_t>();
        int64_t *dst       = result.data<int64_t>();
        for (size_t i = 0; i < numel(); ++i)
            dst[i] = src[i] - static_cast<int64_t>(scalar);
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for subtraction with scalar");
    }

    // 自动微分记录（如启用）
    if (autograd_ctx) {
        autograd_ctx->record_op(result, op::Sub, {const_cast<Tensor *>(this)});
    }

    return result;
}

inline Tensor operator-(float scalar, const Tensor &tensor) {
    Tensor result(ShapeTag{}, tensor._shape, tensor._dtype, tensor._device);
    switch (tensor.dtype()) {
    case DType::kFloat: {
        const float *src = tensor.data<float>();
        float *dst       = result.data<float>();
        for (size_t i = 0; i < tensor.numel(); ++i)
            dst[i] = scalar - src[i];
        break;
    }
    case DType::kDouble: {
        const double *src = tensor.data<double>();
        double *dst       = result.data<double>();
        for (size_t i = 0; i < tensor.numel(); ++i)
            dst[i] = scalar - src[i];
        break;
    }
    case DType::kInt: {
        const int32_t *src = tensor.data<int32_t>();
        int32_t *dst       = result.data<int32_t>();
        for (size_t i = 0; i < tensor.numel(); ++i)
            dst[i] = static_cast<int32_t>(scalar) - src[i];
        break;
    }
    case DType::kLong: {
        const int64_t *src = tensor.data<int64_t>();
        int64_t *dst       = result.data<int64_t>();
        for (size_t i = 0; i < tensor.numel(); ++i)
            dst[i] = static_cast<int64_t>(scalar) - src[i];
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for scalar - tensor");
    }

    // 自动微分记录（注意：此时 scalar 是常数，不参与反向传播）
    if (tensor.autograd_ctx) {
        // 这里可以记录为 sub_from_const 或手动处理
        // 由于 scalar 是常数，反向传播只影响 tensor 的梯度
        tensor.autograd_ctx->record_op(result, op::Sub, {const_cast<Tensor *>(&tensor)});
    }

    return result;
}

// TENSOR_CPPM
