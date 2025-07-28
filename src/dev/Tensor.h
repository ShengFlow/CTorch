/*
 * Tensor.h
 * Created by Beapoe & GhostFace on 2025.7
 * Main Classes: Storage & Tensor & Auto_diff
 * Version : v1.6 (fixed on 2025.7.28 21:48)
 * Log 1.3: 增加了注释及代码易读性
 * Log 1.4: 增加了AutoDiff自动微分类
 * Log 1.5: 增加了连续性检查，修复了变量命名，增加了对自动微分状态的输出，修复了移动时不移动自动微分状态的bug
 * Log 1.6: 修复了广播操作并对所有二元操作进行广播处理，优化了矩阵乘法
 * Unfix : matMul
 */

// includes
#include <algorithm>
#include <cstddef>
#include <initializer_list>
// #include <immintrin.h> 未支持ARM
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <functional>
#include <cstring>
#include <iomanip>
#include <string>
#include <unordered_set>
// ======================= 类型定义和枚举 =======================

// 设备类型 - 定义张量存储的位置
enum class DeviceType {
    kCPU,    //< 主存储器 (RAM)
    kCUDA,   //< NVIDIA GPU (暂未实现)
    kMPS,    //< Apple Silicon (暂未实现)
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
enum class op{

    // 基本运算
    Add,        // 加
    Sub,        // 减
    Mul,        // 乘
    Div,        // 除
    MatMul,     // 矩阵乘法
    Dot,        // 点乘
    Cos,
    Sin,

    // 卷积操作
    Conv,       // 卷积
    Pool,       // 池化

    // 激活函数
    ReLU,       // 线性整流函数
    Tanh,       // 双曲正切函数
    Sigmoid,
    Softmax,

    // 激活函数变种
    LReLU,      // 渗漏线性整流函数
    PReLU,      // 参数化线性整流函数

    // 损失函数
    MSE,         // 均方误差
    MAE,         // 平均绝对误差
    CE,          // 交叉熵损失
    BCE,         // 二元交叉熵损失

    // 其他操作
    Sum,
};

// 广播变形数据结构体
//struct LogicData{
    //std::vector<size_t> logicShape; // 由于广播之后shape和strides都发生了改变，所以广播之后的计算应该依赖于logicShape和logicStrides
  //  std::vector<size_t> logicStrides;
//};

struct BroadCastResult {
    std::vector<size_t> logicShape;    // 广播后的逻辑形状
    std::vector<size_t> logicStridesA; // 张量A的逻辑步幅
    std::vector<size_t> logicStridesB; // 张量B的逻辑步幅
};

// ======================= 辅助函数 =======================

// 将数据类型转换为字符串表示
constexpr const char* dtypeToString(DType dtype) {
    switch (dtype) {
        case DType::kFloat:  return "float32";
        case DType::kDouble: return "float64";
        case DType::kInt:    return "int32";
        case DType::kLong:   return "int64";
        case DType::kBool:   return "bool";
        default:             return "unknown";
    }
}

// 获取数据类型的字节大小
constexpr size_t dtypeSize(DType dtype) {
    switch (dtype) {
        case DType::kFloat:  return sizeof(float);
        case DType::kDouble: return sizeof(double);
        case DType::kInt:    return sizeof(int32_t);
        case DType::kLong:   return sizeof(int64_t);
        case DType::kBool:   return sizeof(bool);
        default: throw std::runtime_error("Unsupported dtype");
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

class AutoDiff;                         // 前置声明，避免循环引用

class Storage {
private:
    size_t _size{};                     // 存储的元素数量，此处使用C++11的新特性花括号初始化，避免类型转换，实际上等同于size_t _size = 0;
    DType _dtype;                       // 数据类型 用于枚举
    DeviceType _device;                 // 设备类型 用于枚举
    std::shared_ptr<char[]> _data;      // 原始内存指针（使用shared_ptr实现共享所有权）避免出现手动delete的问题和delete数组和默认方法不匹配
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
    Storage(size_t size, DType dtype, DeviceType device = DeviceType::kCPU): _size(size), _dtype(dtype), _device(device),_data(size > 0 ? std::shared_ptr<char[]>(new char[size * dtypeSize(dtype)]) : nullptr) {}
    // 如果初始化列表中_size为0，那么初始化为nullptr

    // 构造函数：从现有数据复制
    template <typename T>
    Storage(const T* data, size_t size, DType dtype, DeviceType device = DeviceType::kCPU): Storage(size, dtype, device) {
        if (size > 0 && _data.get()) {
            std::memcpy(_data.get(), data, size * dtypeSize(dtype));
        }
    }
    // 注意，此处实际上将从现有数据复制的操作委托给了第一个构造函数，然后进行memcpy的操作进行复制

    // 默认拷贝构造函数和拷贝赋值运算符（使用shared_ptr，所以是浅拷贝）
    // 此处同上，减少内存开销
    Storage(const Storage&) = default;
    Storage& operator=(const Storage&) = default;

    // 默认移动构造函数和移动赋值运算符
    Storage(Storage&&) = default;
    Storage& operator=(Storage&&) = default;

    Storage() : _size(0), _dtype(DType::kFloat), _device(DeviceType::kCPU) {}

    ~Storage() = default;

    // 获取原始数据的类型化指针
    template <typename T>
    T* data() {
        if (_size == 0 || !_data) return nullptr;
        checkDType<T>();
        return reinterpret_cast<T*>(_data.get());
    }

    // 获取常量原始数据的类型化指针
    template <typename T>
    const T* data() const {
        if (_size == 0 || !_data) return nullptr;
        checkDType<T>();
        return reinterpret_cast<const T*>(_data.get());
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

// ======================= 张量类 (Tensor) =======================
struct ShapeTag {}; // 此处结构体为了使编译器区分构造函数

class Tensor {
private:
    bool _requires_grad = false;   // 是否参与自动微分计算，默认不参与
    // 张量的维度大小
    std::vector<size_t> _strides; // 每个维度的步幅
    size_t _storage_offset;       // 存储中的起始偏移量
    DeviceType _device;           // 张量所在的设备
    DType _dtype;                 // 张量元素的数据类型
    Storage _storage;             // 存储张量数据的对象
    AutoDiff* autograd_ctx = nullptr; // 自动微分上下文指针

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

    // 检查数据类型是否匹配
    template <typename T>
    void checkDType() const {
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
    void elementwiseOp(Tensor& result, const Tensor& a, const Tensor& b, Op op) const {
        const size_t n = a.numel();
        T* out = result.data<T>();
        const T* a_data = a.data<T>();
        const T* b_data = b.data<T>();

        for (size_t i = 0; i < n; ++i) {
            out[i] = op(a_data[i], b_data[i]);
        }
    }

    // 支持广播的逐元素操作
    template <typename T, typename Op>
    void broadcast_elementwise_op(Tensor& result, const Tensor& a, const Tensor& b,
                                  const BroadCastResult& bc, Op op) const {
        const std::vector<size_t>& shape = bc.logicShape;
        const std::vector<size_t>& stridesA = bc.logicStridesA;
        const std::vector<size_t>& stridesB = bc.logicStridesB;

        T* out = result.data<T>();
        const T* a_data = a.data<T>();
        const T* b_data = b.data<T>();

        size_t total_elements = 1;
        for (auto dim : shape) total_elements *= dim;

        // 遍历广播后的每个元素
        for (size_t flat_idx = 0; flat_idx < total_elements; ++flat_idx) {
            size_t a_idx = 0;
            size_t b_idx = 0;
            size_t tmp_idx = flat_idx;

            // 计算每个维度上的坐标
            for (int i = shape.size() - 1; i >= 0; --i) {
                size_t dim_size = shape[i];
                size_t coord = tmp_idx % dim_size;
                tmp_idx /= dim_size;

                a_idx += coord * stridesA[i];
                b_idx += coord * stridesB[i];
            }

            out[flat_idx] = op(a_data[a_idx], b_data[b_idx]);
        }
    }
    // 递归打印张量内容（改进版）
    template <typename T>
    void printRecursive(std::ostream& os, size_t dim, std::vector<size_t> indices) const {
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
            for (size_t i = 0; i < dim; ++i) os << "  ";
        }
        os << "[";

        const size_t max_display = 3; // 每维度最大显示元素数
        const size_t display_count = std::min(_shape[dim], max_display);
        const bool truncated = _shape[dim] > max_display;

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
    Tensor(float value) : _shape({}), _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kFloat) {
        computeStrides();
        _storage = Storage(1, _dtype, _device);
        *_storage.data<float>() = value;
    }

    // 构造函数：从初始值列表创建1D张量
    Tensor(std::initializer_list<float> values): _shape({values.size()}), _storage_offset(0),_device(DeviceType::kCPU), _dtype(DType::kFloat) {
        computeStrides();
        _storage = Storage(values.begin(), values.size(), _dtype, _device);
    }

    // 添加布尔张量构造函数
    Tensor(std::initializer_list<bool> values)
            : _shape({values.size()}), _storage_offset(0),
              _device(DeviceType::kCPU), _dtype(DType::kBool) {
        computeStrides();
        _storage = Storage(values.size(), _dtype, _device);
        bool* data = _storage.data<bool>();
        size_t i = 0;
        for (bool val : values) {
            data[i++] = val;
        }
    }

    // 构造函数：指定形状和数据类型（使用 ShapeTag 避免歧义）
    Tensor(ShapeTag, const std::vector<size_t>& shape, DType dtype = DType::kFloat, DeviceType device = DeviceType::kCPU, bool zero_init = true): _shape(shape), _storage_offset(0), _device(device), _dtype(dtype) {
        computeStrides();
        _storage = Storage(numel(), _dtype, _device);
        if(zero_init) zero();
    }

    // 拷贝构造函数：创建深拷贝
    Tensor(const Tensor& other)
            : _shape(other._shape), _strides(other._strides),
              _storage_offset(other._storage_offset),
              _device(other._device), _dtype(other._dtype),
              _storage(other._storage.clone()) {}  // 深拷贝存储

    // 移动构造函数
    Tensor(Tensor&& other) noexcept
            : _shape(std::move(other._shape)),
              _strides(std::move(other._strides)),
              _storage_offset(other._storage_offset),
              _device(other._device), _dtype(other._dtype),
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
    const std::vector<size_t>& sizes() const { return _shape; }

    // 获取张量中元素的总数
    size_t numel() const {
        if (_shape.empty()) return 1; // 标量有1个元素
        return std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<>());
    }

    // 获取张量的数据类型
    DType dtype() const { return _dtype; }

    // 获取张量所在的设备
    DeviceType device() const { return _device; }

    // ======================= 索引和访问 =======================

    // 1D张量的索引访问
    template <typename T = float>
    T& operator[](size_t index) {
        checkDType<T>();
        if (dim() != 1) throw std::runtime_error("Requires 1D tensor");
        if (index >= _shape[0]) throw std::out_of_range("Tensor index out of bounds");
        return _storage.data<T>()[_storage_offset + index];
    }

    // 1D张量的常量索引访问
    template <typename T = float>
    const T& operator[](size_t index) const {
        checkDType<T>();
        if (dim() != 1) throw std::runtime_error("Requires 1D tensor");
        if (index >= _shape[0]) throw std::out_of_range("Tensor index out of bounds");
        return _storage.data<T>()[_storage_offset + index];
    }

    // 多维张量的索引访问
    template <typename T = float>
    T& operator()(std::initializer_list<size_t> indices) {
        return _storage.data<T>()[computeStorageIndex(indices)];
    }

    // 多维张量的常量索引访问
    template <typename T = float>
    const T& operator()(std::initializer_list<size_t> indices) const {
        return _storage.data<T>()[computeStorageIndex(indices)];
    }

    // 标量访问（0维张量）
    template <typename T = float>
    T& item() {
        if (dim() != 0) throw std::runtime_error("item() only works on 0-dimensional tensors");
        return *_storage.data<T>();
    }

    // 常量标量访问
    template <typename T = float>
    const T& item() const {
        if (dim() != 0) throw std::runtime_error("item() only works on 0-dimensional tensors");
        return *_storage.data<T>();
    }

    // 获取原始数据的类型化指针
    template <typename T = float>
    T* data() {
        checkDType<T>();
        if (_storage.empty()) return nullptr;
        return _storage.data<T>() + _storage_offset;
    }

    // 获取常量原始数据的类型化指针
    template <typename T = float>
    const T* data() const {
        checkDType<T>();
        if (_storage.empty()) return nullptr;
        return _storage.data<T>() + _storage_offset;
    }

    // ======================= 张量操作 =======================

    // 创建张量的深拷贝
    Tensor clone() const {
        Tensor copy;
        copy._shape = _shape;
        copy._strides = _strides;
        copy._storage_offset = 0;
        copy._device = _device;
        copy._dtype = _dtype;
        copy._storage = _storage.clone(); // 深拷贝存储
        copy._requires_grad = _requires_grad;
        copy.autograd_ctx = autograd_ctx;  // 修复:继承上下文
        return copy;
    }

    // 改变张量的形状 (不改变内存布局)
    Tensor view(const std::vector<size_t>& new_shape) const {
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
        result.autograd_ctx = autograd_ctx;  // 修复:继承上下文
        return result;
    }

    Tensor sum(const std::vector<int>& dims, bool keepdim = false) const;

    Tensor sum(int dim, bool keepdim = false) const {
        return sum(std::vector<int>{dim}, keepdim);
    }

    Tensor sum() const {
        return sum(std::vector<int>{}, false);
    }

    // 转置最后两个维度
    Tensor transpose() const {
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

    // 设置上下文并传播到新张量
    Tensor with_ctx(AutoDiff* ctx) const {
        Tensor result = *this;
        result.autograd_ctx = ctx;
        return result;
    }

    // 创建新张量时继承上下文
    static Tensor create_with_ctx(AutoDiff* ctx, const std::vector<size_t>& shape,
                                  DType dtype, DeviceType device) {
        Tensor result(ShapeTag{}, shape, dtype, device);
        result.autograd_ctx = ctx;
        return result;
    }

    // ======================= 运算符重载 =======================

    Tensor transpose_last_two() const {
        return this->transpose();
    }

    // 大于运算符，用于ad类
    Tensor operator>(float scalar) const;

    // 张量加法 (逐元素)
    Tensor operator+(const Tensor& rhs) const;

    // 逐元素减法
    Tensor operator-(const Tensor& rhs) const;

    Tensor operator*(const Tensor& rhs) const;

    // 逐元素除法
    Tensor operator/(const Tensor& rhs) const;
    // 负号
    Tensor operator-() const;

    // 成员函数：Tensor - float
    Tensor operator-(float scalar) const;

    // 成员函数：float - Tensor（友元，需放在类外）
    friend Tensor operator-(float scalar, const Tensor& tensor);

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
    Tensor& operator=(const Tensor& other) {
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

    // 张量移动赋值运算符
    Tensor& operator=(Tensor&& other) noexcept {
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

    // 张量相等比较
    bool operator==(const Tensor& other) const {
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

    // ======================= 输出和调试 =======================

    // 将张量转换为字符串表示
    std::string toString() const {
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

    // 打印张量信息
    void print() const {
        std::cout << toString() << std::endl;
    }

    // 检查张量是否连续
    bool is_contiguous() const {
        size_t stride = 1;
        for (int i = dim()-1; i >= 0; --i) {
            if (_strides[i] != stride) return false;
            stride *= _shape[i];
        }
        return true;
    }


    // 用指定值填充整个张量
    void fill(float value) {
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

    // 将张量所有元素设置为0
    void zero() {
        switch (_dtype) {
            case DType::kFloat:   fill(0.0f); break;
            case DType::kDouble:  fill(0.0); break;
            case DType::kInt:     fill(0); break;
            case DType::kLong:    fill(0L); break;
            case DType::kBool:    fill(false); break;
            default: throw std::runtime_error("Unsupported dtype for zero()");
        }
    }

    // 将张量所有元素设置为1
    void ones() {
        switch (_dtype) {
            case DType::kFloat:   fill(1.0f); break;
            case DType::kDouble:  fill(1.0); break;
            case DType::kInt:     fill(1); break;
            case DType::kLong:    fill(1); break;
            case DType::kBool:    fill(true); break;
            default: throw std::runtime_error("Unsupported dtype for ones()");
        }
    }

    // 是否设置自动微分
    bool isAuto_diff(){
        return _requires_grad;
    }

    void requires_grad(bool key){
        _requires_grad = key;
    }
    void set_autograd_ctx(AutoDiff* ctx);

    bool empty() const {
        return numel() == 0;
    }

    Tensor grad() const;

    void setDtype(const DType dtype);
};

Tensor matMul(Tensor &a, Tensor &b);        // 矩阵乘前置声明

// ======================= 自动微分类 (AutoDiff) =======================
class AutoDiff {
private:
    // 计算图节点定义
    struct Node {
        Tensor tensor;                  // 存储的Tensor值
        Tensor grad;                    // 梯度值
        std::vector<Node*> inputs;      // 输入节点指针
        op operation;                   // 操作类型
        bool requires_grad;             // 是否需要梯度
        bool is_leaf;                   // 是否为叶子节点
        size_t retain_count = 0;        // 保留计数（用于高阶导数）

        Node(Tensor t, bool req_grad, bool leaf = true)
                : tensor(std::move(t)), requires_grad(req_grad), is_leaf(leaf) {
            if (requires_grad) {
                // 初始化梯度为相同形状的零张量
                grad = Tensor(ShapeTag{}, tensor.sizes(), tensor.dtype(), tensor.device());
                grad.zero();
            }
        }
    };

    std::unordered_map<Tensor*, Node*> tensor_to_node;  // Tensor到节点的映射
    std::vector<std::unique_ptr<Node>> nodes;           // 节点存储
    bool retain_graph = false;                          // 是否保留计算图

public:
    // 创建叶子节点
    void make_leaf(Tensor& t, bool requires_grad) {
        if (tensor_to_node.find(&t) != tensor_to_node.end()) {
            return; // 已注册，跳过
        }
        auto node = std::make_unique<Node>(t.clone(), requires_grad);
        tensor_to_node[&t] = node.get();
        nodes.push_back(std::move(node));
    }

    // 记录操作
    void record_op(Tensor& result, op operation, std::initializer_list<Tensor*> inputs) {
        // 确保结果张量已注册
        if (tensor_to_node.find(&result) == tensor_to_node.end()) {
            // 非叶子节点注册（requires_grad=false）
            auto new_node = std::make_unique<Node>(result.clone(), false, false);
            tensor_to_node[&result] = new_node.get();
            nodes.push_back(std::move(new_node));
        }

        // 更新节点信息
        Node* node = tensor_to_node[&result];
        node->operation = operation;

        // 收集输入节点
        node->inputs.clear();
        for (Tensor* input : inputs) {
            if (tensor_to_node.find(input) == tensor_to_node.end()) {
                make_leaf(*input, input->isAuto_diff());
            }
            node->inputs.push_back(tensor_to_node[input]);
        }

        // 确定是否需要梯度
        node->requires_grad = std::any_of(
                node->inputs.begin(), node->inputs.end(),
                [](Node* n) { return n->requires_grad; }
        );
    }

    // 设置是否保留计算图
    void set_retain_graph(bool retain) {
        retain_graph = retain;
    }

    // 获取节点
    Node* get_node(Tensor *t) {
        auto it = tensor_to_node.find(t);
        return it != tensor_to_node.end() ? it->second : nullptr;
    }

    // 反向传播
    void backward(Tensor& root, Tensor grad_output = Tensor()) {
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
            if (grad_output.sizes() != root.sizes()) {
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

    // 加法反向传播（已支持广播）
    void backward_add(Node* node) {
        Tensor& grad_out = node->grad;
        for (Node* input_node : node->inputs) {
            if (!input_node->requires_grad) continue;

            // 处理广播：将梯度reduce到输入形状
            Tensor grad_input = reduce_to_match(grad_out, input_node->tensor.sizes());
            input_node->grad = input_node->grad + grad_input;
        }
    }

    // 减法反向传播（已支持广播）
    void backward_sub(Node* node) {
        Tensor& grad_out = node->grad;
        if (node->inputs.size() != 2) {
            throw std::runtime_error("Subtraction requires exactly 2 inputs");
        }

        Node* a = node->inputs[0];
        Node* b = node->inputs[1];

        if (a->requires_grad) {
            Tensor grad_a = reduce_to_match(grad_out, a->tensor.sizes());
            a->grad = a->grad + grad_a;
        }

        if (b->requires_grad) {
            Tensor grad_b = reduce_to_match(grad_out, b->tensor.sizes());
            b->grad = b->grad - grad_b;
        }
    }

    // 乘法反向传播
    void backward_mul(Node* node) {
        Tensor& grad_out = node->grad;
        if (node->inputs.size() != 2) {
            throw std::runtime_error("Multiplication requires exactly 2 inputs");
        }

        Node* a = node->inputs[0];
        Node* b = node->inputs[1];

        if (a->requires_grad) {
            Tensor grad_a = grad_out * b->tensor;
            // 处理广播梯度
            grad_a = reduce_to_match(grad_a, a->tensor.sizes());
            a->grad = a->grad + grad_a;
        }

        if (b->requires_grad) {
            Tensor grad_b = grad_out * a->tensor;
            // 处理广播梯度
            grad_b = reduce_to_match(grad_b, b->tensor.sizes());
            b->grad = b->grad + grad_b;
        }
    }

    // 除法反向传播
    void backward_div(Node* node) {
        Tensor& grad_out = node->grad;
        if (node->inputs.size() != 2) {
            throw std::runtime_error("Division requires exactly 2 inputs");
        }

        Node* a = node->inputs[0];
        Node* b = node->inputs[1];

        if (a->requires_grad) {
            Tensor grad_a = grad_out / b->tensor;
            grad_a = reduce_to_match(grad_a, a->tensor.sizes());
            a->grad = a->grad + grad_a;
        }

        if (b->requires_grad) {
            Tensor grad_b = grad_out * (-a->tensor) / (b->tensor * b->tensor);
            grad_b = reduce_to_match(grad_b, b->tensor.sizes());
            b->grad = b->grad + grad_b;
        }
    }

    // 矩阵乘法反向传播
    void backward_matmul(Node* node) {
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
            grad_a = reduce_to_match(grad_a, a->tensor.sizes());
            a->grad = a->grad + grad_a;
        }

        if (b->requires_grad) {
            // dL/dB = A^T * dL/dC
            Tensor a_t = a->tensor.transpose_last_two();
            Tensor grad_b = matMul(a_t,grad_out);
            grad_b = reduce_to_match(grad_b, b->tensor.sizes());
            b->grad = b->grad + grad_b;
        }
    }

    // 点积反向传播
    static void backward_dot(Node* node) {
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

    // 余弦函数反向传播
    static void backward_cos(Node* node) {
        Tensor& grad_out = node->grad;
        Node* input_node = node->inputs[0];

        if (input_node->requires_grad) {
            // ∂cos(x)/∂x = -sin(x)
            Tensor sin_x = input_node->tensor.sin();
            Tensor grad = grad_out * (-sin_x);
            input_node->grad = input_node->grad + grad;
        }
    }

    // 正弦函数反向传播
    static void backward_sin(Node* node) {
        Tensor& grad_out = node->grad;
        Node* input_node = node->inputs[0];

        if (input_node->requires_grad) {
            // ∂sin(x)/∂x = cos(x)
            Tensor cos_x = input_node->tensor.cos();
            Tensor grad = grad_out * cos_x;
            input_node->grad = input_node->grad + grad;
        }
    }

    // ReLU反向传播
    static void backward_relu(Node* node) {
        Tensor& grad_out = node->grad;
        Node* input_node = node->inputs[0];

        if (input_node->requires_grad) {
            // ∂ReLU(x)/∂x = {1 if x > 0, else 0}
            Tensor mask = input_node->tensor > 0.0f;
            Tensor grad = grad_out * mask;
            input_node->grad = input_node->grad + grad;
        }
    }

    // Sigmoid反向传播
    static void backward_sigmoid(Node* node) {
        Tensor& grad_out = node->grad;
        Node* input_node = node->inputs[0];

        if (input_node->requires_grad) {
            // ∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x))
            Tensor sig_x = node->tensor;
            Tensor grad = grad_out * sig_x * (1.0f - sig_x);
            input_node->grad = input_node->grad + grad;
        }
    }

    // Tanh反向传播
    static void backward_tanh(Node* node) {
        Tensor& grad_out = node->grad;
        Node* input_node = node->inputs[0];

        if (input_node->requires_grad) {
            // ∂tanh(x)/∂x = 1 - tanh²(x)
            Tensor tanh_x = node->tensor;
            Tensor grad = grad_out * (1.0f - tanh_x * tanh_x);
            input_node->grad = input_node->grad + grad;
        }
    }

    // Softmax反向传播
    static void backward_softmax(Node* node) {
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

    // 降维操作反向传播
    static void backward_sum(Node* node) {
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
            else if (grad_out.sizes() == input_node->tensor.sizes()) {
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

    // ======================= 辅助函数 =======================

    // 将梯度减少到目标形状（处理广播）
    static Tensor reduce_to_match(Tensor grad, const std::vector<size_t>& target_shape) {
        if (grad.sizes() == target_shape) {
            return grad;
        }

        // 计算需要求和的维度
        std::vector<int> reduce_dims;
        std::vector<size_t> grad_shape = grad.sizes();
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
            std::vector<size_t> reduced_shape = reduced.sizes();
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
};

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

Tensor Tensor::operator+(const Tensor &rhs) const {
    // 验证形状和类型匹配
    if (_dtype != rhs._dtype) throw std::runtime_error("DType mismatch in addition");
    if (_device != rhs._device) throw std::runtime_error("Device mismatch in addition");
    if (_shape == rhs._shape){
        // 创建结果张量时继承上下文
        AutoDiff* ctx = this->autograd_ctx ? this->autograd_ctx : rhs.autograd_ctx;
        Tensor result = Tensor::create_with_ctx(ctx, _shape, _dtype, _device);
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
        if (ctx) {
            ctx->record_op(
                    result,
                    op::Add,
                    {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)}
            );
        }
        return result;
    } else {
        // 形状不同时进行广播
        BroadCastResult bc = broadCast(*this, rhs);

        // 创建结果张量（使用广播后的形状）
        AutoDiff* ctx = this->autograd_ctx ? this->autograd_ctx : rhs.autograd_ctx;
        Tensor result = Tensor::create_with_ctx(ctx, bc.logicShape, _dtype, _device);

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

        // 记录自动微分操作
        if (ctx) {
            ctx->record_op(
                    result,
                    op::Add,
                    {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)}
            );
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
        AutoDiff* ctx = this->autograd_ctx ? this->autograd_ctx : rhs.autograd_ctx;
        Tensor result = Tensor::create_with_ctx(ctx, _shape, _dtype, _device);
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
        if (ctx) {
            ctx->record_op(
                    result,
                    op::Div,
                    {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)}
            );
        }
        return result;
    } else {
        // 形状不同时进行广播
        BroadCastResult bc = broadCast(*this, rhs);

        // 创建结果张量（使用广播后的形状）
        AutoDiff* ctx = this->autograd_ctx ? this->autograd_ctx : rhs.autograd_ctx;
        Tensor result = Tensor::create_with_ctx(ctx, bc.logicShape, _dtype, _device);

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

        // 记录自动微分操作
        if (ctx) {
            ctx->record_op(
                    result,
                    op::Div,
                    {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)}
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
        AutoDiff* ctx = this->autograd_ctx ? this->autograd_ctx : rhs.autograd_ctx;
        Tensor result = Tensor::create_with_ctx(ctx, _shape, _dtype, _device);
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
        if (ctx) {
            ctx->record_op(
                    result,
                    op::Sub,
                    {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)}
            );
        }
        return result;
    } else {
        // 形状不同时进行广播
        BroadCastResult bc = broadCast(*this, rhs);

        // 创建结果张量（使用广播后的形状）
        AutoDiff* ctx = this->autograd_ctx ? this->autograd_ctx : rhs.autograd_ctx;
        Tensor result = Tensor::create_with_ctx(ctx, bc.logicShape, _dtype, _device);

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
        if (ctx) {
            ctx->record_op(
                    result,
                    op::Sub,
                    {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)}
            );
        }

        return result;
    }
}

Tensor matMul(Tensor &a, Tensor &b) {
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

Tensor Tensor::operator*(const Tensor &rhs) const {
    // 验证形状和类型匹配
    if (_dtype != rhs._dtype) throw std::runtime_error("DType mismatch in addition");
    if (_device != rhs._device) throw std::runtime_error("Device mismatch in addition");
    if (_shape == rhs._shape){
        // 创建结果张量时继承上下文
        AutoDiff* ctx = this->autograd_ctx ? this->autograd_ctx : rhs.autograd_ctx;
        Tensor result = Tensor::create_with_ctx(ctx, _shape, _dtype, _device);
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
        if (ctx) {
            ctx->record_op(
                    result,
                    op::Mul,
                    {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)}
            );
        }
        return result;
    } else {
        // 形状不同时进行广播
        BroadCastResult bc = broadCast(*this, rhs);

        // 创建结果张量（使用广播后的形状）
        AutoDiff* ctx = this->autograd_ctx ? this->autograd_ctx : rhs.autograd_ctx;
        Tensor result = Tensor::create_with_ctx(ctx, bc.logicShape, _dtype, _device);

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
        if (ctx) {
            ctx->record_op(
                    result,
                    op::Mul,
                    {const_cast<Tensor*>(this), const_cast<Tensor*>(&rhs)}
            );
        }

        return result;
    }
}

void Tensor::set_autograd_ctx(AutoDiff *ctx) {
    autograd_ctx = ctx;
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
    result.autograd_ctx = this->autograd_ctx;  // 继承上下文
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
    if (autograd_ctx) {
        autograd_ctx->record_op(
                result,
                op::Softmax,
                {const_cast<Tensor*>(this)}
        );
    }

    return result;
}

Tensor Tensor::tanh() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    result.autograd_ctx = this->autograd_ctx;  // 继承上下文
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
    if (autograd_ctx) {
        autograd_ctx->record_op(
                result,
                op::Tanh,
                {const_cast<Tensor*>(this)}
        );
    }

    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    result.autograd_ctx = this->autograd_ctx;  // 继承上下文
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
    if (autograd_ctx) {
        autograd_ctx->record_op(
                result,
                op::Sigmoid,
                {const_cast<Tensor*>(this)}
        );
    }

    return result;
}

Tensor Tensor::relu() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    result.autograd_ctx = this->autograd_ctx;  // 继承上下文
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
    if (autograd_ctx) {
        autograd_ctx->record_op(
                result,
                op::ReLU,
                {const_cast<Tensor*>(this)}
        );
    }

    return result;
}

Tensor Tensor::sin() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    result.autograd_ctx = this->autograd_ctx;  // 继承上下文
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
    if (autograd_ctx) {
        autograd_ctx->record_op(
                result,
                op::Sin,
                {const_cast<Tensor*>(this)}
        );
    }

    return result;
}

Tensor Tensor::cos() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    result.autograd_ctx = this->autograd_ctx;  // 继承上下文
    switch (_dtype) {
        case DType::kFloat: {
            const float* src = data<float>();
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
    if (autograd_ctx) {
        autograd_ctx->record_op(
                result,
                op::Cos,
                {const_cast<Tensor*>(this)}
        );
    }

    return result;
}

Tensor Tensor::grad() const {
    if (!autograd_ctx) throw std::runtime_error("No autograd context");
    auto* node = autograd_ctx->get_node(const_cast<Tensor*>(this));
    if (!node || node->grad.empty()) throw std::runtime_error("No gradient available");
    return node->grad;
}

Tensor Tensor::operator-() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    result.autograd_ctx = this->autograd_ctx;  // 继承上下文
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

Tensor Tensor::operator>(float scalar) const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    result.autograd_ctx = this->autograd_ctx;  // 继承上下文
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

Tensor Tensor::operator-(float scalar) const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    result.autograd_ctx = this->autograd_ctx;  // 继承上下文
    switch (_dtype) {
        case DType::kFloat: {
            const float* src = data<float>();
            float* dst = result.data<float>();
            for (size_t i = 0; i < numel(); ++i) dst[i] = src[i] - scalar;
            break;
        }
        case DType::kDouble: {
            const double* src = data<double>();
            double* dst = result.data<double>();
            for (size_t i = 0; i < numel(); ++i) dst[i] = src[i] - scalar;
            break;
        }
        case DType::kInt: {
            const int32_t* src = data<int32_t>();
            int32_t* dst = result.data<int32_t>();
            for (size_t i = 0; i < numel(); ++i) dst[i] = src[i] - static_cast<int32_t>(scalar);
            break;
        }
        case DType::kLong: {
            const int64_t* src = data<int64_t>();
            int64_t* dst = result.data<int64_t>();
            for (size_t i = 0; i < numel(); ++i) dst[i] = src[i] - static_cast<int64_t>(scalar);
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for subtraction with scalar");
    }

    // 自动微分记录（如启用）
    if (autograd_ctx) {
        autograd_ctx->record_op(result, op::Sub, {const_cast<Tensor*>(this)});
    }

    return result;
}

inline Tensor operator-(float scalar, const Tensor& tensor) {
    Tensor result(ShapeTag{}, tensor._shape, tensor._dtype, tensor._device);
    switch (tensor.dtype()) {
        case DType::kFloat: {
            const float* src = tensor.data<float>();
            float* dst = result.data<float>();
            for (size_t i = 0; i < tensor.numel(); ++i) dst[i] = scalar - src[i];
            break;
        }
        case DType::kDouble: {
            const double* src = tensor.data<double>();
            double* dst = result.data<double>();
            for (size_t i = 0; i < tensor.numel(); ++i) dst[i] = scalar - src[i];
            break;
        }
        case DType::kInt: {
            const int32_t* src = tensor.data<int32_t>();
            int32_t* dst = result.data<int32_t>();
            for (size_t i = 0; i < tensor.numel(); ++i) dst[i] = static_cast<int32_t>(scalar) - src[i];
            break;
        }
        case DType::kLong: {
            const int64_t* src = tensor.data<int64_t>();
            int64_t* dst = result.data<int64_t>();
            for (size_t i = 0; i < tensor.numel(); ++i) dst[i] = static_cast<int64_t>(scalar) - src[i];
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for scalar - tensor");
    }

    // 自动微分记录（注意：此时 scalar 是常数，不参与反向传播）
    if (tensor.autograd_ctx) {
        // 这里可以记录为 sub_from_const 或手动处理
        // 由于 scalar 是常数，反向传播只影响 tensor 的梯度
        tensor.autograd_ctx->record_op(result, op::Sub, {const_cast<Tensor*>(&tensor)});
    }

    return result;
}

Tensor Tensor::sum(const std::vector<int>& dims, bool keepdim) const {
    // 处理全局求和（dims 为空）
    if (dims.empty()) {
        Tensor result(ShapeTag{}, {}, _dtype, _device);
        result._storage = Storage(1, _dtype, _device);
        result.autograd_ctx = this->autograd_ctx;  // 继承上下文
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
        if (autograd_ctx) {
            autograd_ctx->record_op(
                    result,
                    op::Sum,
                    {const_cast<Tensor*>(this)}
            );
        }
        return result;
    }

    // 处理指定维度求和
    // 1. 计算输出形状
    std::vector<size_t> result_shape = _shape;
    for (int dim : dims) {
        // 处理负索引
        int actual_dim = (dim < 0) ? dim + static_cast<int>(_shape.size()) : dim;
        if (actual_dim < 0 || actual_dim >= static_cast<int>(_shape.size())) {
            throw std::runtime_error("Invalid dimension in sum");
        }
        result_shape[actual_dim] = 1; // 求和后维度变为1
    }

    // 如果不保留维度，移除大小为1的维度
    if (!keepdim) {
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < result_shape.size(); ++i) {
            bool is_sum_dim = std::find(dims.begin(), dims.end(), static_cast<int>(i)) != dims.end();
            if (!is_sum_dim || keepdim) {
                new_shape.push_back(result_shape[i]);
            }
        }
        result_shape = new_shape;
    }

    // 2. 创建结果张量
    Tensor result(ShapeTag{}, result_shape, _dtype, _device);
    result.zero();

    // 3. 执行求和计算
    switch (_dtype) {
        case DType::kFloat: {
            const float* src = data<float>();
            float* dst = result.data<float>();

            // 遍历所有元素
            for (size_t i = 0; i < numel(); ++i) {
                // 计算在结果张量中的位置
                size_t result_index = 0;
                size_t stride = 1;
                size_t temp = i;

                for (int d = static_cast<int>(_shape.size()) - 1; d >= 0; --d) {
                    size_t dim_size = _shape[d];
                    size_t coord = temp % dim_size;
                    temp /= dim_size;

                    // 如果此维度需要求和，坐标设为0
                    if (std::find(dims.begin(), dims.end(), d) != dims.end()) {
                        coord = 0;
                    }

                    result_index += coord * stride;
                    stride *= result_shape[d];
                }

                dst[result_index] += src[i];
            }
            break;
        }
            break;
        case DType::kDouble: {
            const double * src = data<double>();
            double * dst = result.data<double>();

            // 遍历所有元素
            for (size_t i = 0; i < numel(); ++i) {
                // 计算在结果张量中的位置
                size_t result_index = 0;
                size_t stride = 1;
                size_t temp = i;

                for (int d = static_cast<int>(_shape.size()) - 1; d >= 0; --d) {
                    size_t dim_size = _shape[d];
                    size_t coord = temp % dim_size;
                    temp /= dim_size;

                    // 如果此维度需要求和，坐标设为0
                    if (std::find(dims.begin(), dims.end(), d) != dims.end()) {
                        coord = 0;
                    }

                    result_index += coord * stride;
                    stride *= result_shape[d];
                }

                dst[result_index] += src[i];
            }
            break;
        }
            break;
        case DType::kInt: {
            const int* src = data<int>();
            int* dst = result.data<int>();

            // 遍历所有元素
            for (size_t i = 0; i < numel(); ++i) {
                // 计算在结果张量中的位置
                size_t result_index = 0;
                size_t stride = 1;
                size_t temp = i;

                for (int d = static_cast<int>(_shape.size()) - 1; d >= 0; --d) {
                    size_t dim_size = _shape[d];
                    size_t coord = temp % dim_size;
                    temp /= dim_size;

                    // 如果此维度需要求和，坐标设为0
                    if (std::find(dims.begin(), dims.end(), d) != dims.end()) {
                        coord = 0;
                    }

                    result_index += coord * stride;
                    stride *= result_shape[d];
                }

                dst[result_index] += src[i];
            }
            break;
        }
            break;
        case DType::kLong: {
            const long* src = data<long>();
            long * dst = result.data<long>();

            // 遍历所有元素
            for (size_t i = 0; i < numel(); ++i) {
                // 计算在结果张量中的位置
                size_t result_index = 0;
                size_t stride = 1;
                size_t temp = i;

                for (int d = static_cast<int>(_shape.size()) - 1; d >= 0; --d) {
                    size_t dim_size = _shape[d];
                    size_t coord = temp % dim_size;
                    temp /= dim_size;

                    // 如果此维度需要求和，坐标设为0
                    if (std::find(dims.begin(), dims.end(), d) != dims.end()) {
                        coord = 0;
                    }

                    result_index += coord * stride;
                    stride *= result_shape[d];
                }

                dst[result_index] += src[i];
            }
            break;
        }
            break;
        default:
            throw std::runtime_error("Unsupported dtype for sum with dimensions");
    }

    // 记录自动微分操作
    if (autograd_ctx) {
        autograd_ctx->record_op(
                result,
                op::Sum,
                {const_cast<Tensor*>(this)}
        );
    }

    return result;
}

//TENSOR_CPPM
