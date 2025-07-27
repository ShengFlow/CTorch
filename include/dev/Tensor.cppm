/*
 * Tensor.cppm
 * Created by Beapoe & GhostFace on 2025.7
 * Main Classes: Storage & Tensor & Auto_diff
 * Version : v1.6 (fixed on 2025.7.25 10:18)
 * Log 1.3: 增加了注释及代码易读性
 * Log 1.4: 增加了AutoDiff自动微分类
 * Log 1.5:
增加了连续性检查，修复了变量命名，增加了对自动微分状态的输出，修复了移动时不移动自动微分状态的bug
*  Log 1.6:完成了矩阵乘法
 */

// ======================= 类型定义和枚举 =======================
module;
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <iomanip>
export module tensor;

// 广播变形数据结构体
export struct LogicData {
    std::vector<size_t>
        logicShape; // 由于广播之后shape和strides都发生了改变，所以广播之后的计算应该依赖于logicShape和logicStrides
    std::vector<size_t> logicStrides;
};

// 设备类型 - 定义张量存储的位置
export enum class DeviceType {
    kCPU,  //< 主存储器 (RAM)
    kCUDA, //< NVIDIA GPU (暂未实现)
    kMPS,  //< Apple Silicon (暂未实现)
};

// 数据类型 - 定义张量元素的类型
export enum class DType {
    kFloat,  //< 32位浮点数 (torch.float32)
    kDouble, //< 64位浮点数 (torch.float64)
    kInt,    //< 32位整数 (torch.int32)
    kLong,   //< 64位整数 (torch.int64)
    kBool,   //< 布尔值 (torch.bool)
};

// ======================= 辅助函数 =======================

// 将数据类型转换为字符串表示
export constexpr const char *dtypeToString(DType dtype);

// 获取数据类型的字节大小
export constexpr size_t dtypeSize(DType dtype);

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

export class Storage {
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
    // 在如下的checkDType函数中，std::is_same_v的用法为is_same_v<type,type>，返回true/false，用以判断两个类型是否相同
    // 此函数用来强制类型检查，避免不必要的内存问题
    template <typename T> void checkDType() const { // 此处T应该是基础数据类型
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

  public:
    // 构造函数：分配未初始化的内存
    Storage(size_t size, DType dtype, DeviceType device);
    // 如果初始化列表中_size为0，那么初始化为nullptr

    // 构造函数：从现有数据复制
    template <typename T>
    Storage(const T *data, size_t size, DType dtype, DeviceType device)
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

    Storage();

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
    size_t size() const;

    // 获取数据类型
    DType dtype() const;

    // 获取设备类型
    DeviceType device() const;

    // 创建存储的深拷贝
    Storage clone() const;

    // 检查存储是否为空
    bool empty() const;
};

// ======================= 张量类 (Tensor) =======================
export struct ShapeTag {}; // 此处结构体为了使编译器区分构造函数

export class Tensor {
  private:
    bool _requires_grad = false;  // 是否参与自动微分计算，默认不参与
    std::vector<size_t> _shape;   // 张量的维度大小
    std::vector<size_t> _strides; // 每个维度的步幅
    size_t _storage_offset;       // 存储中的起始偏移量
    DeviceType _device;           // 张量所在的设备
    DType _dtype;                 // 张量元素的数据类型
    Storage _storage;             // 存储张量数据的对象

    // ======================= 内部辅助函数 =======================

    // 计算步幅 (基于行优先顺序)
    void computeStrides();

    // 计算存储中的索引(原本步长)
    size_t computeStorageIndex(std::initializer_list<size_t> indices) const;

    // 计算存储中的索引(自定义步长)
    size_t computeStorageIndex(std::initializer_list<size_t> indices, std::vector<size_t> strides,
                               std::vector<size_t> shape) const;

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

  public:
    // ======================= 构造和析构 =======================

    // 默认构造函数：创建空张量
    Tensor();

    // 标量构造函数
    Tensor(float value);

    // 添加布尔张量构造函数
    Tensor(std::initializer_list<bool> values);

    // 构造函数：指定形状和数据类型（使用 ShapeTag 避免歧义）
    Tensor(ShapeTag, const std::vector<size_t> &shape, DType dtype, DeviceType device,
           bool zero_init);

    // 拷贝构造函数：创建深拷贝
    Tensor(const Tensor &other); // 深拷贝存储

    // 移动构造函数
    Tensor(Tensor &&other) noexcept;

    ~Tensor() = default;

    // ======================= 基本属性 =======================

    // 获取张量的维度数
    size_t dim() const;

    // 获取张量的形状
    const std::vector<size_t> &sizes() const;

    // 获取张量中元素的总数
    size_t numel() const;

    // 获取张量的数据类型
    DType dtype() const;

    // 设置张量的数据类型
    void setDtype(const DType dtype); 

    // 获取张量所在的设备
    DeviceType device() const;

    // 获取张量的形状
    std::vector<size_t> shape() const;

    // 获取张量步长
    std::vector<size_t> strides() const;

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
    Tensor clone() const;

    // 改变张量的形状 (不改变内存布局)
    Tensor view(const std::vector<size_t> &new_shape) const;

    // 转置最后两个维度
    Tensor transpose() const;

    // ======================= 运算符重载 =======================

    // 张量加法 (逐元素)
    Tensor operator+(const Tensor &rhs) const;

    // 张量赋值运算符（深拷贝）
    Tensor &operator=(const Tensor &other);

    // 张量移动赋值运算符
    Tensor &operator=(Tensor &&other) noexcept;

    // 张量相等比较
    bool operator==(const Tensor &other) const;

    // ======================= 输出和调试 =======================

    // 将张量转换为字符串表示
    std::string toString() const;

    // 打印张量信息
    void print() const;

    // 检查张量是否连续
    bool is_contiguous() const;

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
    void zero();

    // 将张量所有元素设置为1
    void ones();
    // 是否设置自动微分
    bool isAuto_diff();

    void grad();
};

// 广播变形
export LogicData broadCast(Tensor &a, Tensor &b);

// 矩阵乘法
Tensor matMul(Tensor &a, Tensor &b);

// ======================= 自动微分类 (Auto_Diff) ===================
export class AutoDiff {};
// TENSOR_CPPM