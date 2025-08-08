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

#include <algorithm>
#include <cstddef>
#include <initializer_list>
// #include <immintrin.h> 未支持ARM，等待在x86-64机器测试
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>  // 使用Apple的BLAS实现
#endif
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <functional>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <stack>
#include <optional>
#include <limits>
// #include<omp.h>   !!!目前不确定在哪些机器上需要这个头文件，如果编译错误，可以尝试加上

export module Tensor_dev;

#ifndef HOOK_RET
#define HOOK_RET std::optional<Tensor>
#endif

// ======================= 类型定义和枚举 =======================

// 设备类型 - 定义张量存储的位置
export enum class DeviceType {
    kCPU,    //< 主存储器 (RAM)
    kCUDA,   //< NVIDIA GPU (暂未实现)
    kMPS,    //< Apple Silicon (暂未实现)
 };

// 数据类型 - 定义张量元素的类型
export enum class DType {
    kFloat,  //< 32位浮点数 (torch.float32)
    kDouble, //< 64位浮点数 (torch.float64)
    kInt,    //< 32位整数 (torch.int32)
    kLong,   //< 64位整数 (torch.int64)
    kBool,   //< 布尔值 (torch.bool)
 };

// 自动微分类操作符枚举
export enum class op{

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
export struct BroadCastResult {
    std::vector<size_t> logicShape;    // 广播后的逻辑形状
    std::vector<size_t> logicStridesA; // 张量A的逻辑步幅
    std::vector<size_t> logicStridesB; // 张量B的逻辑步幅
};

// ======================= 辅助函数 =======================

// 将数据类型转换为字符串表示
export constexpr const char* dtypeToString(DType dtype);

// 获取数据类型的字节大小
export constexpr size_t dtypeSize(DType dtype);

// 将c++类型转换为dtype
template <typename T>
    constexpr DType cpp2DType() noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return DType::kFloat;
    } else if constexpr (std::is_same_v<T, double>) {
        return DType::kDouble;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DType::kInt;
    } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, long>) {
        return DType::kLong;
    } else if constexpr (std::is_same_v<T, bool>) {
        return DType::kBool;
    } else if constexpr (std::is_same_v<T, int>) {
        // 处理int类型，根据系统架构选择
        if constexpr (sizeof(int) == sizeof(int32_t)) {
            return DType::kInt;
        } else {
            return DType::kLong;
        }
    } else {
        // 不支持的类型（运行时错误）
        throw std::runtime_error("Unsupported type for DType conversion");
    }
}

export int minx(int a, int b);

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
* 11.序列化serialize(std::ofstream& os) const
* 12.逆序列化deserialize(std::ifstream& is)
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

export class AutoGrad;// 前置声明，避免循环引用

export class Storage {
private:
    size_t _size{};                     // 存储的元素数量，此处使用C++11的新特性花括号初始化，避免类型转换，实际上等同于size_t _size = 0;
    DType _dtype;                       // 数据类型 用于枚举
    DeviceType _device;                 // 设备类型 用于枚举
    std::shared_ptr<char[]> _data;      // 原始内存指针（使用shared_ptr实现共享所有权）避免出现手动delete的问题和delete数组和默认方法不匹配
    // 此处定义为char[]能够最大限度的节省内存并支持存储任意类型的数据
    // 使用shared_ptr能够共享对内存的所有权，使得同等的tensor可以共用一块内存，减少不必要的内存占用
    // 在需要深拷贝时，提供了一个clone函数，可以调用

    // 检查模板类型是否与存储类型匹配
    // 在如下的checkDType函数中，std::is_same_v的用法为is_same_v<type,type>，返回true/false，用以判断两个类型是否相同
    // 此函数用来强制类型检查，避免不必要的内存问题
    template <typename T>
    void checkDType() const {
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
    // ======================= 构造函数 =======================
    // 原始构造
    Storage();

    // 分配未初始化的内存,如果初始化列表中_size为0，那么初始化为nullptr
    Storage(size_t size, DType dtype, DeviceType device = DeviceType::kCPU);

    // 从现有数据复制,注意，此处实际上将从现有数据复制的操作委托给了第一个构造函数，然后进行memcpy的操作进行复制
    template <typename T>
    Storage(const T* data, size_t size, DType dtype, DeviceType device = DeviceType::kCPU): Storage(size, dtype, device) {
        if (size > 0 && _data.get()) {
            std::memcpy(_data.get(), data, size * dtypeSize(dtype));
        }
    }

    // 默认拷贝构造函数和拷贝赋值运算符（使用shared_ptr，所以是浅拷贝）
    // 此处同上，减少内存开销
    Storage(const Storage&) = default;
    Storage& operator=(const Storage&) = default;

    // 默认移动构造函数和移动赋值运算符
    Storage(Storage&&) = default;
    Storage& operator=(Storage&&) = default;

    // ======================= 析构函数 =======================
    ~Storage() = default;

    // ======================= 基本属性 =======================
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
    [[nodiscard]] size_t size() const;

    // 获取数据类型
    [[nodiscard]] DType dtype() const;

    // 获取设备类型
    [[nodiscard]] DeviceType device() const;

    // 创建存储的深拷贝
    [[nodiscard]] Storage clone() const;

    // 检查存储是否为空
    [[nodiscard]] bool empty() const;

    // ======================= IO函数 =======================
    // 序列化
    void serialize(std::ofstream& os) const;

    // 逆序列化
    void deserialize(std::ifstream& is);
};

// ======================= 张量类 (Tensor) =======================
/* class Tensor
*
* 成员变量：
* bool _requires_grad = false;   // 是否参与自动微分计算，默认不参与
* std::vector<size_t> _strides; // 每个维度的步幅
* size_t _storage_offset;       // 存储中的起始偏移量
* DeviceType _device;           // 张量所在的设备
* DType _dtype;                 // 张量元素的数据类型
* Storage _storage;             // 存储张量数据的对象
* AutoGrad* autograd_ctx = nullptr; // 自动微分上下文指针
* friend class AutoGrad;        // 允许自动微分类访问私有
成员函数：
内部辅助函数
* void computeStrides ()：计算步幅 (基于行优先顺序)
* size_t computeStorageIndex (std::initializer_list<size_t> indices) const：计算存储中的索引
* template <typename T> void checkDType () const：检查数据类型是否匹配
* template <typename T, typename Op> void elementwiseOp (Tensor& result, const Tensor& a, const Tensor& b, Op op) const：通用逐元素操作
* template <typename T, typename Op> void broadcast_elementwise_op (Tensor& result, const Tensor& a, const Tensor& b, const BroadCastResult& bc, Op op) const：支持广播的逐元素操作
* template <typename T> void printRecursive (std::ostream& os, size_t dim, std::vector<size_t> indices) const：递归打印张量内容（改进版）
* AutoGrad 算子
* Tensor cos () const：逐元素余弦
* Tensor sin () const：逐元素正弦
* Tensor relu () const：ReLU 激活函数
* Tensor sigmoid () const：Sigmoid 激活函数
* Tensor tanh () const：Tanh 激活函数
* Tensor softmax (int dim = -1) const：Softmax 激活函数
* 构造函数
* Tensor ()：默认构造函数，创建空张量
* Tensor (float value)：标量构造函数
* Tensor(std::initializer_list<float> values)：从初始值列表创建 1D 张量
* Tensor(std::initializer_list<bool> values)：布尔张量构造函数
* Tensor (ShapeTag, const std::vector<size_t>& shape, DType dtype = DType::kFloat, DeviceType device = DeviceType::kCPU, bool zero_init = true)：指定形状和数据类型的构造函数（使用 ShapeTag 避免歧义）
* template <typename T> Tensor(std::initializer_list<T> data, std::initializer_list<size_t> shape)：初始化构造
* Tensor (const Tensor& other)：拷贝构造函数，创建深拷贝
* Tensor (Tensor&& other) noexcept：移动构造函数
* 析构函数
* ~Tensor () = default：默认析构函数
* 基本属性
* const std::vector<size_t>& shape () const：获取张量形状
* const std::vector<size_t> strides () const：获取张量步长
* size_t dim () const：获取张量的维度数
* size_t numel () const：获取张量中元素的总数
* DType dtype () const：获取张量的数据类型
* DeviceType device () const：获取张量所在的设备
* template <typename T = float> T* data ()：获取原始数据的类型化指针
* template <typename T = float> const T* data () const：获取常量原始数据的类型化指针
* bool is_contiguous () const：检查张量是否连续
* bool isGradRequired () const：判断是否设置自动微分
* void requires_grad (bool key)：设置自动微分标志
* void set_autograd_ctx (AutoGrad* ctx)：设置自动微分上下文
* Tensor &grad () const：获取 Tensor 当前梯度
* void setDtype (const DType dtype)：设置 Tensor 的 DType
* bool hasGrad () const：判断是否有自动微分上下文
* size_t storageOffset () const：获取原始储存内存偏移
* 索引和访问
* template <typename T = float> T& operator [](size_t index)：1D 张量的索引访问
* template <typename T = float> const T& operator [](size_t index) const：1D 张量的常量索引访问
* template <typename T = float> T& operator ()(std::initializer_list<size_t> indices)：多维张量的索引访问
* template <typename T = float> const T& operator ()(std::initializer_list<size_t> indices) const：多维张量的常量索引访问
* template <typename T = float> T& item ()：标量访问（0 维张量）
* template <typename T = float> const T& item () const：常量标量访问
* 张量操作
* Tensor clone () const：创建张量的深拷贝
* Tensor view (const std::vector<size_t>& new_shape) const：改变张量的形状 (不改变内存布局)
* * Tensor sum(const std::vector<int>& dims, bool keepdim = false) const：降维（多维度）
* Tensor sum (int dim, bool keepdim = false) const：降维（单维度）
* Tensor sum () const：降维（所有维度）
* Tensor transpose () const：转置最后两个维度
* Tensor with_ctx (AutoGrad *ctx) const：设置上下文并传播到新张量
* static Tensor create_with_ctx (AutoGrad* ctx, const std::vector<size_t>& shape, DType dtype, DeviceType device)：创建新张量时继承上下文
* void fill (float value)：用指定值填充整个张量
* void zero ()：将张量所有元素设置为 0
* void ones ()：将张量所有元素设置为 1
* bool empty () const：判断张量是否为空
* void zeroGrad () const：清空梯度
* void clearCtx ()：清空自动微分上下文
* IO 函数
* void serialize (std::ofstream &os) const：序列化
* void deserialize (std::ifstream & is)：逆序列化
* std::string toString () const：将张量转换为字符串表示
* void print () const：打印张量信息
* AutoGrad 操作
* void recordOp (Tensor &result, op operation, std::initializer_list<Tensor *> inputs)：记录当前运算
* void setRetainGraph (bool retain) const：设置保留计算图标志
* Hook 相关
* void registerHook (Hook _fn)：注册钩子
* void removeHook (size_t idx)：移除钩子
* void removeAllHooks ()：移除所有钩子
* std::vector<Hook> hooks () const：获取所有钩子
* Hook hook (size_t idx) const：获取指定索引处钩子
* 成员变量：
* bool _requires_grad = false;：是否参与自动微分计算，默认不参与
* std::vector<size_t> _strides;：每个维度的步幅
* size_t _storage_offset;：存储中的起始偏移量
* DeviceType _device;：张量所在的设备
* DType _dtype;：张量元素的数据类型
* Storage _storage;：存储张量数据的对象
* AutoGrad* autograd_ctx = nullptr;：自动微分上下文指针
* using Hook = HOOK_RET (*)(Tensor& self);：钩子函数类型定义
* std::vector<Hook> _hooks;：钩子函数列表
* std::vector<size_t> _shape;：张量形状（protected）
* friend class AutoGrad;：允许自动微分类访问私有成员
* 运算符重载：
* Tensor transpose_last_two () const：转置最后两个维度
* Tensor operator>(float scalar) const：大于运算符，用于 ag 类
* Tensor operator+(const Tensor& rhs) const：张量加法 (逐元素)
* Tensor operator-(const Tensor& rhs) const：逐元素减法
* Tensor operator*(const Tensor& rhs) const：张量乘法
* Tensor operator/(const Tensor& rhs) const：逐元素除法
* Tensor operator-() const：负号
* Tensor operator*(double scalar) const：double 型张量 - 标量乘法
* Tensor operator*(float scalar) const：float 型张量 - 标量乘法
* Tensor operator*(int scalar) const：int 型张量 - 标量乘法
* Tensor operator*(long scalar) const：long 型张量 - 标量乘法
* Tensor operator/(double scalar) const：double 型张量 - 标量除法
* Tensor operator/(float scalar) const：float 型张量 - 标量除法
* Tensor operator/(int scalar) const：int 型张量 - 标量除法
* Tensor operator/(long scalar) const：long 型张量 - 标量除法
* Tensor& operator=(const Tensor& other)：张量赋值运算符（深拷贝）
* Tensor& operator=(Tensor&& other) noexcept：张量移动赋值运算符
* bool operator==(const Tensor& other) const：张量相等比较
*/
export struct ShapeTag {}; // 此处结构体为了使编译器区分构造函数

export class Tensor {
private:
    bool _requires_grad = false;   // 是否参与自动微分计算，默认不参与
    std::vector<size_t> _strides; // 每个维度的步幅
    size_t _storage_offset;       // 存储中的起始偏移量
    DeviceType _device;           // 张量所在的设备
    DType _dtype;                 // 张量元素的数据类型
    Storage _storage;             // 存储张量数据的对象
    // AutoGrad* autograd_ctx = nullptr; // 自动微分上下文指针
    friend class AutoGrad;        // 允许自动微分类访问私有

    using Hook = HOOK_RET(*)(Tensor& self);
    std::vector<Hook> _hooks; // 钩子

    // ======================= 内部辅助函数 =======================

    // 计算步幅 (基于行优先顺序)
    void computeStrides();

    // 计算存储中的索引
    [[nodiscard]] size_t computeStorageIndex(std::initializer_list<size_t> indices) const;

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

        constexpr size_t max_display = 3; // 每维度最大显示元素数
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

    // ======================= AutoGrad算子 =======================
    // 逐元素余弦
    [[nodiscard]] Tensor cos() const;

    // 逐元素正弦
    [[nodiscard]] Tensor sin() const;

    // ReLU激活函数
    [[nodiscard]] Tensor relu() const;

    // Sigmoid激活函数
    [[nodiscard]] Tensor sigmoid() const;

    // Tanh激活函数
    [[nodiscard]] Tensor tanh() const;

    // Softmax激活函数
    [[nodiscard]] Tensor softmax(int dim = -1) const;

protected:
    std::vector<size_t> _shape;   // 张量形状

public:
    // ======================= 构造函数 =======================
    // 默认构造函数：创建空张量
    Tensor();

    // 标量构造函数
    explicit Tensor(float value);

    // 构造函数：从初始值列表创建1D张量
    Tensor(std::initializer_list<float> values);

    // 添加布尔张量构造函数
    Tensor(std::initializer_list<bool> values);

    // 构造函数：指定形状和数据类型（使用 ShapeTag 避免歧义）
    Tensor(ShapeTag, const std::vector<size_t>& shape, DType dtype = DType::kFloat, DeviceType device = DeviceType::kCPU, bool zero_init = true);

    // 初始化构造
    template <typename T>
    Tensor(std::initializer_list<T> data, std::initializer_list<size_t> shape):
    _storage(Storage(data,data.size(),cpp2DType<T>(),DeviceType::kCPU)),
    _shape(std::move(shape)),
    _storage_offset(0),
    _device(DeviceType::kCPU),
    _dtype(cpp2DType<T>()){computeStrides();}

    // 拷贝构造函数：创建深拷贝
    Tensor(const Tensor& other);

    // 移动构造函数
    Tensor(Tensor&& other) noexcept;

    // ======================= 析构函数 =======================
    ~Tensor() = default;

    // ======================= 基本属性 =======================

    // 张量形状
    [[nodiscard]] const std::vector<size_t>& shape() const;

    // 张量步长
    [[nodiscard]] const std::vector<size_t> strides() const;

    // 设置形状
    void setShape(const std::vector<size_t>& shape);

    // 设置步长
    void setStrides(const std::vector<size_t>& strides);

    // 获取张量的维度数
    [[nodiscard]] size_t dim() const;

    // 获取张量中元素的总数
    [[nodiscard]] size_t numel() const;

    // 获取张量的数据类型
    [[nodiscard]] DType dtype() const;

    // 获取张量所在的设备
    [[nodiscard]] DeviceType device() const;

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

    // 检查张量是否连续
    [[nodiscard]] bool is_contiguous() const;

    // 是否设置自动微分
    [[nodiscard]] bool isGradRequired() const;

    // 设置自动微分标志
    void requires_grad(bool key);

    // // 设置自动微分上下文
    // void set_autograd_ctx(AutoGrad* ctx);

    // 获取Tensor当前梯度
    [[nodiscard]] Tensor grad() const;

    // 设置Tensor的DType
    void setDtype(DType dtype);

    // // 是否有自动微分上下文
    // [[nodiscard]] bool hasGrad() const;

    // 获取原始储存内存偏移
    [[nodiscard]] size_t storageOffset() const;

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

    // ======================= 张量操作 =======================
    // 创建张量的深拷贝
    [[nodiscard]] Tensor clone() const;

    // 改变张量的形状 (不改变内存布局)
    [[nodiscard]] Tensor view(const std::vector<size_t>& new_shape) const;

    // 降维
    [[nodiscard]] Tensor sum(const std::vector<int>& dims, bool keepdim = false) const;

    // 降维
    [[nodiscard]] Tensor sum(int dim, bool keepdim = false) const;

    // 降维
    [[nodiscard]] Tensor sum() const;

    // 转置最后两个维度
    [[nodiscard]] Tensor transpose() const;

    // // 设置上下文并传播到新张量
    // Tensor with_ctx(AutoGrad *ctx) const;
    //
    // // 创建新张量时继承上下文
    // static Tensor create_with_ctx(AutoGrad* ctx, const std::vector<size_t>& shape,
    //                               DType dtype, DeviceType device);

    // 用指定值填充整个张量
    void fill(float value);

    // 将张量所有元素设置为0
    void zero();

    // 将张量所有元素设置为1
    void ones();

    // 张量是否为空
    [[nodiscard]] bool empty() const;

    // 清空梯度
    void zeroGrad() const;

    // ======================= IO函数 =======================
    // 序列化
    void serialize(std::ofstream &os) const;
    // 逆序列化
    void deserialize(std::ifstream & is);

    // 打印张量信息
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    // 将张量转换为字符串表示
    [[nodiscard]] std::string toString() const;

    // ======================= 运算符重载 =======================
    // 转置最后两个维度
    [[nodiscard]] Tensor transpose_last_two() const;

    // 大于运算符，用于ag类
    Tensor operator>(float scalar) const;

    // 张量加法 (逐元素)
    Tensor operator+(const Tensor& rhs) const;

    // 逐元素减法
    Tensor operator-(const Tensor& rhs) const;

    // 张量乘法
    Tensor operator*(const Tensor& rhs) const;

    // 逐元素除法
    Tensor operator/(const Tensor& rhs) const;
    // 负号
    Tensor operator-() const;

    // float型张量-标量减法
    Tensor operator-(float scalar) const;

    // float型标量-张量减法
    friend  Tensor operator-(float scalar, const Tensor& tensor);
    //
    // // float - Tensor
    // Tensor operator-(float scalar, const Tensor& tensor);

    // double型张量-标量乘法
    Tensor operator*(double scalar) const;

    // float型张量-标量乘法
    Tensor operator*(float scalar) const;

    // int型张量-标量乘法
    Tensor operator*(int scalar) const;

    // longe型张量-标量乘法
    Tensor operator*(long scalar) const;

    // double型张量-标量除法
    Tensor operator/(double scalar) const;

    // float型张量-标量除法
    Tensor operator/(float scalar) const;

    // int型张量-标量除法
    Tensor operator/(int scalar) const;

    // longe型张量-标量除法
    Tensor operator/(long scalar) const;

    // 张量赋值运算符（深拷贝）
    Tensor& operator=(const Tensor& other);

    // 张量移动赋值运算符
    Tensor& operator=(Tensor&& other) noexcept;

    // 张量相等比较
    bool operator==(const Tensor& other) const;

    // ======================= AutoGrad操作 =======================
    // // 记录当前运算
    // void recordOp(Tensor &result, op operation, std::initializer_list<Tensor *> inputs);
    //
    // // 设置保留计算图标志
    // void setRetainGraph(bool retain) const;

    // 反向传播
    void backward(Tensor& root,Tensor grad_output = Tensor());

    // ======================= Hook =======================
    // 注册钩子
    void registerHook(Hook _fn);

    // 移除钩子
    void removeHook(size_t idx);

    // 移除所有钩子
    void removeAllHooks();

    // 获取所有钩子
    [[nodiscard]] std::vector<Hook> hooks() const;

    // 获取指定索引处钩子
    [[nodiscard]] Hook hook(size_t idx) const;
};

// ======================= 矩阵乘(MatMul) =======================
// Tensor matMul(const Tensor &a, const Tensor &b);        // 矩阵乘前置声明（8.3 upt:这里为老版的声明，暂时保留）

export Tensor matMul(const Tensor &a, const Tensor &b);  // 矩阵乘新版声明

export Tensor matMulNative(const Tensor &a,const Tensor &b); // 循环优化矩阵乘

export Tensor matMulBlocked(const Tensor &a,const Tensor &b); // 分块算法矩阵乘

export Tensor matMulAMX(const Tensor &a, const Tensor &b); // AMX优化矩阵乘

// 定义矩阵大小的阈值
constexpr size_t SMALL_SIZE = 64;

constexpr size_t MEDIUM_SIZE = 512;

export Tensor matMulRecursive(const Tensor &a, const Tensor &b); // 递归的Strassen矩阵乘法

export Tensor matMulTest(const Tensor &a, const Tensor &b); // 矩阵乘主函数

// ======================= 自动微分上下文类 (AutoGradContext) =======================
/*
 * class AutoGradContext
 * 成员函数：
 * 1.current() 获取自动微分公共上下文单例
 *
 * 内部类：
 * Guard：
 *      成员函数：
 *      1.Guard(AutoGrad* ctx) 哨兵构造函数
 *      2.~Guard() 哨兵析构函数
 *      成员变量：
 *      AutoGrad* ctx 公共上下文指针
 */
export class AutoGradContext {
public:
    static AutoGrad*& current();

    class Guard {
    public:
        explicit Guard(AutoGrad* ctx);
        ~Guard();

    private:
        AutoGrad* prev_ctx;
    };
};

// ======================= 自动微分类 (AutoGrad) =======================
/* class AutoGrad
*
内部结构体：
struct Node：计算图节点定义
Tensor tensor：存储的 Tensor 值
Tensor input_grad：传出梯度值
Tensor output_grad：传入梯度值
std::vector<Node*> inputs：输入节点指针
op operation：操作类型
bool requires_grad：是否需要梯度
bool is_leaf：是否为叶子节点
size_t retain_count = 0：保留计数（用于高阶导数）
构造函数：Node (Tensor t, bool req_grad, bool leaf = true)

std::unordered_map<Tensor*, Node*> tensor_to_node;  // Tensor到节点的映射
std::vector<std::unique_ptr<Node>> nodes;           // 节点存储
bool retain_graph = false;                          // 是否保留计算图
成员函数：
反向传播算子
void backward_add (Node* node)：加法反向传播（已支持广播）
void backward_sub (Node* node)：减法反向传播（已支持广播）
void backward_mul (Node* node)：乘法反向传播
void backward_div (Node* node)：除法反向传播
void backward_matmul (Node* node)：矩阵乘法反向传播
static void backward_dot (Node* node)：点积反向传播
static void backward_cos (Node* node)：余弦函数反向传播
static void backward_sin (Node* node)：正弦函数反向传播
static void backward_relu (Node* node)：ReLU 反向传播
static void backward_sigmoid (Node* node)：Sigmoid 反向传播
static void backward_tanh (Node* node)：Tanh 反向传播
static void backward_softmax (Node* node)：Softmax 反向传播
static void backward_sum (Node* node)：降维操作反向传播
辅助函数
static Tensor reduce_to_match (Tensor grad, const std::vector<size_t>& target_shape)：将梯度减少到目标形状（处理广播）
基本属性
const Tensor inputGrad ()：获取输出梯度
const Tensor &outputGrad ()：获取输入梯度
void set_retain_graph (bool retain)：设置是否保留计算图
Node* get_node (Tensor *t)：获取节点
const Node* rootPtr ()：获取计算图起始节点
const Node* topPtr ()：获取计算图终节点
操作
void make_leaf (Tensor& t, bool requires_grad)：创建叶子节点
void record_op (Tensor& result, op operation, std::initializer_list<Tensor*> inputs)：记录操作
void backward (Tensor& root, Tensor grad_output = Tensor ())：反向传播
static void zero_grad (Node* root)：清空梯度
成员变量：
std::unordered_map<Tensor*, Node*> tensor_to_node：Tensor 到节点的映射
std::vector<std::unique_ptr<Node>> nodes：节点存储
bool retain_graph = false：是否保留计算图
*/
class AutoGrad {
private:
    // 计算图节点定义
    struct Node {
        Tensor tensor;                   // 存储的Tensor值
        Tensor grad;                     // 传出梯度值
        std::vector<Tensor> output_grads;// 传入梯度值
        std::vector<Node*> inputs;       // 输入节点指针
        op operation;                    // 操作类型
        bool requires_grad;              // 是否需要梯度
        bool is_leaf;                    // 是否为叶子节点
        size_t retain_count = 0;         // 保留计数（用于高阶导数）

        Node(Tensor t, bool req_grad, bool leaf = true);
    };

    std::unordered_map<Tensor*, Node*> tensor_to_node;  // Tensor到节点的映射
    std::vector<std::unique_ptr<Node>> nodes;           // 节点存储
    bool retain_graph = false;                          // 是否保留计算图

    // ======================= 反向传播算子 =======================
    // 加法反向传播（已支持广播）
    void backward_add(Node* node);

    // 减法反向传播（已支持广播）
    void backward_sub(Node* node);

    // 乘法反向传播
    void backward_mul(Node* node);

    // 除法反向传播
    void backward_div(Node* node);

    // 矩阵乘法反向传播
    void backward_matmul(Node* node);

    // 点积反向传播
    static void backward_dot(Node* node);

    // 余弦函数反向传播
    static void backward_cos(Node* node);

    // 正弦函数反向传播
    static void backward_sin(Node* node);

    // ReLU反向传播
    static void backward_relu(Node* node);

    // Sigmoid反向传播
    static void backward_sigmoid(Node* node);

    // Tanh反向传播
    static void backward_tanh(Node* node);

    // Softmax反向传播
    static void backward_softmax(Node* node);

    // 降维操作反向传播
    static void backward_sum(Node* node);

    // ======================= 辅助函数 =======================
    // 将梯度减少到目标形状（处理广播）
    static Tensor reduce_to_match(Tensor grad, const std::vector<size_t>& target_shape);


public:
    // ======================= 基本属性 =======================
    // // 获取输出梯度
    // const Tensor inputGrad();
    //
    // // 获取输入梯度
    // const Tensor &outputGrad();

    // 获取梯度
    Tensor getGrad(Tensor* t);

    // 清空梯度
    void zeroGrad(Tensor* t);

    // 设置是否保留计算图
    void set_retain_graph(bool retain);

    // 获取节点
    Node* get_node(Tensor *t);

    // 获取计算图起始节点
    const Node* rootPtr();

    // 获取计算图终节点
    const Node* topPtr();

    // ======================= 操作 =======================
    // 创建叶子节点
    void make_leaf(Tensor& t, bool requires_grad);

    // 记录操作
    void record_op(const std::vector<Tensor*>& outputs, op operation,
                  const std::vector<Tensor*>& inputs);

    // 反向传播
    void backward(Tensor& root, Tensor grad_output = Tensor());

    // // 清空梯度
    // static void zero_grad(Node* root);

    // 清空计算图
    void clearGraph();
};

export BroadCastResult broadCast(const Tensor& a, const Tensor& b); // 广播函数