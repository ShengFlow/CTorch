#ifndef TENSOR_H
#define TENSOR_H
/*
* Tensor.h
* Created by Beapoe & GhostFace on 2025.7
* Main Classes: Storage & Tensor & Auto_diff
* Version : v2.1 (fixed on 2025.9.27)
* Log : Fixed AD
*/

// includes
#include <algorithm>
#include <cstddef>
#include "Ctorch_Error.h"
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>  // 使用Apple的BLAS实现
#endif

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
#include <sstream>
#include <limits>
#include <map>
//#include<omp.h>   !!!目前不确定在哪些机器上需要这个头文件，如果编译错误，可以尝试加上
// ======================= 类型定义和枚举 =======================
// ======================= 前向声明 =======================
class AutoDiff;
class Tensor;

// ==================== 统一矩阵乘法接口 ====================

// 矩阵乘法算法选择策略
enum class MatMulStrategy {
    AUTO,           // 自动选择
    NAIVE,          // 朴素算法
    BLOCKED,        // 分块优化
    STRASSEN,       // Strassen递归算法
    OPTIMIZED       // 最优算法组合
};

// 矩阵乘法性能配置
struct MatMulConfig {
    // 分块大小阈值
    static constexpr size_t BLOCK_SIZE_THRESHOLD = 64;

    // Strassen算法阈值
    static constexpr size_t STRASSEN_THRESHOLD = 128;

    // 小矩阵阈值（使用朴素算法）
    static constexpr size_t SMALL_MATRIX_THRESHOLD = 32;

    // 是否启用性能分析
    static constexpr bool ENABLE_PROFILING = true;

    // 是否启用缓存优化
    static constexpr bool ENABLE_CACHE_OPTIMIZATION = true;
};
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
    Neg,        // 负号 - 新增
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
       default: Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::DATATYPE,"未知数据类型！");
   }
}

// 辅助函数
inline int minx(int a, int b){
   int diff = b - a;
   return a + (diff & (diff >> 31));
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
           Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::DATATYPE,"数据类型不匹配！");
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
        if (_size == 0 || !_data) {
            return nullptr;
        }
        checkDType<T>();
        T* result = reinterpret_cast<T*>(_data.get());
        return result;
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
    // 在 Storage 类中修复 clone 方法
    Storage clone() const {
        Storage new_storage(_size, _dtype, _device);
        if (_size > 0 && _data) {
            std::memcpy(new_storage._data.get(), _data.get(), _size * dtypeSize(_dtype));

            // 验证复制是否正确
            if (_dtype == DType::kFloat && _size > 0) {
                const float* src = reinterpret_cast<const float*>(_data.get());
                float* dst = reinterpret_cast<float*>(new_storage._data.get());
            }
        }
        return new_storage;
    }
    // 添加清空方法
    // 修复 clear 方法
    void clear() {
       _data.reset();
       _size = 0;
   }

    // 确保 empty 方法正确
    bool empty() const {
       return _size == 0 || !_data;
   }
};

// ======================= 张量类 (Tensor) =======================
struct ShapeTag {}; // 此处结构体为了使编译器区分构造函数

class AutoDiffContext {
public:
    // 修复：返回可修改的引用而不是右值
    static AutoDiff*& current() {
        static thread_local AutoDiff* ctx = nullptr;
        return ctx;
    }

    class Guard {
    public:
        explicit Guard(AutoDiff* new_ctx) : prev_ctx(current()) {
            current() = new_ctx;  // 现在可以正确赋值
        }

        ~Guard() {
            current() = prev_ctx;
        }

        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;

    private:
        AutoDiff* prev_ctx;
    };
};
Tensor matMul(const Tensor &a, const Tensor &b);
class Tensor {
private:
    static std::atomic<size_t> global_tensor_id;
    size_t tensor_id_;
    bool record_committed_ = false;
   bool _requires_grad = false;   // 是否参与自动微分计算，默认不参与
   // 张量的维度大小
   std::vector<size_t> _strides; // 每个维度的步幅
   size_t _storage_offset;       // 存储中的起始偏移量
   DeviceType _device;           // 张量所在的设备
   DType _dtype;                 // 张量元素的数据类型
   Storage _storage;             // 存储张量数据的对象
   // ======================= 内部辅助函数 =======================

   // 计算步幅 (基于行优先顺序)
   void computeStrides();

    // 计算存储中的索引
   size_t computeStorageIndex(std::initializer_list<size_t> indices) const;

    // 检查数据类型是否匹配
   template <typename T>
   void checkDType() const;

    // 通用逐元素操作
   template <typename T, typename Op>
   void elementwiseOp(Tensor& result, const Tensor& a, const Tensor& b, Op op) const;

    // 支持广播的逐元素操作
    // 在 broadcast_elementwise_op 中也添加调试
    template<typename T, typename Op>
    void broadcast_elementwise_op(Tensor& result, const Tensor& a, const Tensor& b,
                                     const BroadCastResult& bc, Op op) const;

    // 递归打印张量内容（改进版）
   template <typename T>
   void printRecursive(std::ostream& os, size_t dim, std::vector<size_t> indices) const;

protected:
   std::vector<size_t> _shape;
public:
    // 添加清空存储的方法，避免创建新Tensor
    void clear_storage();

    // 添加判断是否为空的辅助方法
    bool is_cleared() const;

    // 增强调试信息
    void debug_info_detailed(const std::string& name = "") const;

    // 提交未完成的记录

    void commit_pending_record();
   // ======================= 构造和析构 =======================

// 修复所有构造函数
    // 添加构造和析构的调试
    Tensor();


    Tensor(float value) : tensor_id_(global_tensor_id++), _shape({}),
                          _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kFloat) {
        std::cout << ">>> Tensor标量构造, ID: " << tensor_id_ << ", 值: " << value << std::endl;
        computeStrides();
        _storage = Storage(1, _dtype, _device);
        if (_storage.data<float>()) {
            *_storage.data<float>() = value;
            std::ostringstream oss;
            oss << ">>> 标量Tensor设置完成, 存储值: " << *_storage.data<float>();
            std::string msg = oss.str();
            Ctorch_Error::info(ErrorPlatform::kCPU,msg);
        } else {
            Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kCPU,ErrorType::MEMORY,"!!! 错误: 无法分配存储");
        }
    }

    Tensor(std::initializer_list<float> values)
        : tensor_id_(global_tensor_id++), _shape({values.size()}),
          _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kFloat) {
        computeStrides();
        _storage = Storage(values.begin(), values.size(), _dtype, _device);
    }

    Tensor(ShapeTag tag, const std::vector<size_t>& shape, DType dtype = DType::kFloat,
           DeviceType device = DeviceType::kCPU, bool zero_init = true)
        : tensor_id_(global_tensor_id++), _shape(shape), _storage_offset(0),
          _device(device), _dtype(dtype) {
        computeStrides();
        _storage = Storage(numel(), _dtype, _device);
        if(zero_init) zero();
    }

    // 修复拷贝构造函数 - 新对象需要新ID
    // 在 Tensor 类中添加拷贝构造和赋值的调试
    Tensor(const Tensor& other)
        : tensor_id_(global_tensor_id++),
          _shape(other._shape), _strides(other._strides),
          _storage_offset(other._storage_offset),
          _device(other._device), _dtype(other._dtype),
          _storage(other._storage.clone()),  // 注意：这里调用了clone()
          _requires_grad(other._requires_grad),
          record_committed_(false) {
        // std::cout << ">>> Tensor拷贝构造, 新ID: " << tensor_id_ << ", 原ID: " << other.tensor_id_ << std::endl;
        std::ostringstream oss;
        oss << ">>> Tensor拷贝构造, 新ID: " << tensor_id_ << ", 原ID: " << other.tensor_id_;
        std::string msg = oss.str();
        Ctorch_Error::info(ErrorPlatform::kCPU,msg);
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            commit_pending_record();

            tensor_id_ = global_tensor_id++;
            _shape = other._shape;
            _strides = other._strides;
            _storage_offset = other._storage_offset;
            _device = other._device;
            _dtype = other._dtype;
            _storage = other._storage.clone();  // 注意：这里调用了clone()
            _requires_grad = other._requires_grad;
            record_committed_ = false;

            std::ostringstream oss;
            oss << ">>> Tensor拷贝赋值, 新ID: " << tensor_id_ << ", 原ID: " << other.tensor_id_;
            std::string msg = oss.str();
            Ctorch_Error::info(ErrorPlatform::kCPU,msg);
        }
        return *this;
    }

    // 修复移动构造函数 - 保持原ID
    Tensor(Tensor&& other) noexcept
        : tensor_id_(other.tensor_id_),
          _shape(std::move(other._shape)),
          _strides(std::move(other._strides)),
          _storage_offset(other._storage_offset),
          _device(other._device), _dtype(other._dtype),
          _storage(std::move(other._storage)),
          _requires_grad(other._requires_grad),
          record_committed_(other.record_committed_) {
        // 使原对象无效
        other.tensor_id_ = 0;
        other.record_committed_ = true;
        other._storage_offset = 0;
        other._shape.clear();
        other._strides.clear();
    }

    // 修复析构函数 - 提交未完成的记录
    // 在 Tensor 析构函数中避免无限递归
    ~Tensor() {
        // 只在有效ID时提交记录
        if (tensor_id_ != 0 && !record_committed_) {
            std::ostringstream oss;
            oss << ">>> Tensor析构, ID: " << tensor_id_;
            std::string msg = oss.str();
            Ctorch_Error::info(ErrorPlatform::kCPU,msg);
            commit_pending_record();
        }
    }

    // 修复移动赋值运算符
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            commit_pending_record();

            tensor_id_ = other.tensor_id_;
            _shape = std::move(other._shape);
            _strides = std::move(other._strides);
            _storage_offset = other._storage_offset;
            _device = other._device;
            _dtype = other._dtype;
            _storage = std::move(other._storage);
            _requires_grad = other._requires_grad;
            record_committed_ = other.record_committed_;

            other.tensor_id_ = 0;
            other.record_committed_ = true;
            other._storage_offset = 0;
            other._shape.clear();
            other._strides.clear();
        }
        return *this;
    }

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
        if (_storage.empty()) {
            std::ostringstream oss;
            oss << "!!! Tensor::item() - 存储为空" ;
            std::string msg = oss.str();
            Ctorch_Error::log(ErrorLevel::WARN,ErrorPlatform::kCPU,ErrorType::MEMORY,msg);
            static T default_value;
            return default_value;
        }
        T* data_ptr = _storage.data<T>();
        if (!data_ptr) {
            std::ostringstream oss;
            oss << "!!! Tensor::item() - 数据指针为空" ;
            std::string msg = oss.str();
            Ctorch_Error::log(ErrorLevel::WARN,ErrorPlatform::kCPU,ErrorType::MEMORY,msg);
            static T default_value;
            return default_value;
        }
        return *data_ptr;
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
    // 获取唯一ID
    size_t id() const { return tensor_id_; }

    // 修复requires_grad设置器
    void requires_grad(bool key) ;

    // 梯度需求传播辅助函数
    static bool should_require_grad(const Tensor& a, const Tensor& b) {
       return a.requires_grad() || b.requires_grad();
   }

    static bool should_require_grad(const std::vector<Tensor*>& inputs) {
       for (const Tensor* t : inputs) {
           if (t && t->requires_grad()) return true;
       }
       return false;
   }

    // 调试信息
    void debug_info(const std::string& name = "") const {
        std::ostringstream oss;
        oss << (name.empty() ? "Tensor" : name)
                << " ID: " << tensor_id_
                << ", requires_grad: " << (_requires_grad ? "true" : "false")
                << ", shape: [";
        for (size_t i = 0; i < _shape.size(); ++i) {
            oss << _shape[i];
            if (i < _shape.size() - 1) oss << ", ";
        }
        oss << "], record_committed: " << (record_committed_ ? "true" : "false");

        std::string msg = oss.str();
        Ctorch_Error::info(ErrorPlatform::kCPU, msg);
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
   bool empty() const {
       return numel() == 0;
   }

   Tensor grad() const;

   void setDtype(const DType dtype);

   bool hasGrad() const;
   // 添加 const 版本的 requires_grad 访问器
   bool requires_grad() const { return _requires_grad; }

   // ==================== 统一矩阵乘法接口 ====================

   // 统一矩阵乘法 - 自动选择算法并支持自动微分
   Tensor matmul_unified(const Tensor& other) const;

    // 矩阵乘法操作符重载 - 使用统一接口
   // 注意：这里不重载operator*，因为已经存在了，使用matmul_unified方法
    // 添加一个辅助函数来创建真正的空 Tensor
   // 保留设置方法
};

template<typename T>
void Tensor::checkDType() const {
    if ((std::is_same_v<T, float> && _dtype != DType::kFloat) ||
        (std::is_same_v<T, double> && _dtype != DType::kDouble) ||
        (std::is_same_v<T, int32_t> && _dtype != DType::kInt) ||
        (std::is_same_v<T, int64_t> && _dtype != DType::kLong) ||
        (std::is_same_v<T, bool> && _dtype != DType::kBool)) {
        throw std::runtime_error("Tensor data type mismatch");
    }
}

template<typename T, typename Op>
void Tensor::elementwiseOp(Tensor &result, const Tensor &a, const Tensor &b, Op op) const {
    const size_t n = a.numel();
    T* out = result.data<T>();
    const T* a_data = a.data<T>();
    const T* b_data = b.data<T>();

    for (size_t i = 0; i < n; ++i) {
        out[i] = op(a_data[i], b_data[i]);
    }
}

template<typename T, typename Op>
void Tensor::broadcast_elementwise_op(Tensor &result, const Tensor &a, const Tensor &b, const BroadCastResult &bc,
    Op op) const {
    Ctorch_Error::info(ErrorPlatform::kCPU,">>> 进入 broadcast_elementwise_op");
    const std::vector<size_t>& shape = bc.logicShape;
    const std::vector<size_t>& stridesA = bc.logicStridesA;
    const std::vector<size_t>& stridesB = bc.logicStridesB;

    T* out = result.data<T>();
    const T* a_data = a.data<T>();
    const T* b_data = b.data<T>();

    size_t total_elements = 1;
    for (auto dim : shape) total_elements *= dim;

    std::ostringstream oss;
    oss << ">>> 总元素数:  " << total_elements;
    std::string msg = oss.str();
    Ctorch_Error::info(ErrorPlatform::kCPU,msg);

    // 遍历广播后的每个元素
    for (size_t flat_idx = 0; flat_idx < total_elements; ++flat_idx) {
        if (flat_idx % 1000000 == 0) { // 每100万次输出一次进度
            std::ostringstream osss;
            osss << ">>> 处理元素: " << flat_idx << "/" << total_elements;
            std::string msgs = osss.str();
            Ctorch_Error::info(ErrorPlatform::kCPU,msgs);
        }

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

    Ctorch_Error::info(ErrorPlatform::kCPU,"<<< 离开 broadcast_elementwise_op");

}

template<typename T>
void Tensor::printRecursive(std::ostream &os, size_t dim, std::vector<size_t> indices) const {
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

std::atomic<size_t> Tensor::global_tensor_id{1};

Tensor matMul_(Tensor &a, Tensor &b);        // 矩阵乘前置声明（8.3 upt:这里为老版的声明，暂时保留）

Tensor create_empty_tensor() {
    return Tensor(ShapeTag{}, std::vector<size_t>{}, DType::kFloat, DeviceType::kCPU);
}
// 在 AutoDiff 类声明之前添加

// 在 Tensor 类中注释掉有问题的函数，或者移到实现文件中
// ======================= 自动微分类 (AutoDiff) =======================
class AutoDiff {
private:
    // 计算图节点定义 - 修复后的版本
    struct Node {
        size_t tensor_id;
        Tensor tensor;
        Tensor grad;
        std::vector<size_t> input_ids;
        op operation;
        bool requires_grad;
        bool is_leaf;

        Node(size_t id, Tensor t, bool req_grad, bool leaf = true)
            : tensor_id(id), tensor(std::move(t)), requires_grad(req_grad), is_leaf(leaf) {
            operation = op::Add; // 默认值
        }

        // 安全清理梯度的方法
        void clear_grad_safely() {
            std::ostringstream oss;
            oss << ">>> Node::clear_grad_safely - 开始, 节点ID: " << tensor_id;
            std::string msg = oss.str();
            Ctorch_Error::info(ErrorPlatform::kCPU,msg);
            if (!grad.empty()) {
                grad.clear_storage();
            }
            Ctorch_Error::info(ErrorPlatform::kCPU,"<<< Node::clear_grad_safely - 完成");
        }
    };

    struct PendingRecord {
        op operation;
        std::vector<size_t> input_ids;
        std::vector<std::vector<size_t>> input_shapes;
        bool committed = false;
    };

    std::unordered_map<size_t, std::unique_ptr<Node>> id_to_node;
    std::unordered_map<size_t, PendingRecord> pending_records;
    std::mutex records_mutex;
    bool retain_graph = false;
    // 添加详细的调试方法


public:
    void debug_print_state(const std::string& context) {
        std::lock_guard<std::mutex> lock(records_mutex);
        std::ostringstream oss;
        oss << "=== AutoDiff状态 [" << context << "] ===" << std::endl;
        oss << "计算图节点 (" << id_to_node.size() << "): ";
        for (const auto& pair : id_to_node) {
            if (pair.second) {
                oss << pair.first << " ";
            }
        }
        oss << std::endl;

        oss << "待处理记录 (" << pending_records.size() << "): ";
        for (const auto& pair : pending_records) {
            oss << pair.first << "(committed=" << pair.second.committed << ") ";
        }
        oss << std::endl;
        oss << "=================================" << std::endl;
        std::string msg = oss.str();
        Ctorch_Error::info(ErrorPlatform::kCPU,msg);
    }
    // 修复get_grad函数
Tensor get_grad(const Tensor* t) {
        if (!t || t->id() == 0) {
            Ctorch_Error::info(ErrorPlatform::kCPU, ">>> get_grad: 无效输入");
            return Tensor();
        }

        std::ostringstream oss;
        oss << ">>> get_grad - 开始, 目标ID: " << t->id();
        std::string msg = oss.str();
        Ctorch_Error::info(ErrorPlatform::kCPU, msg);

        // 第一步：在锁内获取必要信息（不创建新Tensor）
        bool has_grad = false;
        std::vector<size_t> grad_shape;
        DType grad_dtype;
        DeviceType grad_device;
        float grad_value = 0.0f; // 存储梯度值用于调试

        {
            std::lock_guard<std::mutex> lock(records_mutex);
            auto it = id_to_node.find(t->id());
            if (it != id_to_node.end() && it->second && it->second->requires_grad && !it->second->grad.empty()) {
                has_grad = true;
                grad_shape = it->second->grad.sizes();
                grad_dtype = it->second->grad.dtype();
                grad_device = it->second->grad.device();

                // 获取梯度值用于调试
                if (grad_dtype == DType::kFloat && !grad_shape.empty()) {
                    const float *grad_data = it->second->grad.data<float>();
                    if (grad_data) {
                        grad_value = grad_data[0]; // 对于标量，取第一个元素
                    }
                }
                std::ostringstream osss;
                osss << ">>> 找到梯度，形状: [";
                for (auto s: grad_shape) osss << s << " ";
                osss << "], 值: " << grad_value;
                std::string msgs = osss.str();
                Ctorch_Error::info(ErrorPlatform::kCPU, msgs);
            }
        } // 释放锁

        // 第二步：在锁外创建结果Tensor
        if (has_grad) {
            Ctorch_Error::info(ErrorPlatform::kCPU, ">>> 创建梯度副本");
            Tensor result(ShapeTag{}, grad_shape, grad_dtype, grad_device);

            // 第三步：重新加锁复制数据
            {
                std::lock_guard<std::mutex> lock(records_mutex);
                auto it = id_to_node.find(t->id());
                if (it != id_to_node.end() && it->second && !it->second->grad.empty()) {
                    Ctorch_Error::info(ErrorPlatform::kCPU, ">>> 复制梯度数据");
                    // 直接复制数据，避免调用clone()
                    switch (grad_dtype) {
                        case DType::kFloat: {
                            const float *src = it->second->grad.data<float>();
                            float *dst = result.data<float>();
                            size_t count = result.numel();
                            for (size_t i = 0; i < count; ++i) {
                                dst[i] = src[i];
                            }
                            std::ostringstream osss;
                            osss << ">>> 复制了 " << count << " 个float值，第一个值: " << dst[0];
                            std::string msgs = osss.str();
                            Ctorch_Error::info(ErrorPlatform::kCPU, msgs);
                            break;
                        }
                        case DType::kDouble: {
                            const double *src = it->second->grad.data<double>();
                            double *dst = result.data<double>();
                            for (size_t i = 0; i < result.numel(); ++i) {
                                dst[i] = src[i];
                            }
                            break;
                        }
                        case DType::kInt: {
                            const int32_t *src = it->second->grad.data<int32_t>();
                            int32_t *dst = result.data<int32_t>();
                            for (size_t i = 0; i < result.numel(); ++i) {
                                dst[i] = src[i];
                            }
                            break;
                        }
                        case DType::kLong: {
                            const int64_t *src = it->second->grad.data<int64_t>();
                            int64_t *dst = result.data<int64_t>();
                            for (size_t i = 0; i < result.numel(); ++i) {
                                dst[i] = src[i];
                            }
                            break;
                        }
                        default:
                            throw std::runtime_error("Unsupported dtype in get_grad");
                    }
                }
            }
            std::ostringstream osss;
            osss << ">>> 最终梯度副本值: " << result.item<float>() << std::endl;
            osss << "<<< get_grad - 完成" << std::endl;

            std::string msgs = osss.str();
            Ctorch_Error::info(ErrorPlatform::kCPU, msgs);
            return result;
        }
        Ctorch_Error::info(ErrorPlatform::kCPU, "<<< get_grad - 未找到梯度");
        return Tensor();
    }

    // 修复make_leaf函数
    // 修改 make_leaf 函数，避免可能的死锁
    // 在关键方法中调用调试
    void make_leaf(Tensor& t, bool requires_grad) {
        size_t id = t.id();
        if (id == 0) {
            Ctorch_Error::info(ErrorPlatform::kCPU, "!!! 错误: 尝试注册ID为0的张量");
            return;
        }

        std::ostringstream osss;
        osss << ">>> AutoDiff::make_leaf - 开始, ID: " << id << std::endl;

        std::string msgs = osss.str();
        Ctorch_Error::info(ErrorPlatform::kCPU, msgs);
        debug_print_state("make_leaf开始前");
        {
            std::lock_guard<std::mutex> lock(records_mutex);
            if (id_to_node.find(id) != id_to_node.end()) {
                std::ostringstream oss;
                oss << ">>> 节点 " << id << " 已存在，跳过创建" << std::endl;
                std::string msg = oss.str();
                Ctorch_Error::info(ErrorPlatform::kCPU, msg);
                return;
            }
        }

        // 创建节点时不持有锁
        auto node = std::make_unique<Node>(id, t.clone(), requires_grad, true);

        {
            std::lock_guard<std::mutex> lock(records_mutex);
            id_to_node[id] = std::move(node);
        }


        std::ostringstream oss;
        oss << "<<< AutoDiff::make_leaf - 完成, ID: " << id << std::endl;
        std::string msg = oss.str();
        Ctorch_Error::info(ErrorPlatform::kCPU, msg);
        debug_print_state("make_leaf完成后");
    }
    // 在 AutoDiff 类中添加
    ~AutoDiff() {
        Ctorch_Error::info(ErrorPlatform::kCPU, ">>> AutoDiff 析构");
        // 避免在析构时进行复杂操作
        // 直接清空，不持有锁
        id_to_node.clear();
        pending_records.clear();
    }
    // 删除旧的record_op函数，使用新的defer_record
    void defer_record(size_t output_id, op operation, const std::vector<Tensor*>& inputs) {

        std::ostringstream oss;
        oss << ">>> 进入 defer_record, output_id: " << output_id << std::endl;
        std::string msg = oss.str();
        Ctorch_Error::info(ErrorPlatform::kCPU, msg);
        if (output_id == 0) {
            Ctorch_Error::log(ErrorLevel::WARN,ErrorPlatform::kCPU,ErrorType::TENSOR_STATE,"警告: output_id 为0");
            return;
        }

        // 创建记录对象（不持有锁）
        PendingRecord record;
        record.operation = operation;

        std::vector<size_t> input_ids;
        std::vector<std::vector<size_t>> input_shapes;

        // 收集输入信息（不持有锁）
        for (Tensor* input : inputs) {
            if (input && input->id() != 0) {
                input_ids.push_back(input->id());
                input_shapes.push_back(input->shape());
                std::ostringstream osss;
                osss << ">>> 处理输入: " << input->id() << std::endl;
                std::string msgs = osss.str();
                Ctorch_Error::info(ErrorPlatform::kCPU, msgs);

                // 检查是否已注册，避免不必要的锁
                bool needs_registration = false;
                {
                    std::lock_guard<std::mutex> lock(records_mutex);
                    needs_registration = (id_to_node.find(input->id()) == id_to_node.end());
                }

                if (needs_registration) {
                    std::ostringstream ost;
                    ost << ">>> 注册叶子节点: " << input->id() << std::endl;
                    std::string msgt = ost.str();
                    Ctorch_Error::info(ErrorPlatform::kCPU, msgt);

                    make_leaf(*input, input->requires_grad());
                }
            }
        }

        // 现在持有锁并设置记录
        {
            std::lock_guard<std::mutex> lock(records_mutex);
            record.input_ids = input_ids;
            record.input_shapes = input_shapes;
            pending_records[output_id] = record;
        }

        Ctorch_Error::info(ErrorPlatform::kCPU, "<<< 离开 defer_record");
    }
    // 新增：提交延迟记录
    // 1. 在 AutoDiff 里把“需要提交”的信息先捞出来，锁外再构造 Node
    void commit_record(Tensor& output) {
    size_t output_id = output.id();
    std::cout << ">>> AutoDiff::commit_record - 开始, ID: " << output_id << std::endl;

    if (output_id == 0) {
        std::cout << "!!! 错误: 输出ID为0" << std::endl;
        return;
    }

    debug_print_state("commit_record开始前");

    // 第一步：收集必要信息（持有锁，但时间很短）
    PendingRecord record_copy;
    std::vector<size_t> input_ids_copy;
    bool should_create_node = false;

    {
        std::lock_guard<std::mutex> lock(records_mutex);

        auto it = pending_records.find(output_id);
        if (it == pending_records.end()) {
            std::cout << "!!! 警告: 找不到待处理记录 " << output_id << std::endl;
            return;
        }

        PendingRecord& record = it->second;
        if (record.committed) {
            std::cout << ">>> 记录 " << output_id << " 已提交，跳过" << std::endl;
            return;
        }

        std::cout << ">>> 开始提交记录 " << output_id << std::endl;
        std::cout << ">>> 操作类型: " << static_cast<int>(record.operation) << std::endl;
        std::cout << ">>> 输入IDs: ";
        for (auto id : record.input_ids) std::cout << id << " ";
        std::cout << std::endl;

        // 验证输入节点
        bool valid = true;
        for (size_t input_id : record.input_ids) {
            if (id_to_node.find(input_id) == id_to_node.end()) {
                std::cout << "!!! 错误: 输入节点 " << input_id << " 不存在" << std::endl;
                valid = false;
                break;
            } else {
                std::cout << ">>> 输入节点 " << input_id << " 存在" << std::endl;
            }
        }

        if (!valid) {
            std::cout << "!!! 记录验证失败，清除记录 " << output_id << std::endl;
            pending_records.erase(it);
            return;
        }

        // 复制必要信息，然后释放锁
        record_copy = record;
        input_ids_copy = record.input_ids;
        should_create_node = true;

        // 立即标记为已提交，避免重复处理
        record.committed = true;
    } // 释放锁

    // 第二步：在锁外创建节点（避免死锁）
    if (should_create_node) {
        std::cout << ">>> 在锁外创建操作节点 " << output_id << std::endl;

        // 确定梯度需求（在锁外检查）
        bool requires_grad = false;
        for (size_t input_id : input_ids_copy) {
            Node* input_node = get_node_by_id(input_id);
            if (input_node && input_node->requires_grad) {
                requires_grad = true;
                break;
            }
        }

        std::cout << ">>> 节点 " << output_id << " 需要梯度: " << requires_grad << std::endl;

        // 创建节点（不持有锁）
        auto output_node = std::make_unique<Node>(output_id, output.clone(), requires_grad, false);
        output_node->operation = record_copy.operation;
        output_node->input_ids = input_ids_copy;

        // 延迟创建梯度存储
        if (requires_grad) {
            std::cout << ">>> 为节点 " << output_id << " 延迟分配梯度存储" << std::endl;
            // 注意：不在构造函数中创建 grad，避免死锁
        }

        // 第三步：重新加锁，快速插入节点
        {
            std::lock_guard<std::mutex> lock(records_mutex);
            id_to_node[output_id] = std::move(output_node);

            // 清理已提交的记录
            auto it = pending_records.find(output_id);
            if (it != pending_records.end() && it->second.committed) {
                pending_records.erase(it);
            }
        }

        // 更新输出张量的梯度需求（不需要锁）
        output.requires_grad(requires_grad);

        std::cout << "<<< 记录提交完成 " << output_id << std::endl;
        debug_print_state("commit_record完成后");
    }
}
    // 新增：更新梯度需求
    void update_requires_grad(Tensor& t, bool requires_grad) {
        size_t id = t.id();
        if (id == 0) return;

        std::cout << ">>> 更新梯度需求: " << id << " -> " << requires_grad << std::endl;

        // 使用更短的锁作用域
        {
            std::lock_guard<std::mutex> lock(records_mutex);
            auto it = id_to_node.find(id);
            if (it != id_to_node.end() && it->second) {
                it->second->requires_grad = requires_grad;
                if (requires_grad && it->second->grad.empty()) {
                    it->second->grad = Tensor(ShapeTag{}, t.sizes(), t.dtype(), t.device());
                    it->second->grad.zero();
                }
            } else {
                // 如果节点不存在，可能需要创建它
                std::cout << ">>> 节点不存在，可能需要创建: " << id << std::endl;
            }
        }

        std::cout << "<<< 更新梯度需求完成" << std::endl;
    }

    // 设置是否保留计算图
    void set_retain_graph(bool retain) {
        retain_graph = retain;
    }

    // 删除旧的get_node函数，使用新的get_node_by_id
    Node* get_node_by_id(size_t id) {
        // 使用更短的锁作用域
        std::lock_guard<std::mutex> lock(records_mutex);
        auto it = id_to_node.find(id);
        if (it != id_to_node.end()) {
            return it->second.get();
        }
        std::cout << ">>> 警告: 找不到节点 " << id << std::endl;
        return nullptr;
    }
    // 版本1：用户不提供grad_output，使用默认的1.0
    void backward(Tensor& root) {
        backward(root, Tensor(1.0f)); // 直接传入值为1的Tensor
    }
    // 修复backward函数
    void backward(Tensor& root, Tensor grad_output) {
    std::cout << ">>> =========================================" << std::endl;
    std::cout << ">>> 进入 backward 函数，root ID: " << root.id() << std::endl;

    if (root.id() == 0) {
        throw std::runtime_error("Invalid root tensor for backward (ID is 0)");
    }

    // 阶段1：准备阶段（不持有锁）
    std::cout << ">>> 阶段1: 准备阶段" << std::endl;

    // 先提交所有未完成的记录（不持有backward的主锁）
    {
        std::lock_guard<std::mutex> lock(records_mutex);
        for (auto& [output_id, record] : pending_records) {
            if (!record.committed) {
                auto node_it = id_to_node.find(output_id);
                if (node_it != id_to_node.end() && node_it->second) {
                    record.committed = true;
                    node_it->second->operation = record.operation;
                    node_it->second->input_ids = record.input_ids;
                }
            }
        }
    }

     // 阶段2：验证阶段（持有锁但时间很短）
    std::cout << ">>> 阶段2: 验证阶段" << std::endl;
    Node* root_node = nullptr;
    bool root_requires_grad = false;
    std::vector<size_t> root_shape;

    {
        std::lock_guard<std::mutex> lock(records_mutex);
        auto it = id_to_node.find(root.id());
        if (it == id_to_node.end() || !it->second) {
            std::cout << ">>> 错误: 根张量 " << root.id() << " 不在计算图中" << std::endl;
            std::cout << ">>> 当前计算图中的节点: ";
            for (const auto& pair : id_to_node) {
                std::cout << pair.first << " ";
            }
            std::cout << std::endl;
            throw std::runtime_error("Root tensor not in computation graph");
        }
        root_node = it->second.get();
        root_requires_grad = root_node->requires_grad;
        root_shape = root_node->tensor.sizes();
    } // 释放锁

    if (!root_requires_grad) {
        throw std::runtime_error("Root tensor doesn't require gradient");
    }

    // 简化的初始梯度设置
    float initial_grad_value = 1.0f;  // 默认值为1.0

    // 只有当用户明确提供了非空grad_output时才使用它
    if (!grad_output.empty() && grad_output.numel() > 0) {
        std::cout << ">>> 使用用户提供的梯度输出" << std::endl;
        if (grad_output.sizes() != root.sizes()) {
            throw std::runtime_error("Grad output shape mismatch");
        }
        initial_grad_value = grad_output.item<float>();
    } else {
        std::cout << ">>> 使用默认初始梯度值: " << initial_grad_value << std::endl;
    }

    std::cout << ">>> 最终初始梯度值: " << initial_grad_value << std::endl;

    // 重新加锁设置梯度
    {
        std::lock_guard<std::mutex> lock(records_mutex);
        // 确保梯度已分配
        if (root_node->grad.empty()) {
            std::cout << ">>> 为根节点分配梯度存储" << std::endl;
            root_node->grad = Tensor(ShapeTag{}, root_shape, root.dtype(), root.device());
            std::cout << ">>> 根节点梯度分配完成，初始值: " << root_node->grad.item<float>() << std::endl;
        }

        // 设置初始梯度
        std::cout << ">>> 设置根节点初始梯度" << std::endl;
        std::cout << ">>> root_node->grad值(设置前): " << root_node->grad.item<float>() << std::endl;

        switch (root_node->grad.dtype()) {
            case DType::kFloat: {
                float* grad_data = root_node->grad.data<float>();
                std::cout << ">>> 设置梯度数据: " << initial_grad_value << " -> grad_data[0]" << std::endl;
                for (size_t i = 0; i < root_node->grad.numel(); ++i) {
                    grad_data[i] = initial_grad_value;
                }
                std::cout << ">>> 设置后根节点梯度值: " << root_node->grad.item<float>() << std::endl;
                break;
            }
            case DType::kDouble: {
                double* grad_data = root_node->grad.data<double>();
                for (size_t i = 0; i < root_node->grad.numel(); ++i) {
                    grad_data[i] = static_cast<double>(initial_grad_value);
                }
                break;
            }
            case DType::kInt: {
                int32_t* grad_data = root_node->grad.data<int32_t>();
                for (size_t i = 0; i < root_node->grad.numel(); ++i) {
                    grad_data[i] = static_cast<int32_t>(initial_grad_value);
                }
                break;
            }
            case DType::kLong: {
                int64_t* grad_data = root_node->grad.data<int64_t>();
                for (size_t i = 0; i < root_node->grad.numel(); ++i) {
                    grad_data[i] = static_cast<int64_t>(initial_grad_value);
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype in backward");
        }
    }


    // ... 后面的代码不变 ...
    // 阶段3：拓扑排序（不持有锁）
    std::cout << ">>> 阶段3: 拓扑排序" << std::endl;
    std::vector<size_t> order;
    std::unordered_set<size_t> visited;
    std::unordered_set<size_t> in_stack;

    std::function<void(size_t)> dfs = [&](size_t node_id) {
        if (visited.find(node_id) != visited.end()) return;
        if (in_stack.find(node_id) != in_stack.end()) {
            throw std::runtime_error("Cycle detected in computation graph");
        }

        in_stack.insert(node_id);

        // 获取节点信息（短暂持有锁）
        Node* node = nullptr;
        {
            std::lock_guard<std::mutex> lock(records_mutex);
            auto it = id_to_node.find(node_id);
            if (it != id_to_node.end()) {
                node = it->second.get();
            }
        }

        if (node) {
            for (size_t input_id : node->input_ids) {
                dfs(input_id);
            }
        }

        in_stack.erase(node_id);
        visited.insert(node_id);
        order.push_back(node_id);
    };

    dfs(root.id());
    std::reverse(order.begin(), order.end());
    std::cout << ">>> 拓扑排序完成，节点数: " << order.size() << std::endl;

        // 在 backward 函数中修改梯度清理部分
        // 阶段4：反向传播（每个节点独立处理）
        std::cout << ">>> 阶段4: 反向传播 - 开始处理 " << order.size() << " 个节点" << std::endl;
        for (size_t node_id : order) {

            std::cout << ">>> 处理节点 " << node_id << " (" << (order.size() - (&node_id - order.data())) << "/" << order.size() << ")" << std::endl;

            // 获取节点信息（短暂持有锁）
            Node* node = nullptr;
            {
                std::lock_guard<std::mutex> lock(records_mutex);
                auto it = id_to_node.find(node_id);
                if (it != id_to_node.end()) {
                    node = it->second.get();
                }
            }

            if (!node || node->is_leaf) continue;

            // 确保梯度已分配（在锁外）
            if (node->requires_grad && node->grad.empty()) {
                node->grad = Tensor(ShapeTag{}, node->tensor.sizes(), node->tensor.dtype(), node->tensor.device());
                node->grad.zero();
            }

            if (node->grad.empty()) {
                continue;
            }

            // 执行反向传播（不持有锁）
            try {
                switch (node->operation) {
                    case op::Add: backward_add(node); break;
                    case op::Sub: backward_sub(node); break;
                    case op::Neg: backward_neg(node); break;
                    case op::Mul: backward_mul(node); break;
                    case op::Div: backward_div(node); break;
                    case op::MatMul: backward_matmul(node); break;
                    case op::ReLU: backward_relu(node); break;
                    case op::Sigmoid: backward_sigmoid(node); break;
                    case op::Tanh: backward_tanh(node); break;
                    case op::Softmax: backward_softmax(node); break;
                    case op::Sum: backward_sum(node); break;
                    default:
                        std::cerr << "Unsupported operation in backward: "
                                  << static_cast<int>(node->operation) << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error during backward pass for node " << node_id
                          << ": " << e.what() << std::endl;
                throw;
            }

            // 修复：避免在锁内创建新Tensor
            // 在 backward 函数中修改梯度清理部分

            if (!retain_graph && !node->is_leaf) {
                std::cout << ">>> 准备清理节点 " << node_id << " 的中间梯度" << std::endl;
                std::cout << ">>> 当前梯度状态: " << (node->grad.empty() ? "空" : "非空") << std::endl;

                // 使用新的安全清理方法
                {
                    std::lock_guard<std::mutex> lock(records_mutex);
                    node->clear_grad_safely();
                }
                std::cout << ">>> 清理后梯度状态: " << (node->grad.empty() ? "空" : "非空") << std::endl;
                std::cout << ">>> 节点 " << node_id << " 的梯度已清理" << std::endl;
            }
        }

    std::cout << "<<< 离开 backward 函数" << std::endl;
}
    // 清除计算图
    void clear_graph() {
        std::lock_guard<std::mutex> lock(records_mutex);
        id_to_node.clear();
        pending_records.clear();
    }

private:
    // 验证记录有效性
    bool validate_record(const PendingRecord& record, const Tensor& output) {
        for (size_t i = 0; i < record.input_ids.size(); ++i) {
            Node* input_node = get_node_by_id(record.input_ids[i]);
            if (!input_node) {
                std::cerr << "Input node " << record.input_ids[i] << " not found" << std::endl;
                return false;
            }

            // 验证形状匹配（简化验证）
            if (!record.input_shapes[i].empty() &&
                input_node->tensor.sizes() != record.input_shapes[i]) {
                std::cerr << "Shape mismatch for input " << i << std::endl;
                return false;
            }
        }
        return true;
    }

    // 创建操作节点
    void create_operation_node(Tensor& output, const PendingRecord& record) {
        // 确定输出是否需要梯度
        bool requires_grad = false;
        for (size_t input_id : record.input_ids) {
            if (Node* input_node = get_node_by_id(input_id)) {
                if (input_node->requires_grad) {
                    requires_grad = true;
                    break;
                }
            }
        }

        auto output_node = std::make_unique<Node>(output.id(), output.clone(), requires_grad, false);
        output_node->operation = record.operation;
        output_node->input_ids = record.input_ids;

        id_to_node[output.id()] = std::move(output_node);
        output.requires_grad(requires_grad);
    }

    // 提交所有未完成记录
    void commit_all_pending_records() {
        std::cout << ">>> 进入 commit_all_pending_records" << std::endl;

        for (auto& [output_id, record] : pending_records) {
            if (!record.committed) {
                std::cout << ">>> 提交记录: " << output_id << std::endl;

                // 需要output tensor，这里简化处理
                auto it = id_to_node.find(output_id);
                if (it != id_to_node.end() && it->second) {
                    record.committed = true;
                    // 这里假设节点已经创建，只需要设置操作类型
                    it->second->operation = record.operation;
                    it->second->input_ids = record.input_ids;
                    std::cout << ">>> 记录 " << output_id << " 提交成功" << std::endl;
                } else {
                    std::cout << ">>> 警告: 找不到节点 " << output_id << "，跳过记录提交" << std::endl;
                }
            }
        }

        std::cout << "<<< 离开 commit_all_pending_records" << std::endl;
    }

    // 修复反向传播函数 - 使用新的Node结构
void backward_add(Node* node) {
    std::cout << ">>> 进入 backward_add, 节点: " << node->tensor_id << std::endl;

    // 确保梯度已分配
    if (node->requires_grad && node->grad.empty()) {
        std::cout << ">>> 为节点 " << node->tensor_id << " 分配梯度存储" << std::endl;
        node->grad = Tensor(ShapeTag{}, node->tensor.sizes(), node->tensor.dtype(), node->tensor.device());
        node->grad.zero();
    }

    Tensor& grad_out = node->grad;
    std::cout << ">>> grad_out 形状: [";
    for (auto s : grad_out.shape()) std::cout << s << " ";
    std::cout << "], 值: " << grad_out.item<float>() << std::endl;

    for (size_t input_id : node->input_ids) {
        std::cout << ">>> 处理输入: " << input_id << std::endl;
        Node* input_node = get_node_by_id(input_id);
        if (!input_node || !input_node->requires_grad) {
            std::cout << ">>> 输入节点不存在或不需要梯度，跳过" << std::endl;
            continue;
        }

        // 确保输入节点的梯度已分配
        if (input_node->requires_grad && input_node->grad.empty()) {
            std::cout << ">>> 为输入节点 " << input_id << " 分配梯度存储" << std::endl;
            input_node->grad = Tensor(ShapeTag{}, input_node->tensor.sizes(),
                                     input_node->tensor.dtype(), input_node->tensor.device());
            input_node->grad.zero();
        }

        std::cout << ">>> 计算输入梯度" << std::endl;
        std::cout << ">>> 输入节点 " << input_id << " 当前梯度值: " << input_node->grad.item<float>() << std::endl;
        std::cout << ">>> 传播梯度值: " << grad_out.item<float>() << std::endl;

        // 对于加法，梯度直接传播，不需要reduce
        // 直接操作数据，避免创建临时Tensor
        switch (input_node->grad.dtype()) {
            case DType::kFloat: {
                float* dst = input_node->grad.data<float>();
                const float* src = grad_out.data<float>();
                for (size_t i = 0; i < input_node->grad.numel(); ++i) {
                    dst[i] += src[i];  // 累加梯度
                }
                std::cout << ">>> 累加后输入节点 " << input_id << " 梯度值: " << input_node->grad.item<float>() << std::endl;
                break;
            }
            case DType::kDouble: {
                double* dst = input_node->grad.data<double>();
                const double* src = grad_out.data<double>();
                for (size_t i = 0; i < input_node->grad.numel(); ++i) {
                    dst[i] += src[i];
                }
                break;
            }
            case DType::kInt: {
                int32_t* dst = input_node->grad.data<int32_t>();
                const int32_t* src = grad_out.data<int32_t>();
                for (size_t i = 0; i < input_node->grad.numel(); ++i) {
                    dst[i] += src[i];
                }
                break;
            }
            case DType::kLong: {
                int64_t* dst = input_node->grad.data<int64_t>();
                const int64_t* src = grad_out.data<int64_t>();
                for (size_t i = 0; i < input_node->grad.numel(); ++i) {
                    dst[i] += src[i];
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype in backward_add");
        }

        std::cout << ">>> 输入 " << input_id << " 梯度更新完成" << std::endl;
    }

    std::cout << "<<< 离开 backward_add" << std::endl;
}

    void backward_neg(Node* node) {
        std::cout << ">>> 进入 backward_neg, 节点: " << node->tensor_id << std::endl;

        Tensor& grad_out = node->grad;
        if (node->input_ids.size() != 1) {
            throw std::runtime_error("Negation requires exactly 1 input");
        }

        Node* input_node = get_node_by_id(node->input_ids[0]);
        if (!input_node || !input_node->requires_grad) {
            std::cout << ">>> 输入节点不存在或不需要梯度，跳过" << std::endl;
            return;
        }

        // 确保梯度已分配
        if (input_node->grad.empty()) {
            input_node->grad = Tensor(ShapeTag{}, input_node->tensor.sizes(),
                                     input_node->tensor.dtype(), input_node->tensor.device());
            input_node->grad.zero();
        }

        // 对于负号：∂(-x)/∂x = -grad_out
        std::cout << ">>> 计算输入梯度" << std::endl;
        Tensor grad_input = reduce_to_match(grad_out, input_node->tensor.sizes());

        // 直接累加到现有梯度（注意是减法）
        switch (input_node->grad.dtype()) {
            case DType::kFloat: {
                float* dst = input_node->grad.data<float>();
                const float* src = grad_input.data<float>();
                for (size_t i = 0; i < input_node->grad.numel(); ++i) {
                    dst[i] -= src[i];  // 注意这里是减法
                }
                break;
            }
            case DType::kDouble: {
                double* dst = input_node->grad.data<double>();
                const double* src = grad_input.data<double>();
                for (size_t i = 0; i < input_node->grad.numel(); ++i) {
                    dst[i] -= src[i];  // 注意这里是减法
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype in backward_neg");
        }

        std::cout << "<<< 离开 backward_neg" << std::endl;
    }

    void backward_sub(Node* node) {
        std::cout << ">>> 进入 backward_sub, 节点: " << node->tensor_id << std::endl;

        Tensor& grad_out = node->grad;
        if (node->input_ids.size() != 2) {
            throw std::runtime_error("Subtraction requires exactly 2 inputs");
        }

        Node* a = get_node_by_id(node->input_ids[0]);
        Node* b = get_node_by_id(node->input_ids[1]);

        // 确保梯度已分配
        if (a && a->requires_grad && a->grad.empty()) {
            a->grad = Tensor(ShapeTag{}, a->tensor.sizes(), a->tensor.dtype(), a->tensor.device());
            a->grad.zero();
        }

        if (b && b->requires_grad && b->grad.empty()) {
            b->grad = Tensor(ShapeTag{}, b->tensor.sizes(), b->tensor.dtype(), b->tensor.device());
            b->grad.zero();
        }

        // 对于减法：∂(a-b)/∂a = grad_out, ∂(a-b)/∂b = -grad_out
        if (a && a->requires_grad) {
            std::cout << ">>> 计算a的梯度" << std::endl;
            Tensor grad_a = reduce_to_match(grad_out, a->tensor.sizes());

            // 直接累加到现有梯度
            switch (a->grad.dtype()) {
                case DType::kFloat: {
                    float* dst = a->grad.data<float>();
                    const float* src = grad_a.data<float>();
                    for (size_t i = 0; i < a->grad.numel(); ++i) {
                        dst[i] += src[i];
                    }
                    break;
                }
                case DType::kDouble: {
                    double* dst = a->grad.data<double>();
                    const double* src = grad_a.data<double>();
                    for (size_t i = 0; i < a->grad.numel(); ++i) {
                        dst[i] += src[i];
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported dtype in backward_sub");
            }
        }

        if (b && b->requires_grad) {
            std::cout << ">>> 计算b的梯度" << std::endl;
            Tensor grad_b = reduce_to_match(grad_out, b->tensor.sizes());

            // 直接累加到现有梯度（注意是减法）
            switch (b->grad.dtype()) {
                case DType::kFloat: {
                    float* dst = b->grad.data<float>();
                    const float* src = grad_b.data<float>();
                    for (size_t i = 0; i < b->grad.numel(); ++i) {
                        dst[i] -= src[i];  // 注意这里是减法
                    }
                    break;
                }
                case DType::kDouble: {
                    double* dst = b->grad.data<double>();
                    const double* src = grad_b.data<double>();
                    for (size_t i = 0; i < b->grad.numel(); ++i) {
                        dst[i] -= src[i];  // 注意这里是减法
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported dtype in backward_sub");
            }
        }

        std::cout << "<<< 离开 backward_sub" << std::endl;
    }

    void backward_mul(Node* node) {
        std::cout << ">>> 进入 backward_mul, 节点: " << node->tensor_id << std::endl;

        Tensor& grad_out = node->grad;
        if (node->input_ids.size() != 2) {
            throw std::runtime_error("Multiplication requires exactly 2 inputs");
        }

        Node* a = get_node_by_id(node->input_ids[0]);
        Node* b = get_node_by_id(node->input_ids[1]);

        // 确保梯度已分配
        if (a && a->requires_grad && a->grad.empty()) {
            a->grad = Tensor(ShapeTag{}, a->tensor.sizes(), a->tensor.dtype(), a->tensor.device());
            a->grad.zero();
        }

        if (b && b->requires_grad && b->grad.empty()) {
            b->grad = Tensor(ShapeTag{}, b->tensor.sizes(), b->tensor.dtype(), b->tensor.device());
            b->grad.zero();
        }

        // 对于乘法：∂(a*b)/∂a = b * grad_out, ∂(a*b)/∂b = a * grad_out
        // 使用广播操作直接计算
        if (a && a->requires_grad) {
            std::cout << ">>> 计算a的梯度" << std::endl;
            Tensor grad_a = grad_out * b->tensor;
            grad_a = reduce_to_match(grad_a, a->tensor.sizes());

            // 直接累加到现有梯度
            switch (a->grad.dtype()) {
                case DType::kFloat: {
                    float* dst = a->grad.data<float>();
                    const float* src = grad_a.data<float>();
                    for (size_t i = 0; i < a->grad.numel(); ++i) {
                        dst[i] += src[i];
                    }
                    break;
                }
                case DType::kDouble: {
                    double* dst = a->grad.data<double>();
                    const double* src = grad_a.data<double>();
                    for (size_t i = 0; i < a->grad.numel(); ++i) {
                        dst[i] += src[i];
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported dtype in backward_mul");
            }
        }

        if (b && b->requires_grad) {
            std::cout << ">>> 计算b的梯度" << std::endl;
            Tensor grad_b = grad_out * a->tensor;
            grad_b = reduce_to_match(grad_b, b->tensor.sizes());

            // 直接累加到现有梯度
            switch (b->grad.dtype()) {
                case DType::kFloat: {
                    float* dst = b->grad.data<float>();
                    const float* src = grad_b.data<float>();
                    for (size_t i = 0; i < b->grad.numel(); ++i) {
                        dst[i] += src[i];
                    }
                    break;
                }
                case DType::kDouble: {
                    double* dst = b->grad.data<double>();
                    const double* src = grad_b.data<double>();
                    for (size_t i = 0; i < b->grad.numel(); ++i) {
                        dst[i] += src[i];
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported dtype in backward_mul");
            }
        }

        std::cout << "<<< 离开 backward_mul" << std::endl;
    }

    void backward_div(Node* node) {
        std::cout << ">>> 进入 backward_div, 节点: " << node->tensor_id << std::endl;

        Tensor& grad_out = node->grad;
        if (node->input_ids.size() != 2) {
            throw std::runtime_error("Division requires exactly 2 inputs");
        }

        Node* a = get_node_by_id(node->input_ids[0]);
        Node* b = get_node_by_id(node->input_ids[1]);

        // 确保梯度已分配
        if (a && a->requires_grad && a->grad.empty()) {
            a->grad = Tensor(ShapeTag{}, a->tensor.sizes(), a->tensor.dtype(), a->tensor.device());
            a->grad.zero();
        }

        if (b && b->requires_grad && b->grad.empty()) {
            b->grad = Tensor(ShapeTag{}, b->tensor.sizes(), b->tensor.dtype(), b->tensor.device());
            b->grad.zero();
        }

        // 对于除法：∂(a/b)/∂a = grad_out / b, ∂(a/b)/∂b = -grad_out * a / (b^2)
        if (a && a->requires_grad) {
            std::cout << ">>> 计算a的梯度" << std::endl;
            Tensor grad_a = grad_out / b->tensor;
            grad_a = reduce_to_match(grad_a, a->tensor.sizes());

            // 直接累加到现有梯度
            switch (a->grad.dtype()) {
                case DType::kFloat: {
                    float* dst = a->grad.data<float>();
                    const float* src = grad_a.data<float>();
                    for (size_t i = 0; i < a->grad.numel(); ++i) {
                        dst[i] += src[i];
                    }
                    break;
                }
                case DType::kDouble: {
                    double* dst = a->grad.data<double>();
                    const double* src = grad_a.data<double>();
                    for (size_t i = 0; i < a->grad.numel(); ++i) {
                        dst[i] += src[i];
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported dtype in backward_div");
            }
        }

        if (b && b->requires_grad) {
            std::cout << ">>> 计算b的梯度" << std::endl;
            Tensor b_squared = b->tensor * b->tensor;
            Tensor grad_b = grad_out * (-a->tensor) / b_squared;
            grad_b = reduce_to_match(grad_b, b->tensor.sizes());

            // 直接累加到现有梯度
            switch (b->grad.dtype()) {
                case DType::kFloat: {
                    float* dst = b->grad.data<float>();
                    const float* src = grad_b.data<float>();
                    for (size_t i = 0; i < b->grad.numel(); ++i) {
                        dst[i] += src[i];
                    }
                    break;
                }
                case DType::kDouble: {
                    double* dst = b->grad.data<double>();
                    const double* src = grad_b.data<double>();
                    for (size_t i = 0; i < b->grad.numel(); ++i) {
                        dst[i] += src[i];
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported dtype in backward_div");
            }
        }

        std::cout << "<<< 离开 backward_div" << std::endl;
    }

    void backward_matmul(Node* node) {
        Tensor& grad_out = node->grad;
        if (node->input_ids.size() != 2) {
            throw std::runtime_error("Matrix multiplication requires exactly 2 inputs");
        }

        Node* a = get_node_by_id(node->input_ids[0]);
        Node* b = get_node_by_id(node->input_ids[1]);

        if (a && a->requires_grad) {
            Tensor b_t = b->tensor.transpose_last_two();
            Tensor grad_a = matMul(grad_out, b_t);
            grad_a = reduce_to_match(grad_a, a->tensor.sizes());
            a->grad = a->grad.empty() ? grad_a : a->grad + grad_a;
        }

        if (b && b->requires_grad) {
            Tensor a_t = a->tensor.transpose_last_two();
            Tensor grad_b = matMul(a_t, grad_out);
            grad_b = reduce_to_match(grad_b, b->tensor.sizes());
            b->grad = b->grad.empty() ? grad_b : b->grad + grad_b;
        }
    }

    void backward_relu(Node* node) {
        Tensor& grad_out = node->grad;
        if (node->input_ids.size() != 1) return;

        Node* input_node = get_node_by_id(node->input_ids[0]);
        if (!input_node || !input_node->requires_grad) return;

        // 确保输入节点梯度已分配
        if (input_node->grad.empty()) {
            input_node->grad = Tensor(ShapeTag{}, input_node->tensor.sizes(), input_node->tensor.dtype(), input_node->tensor.device());
            input_node->grad.zero();
        }

        // 直接计算ReLU梯度：如果输入>0则梯度为1，否则为0
        switch (input_node->grad.dtype()) {
            case DType::kFloat: {
                float* grad_data = input_node->grad.data<float>();
                const float* grad_out_data = grad_out.data<float>();
                const float* input_data = input_node->tensor.data<float>();

                for (size_t i = 0; i < input_node->grad.numel(); ++i) {
                    grad_data[i] += (input_data[i] > 0.0f) ? grad_out_data[i] : 0.0f;
                }
                break;
            }
            case DType::kDouble: {
                double* grad_data = input_node->grad.data<double>();
                const double* grad_out_data = grad_out.data<double>();
                const double* input_data = input_node->tensor.data<double>();

                for (size_t i = 0; i < input_node->grad.numel(); ++i) {
                    grad_data[i] += (input_data[i] > 0.0) ? grad_out_data[i] : 0.0;
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype in backward_relu");
        }
    }

    void backward_sigmoid(Node* node) {
        Tensor& grad_out = node->grad;
        if (node->input_ids.size() != 1) return;

        Node* input_node = get_node_by_id(node->input_ids[0]);
        if (!input_node || !input_node->requires_grad) return;

        Tensor sig_x = node->tensor;
        Tensor grad = grad_out * sig_x * (1.0f - sig_x);
        input_node->grad = input_node->grad.empty() ? grad : input_node->grad + grad;
    }

    void backward_tanh(Node* node) {
        Tensor& grad_out = node->grad;
        if (node->input_ids.size() != 1) return;

        Node* input_node = get_node_by_id(node->input_ids[0]);
        if (!input_node || !input_node->requires_grad) return;

        Tensor tanh_x = node->tensor;
        Tensor grad = grad_out * (1.0f - tanh_x * tanh_x);
        input_node->grad = input_node->grad.empty() ? grad : input_node->grad + grad;
    }

    void backward_softmax(Node* node) {
        Tensor& grad_out = node->grad;
        if (node->input_ids.size() != 1) return;

        Node* input_node = get_node_by_id(node->input_ids[0]);
        if (!input_node || !input_node->requires_grad) return;

        Tensor s = node->tensor;
        Tensor sum_term = (grad_out * s).sum(-1, true);
        Tensor grad_input = s * (grad_out - sum_term);
        input_node->grad = input_node->grad.empty() ? grad_input : input_node->grad + grad_input;
    }

    void backward_sum(Node* node) {
        std::cout << ">>> 进入 backward_sum, 节点: " << node->tensor_id << std::endl;

        Tensor& grad_out = node->grad;
        if (node->input_ids.size() != 1) {
            std::cout << ">>> 警告: Sum操作应该有1个输入，实际有 " << node->input_ids.size() << " 个" << std::endl;
            return;
        }

        Node* input_node = get_node_by_id(node->input_ids[0]);
        if (!input_node || !input_node->requires_grad) {
            std::cout << ">>> 输入节点不存在或不需要梯度，跳过" << std::endl;
            return;
        }

        std::cout << ">>> 扩展梯度到输入形状" << std::endl;
        Tensor expanded_grad;
        if (input_node->tensor.shape().empty()) {
            expanded_grad = grad_out;
        } else if (grad_out.sizes() == input_node->tensor.sizes()) {
            expanded_grad = grad_out;
        } else {
            expanded_grad = Tensor(ShapeTag{}, input_node->tensor.shape(),
                                  grad_out.dtype(), grad_out.device());

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
                default:
                    throw std::runtime_error("Unsupported dtype in sum backward");
            }
        }

        std::cout << ">>> 累加梯度" << std::endl;
        if (input_node->grad.empty()) {
            input_node->grad = expanded_grad;
        } else {
            input_node->grad = input_node->grad + expanded_grad;
        }

        std::cout << "<<< 离开 backward_sum" << std::endl;
    }

    // reduce_to_match 函数 - 完善实现
    Tensor reduce_to_match(Tensor grad, const std::vector<size_t>& target_shape) {
        std::cout << ">>> 进入 reduce_to_match" << std::endl;
        std::cout << ">>> 梯度形状: [";
        for (auto s : grad.shape()) std::cout << s << " ";
        std::cout << "], 目标形状: [";
        for (auto s : target_shape) std::cout << s << " ";
        std::cout << "]" << std::endl;

        // 如果形状完全匹配，直接返回
        if (grad.sizes() == target_shape) {
            std::cout << "<<< 形状匹配，直接返回" << std::endl;
            return grad;
        }

        // 如果目标形状为空（标量），需要求和
        if (target_shape.empty()) {
            std::cout << ">>> 目标为标量，执行求和" << std::endl;
            return grad.sum();
        }

        // 如果梯度形状为空（标量），需要广播到目标形状
        if (grad.shape().empty()) {
            std::cout << ">>> 梯度为标量，广播到目标形状" << std::endl;
            Tensor result(ShapeTag{}, target_shape, grad.dtype(), grad.device());
            float value = grad.item<float>();
            result.fill(value);
            return result;
        }

        // 处理广播情况：如果某个维度为1，需要求和
        std::vector<size_t> grad_shape = grad.shape();
        std::vector<int> sum_dims;

        // 从右对齐维度
        int grad_idx = grad_shape.size() - 1;
        int target_idx = target_shape.size() - 1;

        while (grad_idx >= 0 && target_idx >= 0) {
            if (grad_shape[grad_idx] != target_shape[target_idx]) {
                if (grad_shape[grad_idx] == 1) {
                    // 梯度维度为1，需要求和
                    sum_dims.push_back(grad_idx);
                } else if (target_shape[target_idx] == 1) {
                    // 目标维度为1，需要广播（不需要操作）
                } else {
                    throw std::runtime_error("Incompatible shapes for gradient reduction");
                }
            }
            grad_idx--;
            target_idx--;
        }

        // 如果还有剩余的梯度维度，需要求和
        while (grad_idx >= 0) {
            sum_dims.push_back(grad_idx);
            grad_idx--;
        }

        if (!sum_dims.empty()) {
            std::cout << ">>> 需要求和的维度: ";
            for (auto dim : sum_dims) std::cout << dim << " ";
            std::cout << std::endl;

            Tensor result = grad.sum(sum_dims, true);

            // 移除keepdim添加的维度
            std::vector<size_t> new_shape;
            for (size_t i = 0; i < result.shape().size(); ++i) {
                bool is_sum_dim = std::find(sum_dims.begin(), sum_dims.end(), static_cast<int>(i)) != sum_dims.end();
                if (!is_sum_dim || result.shape()[i] != 1) {
                    new_shape.push_back(result.shape()[i]);
                }
            }

            if (new_shape != result.shape()) {
                result = result.view(new_shape);
            }

            std::cout << "<<< 求和完成" << std::endl;
            return result;
        }

        std::cout << "<<< 离开 reduce_to_match" << std::endl;
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

// 通用二元操作符模板
// 修改 apply_binary_operator 函数，添加调试信息
template<typename OpFunc>
Tensor apply_binary_operator(const Tensor& lhs, const Tensor& rhs,
                           OpFunc op_func, op operation_type,
                           const char* op_name) {
    std::cout << ">>> =========================================" << std::endl;
    std::cout << ">>> 进入 apply_binary_operator: " << op_name << std::endl;
    std::cout << ">>> 操作类型: " << static_cast<int>(operation_type) << std::endl;
    // 输入张量信息
    lhs.debug_info_detailed("左操作数");
    rhs.debug_info_detailed("右操作数");

    // 基本检查
    if (lhs.dtype() != rhs.dtype()) {
        std::cout << "!!! 错误: 数据类型不匹配" << std::endl;
        throw std::runtime_error(std::string(op_name) + ": DType mismatch");
    }

    // 广播处理
    std::cout << ">>> 开始广播计算..." << std::endl;
    BroadCastResult bc = broadCast(lhs, rhs);
    std::cout << ">>> 广播完成" << std::endl;

    // 创建结果张量
    Tensor result(ShapeTag{}, bc.logicShape, lhs.dtype(), lhs.device());
    std::cout << ">>> 结果张量创建完成" << std::endl;
    result.debug_info_detailed("结果张量");

    // 执行计算
    std::cout << ">>> 开始执行计算..." << std::endl;
    switch (lhs.dtype()) {
    case DType::kFloat:
        apply_broadcast_op_impl<float>(result, lhs, rhs, bc, op_func);
        break;
    case DType::kDouble:
        apply_broadcast_op_impl<double>(result, lhs, rhs, bc, op_func);
        break;
    case DType::kInt:
        apply_broadcast_op_impl<int32_t>(result, lhs, rhs, bc, op_func);
        break;
    case DType::kLong:
        apply_broadcast_op_impl<int64_t>(result, lhs, rhs, bc, op_func);
        break;
    case DType::kBool:
        apply_broadcast_op_impl<bool>(result, lhs, rhs, bc, op_func);
        break;
    default:
        throw std::runtime_error("Unsupported dtype in apply_binary_operator");
    }
    std::cout << ">>> 计算完成" << std::endl;

    // 梯度需求传播
    bool requires_grad = Tensor::should_require_grad(lhs, rhs);
    std::cout << ">>> 设置梯度需求: " << requires_grad << std::endl;
    result.requires_grad(requires_grad);

    // 延迟记录操作
    AutoDiff* ctx = AutoDiffContext::current();
    if (ctx) {
        std::cout << ">>> 找到AutoDiff上下文，开始记录操作..." << std::endl;
        std::vector<Tensor*> inputs = {
            const_cast<Tensor*>(&lhs),
            const_cast<Tensor*>(&rhs)
        };

        std::cout << ">>> 调用 defer_record..." << std::endl;
        ctx->defer_record(result.id(), operation_type, inputs);
        std::cout << ">>> defer_record 完成" << std::endl;

        // 立即提交记录
        std::cout << ">>> 立即提交记录..." << std::endl;
        ctx->commit_record(result);
        std::cout << ">>> 记录提交完成" << std::endl;
    } else {
        std::cout << ">>> 未找到AutoDiff上下文，跳过记录" << std::endl;
    }

    std::cout << "<<< 离开 apply_binary_operator" << std::endl;
    std::cout << "<<< =========================================" << std::endl;
    return result;
}

// 广播操作实现
template<typename T, typename OpFunc>
void apply_broadcast_op_impl(Tensor& result, const Tensor& a, const Tensor& b,
                            const BroadCastResult& bc, OpFunc op_func) {
    const std::vector<size_t>& shape = bc.logicShape;
    const std::vector<size_t>& stridesA = bc.logicStridesA;
    const std::vector<size_t>& stridesB = bc.logicStridesB;

    T* out = result.data<T>();
    const T* a_data = a.data<T>();
    const T* b_data = b.data<T>();

    size_t total_elements = 1;
    for (auto dim : shape) total_elements *= dim;

    for (size_t flat_idx = 0; flat_idx < total_elements; ++flat_idx) {
        size_t a_idx = 0, b_idx = 0;
        size_t tmp_idx = flat_idx;

        for (int i = shape.size() - 1; i >= 0; --i) {
            size_t dim_size = shape[i];
            size_t coord = tmp_idx % dim_size;
            tmp_idx /= dim_size;

            a_idx += coord * stridesA[i];
            b_idx += coord * stridesB[i];
        }

        out[flat_idx] = op_func(a_data[a_idx], b_data[b_idx]);
    }
}
// 加法操作符
Tensor Tensor::operator+(const Tensor& rhs) const {
    return apply_binary_operator(*this, rhs,
        [](auto a, auto b) { return a + b; }, op::Add, "Addition");
}

// 乘法操作符
Tensor Tensor::operator*(const Tensor& rhs) const {
    return apply_binary_operator(*this, rhs,
        [](auto a, auto b) { return a * b; }, op::Mul, "Multiplication");
}

// 减法操作符
Tensor Tensor::operator-(const Tensor& rhs) const {
    return apply_binary_operator(*this, rhs,
        [](auto a, auto b) { return a - b; }, op::Sub, "Subtraction");
}

// 除法操作符
Tensor Tensor::operator/(const Tensor& rhs) const {
    return apply_binary_operator(*this, rhs,
        [](auto a, auto b) {
            if (b == 0) throw std::runtime_error("Division by zero");
            return a / b;
        }, op::Div, "Division");
}

// 负号操作符
Tensor Tensor::operator-() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);

    switch (_dtype) {
    case DType::kFloat: {
        const float* src = data<float>();
        float* dst = result.data<float>();
        for (size_t i = 0; i < numel(); ++i) dst[i] = -src[i];
        break;
    }
    case DType::kDouble: {
        const double* src = data<double>();
        double* dst = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) dst[i] = -src[i];
        break;
    }
    case DType::kInt: {
        const int32_t* src = data<int32_t>();
        int32_t* dst = result.data<int32_t>();
        for (size_t i = 0; i < numel(); ++i) dst[i] = -src[i];
        break;
    }
    case DType::kLong: {
        const int64_t* src = data<int64_t>();
        int64_t* dst = result.data<int64_t>();
        for (size_t i = 0; i < numel(); ++i) dst[i] = -src[i];
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for negation");
    }

    // 梯度需求传播
    result.requires_grad(this->requires_grad());

    // 延迟记录
    AutoDiff* ctx = AutoDiffContext::current();
    if (ctx) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this)};
        ctx->defer_record(result.id(), op::Neg, inputs);
        ctx->commit_record(result);
    }

    return result;
}

// 循环优化
Tensor matMul_naive(Tensor &a, Tensor &b) {
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
// Update 2025.8.1 分块算法
Tensor matMul_blocked(const Tensor &a, const Tensor &b) {
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
// ======================= 矩阵乘法函数（使用AMX优化） =======================

// 定义矩阵大小的阈值
const size_t SMALL_SIZE = 64;
const size_t MEDIUM_SIZE = 512;
Tensor matMul_recursive(const Tensor &a, const Tensor &b);
// 递归的Strassen矩阵乘法
Tensor matMul_recursive(const Tensor &a, const Tensor &b) {
   // 获取矩阵维度
   size_t M = a.shape()[0];
   size_t K = a.shape()[1];
   size_t N = b.shape()[1];

   // 基准情况：小矩阵使用分块优化
   if (M <= SMALL_SIZE || N <= SMALL_SIZE || K <= SMALL_SIZE) {
       return matMul_blocked(a, b);
   }
   // 中等矩阵使用AMX加速


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
   Tensor M1 = matMul_recursive(A11 + A22, B11 + B22);
   Tensor M2 = matMul_recursive(A21 + A22, B11);
   Tensor M3 = matMul_recursive(A11, B12 - B22);
   Tensor M4 = matMul_recursive(A22, B21 - B11);
   Tensor M5 = matMul_recursive(A11 + A12, B22);
   Tensor M6 = matMul_recursive(A21 - A11, B11 + B12);
   Tensor M7 = matMul_recursive(A12 - A22, B21 + B22);

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

// 矩阵乘法主函数(8.3 upt 目前仅AMX支持)
Tensor matMul_test(const Tensor &a, const Tensor &b) {
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
       return matMul_blocked(a, b);  // 小矩阵使用分块优化
   }
   else {
       return matMul_recursive(a, b); // 大矩阵使用Strassen递归
   }
}
// 分块+多线程版本
Tensor matMul(const Tensor& a, const Tensor& b) {
    // 维度检查
    if (a.dim() < 2 || b.dim() < 2) {
        throw std::runtime_error("matMul: Both tensors must be at least 2D");
    }

    size_t a_cols = a.shape()[a.dim() - 1];
    size_t b_rows = b.shape()[b.dim() - 2];
    if (a_cols != b_rows) {
        throw std::runtime_error("matMul: Inner dimensions must match");
    }

    // 批量矩阵乘法支持
    BroadCastResult bc = broadCast(a, b);
    std::vector<size_t> result_shape = bc.logicShape;
    size_t M = a.shape()[a.dim() - 2];
    size_t N = b.shape()[b.dim() - 1];
    result_shape[result_shape.size() - 2] = M;
    result_shape[result_shape.size() - 1] = N;

    Tensor result(ShapeTag{}, result_shape, a.dtype(), a.device());
    result.zero();

    // 执行计算（使用分块优化版本）
    result = matMul_blocked(a, b);

    // 梯度需求传播
    result.requires_grad(a.requires_grad() || b.requires_grad());

    // 延迟记录
    AutoDiff* ctx = AutoDiffContext::current();
    if (ctx) {
        std::vector<Tensor*> inputs = {
            const_cast<Tensor*>(&a),
            const_cast<Tensor*>(&b)
        };
        ctx->defer_record(result.id(), op::MatMul, inputs);
        ctx->commit_record(result);
    }

    return result;
}

// ReLU激活函数
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

    // 梯度需求传播
    result.requires_grad(this->requires_grad());

    // 延迟记录
    AutoDiff* ctx = AutoDiffContext::current();
    if (ctx) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this)};
        ctx->defer_record(result.id(), op::ReLU, inputs);
        ctx->commit_record(result);
    }

    return result;
}

// Sigmoid激活函数
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

    result.requires_grad(this->requires_grad());

    AutoDiff* ctx = AutoDiffContext::current();
    if (ctx) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this)};
        ctx->defer_record(result.id(), op::Sigmoid, inputs);
        ctx->commit_record(result);
    }

    return result;
}

// Tanh激活函数
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

    result.requires_grad(this->requires_grad());

    AutoDiff* ctx = AutoDiffContext::current();
    if (ctx) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this)};
        ctx->defer_record(result.id(), op::Tanh, inputs);
        ctx->commit_record(result);
    }

    return result;
}
// 修复softmax函数
Tensor Tensor::softmax(int dim) const {
    int actual_dim = dim;
    if (actual_dim < 0) {
        actual_dim = this->dim() + actual_dim;
    }

    if (actual_dim < 0 || actual_dim >= static_cast<int>(this->dim())) {
        throw std::runtime_error("Invalid dimension for softmax");
    }

    Tensor result(ShapeTag{}, _shape, _dtype, _device);
    switch (_dtype) {
    case DType::kFloat: {
        const float* src = data<float>();
        float* dst = result.data<float>();

        size_t slice_size = _shape[actual_dim];
        size_t num_slices = numel() / slice_size;

        for (size_t s = 0; s < num_slices; ++s) {
            float max_val = src[s * slice_size];
            for (size_t i = 1; i < slice_size; ++i) {
                if (src[s * slice_size + i] > max_val) {
                    max_val = src[s * slice_size + i];
                }
            }

            float exp_sum = 0.0f;
            for (size_t i = 0; i < slice_size; ++i) {
                float val = std::exp(src[s * slice_size + i] - max_val);
                dst[s * slice_size + i] = val;
                exp_sum += val;
            }

            for (size_t i = 0; i < slice_size; ++i) {
                dst[s * slice_size + i] /= exp_sum;
            }
        }
        break;
    }
    case DType::kDouble: {
        const double* src = data<double>();
        double* dst = result.data<double>();

        size_t slice_size = _shape[actual_dim];
        size_t num_slices = numel() / slice_size;

        for (size_t s = 0; s < num_slices; ++s) {
            double max_val = src[s * slice_size];
            for (size_t i = 1; i < slice_size; ++i) {
                if (src[s * slice_size + i] > max_val) {
                    max_val = src[s * slice_size + i];
                }
            }

            double exp_sum = 0.0;
            for (size_t i = 0; i < slice_size; ++i) {
                double val = std::exp(src[s * slice_size + i] - max_val);
                dst[s * slice_size + i] = val;
                exp_sum += val;
            }

            for (size_t i = 0; i < slice_size; ++i) {
                dst[s * slice_size + i] /= exp_sum;
            }
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for softmax");
    }

    // 使用新的延迟记录机制
    result.requires_grad(this->requires_grad());

    AutoDiff* ctx = AutoDiffContext::current();
    if (ctx) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this)};
        ctx->defer_record(result.id(), op::Softmax, inputs);
        ctx->commit_record(result);
    }

    return result;
}

inline Tensor Tensor::matmul_unified(const Tensor &other) const {
    // 检查是否需要自动微分
    bool need_grad = _requires_grad || other._requires_grad;

    // 执行矩阵乘法
    Tensor result = matMul(*this, other);

    // 如果需要自动微分，记录操作
    if (need_grad) {
        AutoDiff* ctx = AutoDiffContext::current();
        if (ctx) {
            // 设置梯度需求
            result.requires_grad(true);

            // 记录矩阵乘法操作
            std::vector<Tensor*> inputs = {const_cast<Tensor*>(this), const_cast<Tensor*>(&other)};
            ctx->defer_record(result.id(), op::MatMul, inputs);
            ctx->commit_record(result);
        }
    }

    return result;
}

// 修复sin函数
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

    // 使用新的延迟记录机制
    result.requires_grad(this->requires_grad());

    AutoDiff* ctx = AutoDiffContext::current();
    if (ctx) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this)};
        ctx->defer_record(result.id(), op::Sin, inputs);
        ctx->commit_record(result);
    }

    return result;
}

// 修复cos函数
Tensor Tensor::cos() const {
    Tensor result(ShapeTag{}, _shape, _dtype, _device);
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

    // 使用新的延迟记录机制
    result.requires_grad(this->requires_grad());

    AutoDiff* ctx = AutoDiffContext::current();
    if (ctx) {
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this)};
        ctx->defer_record(result.id(), op::Cos, inputs);
        ctx->commit_record(result);
    }

    return result;
}
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

Tensor Tensor::operator-(float scalar) const {
   Tensor result(ShapeTag{}, _shape, _dtype, _device);
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

   return result;
}

Tensor Tensor::sum(const std::vector<int>& dims, bool keepdim) const {
    std::cout << ">>> 进入 sum 函数, dims.size(): " << dims.size() << std::endl;

    // 处理全局求和（dims 为空）
    if (dims.empty()) {
        std::cout << ">>> 执行全局求和" << std::endl;

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

        // 关键修复：正确传播梯度需求
        result.requires_grad(this->requires_grad());
        std::cout << ">>> sum结果梯度需求: " << result.requires_grad() << std::endl;

        // 记录自动微分操作
        AutoDiff* ctx = AutoDiffContext::current();
        if (ctx) {
            std::cout << ">>> 记录sum操作" << std::endl;
            std::vector<Tensor*> inputs = {const_cast<Tensor*>(this)};
            ctx->defer_record(result.id(), op::Sum, inputs);
            ctx->commit_record(result);
        }

        std::cout << "<<< 离开sum函数" << std::endl;
        return result;
    }

    // 处理指定维度求和
    std::cout << ">>> 执行指定维度求和" << std::endl;

    // 1. 计算输出形状
    std::vector<size_t> result_shape = _shape;
    for (int dim : dims) {
        int actual_dim = (dim < 0) ? dim + static_cast<int>(_shape.size()) : dim;
        if (actual_dim < 0 || actual_dim >= static_cast<int>(_shape.size())) {
            throw std::runtime_error("Invalid dimension in sum");
        }
        result_shape[actual_dim] = 1;
    }

    if (!keepdim) {
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < result_shape.size(); ++i) {
            bool is_sum_dim = std::find(dims.begin(), dims.end(), static_cast<int>(i)) != dims.end();
            if (!is_sum_dim) {
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

        for (size_t i = 0; i < numel(); ++i) {
            size_t result_index = 0;
            size_t stride = 1;
            size_t temp = i;

            for (int d = static_cast<int>(_shape.size()) - 1; d >= 0; --d) {
                size_t dim_size = _shape[d];
                size_t coord = temp % dim_size;
                temp /= dim_size;

                if (std::find(dims.begin(), dims.end(), d) != dims.end()) {
                    coord = 0;
                }

                result_index += coord * stride;
                stride *= (d < static_cast<int>(result_shape.size())) ? result_shape[d] : 1;
            }

            dst[result_index] += src[i];
        }
        break;
    }
    case DType::kDouble: {
        const double* src = data<double>();
        double* dst = result.data<double>();

        for (size_t i = 0; i < numel(); ++i) {
            size_t result_index = 0;
            size_t stride = 1;
            size_t temp = i;

            for (int d = static_cast<int>(_shape.size()) - 1; d >= 0; --d) {
                size_t dim_size = _shape[d];
                size_t coord = temp % dim_size;
                temp /= dim_size;

                if (std::find(dims.begin(), dims.end(), d) != dims.end()) {
                    coord = 0;
                }

                result_index += coord * stride;
                stride *= (d < static_cast<int>(result_shape.size())) ? result_shape[d] : 1;
            }

            dst[result_index] += src[i];
        }
        break;
    }
    case DType::kInt: {
        const int32_t* src = data<int32_t>();
        int32_t* dst = result.data<int32_t>();

        for (size_t i = 0; i < numel(); ++i) {
            size_t result_index = 0;
            size_t stride = 1;
            size_t temp = i;

            for (int d = static_cast<int>(_shape.size()) - 1; d >= 0; --d) {
                size_t dim_size = _shape[d];
                size_t coord = temp % dim_size;
                temp /= dim_size;

                if (std::find(dims.begin(), dims.end(), d) != dims.end()) {
                    coord = 0;
                }

                result_index += coord * stride;
                stride *= (d < static_cast<int>(result_shape.size())) ? result_shape[d] : 1;
            }

            dst[result_index] += src[i];
        }
        break;
    }
    case DType::kLong: {
        const int64_t* src = data<int64_t>();
        int64_t* dst = result.data<int64_t>();

        for (size_t i = 0; i < numel(); ++i) {
            size_t result_index = 0;
            size_t stride = 1;
            size_t temp = i;

            for (int d = static_cast<int>(_shape.size()) - 1; d >= 0; --d) {
                size_t dim_size = _shape[d];
                size_t coord = temp % dim_size;
                temp /= dim_size;

                if (std::find(dims.begin(), dims.end(), d) != dims.end()) {
                    coord = 0;
                }

                result_index += coord * stride;
                stride *= (d < static_cast<int>(result_shape.size())) ? result_shape[d] : 1;
            }

            dst[result_index] += src[i];
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for sum");
    }

    // 关键修复：正确传播梯度需求
    result.requires_grad(this->requires_grad());
    std::cout << ">>> sum结果梯度需求: " << result.requires_grad() << std::endl;

    // 记录自动微分操作
    AutoDiff* ctx = AutoDiffContext::current();
    if (ctx) {
        std::cout << ">>> 记录sum操作" << std::endl;
        std::vector<Tensor*> inputs = {const_cast<Tensor*>(this)};
        ctx->defer_record(result.id(), op::Sum, inputs);
        ctx->commit_record(result);
    }

    std::cout << "<<< 离开sum函数" << std::endl;
    return result;
}

Tensor grad(const Tensor& t) {
    AutoDiff* ctx = AutoDiffContext::current();
    if (!ctx) {
        std::cout << "!!! grad: 没有活动的AutoDiff上下文" << std::endl;
        return Tensor();
    }

    std::cout << ">>> grad - 开始, 目标ID: " << t.id() << std::endl;
    Tensor result = ctx->get_grad(&t);
    std::cout << "<<< grad - 完成, 结果: " << (result.empty() ? "空" : "有数据") << std::endl;
    return result;
}

// 修复 backward 函数 - 为第二个参数提供默认值
void backward(Tensor& root, Tensor grad_output = Tensor(1.0f)) {
    AutoDiff* ctx = AutoDiffContext::current();
    if (!ctx) {
        throw std::runtime_error("No active AutoDiff context");
    }
    ctx->backward(root, grad_output);
}

inline void Tensor::computeStrides() {
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

inline size_t Tensor::computeStorageIndex(std::initializer_list<size_t> indices) const {
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

inline void Tensor::clear_storage() {
    std::cout << ">>> Tensor::clear_storage - 开始, ID: " << tensor_id_ << std::endl;

    // 直接重置存储，不创建新对象
    _storage = Storage();  // 调用默认构造函数创建空Storage
    _shape.clear();
    _strides.clear();
    _storage_offset = 0;

    std::cout << "<<< Tensor::clear_storage - 完成" << std::endl;
}

inline bool Tensor::is_cleared() const {
    return _storage.empty() && _shape.empty();
}

inline void Tensor::debug_info_detailed(const std::string &name) const {
    std::cout << "=== 张量详细信息 [" << (name.empty() ? "Tensor" : name) << "] ===" << std::endl;
    std::cout << "ID: " << tensor_id_ << std::endl;
    std::cout << "requires_grad: " << (_requires_grad ? "true" : "false") << std::endl;
    std::cout << "record_committed: " << (record_committed_ ? "true" : "false") << std::endl;
    std::cout << "形状: [";
    for (size_t i = 0; i < _shape.size(); ++i) {
        std::cout << _shape[i];
        if (i < _shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "步幅: [";
    for (size_t i = 0; i < _strides.size(); ++i) {
        std::cout << _strides[i];
        if (i < _strides.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "数据类型: " << dtypeToString(_dtype) << std::endl;
    std::cout << "元素数量: " << numel() << std::endl;
    std::cout << "==============================" << std::endl;
}

// ======================= Tensor 成员函数实现（在文件末尾） =======================
void Tensor::commit_pending_record() {
    // 只有在有未提交记录且不在析构过程中时才提交
    if (!record_committed_ && tensor_id_ != 0) {
        AutoDiff* ctx = AutoDiffContext::current();
        if (ctx) {
            std::cout << ">>> Tensor::commit_pending_record - 开始, ID: " << tensor_id_ << std::endl;
            ctx->commit_record(*this);
            record_committed_ = true;
            std::cout << "<<< Tensor::commit_pending_record - 完成" << std::endl;
        }
    }
}

inline Tensor::Tensor(): tensor_id_(global_tensor_id++), _storage_offset(0),
                         _device(DeviceType::kCPU), _dtype(DType::kFloat) {
    std::cout << ">>> Tensor默认构造, ID: " << tensor_id_ << std::endl;
    computeStrides();
    _storage = Storage(numel(), _dtype, _device);
}

void Tensor::requires_grad(bool key) {
    std::cout << ">>> Tensor::requires_grad - 开始, ID: " << tensor_id_
              << ", 新值: " << key << ", 原值: " << _requires_grad << std::endl;

    // 先设置标志
    _requires_grad = key;

    // 确保存储空间
    if (key && _storage.empty() && numel() > 0) {
        std::cout << ">>> 为张量 " << tensor_id_ << " 分配存储空间" << std::endl;
        _storage = Storage(numel(), _dtype, _device);
    }

    // 只有在有有效ID时才通知AutoDiff
    if (tensor_id_ != 0) {
        AutoDiff* ctx = AutoDiffContext::current();
        if (ctx) {
            std::cout << ">>> 通知AutoDiff更新梯度需求" << std::endl;
            ctx->update_requires_grad(*this, key);
        }
    }

    std::cout << "<<< Tensor::requires_grad - 完成" << std::endl;
}

// ==================== 统一矩阵乘法接口实现 ====================

// 矩阵乘法性能分析器
class MatMulProfiler {
private:
    static std::map<std::string, std::vector<double>> performance_data_;
    static bool profiling_enabled_;

public:
    static void enable_profiling(bool enable = true) {
        profiling_enabled_ = enable;
    }

    static void record_performance(const std::string& algorithm,
                                 size_t m, size_t n, size_t k,
                                 double time_ms) {
        if (!profiling_enabled_) return;

        std::string key = algorithm + "_" + std::to_string(m) + "x" +
                         std::to_string(n) + "x" + std::to_string(k);
        performance_data_[key].push_back(time_ms);
    }

    static void print_statistics() {
        if (!profiling_enabled_) return;

        std::cout << "\n=== 矩阵乘法性能统计 ===" << std::endl;
        for (const auto& pair : performance_data_) {
            const auto& times = pair.second;
            if (times.empty()) continue;

            double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            double min_time = *std::min_element(times.begin(), times.end());
            double max_time = *std::max_element(times.begin(), times.end());

            std::cout << pair.first << ": "
                      << "平均=" << avg_time << "ms, "
                      << "最小=" << min_time << "ms, "
                      << "最大=" << max_time << "ms, "
                      << "次数=" << times.size() << std::endl;
        }
    }

    static void clear_statistics() {
        performance_data_.clear();
    }
};

// 矩阵乘法统一接口类
class UnifiedMatMul {
private:
    // 性能计时器
    template<typename Func>
    static double measure_time(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    // 分析矩阵特征
    struct MatrixProfile {
        size_t m, n, k;
        bool is_square;
        bool is_small;
        bool is_large;
        double aspect_ratio;
        size_t total_elements;

        MatrixProfile(size_t rows, size_t cols, size_t inner_dim)
            : m(rows), n(cols), k(inner_dim) {
            is_square = (m == n && n == k);
            is_small = (m <= MatMulConfig::SMALL_MATRIX_THRESHOLD &&
                       n <= MatMulConfig::SMALL_MATRIX_THRESHOLD &&
                       k <= MatMulConfig::SMALL_MATRIX_THRESHOLD);
            is_large = (m >= MatMulConfig::STRASSEN_THRESHOLD ||
                       n >= MatMulConfig::STRASSEN_THRESHOLD ||
                       k >= MatMulConfig::STRASSEN_THRESHOLD);
            aspect_ratio = static_cast<double>(std::max({m, n, k})) /
                          static_cast<double>(std::min({m, n, k}));
            total_elements = m * n * k;
        }
    };

    // 自动选择最佳算法
    static MatMulStrategy select_algorithm(const MatrixProfile& profile) {
        // 小矩阵：使用朴素算法
        if (profile.is_small) {
            return MatMulStrategy::NAIVE;
        }

        // 大矩阵且接近正方形：使用Strassen算法
        if (profile.is_large && profile.is_square && profile.aspect_ratio < 2.0) {
            return MatMulStrategy::STRASSEN;
        }

        // 中等大小矩阵：使用分块算法
        if (profile.total_elements > 10000) {
            return MatMulStrategy::BLOCKED;
        }

        // 默认使用朴素算法
        return MatMulStrategy::NAIVE;
    }

    // 执行矩阵乘法并记录性能
    static Tensor execute_with_profiling(const Tensor& a, const Tensor& b,
                                       MatMulStrategy strategy) {
        MatrixProfile profile(a.shape()[0], b.shape()[1], a.shape()[1]);

        Tensor result;
        double time_ms = 0.0;

        switch (strategy) {
            case MatMulStrategy::NAIVE:
                time_ms = measure_time([&]() {
                    result = matMul_naive(const_cast<Tensor&>(a), const_cast<Tensor&>(b));
                });
                MatMulProfiler::record_performance("NAIVE", profile.m, profile.n, profile.k, time_ms);
                break;

            case MatMulStrategy::BLOCKED:
                time_ms = measure_time([&]() {
                    result = matMul_blocked(a, b);
                });
                MatMulProfiler::record_performance("BLOCKED", profile.m, profile.n, profile.k, time_ms);
                break;

            case MatMulStrategy::STRASSEN:
                time_ms = measure_time([&]() {
                    result = matMul_recursive(a, b);
                });
                MatMulProfiler::record_performance("STRASSEN", profile.m, profile.n, profile.k, time_ms);
                break;

            case MatMulStrategy::OPTIMIZED:
                // 使用最优算法组合
                result = execute_optimized_matmul(a, b, profile);
                break;

            default:
                throw std::runtime_error("Unknown matrix multiplication strategy");
        }

        if (MatMulConfig::ENABLE_PROFILING) {
            std::cout << "矩阵乘法 [" << profile.m << "x" << profile.k << "] * ["
                      << profile.k << "x" << profile.n << "] = ["
                      << profile.m << "x" << profile.n << "] "
                      << "算法=" << strategy_to_string(strategy)
                      << " 时间=" << time_ms << "ms" << std::endl;
        }

        return result;
    }

    // 最优算法组合
    static Tensor execute_optimized_matmul(const Tensor& a, const Tensor& b,
                                         const MatrixProfile& profile) {
        // 根据矩阵特征选择最优算法
        MatMulStrategy strategy = select_algorithm(profile);

        // 对于非常大的矩阵，可能需要特殊处理
        if (profile.total_elements > 1000000) {
            // 可以考虑并行化或GPU加速
            std::cout << "检测到大矩阵乘法，建议考虑并行化优化" << std::endl;
        }

        return execute_with_profiling(a, b, strategy);
    }

    // 策略转字符串
    static std::string strategy_to_string(MatMulStrategy strategy) {
        switch (strategy) {
            case MatMulStrategy::AUTO: return "AUTO";
            case MatMulStrategy::NAIVE: return "NAIVE";
            case MatMulStrategy::BLOCKED: return "BLOCKED";
            case MatMulStrategy::STRASSEN: return "STRASSEN";
            case MatMulStrategy::OPTIMIZED: return "OPTIMIZED";
            default: return "UNKNOWN";
        }
    }

public:
    // 统一的矩阵乘法接口 - 自动选择算法
    static Tensor matmul(const Tensor& a, const Tensor& b) {
        // 输入验证
        if (a.dim() < 2 || b.dim() < 2) {
            throw std::runtime_error("矩阵乘法要求至少2维张量");
        }

        if (a.shape()[1] != b.shape()[0]) {
            throw std::runtime_error("矩阵维度不匹配: A的列数(" +
                                   std::to_string(a.shape()[1]) +
                                   ") != B的行数(" +
                                   std::to_string(b.shape()[0]) + ")");
        }

        // 分析矩阵特征
        MatrixProfile profile(a.shape()[0], b.shape()[1], a.shape()[1]);

        // 自动选择算法
        MatMulStrategy strategy = select_algorithm(profile);

        // 执行矩阵乘法
        return execute_with_profiling(a, b, strategy);
    }

    // 配置接口
    static void configure_profiling(bool enable) {
        MatMulProfiler::enable_profiling(enable);
    }

    static void print_config() {
        std::cout << "\n=== 矩阵乘法配置 ===" << std::endl;
        std::cout << "分块大小阈值: " << MatMulConfig::BLOCK_SIZE_THRESHOLD << std::endl;
        std::cout << "Strassen阈值: " << MatMulConfig::STRASSEN_THRESHOLD << std::endl;
        std::cout << "小矩阵阈值: " << MatMulConfig::SMALL_MATRIX_THRESHOLD << std::endl;
        std::cout << "性能分析: " << (MatMulConfig::ENABLE_PROFILING ? "启用" : "禁用") << std::endl;
        std::cout << "缓存优化: " << (MatMulConfig::ENABLE_CACHE_OPTIMIZATION ? "启用" : "禁用") << std::endl;
    }
};

// 静态成员定义
std::map<std::string, std::vector<double>> MatMulProfiler::performance_data_;
bool MatMulProfiler::profiling_enabled_ = MatMulConfig::ENABLE_PROFILING;

//TENSOR_CPPM
//TENSOR_CPPM
#endif //TENSOR_H
