/**
 * @file Tensor.h
 * @brief 张量类和相关操作的定义
 * @author GhostFace, Beapoe
 * @date 2025/12/21
 * @version v3.1
 * @details 定义了张量类(Tensor)及其相关操作，包括存储管理、自动微分、各种数学运算等。
 */
#ifndef TENSOR_H
#define TENSOR_H

// includes
#include <algorithm>
#include <atomic>
#include <cstddef>
#include "Ctorch_Error.h"
#include "Storage.h"
#ifdef __APPLE__
#include <Accelerate/Accelerate.h> // 使用Apple的BLAS实现
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
#include "Ctools.h"
// #include<omp.h>   !!!目前不确定在哪些机器上需要这个头文件，如果编译错误，可以尝试加上
// ======================= 前向声明 =======================
class Tensor;
class Storage;
class Node;
// ======================= 存储类 (Storage) =======================

/**
 * @class Storage
 * @brief Tensor 的存储底层
 * @authors GhostFace Beapoe
 */

// ======================= 张量类 (Tensor) =======================
struct ShapeTag {}; // 此处结构体为了使编译器区分构造函数

// 引入自动微分系统
#include "AutoDiff.h"

class GradFn;

/**
 * @brief 矩阵乘法函数
 * @param a 第一个矩阵张量
 * @param b 第二个矩阵张量
 * @return 矩阵乘法结果
 */
Tensor matMul(const Tensor &a, const Tensor &b);

/**
 * @class Tensor
 * @brief 张量类，用于表示和操作多维数组
 * @details 张量是自动微分系统的核心数据结构，支持各种数学运算和自动微分
 */
class Tensor {
  private:
    /**
     * @var _node 与该张量相关的Node
     */
    std::weak_ptr<Node> _node;

    /**
     * @var global_tensor_id
     * @brief 全局张量ID计数器
     */
    static std::atomic<size_t> global_tensor_id;

    /**
     * @var tensor_id_
     * @brief 张量的唯一标识符
     */
    size_t tensor_id_;

    /**
     * @var record_committed_
     * @brief 记录是否已提交
     */
    bool record_committed_ = false;

    /**
     * @var _requires_grad
     * @brief 是否参与自动微分计算，默认不参与
     */
    bool _requires_grad = false;

    /**
     * @var _strides
     * @brief 每个维度的步幅
     */
    std::vector<size_t> _strides;

    /**
     * @var _storage_offset
     * @brief 存储中的起始偏移量
     */
    size_t _storage_offset;

    /**
     * @var _device
     * @brief 张量所在的设备
     */
    DeviceType _device;

    /**
     * @var _dtype
     * @brief 张量元素的数据类型
     */
    DType _dtype;

    /**
     * @var _storage
     * @brief 存储张量数据的对象
     */
    Storage _storage;
    // ======================= 内部辅助函数 =======================

    /**
     * @brief 计算步幅 (基于行优先顺序)
     */
    void computeStrides();

    /**
     * @brief 计算存储中的索引
     * @param indices 多维索引
     * @return 存储中的一维索引
     */
    size_t computeStorageIndex(std::initializer_list<size_t> indices) const;

    /**
     * @brief 检查数据类型是否匹配
     * @tparam T 期望的数据类型
     * @throw std::runtime_error 如果数据类型不匹配
     */
    template <typename T> void checkDType() const;

    /**
     * @brief 通用逐元素操作
     * @tparam T 数据类型
     * @tparam Op 操作类型
     * @param result 结果张量
     * @param a 输入张量a
     * @param b 输入张量b
     * @param op 操作函数
     */
    template <typename T, typename Op>
    void elementwiseOp(Tensor &result, const Tensor &a, const Tensor &b, Op op) const;

    /**
     * @brief 支持广播的逐元素操作
     * @tparam T 数据类型
     * @tparam Op 操作类型
     * @param result 结果张量
     * @param a 输入张量a
     * @param b 输入张量b
     * @param bc 广播结果
     * @param op 操作函数
     */
    template <typename T, typename Op>
    void broadcast_elementwise_op(Tensor &result, const Tensor &a, const Tensor &b,
                                  const BroadCastResult &bc, Op op) const;

    /**
     * @brief 递归打印张量内容
     * @tparam T 数据类型
     * @param os 输出流
     * @param dim 当前维度
     * @param indices 当前索引
     */
    template <typename T>
    void printRecursive(std::ostream &os, size_t dim, std::vector<size_t> indices) const;

  protected:
    /**
     * @var _shape
     * @brief 张量的维度大小
     */
    std::vector<size_t> _shape;

  public:
    /**
     * @brief 清空存储的方法，避免创建新Tensor
     */
    void clear_storage();

    /**
     * @brief 判断是否为空的辅助方法
     * @return 如果存储为空返回true，否则返回false
     */
    bool is_cleared() const;

    /**
     * @brief 增强调试信息
     * @param name 张量名称，用于调试输出
     */
    void debug_info_detailed(const std::string &name = "") const;

    /**
     * @brief 提交未完成的记录
     */
    void commit_pending_record();

    /**
     * @brief 检查是否有待处理记录
     * @return 如果有待处理记录返回true，否则返回false
     */
    bool has_pending_record() { return !record_committed_; }

    /**
     * @brief 设置张量ID（仅用于自动微分系统）
     * @param id 新的张量ID
     */
    void set_id(size_t id) { tensor_id_ = id; }
    // ======================= 构造和析构 =======================

    /**
     * @brief 默认构造函数
     */
    Tensor();

    /**
     * @brief 标量构造函数
     * @param value 标量值
     */
    Tensor(float value)
        : tensor_id_(global_tensor_id++), _storage_offset(0), _device(DeviceType::kCPU),
          _dtype(DType::kFloat) {
        _shape = {};
        std::ostringstream oss;
        oss << ">>> Tensor标量构造, ID: " << tensor_id_ << ", 值: " << value;
        std::string msg = oss.str();
        Ctorch_Error::trace(ErrorPlatform::kCPU, msg);
        computeStrides();
        _storage = Storage(1, _dtype, _device);
        if (_storage.data<float>()) {
            *_storage.data<float>() = value;
            std::ostringstream oss;
            oss << ">>> 标量Tensor设置完成, 存储值: " << *_storage.data<float>();
            std::string msg = oss.str();
            Ctorch_Error::trace(ErrorPlatform::kCPU, msg);
        } else {
            Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kCPU, ErrorType::MEMORY,
                              "!!! 错误: 无法分配存储");
        }
    }

    /**
     * @brief 初始化列表构造函数
     * @param values 初始化列表
     */
    Tensor(std::initializer_list<float> values)
        : tensor_id_(global_tensor_id++), _storage_offset(0), _device(DeviceType::kCPU),
          _dtype(DType::kFloat) {
        _shape = {values.size()};
        computeStrides();
        _storage = Storage(values.begin(), values.size(), _dtype, _device);
    }

    /**
     * @brief 形状标签构造函数
     * @param tag 形状标签
     * @param shape 张量形状
     * @param dtype 数据类型
     * @param device 设备类型
     * @param zero_init 是否零初始化
     */
    Tensor(ShapeTag /*tag*/, const std::vector<size_t> &shape, DType dtype = DType::kFloat,
           DeviceType device = DeviceType::kCPU, bool zero_init = true)
        : tensor_id_(global_tensor_id++), _storage_offset(0), _device(device), _dtype(dtype) {
        _shape = shape;
        computeStrides();
        _storage = Storage(numel(), _dtype, _device);
        if (zero_init)
            zero();
    }

    /**
     * @brief 1D张量构造函数
     * @param size 张量大小
     * @param dtype 数据类型
     * @param device 设备类型
     * @param zero_init 是否零初始化
     */
    Tensor(size_t size, DType dtype = DType::kFloat, DeviceType device = DeviceType::kCPU,
           bool zero_init = true)
        : tensor_id_(global_tensor_id++), _storage_offset(0), _device(device), _dtype(dtype) {
        _shape = {size};
        computeStrides();
        _storage = Storage(size, _dtype, _device);
        if (zero_init)
            zero();
    }

    /**
     * @brief 拷贝构造函数
     * @param other 被拷贝的张量
     * @details 新对象会分配新的张量ID和深拷贝存储
     */
    Tensor(const Tensor &other)
        : tensor_id_(global_tensor_id++), record_committed_(false),
          _requires_grad(other._requires_grad), _strides(other._strides),
          _storage_offset(other._storage_offset), _device(other._device), _dtype(other._dtype),
          _storage(other._storage.clone()), // 注意：这里调用了clone()
          _shape(other._shape) {
        // std::cout << ">>> Tensor拷贝构造, 新ID: " << tensor_id_ << ", 原ID: " << other.tensor_id_
        // << std::endl;
        std::ostringstream oss;
        oss << ">>> Tensor拷贝构造, 新ID: " << tensor_id_ << ", 原ID: " << other.tensor_id_;
        std::string msg = oss.str();
        Ctorch_Error::trace(ErrorPlatform::kCPU, msg);
    }

    /**
     * @brief 赋值操作符
     * @param other 被赋值的张量
     * @return 引用当前对象
     * @details 深拷贝存储并分配新的张量ID
     */
    Tensor &operator=(const Tensor &other) {
        if (this != &other) {
            commit_pending_record();

            tensor_id_        = global_tensor_id++;
            _shape            = other._shape;
            _strides          = other._strides;
            _storage_offset   = other._storage_offset;
            _device           = other._device;
            _dtype            = other._dtype;
            _storage          = other._storage.clone(); // 深拷贝存储
            _requires_grad    = other._requires_grad;
            record_committed_ = false;
        }
        return *this;
    }

    /**
     * @brief 移动构造函数
     * @param other 被移动的张量
     * @details 移动构造后，原对象的tensor_id变为0，避免冲突
     */
    Tensor(Tensor &&other) noexcept
        : tensor_id_(other.tensor_id_), record_committed_(other.record_committed_),
          _requires_grad(other._requires_grad), _strides(std::move(other._strides)),
          _storage_offset(other._storage_offset), _device(other._device), _dtype(other._dtype),
          _storage(std::move(other._storage)), _shape(std::move(other._shape)) {
        // 移动构造后，原对象的tensor_id变为0，避免冲突
        other.tensor_id_        = 0;
        other.record_committed_ = false;
    }

    /**
     * @brief 移动赋值操作符
     * @param other 被移动的张量
     * @return 引用当前对象
     * @details 移动赋值后，原对象的tensor_id变为0，避免冲突
     */
    Tensor &operator=(Tensor &&other) noexcept {
        if (this != &other) {
            commit_pending_record();

            tensor_id_        = other.tensor_id_;
            _shape            = std::move(other._shape);
            _strides          = std::move(other._strides);
            _storage_offset   = other._storage_offset;
            _device           = other._device;
            _dtype            = other._dtype;
            _storage          = std::move(other._storage);
            _requires_grad    = other._requires_grad;
            record_committed_ = other.record_committed_;

            // 移动赋值后，原对象的tensor_id变为0，避免冲突
            other.tensor_id_        = 0;
            other.record_committed_ = false;
        }
        return *this;
    }

    /**
     * @brief 析构函数
     * @details 如果是计算图的根节点，会清理计算图相关资源
     */
    ~Tensor() {
        // 如果是计算图的根节点，清理计算图
        if (record_committed_ && AutoDiffContext::current()) {
            // 注意：这里不再直接调用AutoDiff的清理方法，避免循环依赖
        }
    }

    /**
     * @brief 设置梯度需求
     * @param key 是否需要梯度
     */
    void requires_grad(bool key);

    /**
     * @brief 获取是否需要梯度
     * @return 如果需要梯度返回true，否则返回false
     */
    bool requires_grad() const { return _requires_grad; }

    /**
     * @brief 设置梯度需求
     * @param requires_grad 是否需要梯度
     */
    void set_requires_grad(bool requires_grad) { _requires_grad = requires_grad; }

    // ======================= 访问器 =======================

    /**
     * @brief 获取张量ID
     * @return 张量的唯一标识符
     */
    [[nodiscard]] size_t id() const { return tensor_id_; }

    /**
     * @brief 获取张量的形状
     * @return 张量的形状向量
     */
    [[nodiscard]] const std::vector<size_t> &shape() const;

    /**
     * @brief 获取张量的大小（元素总数量）
     * @return 张量的元素总数量
     */
    [[nodiscard]] size_t numel() const;

    /**
     * @brief 获取张量的步幅
     * @return 张量的步幅向量
     */
    [[nodiscard]] const std::vector<size_t> &strides() const { return _strides; }

    /**
     * @brief 获取张量的数据类型
     * @return 张量的数据类型
     */
    [[nodiscard]] DType dtype() const { return _dtype; }

    /**
     * @brief 获取张量所在的设备
     * @return 张量所在的设备
     */
    [[nodiscard]] DeviceType device() const { return _device; }

    /**
     * @brief 获取张量的存储
     * @return 张量的存储对象
     */
    [[nodiscard]] Storage &storage() { return _storage; }

    /**
     * @brief 获取常量存储
     * @return 常量存储对象
     */
    [[nodiscard]] const Storage &storage() const { return _storage; }

    /**
     * @brief 获取张量的存储偏移量
     * @return 存储中的起始偏移量
     */
    [[nodiscard]] size_t storage_offset() const { return _storage_offset; }

    /**
     * @brief 检查存储偏移量是否有效
     * @return 如果存储偏移量有效返回true，否则返回false
     */
    [[nodiscard]] bool check_storage_offset() const;

    /**
     * @brief 检查索引是否在边界内
     * @param indices 多维索引
     * @return 如果索引在边界内返回true，否则返回false
     */
    [[nodiscard]] bool check_index_bounds(const std::vector<size_t> &indices) const;

    /**
     * @brief 获取张量的维度
     * @return 张量的维度数量
     */
    [[nodiscard]] int dim() const { return static_cast<int>(_shape.size()); }

    /**
     * @brief 获取张量的维度大小
     * @return 张量的维度大小向量
     */
    [[nodiscard]] const std::vector<size_t> &sizes() const { return _shape; }

    /**
     * @brief 获取张量指定维度的大小
     * @param dim 维度索引
     * @return 指定维度的大小
     */
    [[nodiscard]] size_t size(int dim) const;

    /**
     * @brief 获取张量指定维度的步幅
     * @param dim 维度索引
     * @return 指定维度的步幅
     */
    [[nodiscard]] size_t stride(int dim) const;

    // ======================= 数据访问 =======================

    /**
     * @brief 获取常量原始数据指针
     * @tparam T 数据类型
     * @return 常量数据指针
     * @throw std::runtime_error 如果数据类型不匹配或存储偏移量无效
     */
    template <typename T> [[nodiscard]] const T *data() const {
        checkDType<T>();
        if (!check_storage_offset()) {
            Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
                                         "张量存储偏移量无效");
        }
        return _storage.data<T>() + _storage_offset;
    }

    /**
     * @brief 获取可修改的原始数据指针
     * @tparam T 数据类型
     * @return 可修改的数据指针
     * @throw std::runtime_error 如果数据类型不匹配或存储偏移量无效
     */
    template <typename T> T *data() {
        checkDType<T>();
        if (!check_storage_offset()) {
            Ctorch_Error::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
                                         "张量存储偏移量无效");
        }
        return _storage.data<T>() + _storage_offset;
    }

    /**
     * @brief 获取标量值
     * @tparam T 数据类型
     * @return 标量值
     * @throw std::runtime_error 如果张量不是标量或数据类型不匹配
     */
    template <typename T> [[nodiscard]] T item() const;

    /**
     * @brief 索引操作
     * @param index 索引值
     * @return 索引结果张量
     * @throw std::runtime_error 如果索引操作不支持
     */
    Tensor operator[](size_t index) const;

    // ======================= 操作 =======================

    /**
     * @brief 创建一个新的张量，形状相同，数据不同
     * @return 新的张量
     */
    Tensor clone() const;

    /**
     * @brief 将张量转换为指定数据类型
     * @param dtype 目标数据类型
     * @return 转换后的数据类型
     */
    Tensor to(DType dtype) const;

    /**
     * @brief 转置张量
     * @param dim0 第一个维度
     * @param dim1 第二个维度
     * @return 转置后的张量
     * @throw std::runtime_error 如果转置操作不支持
     */
    Tensor transpose(int dim0, int dim1) const;

    /**
     * @brief 转置张量（二维情况）
     * @return 转置后的张量
     */
    Tensor t() const;

    /**
     * @brief 重塑张量形状
     * @param new_shape 新的形状
     * @return 重塑后的张量
     */
    Tensor reshape(std::initializer_list<size_t> new_shape) const;

    /**
     * @brief 重塑张量形状
     * @param new_shape 新的形状
     * @return 重塑后的张量
     */
    Tensor reshape(const std::vector<size_t> &new_shape) const;

    /**
     * @brief 广播张量到指定形状
     * @param shape 目标形状
     * @return 广播后的张量
     */
    Tensor broadcast_to(const std::vector<size_t> &shape) const;

    /**
     * @brief 零初始化张量
     */
    void zero();

    /**
     * @brief 一初始化张量
     */
    void ones();

    /**
     * @brief 随机初始化张量
     */
    void rand();

    //  ======================= 运算 =======================

    /**
     * @brief 矩阵乘法
     * @param other 另一个矩阵张量
     * @return 矩阵乘法结果
     */
    Tensor matmul(const Tensor &other) const;

    /**
     * @brief 矩阵乘法操作符重载
     * @param other 另一个矩阵张量
     * @return 矩阵乘法结果
     */
    Tensor operator*(const Tensor &other) const;

    /**
     * @brief 加法操作符重载
     * @param other 另一个张量
     * @return 加法结果
     */
    Tensor operator+(const Tensor &other) const;

    /**
     * @brief 减法操作符重载
     * @param other 另一个张量
     * @return 减法结果
     */
    Tensor operator-(const Tensor &other) const;

    /**
     * @brief 除法操作符重载
     * @param other 另一个张量
     * @return 除法结果
     */
    Tensor operator/(const Tensor &other) const;

    /**
     * @brief 标量乘法操作符重载
     * @param scalar 标量值
     * @return 标量乘法结果
     */
    Tensor operator*(float scalar) const;

    /**
     * @brief 标量加法操作符重载
     * @param scalar 标量值
     * @return 标量加法结果
     */
    Tensor operator+(float scalar) const;

    /**
     * @brief 标量减法操作符重载
     * @param scalar 标量值
     * @return 标量减法结果
     */
    Tensor operator-(float scalar) const;

    /**
     * @brief 标量除法操作符重载
     * @param scalar 标量值
     * @return 标量除法结果
     */
    Tensor operator/(float scalar) const;

    /**
     * @brief 一元减操作符重载
     * @return 取反结果
     */
    Tensor operator-() const;

    /**
     * @brief 比较操作符重载：大于标量
     * @param scalar 标量值
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator>(float scalar) const;

    /**
     * @brief 比较操作符重载：小于标量
     * @param scalar 标量值
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator<(float scalar) const;

    /**
     * @brief 比较操作符重载：等于标量
     * @param scalar 标量值
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator==(float scalar) const;

    /**
     * @brief 比较操作符重载：大于等于标量
     * @param scalar 标量值
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator>=(float scalar) const;

    /**
     * @brief 比较操作符重载：小于等于标量
     * @param scalar 标量值
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator<=(float scalar) const;

    /**
     * @brief 比较操作符重载：不等于标量
     * @param scalar 标量值
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator!=(float scalar) const;

    /**
     * @brief 比较操作符重载：大于另一个张量
     * @param other 另一个张量
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator>(const Tensor &other) const;

    /**
     * @brief 比较操作符重载：小于另一个张量
     * @param other 另一个张量
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator<(const Tensor &other) const;

    /**
     * @brief 比较操作符重载：等于另一个张量
     * @param other 另一个张量
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator==(const Tensor &other) const;

    /**
     * @brief 比较操作符重载：大于等于另一个张量
     * @param other 另一个张量
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator>=(const Tensor &other) const;

    /**
     * @brief 比较操作符重载：小于等于另一个张量
     * @param other 另一个张量
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator<=(const Tensor &other) const;

    /**
     * @brief 比较操作符重载：不等于另一个张量
     * @param other 另一个张量
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator!=(const Tensor &other) const;

    /**
     * @brief 求和操作
     * @param dims 要求和的维度列表
     * @param keepdim 是否保持原维度
     * @return 求和结果张量
     */
    Tensor sum(const std::vector<int> &dims, bool keepdim = false) const;

    /**
     * @brief 求和操作（单个维度）
     * @param dim 要求和的维度
     * @param keepdim 是否保持原维度
     * @return 求和结果张量
     */
    Tensor sum(int dim, bool keepdim = false) const;

    /**
     * @brief 求和操作（所有维度）
     * @return 求和结果张量（标量）
     */
    Tensor sum() const;

    /**
     * @brief 求平均值操作
     * @param dims 要求平均值的维度列表
     * @param keepdim 是否保持原维度
     * @return 平均值结果张量
     */
    Tensor mean(const std::vector<int> &dims, bool keepdim = false) const;

    /**
     * @brief 求平均值操作（单个维度）
     * @param dim 要求平均值的维度
     * @param keepdim 是否保持原维度
     * @return 平均值结果张量
     */
    Tensor mean(int dim, bool keepdim = false) const;

    /**
     * @brief 求平均值操作（所有维度）
     * @return 平均值结果张量（标量）
     */
    Tensor mean() const;

    /**
     * @brief 求最大值操作
     * @param dims 要求最大值的维度列表
     * @param keepdim 是否保持原维度
     * @return 最大值结果张量
     */
    Tensor max(const std::vector<int> &dims, bool keepdim = false) const;

    /**
     * @brief 求最大值操作（单个维度）
     * @param dim 要求最大值的维度
     * @param keepdim 是否保持原维度
     * @return 最大值结果张量
     */
    Tensor max(int dim, bool keepdim = false) const;

    /**
     * @brief 求最大值操作（所有维度）
     * @return 最大值结果张量（标量）
     */
    Tensor max() const;

    /**
     * @brief 求最小值操作
     * @param dims 要求最小值的维度列表
     * @param keepdim 是否保持原维度
     * @return 最小值结果张量
     */
    Tensor min(const std::vector<int> &dims, bool keepdim = false) const;

    /**
     * @brief 求最小值操作（单个维度）
     * @param dim 要求最小值的维度
     * @param keepdim 是否保持原维度
     * @return 最小值结果张量
     */
    Tensor min(int dim, bool keepdim = false) const;

    /**
     * @brief 求最小值操作（所有维度）
     * @return 最小值结果张量（标量）
     */
    Tensor min() const;

    /**
     * @brief 求标准差操作
     * @param dims 要求标准差的维度列表
     * @param keepdim 是否保持原维度
     * @return 标准差结果张量
     */
    Tensor std(const std::vector<int> &dims, bool keepdim = false) const;

    /**
     * @brief 求标准差操作（单个维度）
     * @param dim 要求标准差的维度
     * @param keepdim 是否保持原维度
     * @return 标准差结果张量
     */
    Tensor std(int dim, bool keepdim = false) const;

    /**
     * @brief 求标准差操作（所有维度）
     * @return 标准差结果张量（标量）
     */
    Tensor std() const;

    /**
     * @brief 求方差操作
     * @param dims 要求方差的维度列表
     * @param keepdim 是否保持原维度
     * @return 方差结果张量
     */
    Tensor var(const std::vector<int> &dims, bool keepdim = false) const;

    /**
     * @brief 求方差操作（单个维度）
     * @param dim 要求方差的维度
     * @param keepdim 是否保持原维度
     * @return 方差结果张量
     */
    Tensor var(int dim, bool keepdim = false) const;

    /**
     * @brief 求方差操作（所有维度）
     * @return 方差结果张量（标量）
     */
    Tensor var() const;

    /**
     * @brief 求绝对值操作
     * @return 绝对值结果张量
     */
    Tensor abs() const;

    /**
     * @brief 求指数操作
     * @return 指数结果张量
     */
    Tensor exp() const;

    /**
     * @brief 求对数操作
     * @return 对数结果张量
     */
    Tensor log() const;

    /**
     * @brief 求平方根操作
     * @return 平方根结果张量
     */
    Tensor sqrt() const;

    /**
     * @brief 求平方操作
     * @return 平方结果张量
     */
    Tensor square() const;

    /**
     * @brief 求正弦操作
     * @return 正弦结果张量
     */
    Tensor sin() const;

    /**
     * @brief 求余弦操作
     * @return 余弦结果张量
     */
    Tensor cos() const;

    /**
     * @brief 求正切操作
     * @return 正切结果张量
     */
    Tensor tan() const;

    /**
     * @brief 求反正弦操作
     * @return 反正弦结果张量
     */
    Tensor asin() const;

    /**
     * @brief 求反余弦操作
     * @return 反余弦结果张量
     */
    Tensor acos() const;

    /**
     * @brief 求反正切操作
     * @return 反正切结果张量
     */
    Tensor atan() const;

    /**
     * @brief 求双曲正弦操作
     * @return 双曲正弦结果张量
     */
    Tensor sinh() const;

    /**
     * @brief 求双曲余弦操作
     * @return 双曲余弦结果张量
     */
    Tensor cosh() const;

    /**
     * @brief 求双曲正切操作
     * @return 双曲正切结果张量
     */
    Tensor tanh() const;

    /**
     * @brief 求ReLU操作
     * @return ReLU结果张量
     */
    Tensor relu() const;

    /**
     * @brief 求Cos操作
     * @return Cos 结果张量
     */
    Tensor dot(const Tensor &other) const;

    /**
     * @brief 求Leaky ReLU操作
     * @param negative_slope 负斜率值
     * @return Leaky ReLU结果张量
     */
    Tensor leaky_relu(float negative_slope = 0.01f) const;

    /**
     * @brief 求Sigmoid操作
     * @return Sigmoid结果张量
     */
    Tensor sigmoid() const;

    /**
     * @brief 求Softmax操作
     * @param dim 进行Softmax的维度
     * @return Softmax结果张量
     */
    Tensor softmax(int dim = -1) const;

    /**
     * @brief 求LogSoftmax操作
     * @param dim 进行LogSoftmax的维度
     * @return LogSoftmax结果张量
     */
    Tensor log_softmax(int dim = -1) const;

    /**
     * @brief 求交叉熵损失
     * @param target 目标张量
     * @return 交叉熵损失结果张量
     */
    Tensor cross_entropy(const Tensor &target) const;

    /**
     * @brief 求均方误差
     * @param target 目标张量
     * @return 均方误差结果张量
     */
    Tensor mse_loss(const Tensor &target) const;

    /**
     * @brief 求平均绝对误差
     * @param target 目标张量
     * @return 平均绝对误差结果张量
     */
    Tensor mae_loss(const Tensor &target) const;

    //  ======================= 自动微分 =======================

    /**
     * @brief 反向传播
     * @details 从当前张量开始，计算所有梯度
     */
    void backward() const;

    /**
     * @brief 反向传播（带有梯度输出）
     * @param grad_output 输出梯度
     * @details 从当前张量开始，使用指定的输出梯度计算所有梯度
     */
    void backward(const Tensor &grad_output) const;

    //  ======================= 统一矩阵乘法接口 =======================

    /**
     * @brief 统一矩阵乘法 - 自动选择算法并支持自动微分
     * @param other 另一个矩阵张量
     * @return 矩阵乘法结果
     */
    Tensor matmul_unified(const Tensor &other) const;

    // 矩阵乘法操作符重载 - 使用统一接口
    // 注意：这里不重载operator*，因为已经存在了，使用matmul_unified方法
    // 添加一个辅助函数来创建真正的空 Tensor
    // 保留设置方法

    std::weak_ptr<Node> getRelatedNode() const;

    Tensor view(std::initializer_list<size_t> shape);
};

/**
 * @brief 全局的backward函数，用于启动反向传播
 * @param root 根张量
 */
void backward(Tensor &root);

/**
 * @brief 全局的backward函数，用于启动反向传播（带有梯度输出）
 * @param root 根张量
 * @param grad_output 输出梯度
 */
void backward(Tensor &root, Tensor grad_output);

/**
 * @brief 全局的grad函数，用于获取张量的梯度
 * @param t 输入张量
 * @return 梯度张量
 */
Tensor grad(const Tensor &t);

// ======================= 标量操作符重载（右操作数） =======================

/**
 * @brief 标量加法操作符重载（右操作数）
 * @param scalar 标量值
 * @param tensor 张量
 * @return 加法结果
 */
Tensor operator+(float scalar, const Tensor &tensor);

/**
 * @brief 标量减法操作符重载（右操作数）
 * @param scalar 标量值
 * @param tensor 张量
 * @return 减法结果
 */
Tensor operator-(float scalar, const Tensor &tensor);

/**
 * @brief 标量乘法操作符重载（右操作数）
 * @param scalar 标量值
 * @param tensor 张量
 * @return 乘法结果
 */
Tensor operator*(float scalar, const Tensor &tensor);

/**
 * @brief 标量除法操作符重载（右操作数）
 * @param scalar 标量值
 * @param tensor 张量
 * @return 除法结果
 */
Tensor operator/(float scalar, const Tensor &tensor);

// ======================= 比较操作符重载（右操作数） =======================

/**
 * @brief 比较操作符重载：标量大于张量
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator>(float scalar, const Tensor &tensor);

/**
 * @brief 比较操作符重载：标量小于张量
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator<(float scalar, const Tensor &tensor);

/**
 * @brief 比较操作符重载：标量等于张量
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator==(float scalar, const Tensor &tensor);

/**
 * @brief 比较操作符重载：标量大于等于张量
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator>=(float scalar, const Tensor &tensor);

/**
 * @brief 比较操作符重载：标量小于等于张量
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator<=(float scalar, const Tensor &tensor);

/**
 * @brief 比较操作符重载：标量不等于张量
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator!=(float scalar, const Tensor &tensor);

// ======================= 输出操作符 =======================

/**
 * @brief 输出张量信息
 * @param os 输出流
 * @param tensor 张量
 * @return 输出流
 */
std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

/**
 * @brief 全局的matMul函数
 * @param a 第一个矩阵张量
 * @param b 第二个矩阵张量
 * @return 矩阵乘法结果
 */
Tensor matMul(const Tensor &a, const Tensor &b);

// ======================= 辅助函数 =======================

/**
 * @brief 计算两个张量的广播结果
 * @param a 第一个张量
 * @param tensor2 第二个张量
 * @return 广播结果
 */
BroadCastResult broadCast(const Tensor &a, const Tensor &tensor2);

#endif // TENSOR_H