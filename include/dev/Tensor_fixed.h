/*
 * Tensor.h
 * Created by Beapoe & GhostFace on 2025.7
 * Main Classes: Storage & Tensor
 */
// Dev:此文件并未使用cpp20的特性，仅做开发测试使用，后续会支持
// includes

#ifndef CTT_TENSOR_H
#define CTT_TENSOR_FIXED_H
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <functional>
#include <cstring>
#include <iomanip>

// ======================= 类型定义和枚举 =======================

/// 设备类型 - 定义张量存储的位置
enum class DeviceType {
    kCPU,    ///< 主存储器 (RAM)
    kCUDA,   ///< NVIDIA GPU (暂未实现)
    kMPS,    ///< Apple Silicon (暂未实现)
};

/// 数据类型 - 定义张量元素的类型
enum class DType {
    kFloat,  ///< 32位浮点数 (torch.float32)
    kDouble, ///< 64位浮点数 (torch.float64)
    kInt,    ///< 32位整数 (torch.int32)
    kLong,   ///< 64位整数 (torch.int64)
    kBool,   ///< 布尔值 (torch.bool)
};

// ======================= 辅助函数 =======================

/// 将数据类型转换为字符串表示
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

/// 存储类 - 管理张量的原始内存
class Storage {
public:
    // 构造函数：分配未初始化的内存
    Storage(size_t size, DType dtype, DeviceType device = DeviceType::kCPU)
            : size_(size), dtype_(dtype), device_(device),
              data_(size > 0 ? std::shared_ptr<char[]>(new char[size * dtypeSize(dtype)]) : nullptr) {}

    // 构造函数：从现有数据复制
    template <typename T>
    Storage(const T* data, size_t size, DType dtype, DeviceType device = DeviceType::kCPU)
            : Storage(size, dtype, device) {
        if (size > 0 && data_.get()) {
            std::memcpy(data_.get(), data, size * dtypeSize(dtype));
        }
    }

    // 默认拷贝构造函数和拷贝赋值运算符（使用shared_ptr，所以是浅拷贝）
    Storage(const Storage&) = default;
    Storage& operator=(const Storage&) = default;

    // 默认移动构造函数和移动赋值运算符
    Storage(Storage&&) = default;
    Storage& operator=(Storage&&) = default;

    Storage() : size_(0), dtype_(DType::kFloat), device_(DeviceType::kCPU) {}

    ~Storage() = default;

    /// 获取原始数据的类型化指针
    template <typename T>
    T* data() {
        if (size_ == 0 || !data_) return nullptr;
        checkDType<T>();
        return reinterpret_cast<T*>(data_.get());
    }

    /// 获取常量原始数据的类型化指针
    template <typename T>
    const T* data() const {
        if (size_ == 0 || !data_) return nullptr;
        checkDType<T>();
        return reinterpret_cast<const T*>(data_.get());
    }

    /// 获取存储中的元素数量
    size_t size() const { return size_; }

    /// 获取数据类型
    DType dtype() const { return dtype_; }

    /// 获取设备类型
    DeviceType device() const { return device_; }

    /// 创建存储的深拷贝
    Storage clone() const {
        Storage new_storage(size_, dtype_, device_);
        if (size_ > 0 && data_) {
            std::memcpy(new_storage.data_.get(), data_.get(), size_ * dtypeSize(dtype_));
        }
        return new_storage;
    }

    /// 检查存储是否为空
    bool empty() const { return size_ == 0 || !data_; }

private:
    /// 检查模板类型是否与存储类型匹配
    template <typename T>
    void checkDType() const {
        if ((std::is_same_v<T, float> && dtype_ != DType::kFloat) ||
            (std::is_same_v<T, double> && dtype_ != DType::kDouble) ||
            (std::is_same_v<T, int32_t> && dtype_ != DType::kInt) ||
            (std::is_same_v<T, int64_t> && dtype_ != DType::kLong) ||
            (std::is_same_v<T, bool> && dtype_ != DType::kBool)) {
            throw std::runtime_error("Storage data type mismatch");
        }
    }

    size_t size_{};                ///< 存储的元素数量
    DType dtype_;                ///< 数据类型
    DeviceType device_;          ///< 设备类型
    std::shared_ptr<char[]> data_; ///< 原始内存指针（使用shared_ptr实现共享所有权）
};

// ======================= 张量类 (Tensor) =======================
struct ShapeTag {};

class Tensor {
public:
    // ======================= 构造和析构 =======================

    /// 默认构造函数：创建空张量
    Tensor() : storage_offset_(0), device_(DeviceType::kCPU), dtype_(DType::kFloat) {
        computeStrides();
        storage_ = Storage(numel(), dtype_, device_);
    }

    /// 标量构造函数
    Tensor(float value) : shape_({}), storage_offset_(0),
                          device_(DeviceType::kCPU), dtype_(DType::kFloat) {
        computeStrides();
        storage_ = Storage(1, dtype_, device_);
        *storage_.data<float>() = value;
    }

    /// 构造函数：从初始值列表创建1D张量
    Tensor(std::initializer_list<float> values)
            : shape_({values.size()}), storage_offset_(0),
              device_(DeviceType::kCPU), dtype_(DType::kFloat) {
        computeStrides();
        storage_ = Storage(values.begin(), values.size(), dtype_, device_);
    }

    // 添加布尔张量构造函数
    Tensor(std::initializer_list<bool> values)
            : shape_({values.size()}), storage_offset_(0),
              device_(DeviceType::kCPU), dtype_(DType::kBool) {
        computeStrides();
        storage_ = Storage(values.size(), dtype_, device_);
        bool* data = storage_.data<bool>();
        size_t i = 0;
        for (bool val : values) {
            data[i++] = val;
        }
    }

    /// 构造函数：指定形状和数据类型（使用 ShapeTag 避免歧义）
    Tensor(ShapeTag, const std::vector<size_t>& shape,
           DType dtype = DType::kFloat,
           DeviceType device = DeviceType::kCPU)
            : shape_(shape), storage_offset_(0), device_(device), dtype_(dtype) {
        computeStrides();
        storage_ = Storage(numel(), dtype_, device_);
    }

    /// 拷贝构造函数：创建深拷贝
    Tensor(const Tensor& other)
            : shape_(other.shape_), strides_(other.strides_),
              storage_offset_(other.storage_offset_),
              device_(other.device_), dtype_(other.dtype_),
              storage_(other.storage_.clone()) {}  // 深拷贝存储

    /// 移动构造函数
    Tensor(Tensor&& other) noexcept
            : shape_(std::move(other.shape_)),
              strides_(std::move(other.strides_)),
              storage_offset_(other.storage_offset_),
              device_(other.device_), dtype_(other.dtype_),
              storage_(std::move(other.storage_)) {
        other.storage_offset_ = 0;
        other.shape_.clear();
        other.strides_.clear();
    }

    ~Tensor() = default;

    // ======================= 基本属性 =======================

    /// 获取张量的维度数
    size_t dim() const { return shape_.size(); }

    /// 获取张量的形状
    const std::vector<size_t>& sizes() const { return shape_; }

    /// 获取张量中元素的总数
    size_t numel() const {
        if (shape_.empty()) return 1; // 标量有1个元素
        return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
    }

    /// 获取张量的数据类型
    DType dtype() const { return dtype_; }

    /// 获取张量所在的设备
    DeviceType device() const { return device_; }

    // ======================= 索引和访问 =======================

    /// 1D张量的索引访问
    template <typename T = float>
    T& operator[](size_t index) {
        checkDType<T>();
        if (dim() != 1) throw std::runtime_error("Requires 1D tensor");
        if (index >= shape_[0]) throw std::out_of_range("Tensor index out of bounds");
        return storage_.data<T>()[storage_offset_ + index];
    }

    /// 1D张量的常量索引访问
    template <typename T = float>
    const T& operator[](size_t index) const {
        checkDType<T>();
        if (dim() != 1) throw std::runtime_error("Requires 1D tensor");
        if (index >= shape_[0]) throw std::out_of_range("Tensor index out of bounds");
        return storage_.data<T>()[storage_offset_ + index];
    }

    /// 多维张量的索引访问
    template <typename T = float>
    T& operator()(std::initializer_list<size_t> indices) {
        return storage_.data<T>()[computeStorageIndex(indices)];
    }

    /// 多维张量的常量索引访问
    template <typename T = float>
    const T& operator()(std::initializer_list<size_t> indices) const {
        return storage_.data<T>()[computeStorageIndex(indices)];
    }

    /// 标量访问（0维张量）
    template <typename T = float>
    T& item() {
        if (dim() != 0) throw std::runtime_error("item() only works on 0-dimensional tensors");
        return *storage_.data<T>();
    }

    /// 常量标量访问
    template <typename T = float>
    const T& item() const {
        if (dim() != 0) throw std::runtime_error("item() only works on 0-dimensional tensors");
        return *storage_.data<T>();
    }

    /// 获取原始数据的类型化指针
    template <typename T = float>
    T* data() {
        checkDType<T>();
        if (storage_.empty()) return nullptr;
        return storage_.data<T>() + storage_offset_;
    }

    /// 获取常量原始数据的类型化指针
    template <typename T = float>
    const T* data() const {
        checkDType<T>();
        if (storage_.empty()) return nullptr;
        return storage_.data<T>() + storage_offset_;
    }

    // ======================= 张量操作 =======================

    /// 创建张量的深拷贝
    Tensor clone() const {
        Tensor copy;
        copy.shape_ = shape_;
        copy.strides_ = strides_;
        copy.storage_offset_ = 0;
        copy.device_ = device_;
        copy.dtype_ = dtype_;
        copy.storage_ = storage_.clone(); // 深拷贝存储
        return copy;
    }

    /// 改变张量的形状 (不改变内存布局)
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

        Tensor result(ShapeTag{}, new_shape, dtype_, device_);
        result.storage_ = storage_;  // 共享存储（使用shared_ptr实现共享所有权）
        result.storage_offset_ = storage_offset_;
        result.strides_ = strides_;  // 保持原步幅
        return result;
    }

    /// 转置最后两个维度
    Tensor transpose() const {
        if (dim() < 2) {
            throw std::runtime_error("transpose requires at least 2 dimensions");
        }

        // 创建新的形状和步幅
        std::vector<size_t> new_shape = shape_;
        std::vector<size_t> new_strides = strides_;

        // 交换最后两个维度
        std::swap(new_shape[dim()-1], new_shape[dim()-2]);
        std::swap(new_strides[dim()-1], new_strides[dim()-2]);

        Tensor result = *this;
        result.shape_ = new_shape;
        result.strides_ = new_strides;
        return result;
    }

    // ======================= 运算符重载 =======================

    /// 张量加法 (逐元素)
    Tensor operator+(const Tensor& rhs) const {
        // 验证形状和类型匹配
        if (shape_ != rhs.shape_) throw std::runtime_error("Shape mismatch in addition");
        if (dtype_ != rhs.dtype_) throw std::runtime_error("DType mismatch in addition");
        if (device_ != rhs.device_) throw std::runtime_error("Device mismatch in addition");

        Tensor result(ShapeTag{}, shape_, dtype_, device_);
        const size_t n = numel();

        // 根据数据类型分派加法操作
        switch (dtype_) {
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

    /// 张量赋值运算符（深拷贝）
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            shape_ = other.shape_;
            strides_ = other.strides_;
            storage_offset_ = other.storage_offset_;
            device_ = other.device_;
            dtype_ = other.dtype_;
            storage_ = other.storage_.clone(); // 深拷贝存储
        }
        return *this;
    }

    /// 张量移动赋值运算符
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            storage_offset_ = other.storage_offset_;
            device_ = other.device_;
            dtype_ = other.dtype_;
            storage_ = std::move(other.storage_);

            other.storage_offset_ = 0;
            other.shape_.clear();
            other.strides_.clear();
        }
        return *this;
    }

    /// 张量相等比较
    bool operator==(const Tensor& other) const {
        if (shape_ != other.shape_ || dtype_ != other.dtype_) {
            return false;
        }

        const size_t n = numel();
        if (n == 0) return true; // 空张量相等

        switch (dtype_) {
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

    /// 将张量转换为字符串表示
    std::string toString() const {
        std::ostringstream oss;
        oss << "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            oss << shape_[i];
            if (i < shape_.size() - 1) oss << ", ";
        }
        oss << "], dtype=" << dtypeToString(dtype_)
            << ", device=" << (device_ == DeviceType::kCPU ? "cpu" : "gpu") << ")\n";

        // 打印张量内容
        if (numel() == 0) {
            oss << "[]";
        } else {
            oss << "[";
            if (dtype_ == DType::kFloat) {
                printRecursive<float>(oss, 0, std::vector<size_t>());
            } else if (dtype_ == DType::kDouble) {
                printRecursive<double>(oss, 0, std::vector<size_t>());
            } else if (dtype_ == DType::kInt) {
                printRecursive<int32_t>(oss, 0, std::vector<size_t>());
            } else if (dtype_ == DType::kLong) {
                printRecursive<int64_t>(oss, 0, std::vector<size_t>());
            } else if (dtype_ == DType::kBool) {
                printRecursive<bool>(oss, 0, std::vector<size_t>());
            }
            oss << "]";
        }
        return oss.str();
    }

    /// 打印张量信息
    void print() const {
        std::cout << toString() << std::endl;
    }

private:
    // ======================= 内部辅助函数 =======================

    /// 计算步幅 (基于行优先顺序)
    void computeStrides() {
        if (shape_.empty()) {
            strides_.clear();
            return;
        }

        strides_.resize(shape_.size());
        // 行优先布局: 最后一个维度步幅为1
        strides_[shape_.size() - 1] = 1;
        for (int i = shape_.size() - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    /// 计算存储中的索引
    size_t computeStorageIndex(std::initializer_list<size_t> indices) const {
        if (indices.size() != dim()) {
            throw std::runtime_error("Indices count mismatch");
        }

        if (dim() == 0) {
            return storage_offset_; // 标量情况
        }

        size_t index = storage_offset_;
        size_t i = 0;
        for (const auto& idx : indices) {
            if (idx >= shape_[i]) {
                throw std::out_of_range("Tensor index out of bounds");
            }
            index += idx * strides_[i];
            ++i;
        }
        return index;
    }

    /// 检查数据类型是否匹配
    template <typename T>
    void checkDType() const {
        if ((std::is_same_v<T, float> && dtype_ != DType::kFloat) ||
            (std::is_same_v<T, double> && dtype_ != DType::kDouble) ||
            (std::is_same_v<T, int32_t> && dtype_ != DType::kInt) ||
            (std::is_same_v<T, int64_t> && dtype_ != DType::kLong) ||
            (std::is_same_v<T, bool> && dtype_ != DType::kBool)) {
            throw std::runtime_error("Tensor data type mismatch");
        }
    }

    /// 通用逐元素操作
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

    /// 递归打印张量内容（改进版）
    template <typename T>
    void printRecursive(std::ostream& os, size_t dim, std::vector<size_t> indices) const {
        if (dim == this->dim()) {
            // 到达最后一个维度，打印元素
            size_t index = storage_offset_;
            for (size_t i = 0; i < indices.size(); ++i) {
                index += indices[i] * strides_[i];
            }

            if constexpr (std::is_same_v<T, bool>) {
                os << (storage_.data<T>()[index] ? "true" : "false");
            } else if constexpr (std::is_floating_point_v<T>) {
                os << std::fixed << std::setprecision(2) << storage_.data<T>()[index];
            } else {
                os << storage_.data<T>()[index];
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
        const size_t display_count = std::min(shape_[dim], max_display);
        const bool truncated = shape_[dim] > max_display;

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

private:
    std::vector<size_t> shape_;   ///< 张量的维度大小
    std::vector<size_t> strides_; ///< 每个维度的步幅（字节或元素数）
    size_t storage_offset_;       ///< 存储中的起始偏移量
    DeviceType device_;           ///< 张量所在的设备
    DType dtype_;                 ///< 张量元素的数据类型
    Storage storage_;             ///< 存储张量数据的存储对象
};
#endif //CTT_TENSOR_H
