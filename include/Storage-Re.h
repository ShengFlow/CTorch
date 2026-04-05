/**
* @file Storage.h
 * @brief Ctorch 存储类，用于管理张量数据的底层存储
 * @author GhostFace
 * @date 2025/12/21
 * @version v3.1
 * @details 存储类是张量数据的底层容器，支持多种数据类型和设备类型
 */

#ifndef CTORCH_STORAGE_RE_H
#define CTORCH_STORAGE_RE_H

#include "Ctools.h"
#include "Ctorch_Error.h"
#include <memory>
#include <cstring>
#include <iostream>
#include <type_traits>

/**
 * @class Storage
 * @brief 存储类，用于管理张量数据的底层存储
 * @details 存储类支持多种数据类型和设备类型，使用shared_ptr实现共享所有权，减少内存占用
 */
class Storage {
private:
    size_t _size{};                              ///< 存储的元素数量
    DType _dtype;                                ///< 数据类型枚举
    DeviceType _device;                          ///< 设备类型枚举
    std::shared_ptr<char[]> _data;               ///< 原始内存指针（共享所有权）

    /**
     * @brief 检查模板类型是否与存储类型匹配
     * @tparam T 模板类型
     * @throw std::runtime_error 如果类型不匹配
     */
    template <typename T>
    void checkDType() const;

public:
    // ========== 构造函数 ==========

    /**
     * @brief 默认构造函数：创建一个空的float类型、CPU设备的存储
     */
    Storage();

    /**
     * @brief 构造函数：分配未初始化的内存
     * @param size 存储的元素数量
     * @param dtype 数据类型
     * @param device 设备类型，默认CPU
     */
    Storage(size_t size, DType dtype, DeviceType device = DeviceType::kCPU);

    /**
     * @brief 构造函数：从现有数据复制
     * @tparam T 数据类型
     * @param data 现有数据指针
     * @param size 数据元素数量
     * @param dtype 数据类型
     * @param device 设备类型，默认CPU
     */
    template <typename T>
    Storage(const T* data, size_t size, DType dtype, DeviceType device = DeviceType::kCPU);

    /**
     * @brief 拷贝构造函数（浅拷贝，共享数据）
     */
    Storage(const Storage&) = default;

    /**
     * @brief 拷贝赋值运算符（浅拷贝，共享数据）
     */
    Storage& operator=(const Storage&) = default;

    /**
     * @brief 移动构造函数
     */
    Storage(Storage&&) = default;

    /**
     * @brief 移动赋值运算符
     */
    Storage& operator=(Storage&&) = default;

    /**
     * @brief 析构函数
     */
    ~Storage() = default;

    // ========== 数据访问 ==========

    /**
     * @brief 获取原始数据的类型化指针
     * @tparam T 数据类型
     * @return 类型化数据指针，如果存储为空返回nullptr
     */
    template <typename T>
    T* data();

    /**
     * @brief 获取常量原始数据的类型化指针
     * @tparam T 数据类型
     * @return 常量类型化数据指针，如果存储为空返回nullptr
     */
    template <typename T>
    const T* data() const;

    // ========== 属性 ==========

    /**
     * @brief 获取存储中的元素数量
     * @return 元素数量
     */
    size_t size() const { return _size; }

    /**
     * @brief 获取数据类型
     * @return 数据类型
     */
    DType dtype() const { return _dtype; }

    /**
     * @brief 获取设备类型
     * @return 设备类型
     */
    DeviceType device() const { return _device; }

    /**
     * @brief 获取原始指针（不进行类型检查）
     * @return 原始char指针
     */
    char* raw_data() { return _data.get(); }
    const char* raw_data() const { return _data.get(); }

    // ========== 操作 ==========

    /**
     * @brief 创建存储的深拷贝
     * @return 深拷贝的存储对象
     */
    Storage clone() const;

    /**
     * @brief 清空存储
     * @details 释放内存并将大小设置为0
     */
    void clear();

    /**
     * @brief 检查存储是否为空
     * @return 如果存储为空返回true，否则返回false
     */
    bool empty() const { return _size == 0 || !_data; }

    /**
     * @brief 重置存储（分配新内存，丢弃旧数据）
     * @param size 新的大小
     * @param dtype 新的数据类型
     * @param device 新的设备类型
     */
    void reset(size_t size, DType dtype, DeviceType device = DeviceType::kCPU);
};

// ========== 模板方法实现（必须在头文件中） ==========

template <typename T>
void Storage::checkDType() const {
    if ((std::is_same<T, float>::value && _dtype != DType::kFloat) ||
        (std::is_same<T, double>::value && _dtype != DType::kDouble) ||
        (std::is_same<T, int32_t>::value && _dtype != DType::kInt) ||
        (std::is_same<T, int64_t>::value && _dtype != DType::kLong) ||
        (std::is_same<T, bool>::value && _dtype != DType::kBool)) {
        std::cerr << "Storage data type mismatch: T=" << typeid(T).name()
                  << ", dtype=" << dtypeToString(_dtype) << std::endl;
        Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "数据类型不匹配！");
        throw std::runtime_error("Storage data type mismatch");
    }
}

template <typename T>
Storage::Storage(const T* data, size_t size, DType dtype, DeviceType device)
    : Storage(size, dtype, device) {
    if (size > 0 && _data.get()) {
        std::memcpy(_data.get(), data, size * dtypeSize(dtype));
    }
}

template <typename T>
T* Storage::data() {
    if (_size == 0 || !_data) {
        return nullptr;
    }
    checkDType<T>();
    return reinterpret_cast<T*>(_data.get());
}

template <typename T>
const T* Storage::data() const {
    if (_size == 0 || !_data) {
        return nullptr;
    }
    checkDType<T>();
    return reinterpret_cast<const T*>(_data.get());
}

#endif // CTORCH_STORAGE_RE_H
