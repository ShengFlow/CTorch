/**
 * @file Storage.h
 * @brief Ctorch 存储类，用于管理张量数据的底层存储
 * @author GhostFace
 * @date 2025/12/21
 * @version v3.1
 * @details 存储类是张量数据的底层容器，支持多种数据类型和设备类型
 */

#ifndef STORAGE_H
#define STORAGE_H
#include "Ctools.h"
#include "CtorchError.h"
#include "DeviceAllocator.h"
#include <climits>

/**
 * @class Storage
 * @brief 存储类，用于管理张量数据的底层存储
 * @details 存储类支持多种数据类型和设备类型，使用shared_ptr实现共享所有权，减少内存占用
 * @note ABI 注意：本类为公开头文件核心类，任何成员变量布局、Deleter 行为或 sizeof 变化
 *       都会破坏 ABI。后续若需变更分配策略，请将分配器状态封装到不暴露布局的位置
 *       （如 StorageImpl opaque handle），或显式进行主版本升级。
 *       本项目当前不保证 ABI 稳定性，但禁止静默 ABI 破坏；详见 ABI_POLICY.md。
 */
class Storage {
private:
   /**
    * @var _size
    * @brief 存储的元素数量
    * @details 使用C++11的花括号初始化，等同于size_t _size = 0;
    */
   size_t _size{};
   
   /**
    * @var _dtype
    * @brief 数据类型枚举
    */
   DType _dtype;
   
   /**
    * @var _device
    * @brief 设备类型枚举
    */
   DeviceType _device;
   
   /**
    * @var _data
    * @brief 原始内存指针
    * @details 使用shared_ptr<char>实现共享所有权，避免手动delete问题
    * 使用char*能够最大限度节省内存并支持存储任意类型的数据
    * 同等tensor可以共用一块内存，减少不必要的内存占用
    * 内存分配通过DeviceAllocator接口实现，支持多设备
    */
   std::shared_ptr<char> _data;

   struct Deleter {
       std::shared_ptr<DeviceAllocator> _allocator;
       DeviceType _device;
       void operator()(char* ptr) const {
           if (ptr) {
               if (_allocator) {
                   _allocator->deallocate(ptr, _device);
               } else {
                   std::free(ptr);
               }
           }
       }
   };

   /**
    * @brief 检查模板类型是否与存储类型匹配
    * @tparam T 模板类型
    * @throw std::runtime_error 如果类型不匹配
    * @details 强制类型检查，避免不必要的内存问题
    */
   template <typename T>
   void checkDType() const {
       if ((std::is_same<T, float>::value && _dtype != DType::kFloat) ||
           (std::is_same<T, double>::value && _dtype != DType::kDouble) ||
           (std::is_same<T, int32_t>::value && _dtype != DType::kInt) ||
           (std::is_same<T, int64_t>::value && _dtype != DType::kLong) ||
           (std::is_same<T, bool>::value && _dtype != DType::kBool)) {
           CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "Storage data type mismatch");
       }
   }

public:
    /**
     * @brief 构造函数：分配未初始化的内存
     * @param size 存储的元素数量
     * @param dtype 数据类型
     * @param device 设备类型，默认CPU
     */
    Storage(size_t size, DType dtype, DeviceType device = DeviceType::kCPU)
        : _size(size), _dtype(dtype), _device(device) {
        if (size > 0) {
            size_t elem_size = dtypeSize(dtype);
            if (elem_size > 0 && size > SIZE_MAX / elem_size) {
                CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
                                            "Storage size overflow: requested element count too large");
            }
            size_t bytes = size * elem_size;
            std::shared_ptr<DeviceAllocator> allocator = AllocatorManager::getInstance().getAllocator(device);
            if (allocator) {
                char* ptr = static_cast<char*>(allocator->allocate(bytes, device));
                if (ptr) {
                    _data = std::shared_ptr<char>(ptr, Deleter{allocator, device});
                }
            }
            if (!_data) {
                _data = std::shared_ptr<char>(new char[bytes], std::default_delete<char[]>());
            }
        }
    }

    /**
     * @brief 构造函数：从现有数据复制
     * @tparam T 数据类型
     * @param data 现有数据指针
     * @param size 数据元素数量
     * @param dtype 数据类型
     * @param device 设备类型，默认CPU
     * @details 将从现有数据复制的操作委托给第一个构造函数，然后进行memcpy操作
     */
    template <typename T>
    Storage(const T* data, size_t size, DType dtype, DeviceType device = DeviceType::kCPU): Storage(size, dtype, device) {
        if (size > 0 && _data.get()) {
            std::memcpy(_data.get(), data, size * dtypeSize(dtype));
        }
    }

    /**
     * @brief 拷贝构造函数（浅拷贝，共享底层数据）
     * @details 需要深拷贝时请显式调用 clone()。
     */
    Storage(const Storage& other)
        : _size(other._size), _dtype(other._dtype), _device(other._device), _data(other._data) {}

    /**
     * @brief 拷贝赋值运算符（浅拷贝，共享底层数据）
     * @details 需要深拷贝时请显式调用 clone()。
     */
    Storage& operator=(const Storage& other) {
        if (this != &other) {
            _size = other._size;
            _dtype = other._dtype;
            _device = other._device;
            _data = other._data;
        }
        return *this;
    }

    /**
     * @brief 移动构造函数
     * @details 转移底层数据所有权，并将源 Storage 置为空（size=0，data=nullptr）
     */
    Storage(Storage&& other) noexcept
        : _size(other._size), _dtype(other._dtype), _device(other._device), _data(std::move(other._data)) {
        other._size = 0;
    }

    /**
     * @brief 移动赋值运算符
     * @details 转移底层数据所有权，并将源 Storage 置为空
     */
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            _size = other._size;
            _dtype = other._dtype;
            _device = other._device;
            _data = std::move(other._data);
            other._size = 0;
        }
        return *this;
    }

    /**
     * @brief 默认构造函数
     * @details 创建一个空的float类型、CPU设备的存储
     */
    Storage() : _size(0), _dtype(DType::kFloat), _device(DeviceType::kCPU) {}

    /**
     * @brief 默认析构函数
     */
    ~Storage() = default;

    /**
     * @brief 获取原始数据的类型化指针
     * @tparam T 数据类型
     * @return 类型化数据指针，如果存储为空返回nullptr
     */
    template <typename T>
    T* data() {
        if (_size == 0 || !_data) {
            return nullptr;
        }
        checkDType<T>();
        T* result = reinterpret_cast<T*>(_data.get());
        return result;
    }

    /**
     * @brief 获取常量原始数据的类型化指针
     * @tparam T 数据类型
     * @return 常量类型化数据指针，如果存储为空返回nullptr
     */
    template <typename T>
    const T* data() const {
        if (_size == 0 || !_data) return nullptr;
        checkDType<T>();
        return reinterpret_cast<const T*>(_data.get());
    }

    /**
     * @brief 获取存储中的元素数量
     * @return 元素数量
     */
    size_t size() const noexcept { return _size; }

    /**
     * @brief 获取数据类型
     * @return 数据类型
     */
    DType dtype() const noexcept { return _dtype; }

    /**
     * @brief 获取设备类型
     * @return 设备类型
     */
    DeviceType device() const noexcept { return _device; }

    /**
     * @brief 创建存储的深拷贝
     * @return 深拷贝的存储对象
     */
    Storage clone() const {
        Storage new_storage(_size, _dtype, _device);
        if (_size > 0 && _data && new_storage._data) {
            std::shared_ptr<DeviceAllocator> allocator = AllocatorManager::getInstance().getAllocator(_device);
            if (allocator) {
                allocator->memcpy(new_storage._data.get(), _data.get(),
                    _size * dtypeSize(_dtype), _device, _device);
            } else {
                std::memcpy(new_storage._data.get(), _data.get(),
                    _size * dtypeSize(_dtype));
            }
        }
        return new_storage;
    }
    
    /**
     * @brief 清空存储
     * @details 释放内存并将大小设置为0
     */
    void clear() {
       _data.reset();
       _size = 0;
   }

    /**
     * @brief 检查存储是否为空
     * @return 如果存储为空返回true，否则返回false
     */
    bool empty() const noexcept {
       return _size == 0 || !_data;
   }
};
#endif //STORAGE_H
