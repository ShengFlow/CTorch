/**
 * @file Arena.h
 * @author Beapoe
 * @brief 自动微分系统的内存池
 * @date 2026/3/7
 */

#ifndef CTORCH_ARENA_H
#define CTORCH_ARENA_H

#include <vector>
#include <functional>
#include <memory>
#include <mutex>
#include "CtorchError.h"

/**
 * @struct Block
 * @brief 内存块结构，Arena内存池的基本分配单元
 */
struct Block {
    /** @brief 内存块起始地址 */
    char* _base;
    /** @brief 当前分配偏移量 */
    size_t _offset;
    /** @brief 内存块最大容量 */
    size_t _maxOffset;

    /** @brief 构造函数，分配指定大小的内存块 */
    explicit Block(size_t size);
    /** @brief 析构函数，释放内存块 */
    ~Block();

    /** @brief 禁用拷贝构造 */
    Block(const Block&) = delete;
    /** @brief 禁用拷贝赋值 */
    Block& operator=(const Block&) = delete;
};

/**
 * @class Arena
 * @brief 自动微分系统的内存池类
 * @details 使用对象池模式管理计算图节点的内存分配，避免频繁的new/delete操作。
 *          采用线程安全设计，支持并发访问。
 */
class Arena {
    /** @brief 内存块列表 */
    std::vector<std::unique_ptr<Block>> _blocks;
    /** @brief 析构函数列表，用于手动调用非平凡析构 */
    std::vector<std::function<void()>> _destroyFuncs;
    /** @brief 互斥锁，保证线程安全 */
    mutable std::mutex _mtx;

    /**
     * @brief 添加一个新的内存块
     * @param size 内存块大小，默认1MB
     */
    void addBlock(size_t size = 1024*1024);

    /**
     * @brief 分配指定类型大小的内存
     * @tparam T 要分配的类型
     * @return 分配的内存指针，失败返回nullptr
     */
    template <typename T>
    char* allocate() {
        auto allocateFrom = [](std::unique_ptr<Block>& block,size_t align,size_t size)-> char* {
            void* ptr = block->_base + block->_offset;
            size_t space = block->_maxOffset - block->_offset;
            if (std::align(align,size,ptr,space)) {
                block->_offset = static_cast<char*>(ptr) + size - block->_base;
                return static_cast<char*>(ptr);
            }
            return nullptr;
        };

        if (_blocks.empty()) {
            if (sizeof(T) + alignof(T) > 1024*1024) addBlock(sizeof(T) + alignof(T) -1);
            else addBlock();
        }

        char* ptr = allocateFrom(_blocks.back(),alignof(T),sizeof(T));
        if (ptr) return ptr;

        for (auto it = _blocks.rbegin() + 1;it != _blocks.rend();++it) {
            ptr = allocateFrom((*it),alignof(T),sizeof(T));
            if (ptr) return ptr;
        }

        if (sizeof(T) + alignof(T) > 1024*1024) addBlock(sizeof(T) + alignof(T) - 1);
        else addBlock();

        ptr = allocateFrom(_blocks.back(),alignof(T),sizeof(T));
        if (ptr) return ptr;
        CtorchError::error(ErrorPlatform::kAutoDiff,ErrorType::UNKNOWN,"Unable to allocate for the object.");
        return nullptr;
    }

    /** @brief 私有构造函数，防止外部实例化 */
    Arena();
public:
    /**
     * @brief 获取单例实例
     * @return Arena的引用
     */
    static Arena& getInstance() {
        static auto instance = Arena();
        return instance;
    }

    /** @brief 析构函数，释放所有内存块 */
    ~Arena();

    /** @brief 禁用拷贝构造 */
    Arena(const Arena&) = delete;
    /** @brief 禁用拷贝赋值 */
    Arena operator=(const Arena&) = delete;

    /**
     * @brief 在内存池中构造对象
     * @tparam T 对象类型
     * @tparam Args 构造参数类型
     * @param args 构造参数
     * @return 对象的shared_ptr，使用空删除器（内存由Arena管理）
     */
    template <typename T,typename... Args>
    std::shared_ptr<T> invoke(Args&&... args) {
        std::lock_guard lock(_mtx);
        if (char* mem = allocate<T>()) {
            T* obj = new (mem) T(std::forward<Args>(args)...);
            if constexpr (!std::is_trivially_destructible_v<T>)
                _destroyFuncs.push_back([obj](){obj->~T();});
            auto emptyDeleter = [](T*) noexcept {};
            std::shared_ptr<T> ptr(obj, emptyDeleter);
            return ptr;
        }
        CtorchError::error(ErrorPlatform::kAutoDiff,ErrorType::UNKNOWN,"Unable to add for the object.");
        return nullptr;
    }

    /** @brief 重置内存池，释放所有分配的内存 */
    void reset();

    /** @brief 清理内存池，释放所有内存块 */
    void clear();
};

#endif // CTORCH_ARENA_H
