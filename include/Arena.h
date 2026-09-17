/**
 * @file Arena.h
 * @author Beapoe
 * @brief 自动微分系统的内存池
 * @date 2026/3/7
 */

#ifndef CTORCH_ARENA_H
#define CTORCH_ARENA_H

#include <cstddef>  // [Fix] v0.5.2 Linux build: std::max_align_t 定义在此
                     // macOS clang transitive include 拿到, DTK 26.04 clang 17 严格, 必须显式 include
#include <vector>
#include <functional>
#include <memory>
#include <mutex>
#include "CtorchError.h"
#include "CoreDefs.h"

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

    /** @brief 保留的内存块数量 */
    static constexpr size_t KEEP_BLOCKS = 10;

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
        // [Fix 2026-09-17] 改为进程生命周期单例，不参与静态析构。
        //
        // 原实现是 Meyers 单例（`static auto instance = Arena();`）。进程退出时它
        // 会被析构，析构里调用 reset() 去执行池中残留对象的销毁函数；那些对象的
        // 析构又会触碰其他**已完成析构**的静态对象（存储池、日志、C3 单例），
        // 于是退化成对已失效内存的访问。
        //
        // 实测（AddressSanitizer，OpenInspire3 可微动力学测试）：100% 稳定地在
        // exit() 的 __cxa_finalize_ranges 段 SIGSEGV，栈顶为
        // `Tensor::~Tensor() <- Arena::reset() <- Arena::~Arena()`。因为发生在
        // 退出阶段，表现为「测试全部通过但进程以 138/139 退出」，且时崩时不崩，
        // 极易被误判成运算过程中的内存破坏。
        //
        // 池本身是进程级缓存，退出时交给操作系统回收是正确做法；这与
        // C3KernelRegistry（同样改为 new 分配）和 FlatOutPool（§4.93「释放数据、
        // 保留结构」）的处理保持一致。运行期的 reset() 语义不变。
        static Arena* instance = new Arena();
        return *instance;
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

    /**
     * @brief 分配指定大小的原始内存
     * @param bytes 需要分配的字节数
     * @param align 对齐要求，默认最大对齐
     * @return 分配的内存指针，失败返回nullptr
     */
    CT_MALLOC char* allocBytes(size_t bytes, size_t align = alignof(std::max_align_t));

    /**
     * @brief 分配指定大小的内存并返回 shared_ptr（由 Arena 管理生命周期）
     * @param bytes 需要分配的字节数
     * @param align 对齐要求，默认最大对齐
     * @return shared_ptr<char>，空删除器（内存由 Arena 统一管理）
     */
    std::shared_ptr<char> allocShared(size_t bytes, size_t align = alignof(std::max_align_t));

    /** @brief 重置内存池，释放所有分配的内存 */
    void reset();

    /** @brief 清理内存池，释放所有内存块 */
    void clear();
};

#endif // CTORCH_ARENA_H
