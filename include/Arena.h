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

#include <cstdlib>
#include <string>

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
     * @brief 在指定块内按对齐要求切出一段内存
     * @param block 目标块
     * @param align 对齐要求
     * @param size  需要的字节数
     * @return 切出的地址；该块剩余空间不足时返回 nullptr
     *
     * @note 此前这段逻辑在 allocate() 与 allocBytes() 里各写了一份，容易漂移。
     */
    static char* allocateFromBlock(Block& block, size_t align, size_t size);

    /**
     * @brief 分配指定类型大小的内存
     * @tparam T 要分配的类型
     * @return 分配的内存指针，失败返回nullptr
     */
    template <typename T>
    char* allocate() {
        auto allocateFrom = [](std::unique_ptr<Block>& block, size_t align, size_t size) {
            return allocateFromBlock(*block, align, size);
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
     * @brief reset() 后保留的内存块数量
     *
     * 超过该水位的块会在 reset() 时真正释放 —— 池按峰值增长、峰值过后回落。
     * （该常量此前已声明但未被使用，reset() 保留全部块，导致内存只增不减。）
     */
    static constexpr size_t KEEP_BLOCKS = 10;

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
        // [Fix 2026-09-18] 节点生命周期改由 shared_ptr 独占管理。
        //
        // 原实现把对象放在 Arena 的块里，返回「空删除器」的 shared_ptr，并在
        // reset() 时统一执行析构、再把块偏移归零以供复用。这条路径有两个无法
        // 调和的问题：
        //
        //  1. **悬垂**：Tensor 的 `_autograd_meta._node` 是 shared_ptr，会跨 reset
        //     存活。reset 强杀对象后这些引用立即悬垂，而块偏移归零让同一地址被
        //     下一个对象复用 —— 悬垂引用于是指向了**另一个类型**的对象，虚调用
        //     直接越界。实测多轮「建图-反传-更新」循环稳定 SIGBUS/SIGSEGV
        //     （ASan: heap-buffer-overflow in ComputeCore::backward），而把 reset
        //     改成 no-op 或把本函数改为 make_shared 都立即恢复正常 —— 二者互为对照，
        //     确认根因在此而非运算逻辑。
        //  2. **成本**：bench_arena 实测该路径比 make_shared 慢约 3 倍（40 万次创建：
        //     14.3 ms vs 4.7 ms）。它省下了对象的 malloc，却没省下 shared_ptr
        //     控制块的那次 malloc —— 也就是说这条路径既没有安全性也没有收益。
        //
        // 改为标准堆分配后：生命周期由引用计数精确管理（不再有悬垂）、每次创建
        // 少一把全局锁（并发创建不再串行化），reset() 也就不再需要「强杀对象」。
        // Arena 的其余 API（allocBytes / allocShared / reset）保持不变，仍可作
        // 原始的字节级池使用。
        return std::make_shared<T>(std::forward<Args>(args)...);
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

    /** @brief 重置内存池：析构池中对象，并把块回落到保留水位 */
    void reset();

    /** @brief 当前池中的内存块数量（诊断用，非热路径） */
    [[nodiscard]] size_t blockCount() const;

    /** @brief 清理内存池，释放所有内存块 */
    void clear();
};

#endif // CTORCH_ARENA_H
