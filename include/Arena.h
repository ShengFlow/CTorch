/**
*@file Arena.h
 *@author Beapoe
 *@brief 内存池
 *@date 2026/3/7
 **/

#ifndef CTORCH_ARENA_H
#define CTORCH_ARENA_H

#include <vector>
#include <functional>
#include <memory>
#include <mutex>
#include "Ctorch_Error.h"

struct Block {
    char* _base;
    size_t _offset;
    size_t _maxOffset;

    explicit Block(size_t size);
    ~Block();

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
};

class Arena {
    std::vector<std::unique_ptr<Block>> _blocks;
    std::vector<std::function<void()>> _destroyFuncs;
    mutable std::mutex _mtx;

    void addBlock(size_t size = 1024*1024);

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
        Ctorch_Error::error(ErrorPlatform::kAutoDiff,ErrorType::UNKNOWN,"Unable to allocate for the object.");
        return nullptr;
    }

    Arena();
public:
    static Arena& getInstance() {
        static auto instance = Arena();
        return instance;
    }
    ~Arena();

    Arena(const Arena&) = delete;
    Arena operator=(const Arena&) = delete;

    template <typename T,typename... Args>
    std::shared_ptr<T> invoke(Args&&... args) {
        std::lock_guard lock(_mtx);
        if (char* mem = allocate<T>()) {
            T* obj = new (mem) T(std::forward<Args>(args)...);
            if constexpr (!std::is_trivially_destructible_v<T>)
                _destroyFuncs.push_back([obj](){obj->~T();});
            auto emptyDeleter = [](T*) noexcept {};
            std::shared_ptr<T> ptr(obj,emptyDeleter);
            return ptr;
        }
        Ctorch_Error::error(ErrorPlatform::kAutoDiff,ErrorType::UNKNOWN,"Unable to add for the object.");
        return nullptr;
    }

    void reset();
};

#endif // CTORCH_ARENA_H
