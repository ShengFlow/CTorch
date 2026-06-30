/**
 *@file Arena.cpp
 *@author Beapoe
 *@brief 内存池
 *@date 2026/3/7
 **/

#include "../include/Arena.h"
#include "../include/CoreDefs.h"

Block::Block(size_t size)
    :_base(static_cast<char*>(::operator new(size))),_offset(0),_maxOffset(size)
{}

Block::~Block() { ::operator delete(_base); }

Arena::Arena()
    :_destroyFuncs(std::vector<std::function<void()>>())
{}

Arena::~Arena() {
    reset();
}

void Arena::addBlock(size_t size) {
    auto block = std::make_unique<Block>(size);
    _blocks.push_back(std::move(block));
}

void Arena::reset() {
    std::lock_guard lock(_mtx);
    for (auto it = _destroyFuncs.rbegin();it != _destroyFuncs.rend();++it) (*it)();
    _destroyFuncs.clear();

    for (auto& block:_blocks) block->_offset = 0;
}

CT_MALLOC char* Arena::allocBytes(size_t bytes, size_t align) {
    std::lock_guard lock(_mtx);

    auto allocateFrom = [](std::unique_ptr<Block>& block, size_t alignment, size_t size) -> char* {
        void* ptr = block->_base + block->_offset;
        size_t space = block->_maxOffset - block->_offset;
        if (std::align(alignment, size, ptr, space)) [[likely]] {
            block->_offset = static_cast<char*>(ptr) + size - block->_base;
            return static_cast<char*>(ptr);
        }
        return nullptr;
    };

    if (_blocks.empty()) [[unlikely]] {
        size_t blockSize = std::max(bytes + align, static_cast<size_t>(1024 * 1024));
        addBlock(blockSize);
    }

    char* ptr = allocateFrom(_blocks.back(), align, bytes);
    if (ptr) [[likely]] return ptr;

    for (auto it = _blocks.rbegin() + 1; it != _blocks.rend(); ++it) {
        ptr = allocateFrom(*it, align, bytes);
        if (ptr) [[likely]] return ptr;
    }

    size_t blockSize = std::max(bytes + align, static_cast<size_t>(1024 * 1024));
    addBlock(blockSize);

    ptr = allocateFrom(_blocks.back(), align, bytes);
    if (ptr) [[likely]] return ptr;

    CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "Unable to allocate bytes from Arena.");
    return nullptr;
}

std::shared_ptr<char> Arena::allocShared(size_t bytes, size_t align) {
    char* mem = allocBytes(bytes, align);
    if (mem) [[likely]] {
        return std::shared_ptr<char>(mem, [](char*) noexcept {});
    }
    return nullptr;
}

void Arena::clear() {
    std::lock_guard lock(_mtx);
    for (auto it = _destroyFuncs.rbegin();it != _destroyFuncs.rend();++it) (*it)();
    _destroyFuncs.clear();

    _blocks.clear();
}