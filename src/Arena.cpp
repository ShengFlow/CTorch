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

char* Arena::allocateFromBlock(Block& block, size_t align, size_t size) {
    void* ptr = block._base + block._offset;
    size_t space = block._maxOffset - block._offset;
    if (std::align(align, size, ptr, space)) [[likely]] {
        block._offset = static_cast<char*>(ptr) + size - block._base;
        return static_cast<char*>(ptr);
    }
    return nullptr;
}

void Arena::reset() {
    std::lock_guard lock(_mtx);
    for (auto it = _destroyFuncs.rbegin();it != _destroyFuncs.rend();++it) (*it)();
    _destroyFuncs.clear();

    // [Fix 2026-09-17] 把块回落到保留水位，而不是仅把偏移归零。
    //
    // 原实现保留全部块：池按峰值增长，峰值过后永不回落（与 §4.93 的 FlatOutPool
    // 同类问题；当时 KEEP_BLOCKS 常量已声明却未被使用）。一次内存尖峰（例如临时
    // 构造特大的图）会让进程长期占住那份内存。
    //
    // 安全性前提与 reset() 本身的语义一致：调用方保证此刻池中已无存活对象。
    // 池的唯一生产者是 invoke<T>() 构造的图节点，它们在上面的析构循环里已全部
    // 析构；allocShared/allocBytes 目前无生产调用点（仅基准使用），故不构成例外。
    if (_blocks.size() > KEEP_BLOCKS) {
        _blocks.resize(KEEP_BLOCKS);
    }

    for (auto& block:_blocks) block->_offset = 0;
}

size_t Arena::blockCount() const {
    std::lock_guard lock(_mtx);
    return _blocks.size();
}

CT_MALLOC char* Arena::allocBytes(size_t bytes, size_t align) {
    std::lock_guard lock(_mtx);

    auto allocateFrom = [](std::unique_ptr<Block>& block, size_t alignment, size_t size) {
        return allocateFromBlock(*block, alignment, size);
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