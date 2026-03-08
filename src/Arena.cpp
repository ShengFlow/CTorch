/**
*@file Arena.cpp
 *@author Beapoe
 *@brief 内存池
 *@date 2026/3/7
 **/

#include "../include/Arena.h"

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
