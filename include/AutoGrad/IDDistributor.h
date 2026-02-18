/**
 *@file IDDistributor.h
 *@author Beapoe
 *@brief 单线程全局NodeID分配
 *@date 2026/2/18
 **/

#ifndef CTORCH_IDDISTRIBUTOR
#define CTORCH_IDDISTRIBUTOR

#include <cstddef>

/**
 * @class IDDistributor
 * @brief 单线程全局NodeID分配类
 */
class IDDistributor {
    static thread_local size_t currentCount;

    IDDistributor() = default;

  public:
    /**
     * @brief 单线程全局NodeID分配函数
     * @return 返回分配的ID
     */
    static size_t allocateID() { return currentCount++; }
};

#endif // CTORCH_IDDISTRIBUTOR
