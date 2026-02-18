/**
 *@file IDDistributor.cpp
 *@author Beapoe
 *@brief 单线程全局NodeID分配
 *@date 2026/2/18
 **/

#include "../include/AutoGrad/IDDistributor.h"

thread_local size_t IDDistributor::currentCount = 0;