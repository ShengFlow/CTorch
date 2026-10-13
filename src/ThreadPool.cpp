/**
*@file ThreadPool.cpp
 *@brief 线程池
 *@author Beapoe
 *@date 2026/2/18
 **/

#include "ThreadPool.h"

size_t ThreadPool::threadAmount = 0;

void ThreadPool::run(){
    while(1){
        std::function<void()> task;
        {
            std::unique_lock lock(mtx);
            keepRun.wait(lock,[this](){return !tasks.empty() || terminate;});
            if(terminate && tasks.empty()) break;
            task = std::move(tasks.front());
            tasks.pop();
        }
        task();
    }
}

ThreadPool::ThreadPool(size_t nums){
    for(size_t i = 0;i<nums;i++) threads.emplace_back(&ThreadPool::run,this);
}

ThreadPool::~ThreadPool(){
    {
        std::lock_guard lock(mtx);
        terminate = true;
    }
    keepRun.notify_all();

    for(auto& thread:threads)
        if(thread.joinable()) thread.join();
}

ThreadPool& ThreadPool::getInstance(){
    size_t default_threads = std::thread::hardware_concurrency();
    if (default_threads == 0) default_threads = 1;
    static ThreadPool instance = ThreadPool(threadAmount != 0 ? threadAmount : default_threads);
    return instance;
}

void ThreadPool::setThreadAmount(size_t nums){threadAmount = nums;}