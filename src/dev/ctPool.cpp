module;

#include <mutex>
#include <functional>
#include <cwchar>
#include <thread>

module Accelerate;

size_t ThreadPool::threadAmount = 0;

void ThreadPool::run(){
    while(1){
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mtx);
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
        std::lock_guard<std::mutex> lock(mtx);
        terminate = true;
    }
    keepRun.notify_all();

    for(auto& thread:threads)
        if(thread.joinable()) thread.join();
}

ThreadPool& ThreadPool::getInstance(){
    static ThreadPool instance = ThreadPool(threadAmount != 0?threadAmount:std::thread::hardware_concurrency());
    return instance;
}

void ThreadPool::setThreadAmount(size_t nums){threadAmount = nums;}
