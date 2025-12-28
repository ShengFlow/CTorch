module;

#include <functional>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <tuple>
#include <future>
#include <type_traits>

export module Accelerate;

export class ThreadPool{
private:
    std::queue<std::function<void()>> tasks;
    std::vector<std::thread> threads;
    static size_t threadAmount;
    bool terminate = false;
    std::mutex mtx;
    std::condition_variable keepRun;

    ThreadPool(size_t nums);
    void run();
public:
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    static ThreadPool& getInstance();
    static void setThreadAmount(size_t nums);

    template<typename Callable, typename... Args>
auto addTask(Callable&& callable, Args&&... args){
    using Ret = std::invoke_result_t<Callable, Args...>;
    
    auto promise = std::make_shared<std::promise<Ret>>();
    auto future = promise->get_future();
    
    auto task = [promise, 
                 toCall = std::forward<Callable>(callable),
                 argsTuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        try {
            if constexpr(std::is_same_v<Ret, void>){
                std::apply(toCall, argsTuple);
                promise->set_value();
            } else {
                auto val = std::apply(toCall, argsTuple);
                promise->set_value(std::move(val));
            }
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    };
    
    {
        std::lock_guard<std::mutex> lock(mtx);
        tasks.emplace(std::move(task));
    }
    
    keepRun.notify_one();
    return future;
}
};
