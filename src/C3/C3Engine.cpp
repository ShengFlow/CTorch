/**
 * @file C3Engine.cpp
 * @brief C3 JIT 编译引擎实现
 * @details 实现 C3Engine 单例，将 Graph 编译为 ConcreteCompiledKernel。
 *          当前 EXP-1 阶段使用 HandwrittenKernelGen 生成 C++ kernel 并通过
 *          clang++ 编译为 .so 加载。后续阶段将替换为 MLIR+LLVM 后端。
 * @date 2026/7/31
 */

#include "../../include/C3/C3Engine.h"
#include "../../include/C3/Graph.h"
#include "HandwrittenKernelGen.h"

#ifdef CT_ENABLE_MLIR
#include "MLIRKernelGen.h"
#endif

#include <dlfcn.h>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

namespace ct {
namespace c3 {

// ======================= ConcreteCompiledKernel =======================

// ======================= 编译内核实现 =======================

/**
 * @class FusedCompiledKernel
 * @brief 融合编译产物：封装 FusedKernelFunc，支持多输入
 */
class FusedCompiledKernel : public CompiledKernel {
public:
    FusedCompiledKernel(FusedKernelFunc func,
                        std::function<void()> deleter,
                        std::string cache_key, DeviceType device,
                        size_t num_inputs)
        : func_(func), deleter_(std::move(deleter)),
          cache_key_(std::move(cache_key)), device_(device),
          num_inputs_(num_inputs) {}

    ~FusedCompiledKernel() override {
        if (deleter_) deleter_();
    }

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        if (inputs.size() < num_inputs_) {
            throw std::runtime_error(
                "FusedCompiledKernel::execute: need " + std::to_string(num_inputs_) +
                " inputs, got " + std::to_string(inputs.size()));
        }

        size_t n = inputs[0].numel();
        Tensor out(ShapeTag{}, inputs[0].shape());

        std::vector<const float*> in_ptrs;
        for (size_t i = 0; i < num_inputs_; ++i) {
            in_ptrs.push_back(inputs[i].data_read<float>());
        }

        func_(in_ptrs.data(), out.data_write<float>(), n);

        return {out};
    }

    [[nodiscard]] const std::string& cacheKey() const override { return cache_key_; }
    [[nodiscard]] DeviceType targetDevice() const override { return device_; }
    [[nodiscard]] size_t workspaceBytes() const override { return 0; }

    bool installIntoRegistry(op op_type, const KernelShapeInfo& shapes) override {
        (void)op_type; (void)shapes;
        return false; // 融合 kernel 暂不注册到 registry
    }

private:
    FusedKernelFunc func_;
    std::function<void()> deleter_;
    std::string cache_key_;
    DeviceType device_;
    size_t num_inputs_;
};

/**
 * @class ConcreteCompiledKernel
 * @brief 具体编译产物：封装 C3KernelFunc + dlopen 句柄
 */
class ConcreteCompiledKernel : public CompiledKernel {
public:
    ConcreteCompiledKernel(C3KernelFunc func,
                           std::function<void()> deleter,
                           std::string cache_key, DeviceType device,
                           bool is_matmul, size_t M, size_t K, size_t N)
        : func_(func), deleter_(std::move(deleter)),
          cache_key_(std::move(cache_key)), device_(device),
          is_matmul_(is_matmul), M_(M), K_(K), N_(N) {}

    ~ConcreteCompiledKernel() override {
        if (deleter_) deleter_();
    }

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        if (inputs.empty()) {
            throw std::runtime_error("ConcreteCompiledKernel::execute: need at least 1 input");
        }

        const Tensor& a = inputs[0];
        // 二元算子用第二个输入，一元算子复用第一个（kernel 内部忽略不用的参数）
        const Tensor& b = (inputs.size() >= 2) ? inputs[1] : a;

        // 创建输出张量
        Tensor out;
        if (is_matmul_) {
            out = Tensor(ShapeTag{}, {M_, N_});
            func_(
                a.data_read<float>(),
                b.data_read<float>(),
                out.data_write<float>(),
                0, M_, K_, N_
            );
        } else {
            size_t n = a.numel();
            out = Tensor(ShapeTag{}, a.shape());
            func_(
                a.data_read<float>(),
                b.data_read<float>(),
                out.data_write<float>(),
                n, 0, 0, 0
            );
        }

        return {out};
    }

    [[nodiscard]] const std::string& cacheKey() const override { return cache_key_; }
    [[nodiscard]] DeviceType targetDevice() const override { return device_; }
    [[nodiscard]] size_t workspaceBytes() const override { return 0; }

    bool installIntoRegistry(op op_type, const KernelShapeInfo& shapes) override {
        KernelShapeInfo s = shapes;
        if (is_matmul_) {
            s.is_matmul = true;
            s.M = M_; s.K = K_; s.N = N_;
        }
        C3KernelRegistry::getInstance().install(op_type, device_, func_, s);
        return true;
    }

private:
    C3KernelFunc func_;
    std::function<void()> deleter_;
    std::string cache_key_;
    DeviceType device_;
    bool is_matmul_;
    size_t M_, K_, N_;
};

// ======================= C3Engine 实现 =======================

C3Engine& C3Engine::getInstance() {
    static C3Engine instance;
    return instance;
}

// C3Engine 内部缓存
namespace {
    struct CacheEntry {
        std::shared_ptr<CompiledKernel> kernel;
        size_t bytes_approx = 0;
    };

    /// 进行中的异步编译任务
    struct PendingEntry {
        std::shared_ptr<std::promise<std::shared_ptr<CompiledKernel>>> promise;
        /// 缓存的 shared_future（promise->get_future() 只能调用一次，后续去重直接返回此值）
        CompileFuture future;
        std::chrono::steady_clock::time_point created_at;
    };

    struct EngineState {
        std::mutex mutex;
        std::unordered_map<std::string, CacheEntry> cache;
        /// 进行中的异步编译（key → promise），用于去重
        std::unordered_map<std::string, PendingEntry> pending;
        C3CacheStats stats;
    };

    EngineState& getState() {
        static EngineState state;
        return state;
    }
} // anonymous namespace

/// 从 Graph 和 CompileOptions 生成缓存键
static std::string makeCacheKey(const Graph& graph, const CompileOptions& options) {
    if (!options.cache_key_override.empty()) {
        return options.cache_key_override;
    }
    std::ostringstream ss;
    ss << "c3_v3_"
       << static_cast<int>(options.backend) << "_"
       << static_cast<int>(options.target_device) << "_"
       << options.opt_level << "_"
       << (options.enable_fusion ? "f" : "n") << "_"
       << graph.nodeCount() << "n_"
       << graph.inputCount() << "i_"
       << std::hash<std::string>{}(graph.toString());
    return ss.str();
}

/// 实际编译逻辑（同步和异步路径共用）
static std::shared_ptr<CompiledKernel> doCompile(
    const Graph& working_graph, const CompileOptions& options,
    const std::string& /*cache_key*/)
{
    GeneratedKernel gen;
    switch (options.backend) {
#ifdef CT_ENABLE_MLIR
        case C3Backend::MLIR:
            gen = generateFromGraphMLIR(working_graph);
            break;
#endif
        case C3Backend::Handwritten:
        default:
            gen = generateFromGraph(working_graph);
            break;
    }

    std::shared_ptr<CompiledKernel> kernel;
    if (gen.is_fused) {
        kernel = std::make_shared<FusedCompiledKernel>(
            gen.fused_func, gen.deleter, makeCacheKey(working_graph, options),
            options.target_device, gen.num_inputs
        );
    } else {
        kernel = std::make_shared<ConcreteCompiledKernel>(
            gen.func, gen.deleter, makeCacheKey(working_graph, options),
            options.target_device, gen.is_matmul, gen.M, gen.K, gen.N
        );
    }

    return kernel;
}

std::shared_ptr<CompiledKernel> C3Engine::compile(
    const Graph& graph, const CompileOptions& options) {

    // 算子融合：若启用，先对图做融合
    Graph working_graph = graph;
    if (options.enable_fusion) {
        working_graph = working_graph.fuse();
    }

    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.mutex);

    // 缓存查询
    if (options.enable_cache) {
        std::string cache_key = makeCacheKey(working_graph, options);
        auto it = state.cache.find(cache_key);
        if (it != state.cache.end()) {
            state.stats.hits++;
            return it->second.kernel;
        }
        state.stats.misses++;
    }

    auto kernel = doCompile(working_graph, options, makeCacheKey(working_graph, options));

    // 写入缓存
    if (options.enable_cache) {
        state.cache[makeCacheKey(working_graph, options)] = {kernel, 0};
    }

    return kernel;
}

CompileFuture C3Engine::compileAsync(
    const Graph& graph, const CompileOptions& options) {

    // 算子融合
    Graph working_graph = graph;
    if (options.enable_fusion) {
        working_graph = working_graph.fuse();
    }

    auto& state = getState();
    std::string cache_key = makeCacheKey(working_graph, options);

    std::shared_ptr<std::promise<std::shared_ptr<CompiledKernel>>> promise;
    CompileFuture result_future;

    {
        std::lock_guard<std::mutex> lock(state.mutex);

        // 1. 检查常规缓存
        if (options.enable_cache) {
            auto it = state.cache.find(cache_key);
            if (it != state.cache.end()) {
                state.stats.hits++;
                // 返回已就绪的 future
                auto ready_promise = std::make_shared<std::promise<std::shared_ptr<CompiledKernel>>>();
                ready_promise->set_value(it->second.kernel);
                return ready_promise->get_future().share();
            }
            state.stats.misses++;
        }

        // 2. 检查是否有进行中的编译任务（去重）
        auto pending_it = state.pending.find(cache_key);
        if (pending_it != state.pending.end()) {
            return pending_it->second.future;
        }

        // 3. 创建新的异步编译任务
        promise = std::make_shared<std::promise<std::shared_ptr<CompiledKernel>>>();
        result_future = promise->get_future().share();

        state.pending[cache_key] = {promise, result_future, std::chrono::steady_clock::now()};
        state.stats.pending_compiles++;
    }
    // 释放锁，不阻塞后台线程

    // 启动后台编译
    std::thread([working_graph = std::move(working_graph), options, cache_key,
                 promise]() mutable {
        try {
            auto kernel = doCompile(working_graph, options, cache_key);
            promise->set_value(kernel);

            // 编译完成后写入缓存
            auto& state = getState();
            std::lock_guard<std::mutex> lock(state.mutex);
            if (options.enable_cache) {
                state.cache[cache_key] = {kernel, 0};
            }
            state.pending.erase(cache_key);
            state.stats.pending_compiles--;
            state.stats.async_completions++;
        } catch (...) {
            // 编译失败：存储 nullptr，不崩溃，eager 路径继续工作
            try {
                promise->set_value(nullptr);
            } catch (...) {
                // promise 可能已被销毁（极少数情况）
            }
            auto& state = getState();
            std::lock_guard<std::mutex> lock(state.mutex);
            state.pending.erase(cache_key);
            state.stats.pending_compiles--;
            state.stats.async_failures++;
        }
    }).detach();

    return result_future;
}

std::shared_ptr<CompiledKernel> C3Engine::getKernel(const std::string& cache_key) const {
    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.cache.find(cache_key);
    if (it != state.cache.end()) {
        return it->second.kernel;
    }
    return nullptr;
}

C3CacheStats C3Engine::getCacheStats() const {
    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.mutex);
    C3CacheStats stats = state.stats;
    stats.total_entries = state.cache.size();
    stats.pending_compiles = state.pending.size();
    return stats;
}

void C3Engine::clearCache() {
    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.cache.clear();
    // 清空进行中的异步任务（promise 被销毁，future 端会收到 broken_promise）
    state.pending.clear();
    state.stats = C3CacheStats{};
}

} // namespace c3
} // namespace ct