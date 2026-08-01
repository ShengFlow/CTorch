/**
 * @file JITEngine.cpp
 * @brief C3 JIT 编译引擎实现
 * @details 实现 JITEngine 单例，将 Graph 编译为 ConcreteCompiledKernel。
 *          当前 EXP-1 阶段使用 HandwrittenKernelGen 生成 C++ kernel 并通过
 *          clang++ 编译为 .so 加载。后续阶段将替换为 MLIR+LLVM 后端。
 * @date 2026/7/31
 */

#include "../../include/JIT/JITEngine.h"
#include "../../include/JIT/Graph.h"
#include "HandwrittenKernelGen.h"

#include <dlfcn.h>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

namespace ct {
namespace jit {

// ======================= ConcreteCompiledKernel =======================

/**
 * @class ConcreteCompiledKernel
 * @brief 具体编译产物：封装 C3KernelFunc + dlopen 句柄
 */
class ConcreteCompiledKernel : public CompiledKernel {
public:
    ConcreteCompiledKernel(C3KernelFunc func, void* dl_handle,
                           std::string cache_key, DeviceType device,
                           bool is_matmul, size_t M, size_t K, size_t N)
        : func_(func), dl_handle_(dl_handle),
          cache_key_(std::move(cache_key)), device_(device),
          is_matmul_(is_matmul), M_(M), K_(K), N_(N) {}

    ~ConcreteCompiledKernel() override {
        if (dl_handle_) {
            dlclose(dl_handle_);
        }
    }

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        if (inputs.size() < 2) {
            throw std::runtime_error("ConcreteCompiledKernel::execute: need at least 2 inputs");
        }

        const Tensor& a = inputs[0];
        const Tensor& b = inputs[1];

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
    void* dl_handle_;
    std::string cache_key_;
    DeviceType device_;
    bool is_matmul_;
    size_t M_, K_, N_;
};

// ======================= JITEngine 实现 =======================

JITEngine& JITEngine::getInstance() {
    static JITEngine instance;
    return instance;
}

// JITEngine 内部缓存
namespace {
    struct CacheEntry {
        std::shared_ptr<CompiledKernel> kernel;
        size_t bytes_approx = 0;
    };

    struct EngineState {
        std::mutex mutex;
        std::unordered_map<std::string, CacheEntry> cache;
        JITCacheStats stats;
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
    ss << "c3_v1_"
       << static_cast<int>(options.target_device) << "_"
       << options.opt_level << "_"
       << graph.nodeCount() << "n_"
       << graph.inputCount() << "i_"
       << std::hash<std::string>{}(graph.toString());
    return ss.str();
}

std::shared_ptr<CompiledKernel> JITEngine::compile(
    const Graph& graph, const CompileOptions& options) {

    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.mutex);

    // 缓存查询
    if (options.enable_cache) {
        std::string cache_key = makeCacheKey(graph, options);
        auto it = state.cache.find(cache_key);
        if (it != state.cache.end()) {
            state.stats.hits++;
            return it->second.kernel;
        }
        state.stats.misses++;
    }

    // 生成并编译 kernel
    GeneratedKernel gen = generateFromGraph(graph);

    std::string cache_key = makeCacheKey(graph, options);
    auto kernel = std::make_shared<ConcreteCompiledKernel>(
        gen.func, gen.dl_handle, cache_key, options.target_device,
        gen.is_matmul, gen.M, gen.K, gen.N
    );

    // 写入缓存
    if (options.enable_cache) {
        state.cache[cache_key] = {kernel, 0};
    }

    return kernel;
}

std::shared_ptr<CompiledKernel> JITEngine::getKernel(const std::string& cache_key) const {
    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.cache.find(cache_key);
    if (it != state.cache.end()) {
        return it->second.kernel;
    }
    return nullptr;
}

JITCacheStats JITEngine::getCacheStats() const {
    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.mutex);
    JITCacheStats stats = state.stats;
    stats.total_entries = state.cache.size();
    return stats;
}

void JITEngine::clearCache() {
    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.cache.clear();
    state.stats = JITCacheStats{};
}

} // namespace jit
} // namespace ct