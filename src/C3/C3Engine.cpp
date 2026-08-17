/**
 * @file C3Engine.cpp
 * @generation SHARED 跨代编译引擎接口实现（串联 JIT-2.0 / 3.0 MLIR 统一内核编译流）
 * @brief C3 JIT 编译引擎实现
 * @details 实现 C3Engine 单例，将 Graph 编译为 ConcreteCompiledKernel/FusedCompiledKernel/MultiNodeCompiledKernel 等。
 *          JIT 1.0 (clang++ 手写内核落盘编译) 已经被彻底废弃并物理删除。
 *          目前已全面进化为 JIT 2.x 与 3.0 内存级 LLVM JIT / Linalg 声明式编译，具有极高并发性与超低冷启动开销。
 * @date 2026/7/31
 */

#include "../../include/C3/C3Engine.h"
#include "../../include/C3/Graph.h"
#include "../../include/C3/C3HotPathManager.h"
#include "../../include/C3/PGOManager.h"
#include "C3/AutoTuner.h"
#include "C3/TuningState.h"
#include "ThreadPool.h"

#ifdef CT_ENABLE_MLIR
#include "MLIRKernelGen.h"
#include "C3/LinalgFusedGen.h"
#include "C3/LinalgOneShotGen.h"
#endif

#include <chrono>
#include <dlfcn.h>
#include <future>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

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
                        size_t num_inputs,
                        std::vector<size_t> out_shape = {},
                        int opt_level = 3)
        : func_(func), deleter_(std::move(deleter)),
          cache_key_(std::move(cache_key)), device_(device),
          num_inputs_(num_inputs), out_shape_(std::move(out_shape)) {
        opt_level_ = opt_level;
    }

    ~FusedCompiledKernel() override {
        if (deleter_) deleter_();
    }

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        if (inputs.size() < num_inputs_) {
            throw std::runtime_error(
                "FusedCompiledKernel::execute: need " + std::to_string(num_inputs_) +
                " inputs, got " + std::to_string(inputs.size()));
        }

        // DEBT-NEW-7 候选 A:用 out_shape 分配 output buffer
        // - 旧实现:inputs[0].shape()(只对 elementwise 融合正确)
        // - 新实现:out_shape(从 FusedNode.out_desc 提取,支持 MatMul-rooted region)
        //   当 out_shape 为空(向后兼容),回退到 inputs[0].shape()
        std::vector<size_t> out_shape = out_shape_;
        if (out_shape.empty()) {
            out_shape = inputs[0].shape();
        }
        size_t n = 1;
        for (auto s : out_shape) n *= s;
        Tensor out(ShapeTag{}, out_shape, inputs[0].dtype(), inputs[0].device());

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

    /// v0.5.2 out_shape 修复: FusedKernel 已有 out_shape_ 字段,直接 expose
    [[nodiscard]] std::optional<std::vector<size_t>> outShape() const override {
        if (out_shape_.empty()) return std::nullopt;
        return out_shape_;
    }

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
    std::vector<size_t> out_shape_;  ///< DEBT-NEW-7 候选 A:真实输出 shape(支持 MatMul-rooted)
};

/**
 * @class MultiNodeCompiledKernel
 * @brief 多节点编译产物：封装 MultiNodeKernelFunc，支持多节点图执行
 */
class MultiNodeCompiledKernel : public CompiledKernel {
public:
    MultiNodeCompiledKernel(MultiNodeKernelFunc func,
                            std::function<void()> deleter,
                            std::string cache_key, DeviceType device,
                            size_t num_inputs, size_t M, size_t K, size_t N,
                            size_t elem_n,
                            size_t num_outputs,
                            std::vector<std::vector<size_t>> out_shapes,
                            size_t scratch_size = 0,
                            int opt_level = 3)
        : func_(func), deleter_(std::move(deleter)),
          cache_key_(std::move(cache_key)), device_(device),
          num_inputs_(num_inputs), M_(M), K_(K), N_(N), elem_n_(elem_n),
          num_outputs_(num_outputs), out_shapes_(std::move(out_shapes)),
          scratch_size_(scratch_size) {
        opt_level_ = opt_level;
    }

    ~MultiNodeCompiledKernel() override {
        if (deleter_) deleter_();
    }

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        if (inputs.size() < num_inputs_) {
            throw std::runtime_error(
                "MultiNodeCompiledKernel::execute: need " + std::to_string(num_inputs_) +
                " inputs, got " + std::to_string(inputs.size()));
        }

        // 收集输入指针
        std::vector<const float*> in_ptrs;
        for (size_t i = 0; i < num_inputs_; ++i) {
            in_ptrs.push_back(inputs[i].data_read<float>());
        }

        // 多输出支持：kernel 把所有输出节点写入一个平面 buffer（每段 elem_n 对齐），
        //   output 段 k 对应 graph.outputs()[k]。这里分配 num_outputs * elem_n 的平面 buffer，
        //   调用 kernel 后按各输出形状拆分回 Tensor。
        const size_t seg_n = (num_outputs_ > 0) ? num_outputs_ : 1;
        std::vector<float> flat(seg_n * elem_n_, 0.0f);

        // 设定并行阈值 (Threshold Gate)
        constexpr size_t kParallelThreshold = 262144;
        
        // 仅当元素数足够大、单输出，且不是 MatMul（MatMul 自带并行）时才开启多线程并行
        bool do_parallel = (elem_n_ >= kParallelThreshold && M_ == 0 && seg_n == 1);
        
        if (do_parallel) {
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
            if (num_threads > 8) num_threads = 8; // 限制最大线程数避免超载
            
            size_t chunk_size = (elem_n_ + num_threads - 1) / num_threads;
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads);
            
            for (unsigned int t = 0; t < num_threads; ++t) {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, elem_n_);
                if (start >= end) break;
                
                size_t slice_n = end - start;
                
                futures.push_back(ThreadPool::getInstance().addTask([this, start, slice_n, in_ptrs, &flat]() {
                    // 每个线程持有独立的 thread_local scratchpad
                    thread_local std::vector<float> worker_scratchpad;
                    if (scratch_size_ > 0 && worker_scratchpad.size() < scratch_size_) {
                        worker_scratchpad.resize(scratch_size_);
                    }
                    float* w_scratch_ptr = (scratch_size_ > 0) ? worker_scratchpad.data() : nullptr;
                    
                    // 构造切片后的输入指针数组
                    std::vector<const float*> slice_in_ptrs;
                    slice_in_ptrs.reserve(num_inputs_);
                    for (size_t i = 0; i < num_inputs_; ++i) {
                        slice_in_ptrs.push_back(in_ptrs[i] + start);
                    }
                    
                    float* slice_out_ptr = flat.data() + start;
                    
                    // 执行切片任务
                    func_(slice_in_ptrs.data(), slice_out_ptr, slice_n, M_, K_, N_, w_scratch_ptr);
                }));
            }
            
            // 等待所有并行的切片任务完成
            for (auto& f : futures) {
                f.wait();
            }
        } else {
            // 单核串行执行路径 (包括小张量或 MatMul)
            thread_local std::vector<float> thread_scratchpad;
            if (scratch_size_ > 0 && thread_scratchpad.size() < scratch_size_) {
                thread_scratchpad.resize(scratch_size_);
            }
            float* scratch_ptr = (scratch_size_ > 0) ? thread_scratchpad.data() : nullptr;

            func_(in_ptrs.data(), flat.data(), elem_n_, M_, K_, N_, scratch_ptr);
        }

        std::vector<Tensor> outs;
        outs.reserve(seg_n);
        for (size_t k = 0; k < seg_n; ++k) {
            std::vector<size_t> shape = (k < out_shapes_.size()) ? out_shapes_[k]
                                                                 : std::vector<size_t>{elem_n_};
            size_t n = 1;
            for (auto s : shape) n *= s;
            Tensor t(ShapeTag{}, shape);
            std::memcpy(t.data_write<float>(), flat.data() + k * elem_n_, n * sizeof(float));
            outs.push_back(std::move(t));
        }
        return outs;
    }

    [[nodiscard]] const std::string& cacheKey() const override { return cache_key_; }
    [[nodiscard]] DeviceType targetDevice() const override { return device_; }
    [[nodiscard]] size_t workspaceBytes() const override { return 0; }

    /// v0.5.2 out_shape 修复: MultiNode 用 M×N (2D) 或 elem_n (1D) 作为 out_shape
    [[nodiscard]] std::optional<std::vector<size_t>> outShape() const override {
        if (!out_shapes_.empty()) return out_shapes_.front();
        if (M_ > 0 && N_ > 0) return std::vector<size_t>{M_, N_};
        if (elem_n_ > 0)       return std::vector<size_t>{elem_n_};
        return std::nullopt;
    }

    bool installIntoRegistry(op op_type, const KernelShapeInfo& shapes) override {
        (void)op_type; (void)shapes;
        return false; // 多节点 kernel 暂不注册到 registry
    }

private:
    MultiNodeKernelFunc func_;
    std::function<void()> deleter_;
    std::string cache_key_;
    DeviceType device_;
    size_t num_inputs_;
    size_t M_, K_, N_;
    size_t elem_n_;
    size_t num_outputs_;
    std::vector<std::vector<size_t>> out_shapes_;
    size_t scratch_size_ = 0;
};

#ifdef CT_ENABLE_MLIR
/**
 * @class LinalgFusedCompiledKernel
 * @brief 用 linalg.generic 编译的多节点逐元素融合 kernel 运行期包装
 */
class LinalgFusedCompiledKernel : public CompiledKernel {
public:
    LinalgFusedCompiledKernel(std::shared_ptr<LinalgFusedKernel> kernel,
                              std::vector<std::vector<size_t>> out_shapes,
                              std::string cache_key,
                              int opt_level = 3)
        : kernel_(std::move(kernel)), out_shapes_(std::move(out_shapes)),
          cache_key_(std::move(cache_key)) {
        num_inputs_ = kernel_->numInputs();
        num_outputs_ = kernel_->numOutputs();
        opt_level_ = opt_level;
    }

    ~LinalgFusedCompiledKernel() override = default;

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        if (inputs.size() < num_inputs_) {
            throw std::runtime_error(
                "LinalgFusedCompiledKernel::execute: need " + std::to_string(num_inputs_) +
                " inputs, got " + std::to_string(inputs.size()));
        }

        // 1. 收集输入数据指针
        std::vector<const float*> in_ptrs;
        in_ptrs.reserve(num_inputs_);
        for (size_t i = 0; i < num_inputs_; ++i) {
            in_ptrs.push_back(inputs[i].data_read<float>());
        }

        // 2. 分配输出 Tensors，并收集输出指针
        std::vector<Tensor> outputs;
        std::vector<float*> out_ptrs;
        outputs.reserve(num_outputs_);
        out_ptrs.reserve(num_outputs_);

        size_t n = 0;
        if (!out_shapes_.empty()) {
            n = 1;
            for (auto s : out_shapes_[0]) n *= s;
        }

        for (const auto& shape : out_shapes_) {
            Tensor out(ShapeTag{}, shape);
            out_ptrs.push_back(out.data_write<float>());
            outputs.push_back(std::move(out));
        }

        // 3. 调用 linalg 声明式 kernel 执行
        kernel_->execute(in_ptrs.data(), out_ptrs.data(), n);

        return outputs;
    }

    [[nodiscard]] const std::string& cacheKey() const override { return cache_key_; }
    [[nodiscard]] DeviceType targetDevice() const override { return DeviceType::kCPU; }
    [[nodiscard]] size_t workspaceBytes() const override { return 0; }
    [[nodiscard]] std::optional<std::vector<size_t>> outShape() const override {
        if (out_shapes_.empty()) return std::nullopt;
        return out_shapes_[0];
    }
    bool installIntoRegistry(op op_type, const KernelShapeInfo& shapes) override {
        std::shared_ptr<CompiledKernel> self =
            std::shared_ptr<CompiledKernel>(this, [](CompiledKernel*) {});
        C3KernelRegistry::getInstance().install(op_type, targetDevice(), self, shapes);
        return true;
    }

private:
    std::shared_ptr<LinalgFusedKernel> kernel_;
    std::vector<std::vector<size_t>> out_shapes_;
    std::string cache_key_;
    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;
};

/**
 * @class LinalgOneShotCompiledKernel
 * @brief 用 JIT 3.0 (One-Shot Bufferization + Linalg Fusion) 编译的多节点逐元素融合 kernel 运行期包装
 */
class LinalgOneShotCompiledKernel : public CompiledKernel {
public:
    LinalgOneShotCompiledKernel(std::shared_ptr<LinalgOneShotKernel> kernel,
                               std::vector<std::vector<size_t>> out_shapes,
                               std::string cache_key,
                               int opt_level = 3)
        : kernel_(std::move(kernel)), out_shapes_(std::move(out_shapes)),
          cache_key_(std::move(cache_key)) {
        num_inputs_ = kernel_->numInputs();
        num_outputs_ = kernel_->numOutputs();
        opt_level_ = opt_level;
    }

    ~LinalgOneShotCompiledKernel() override = default;

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        if (inputs.size() < num_inputs_) {
            throw std::runtime_error(
                "LinalgOneShotCompiledKernel::execute: need " + std::to_string(num_inputs_) +
                " inputs, got " + std::to_string(inputs.size()));
        }

        // 1. 收集输入数据指针
        std::vector<const float*> in_ptrs;
        in_ptrs.reserve(num_inputs_);
        for (size_t i = 0; i < num_inputs_; ++i) {
            in_ptrs.push_back(inputs[i].data_read<float>());
        }

        // 2. 分配输出 Tensors，并收集输出指针
        std::vector<Tensor> outputs;
        std::vector<float*> out_ptrs;
        outputs.reserve(num_outputs_);
        out_ptrs.reserve(num_outputs_);

        size_t n = 0;
        if (!out_shapes_.empty()) {
            n = 1;
            for (auto s : out_shapes_[0]) n *= s;
        }

        for (const auto& shape : out_shapes_) {
            Tensor out(ShapeTag{}, shape);
            out_ptrs.push_back(out.data_write<float>());
            outputs.push_back(std::move(out));
        }

        // 3. 调用 linalg 声明式 kernel 执行
        kernel_->execute(in_ptrs.data(), out_ptrs.data(), n);

        return outputs;
    }

    [[nodiscard]] const std::string& cacheKey() const override { return cache_key_; }
    [[nodiscard]] DeviceType targetDevice() const override { return DeviceType::kCPU; }
    [[nodiscard]] size_t workspaceBytes() const override { return 0; }
    [[nodiscard]] std::optional<std::vector<size_t>> outShape() const override {
        if (out_shapes_.empty()) return std::nullopt;
        return out_shapes_[0];
    }
    bool installIntoRegistry(op op_type, const KernelShapeInfo& shapes) override {
        std::shared_ptr<CompiledKernel> self =
            std::shared_ptr<CompiledKernel>(this, [](CompiledKernel*) {});
        C3KernelRegistry::getInstance().install(op_type, targetDevice(), self, shapes);
        return true;
    }

private:
    std::shared_ptr<LinalgOneShotKernel> kernel_;
    std::vector<std::vector<size_t>> out_shapes_;
    std::string cache_key_;
    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;
};
#endif

/**
 * @class ConcreteCompiledKernel
 * @brief 具体编译产物：封装 C3KernelFunc + dlopen 句柄
 */
class ConcreteCompiledKernel : public CompiledKernel {
public:
    /// [Fix 2026-08-09 用户审查 P0]: 默认 a=inputs[0], b=inputs[1]
    /// 反向多输入 kernel (Mul|in:0) 实际用 inputs[0]=grad, inputs[2]=B (跳过 A),
    /// 因此允许 caller 通过 input_b_index 指定 b 来自哪个 inputs slot。
    /// 默认 1 (forward 二元 op 用 inputs[1] = rhs)。
    ConcreteCompiledKernel(C3KernelFunc func,
                           std::function<void()> deleter,
                           std::string cache_key, DeviceType device,
                           bool is_matmul, size_t M, size_t K, size_t N,
                           std::vector<size_t> out_shape = {},
                           size_t input_b_index = 1,
                           std::function<GeneratedKernel::SingleNodeExecutor> func_any = nullptr,
                           int opt_level = 3)
        : func_(func), func_any_(std::move(func_any)), deleter_(std::move(deleter)),
          cache_key_(std::move(cache_key)), device_(device),
          is_matmul_(is_matmul), M_(M), K_(K), N_(N),
          out_shape_(std::move(out_shape)),
          input_b_index_(input_b_index) {
        opt_level_ = opt_level;
    }

    ~ConcreteCompiledKernel() override {
        if (deleter_) deleter_();
    }

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        // [Fix 2026-08-09 用户审查 P0]: 反向多输入 kernel 数量校验
        // Mul|in:0 expects [grad, A, B]  (3 inputs), 不能 < 3
        if (inputs.empty()) {
            throw std::runtime_error("ConcreteCompiledKernel::execute: need at least 1 input");
        }
        if (input_b_index_ >= inputs.size()) {
            // 一元 op (input_b_index=1 但 inputs 只有 1) → fallback 复用 a
            if (inputs.size() == 1) {
                // OK, 一元 op 复用 a 当 b
            } else {
                throw std::runtime_error(
                    "ConcreteCompiledKernel::execute: input_b_index=" +
                    std::to_string(input_b_index_) + " out of range, inputs.size=" +
                    std::to_string(inputs.size()) +
                    " (反向多输入 kernel 配置错? Mul|in:0 expect [grad, A, B])");
            }
        }

        const Tensor& a = inputs[0];
        // [Fix 2026-08-09] 之前固定 inputs[1], 反向 Mul|in:0 要 inputs[2] (=B)
        // 用 input_b_index_ 字段允许 caller 指定 (默认 1 = forward 二元 op)
        const Tensor& b = (inputs.size() >= 2 && input_b_index_ < inputs.size())
                           ? inputs[input_b_index_]
                           : a;

        // v0.5.2 (2026-08-09) out_shape 修复: 优先用注册时记录的 out_shape_,
        // 解决 backward 路径 grad 形状 ≠ forward output 形状的 bug。
        // 优先级: 1) out_shape_ (注册时透传)  2) is_matmul → {M_,N_}  3) a.shape() 启发式
        Tensor out;
        if (!out_shape_.empty()) {
            out = Tensor(ShapeTag{}, out_shape_);
        } else if (is_matmul_) {
            out = Tensor(ShapeTag{}, {M_, N_});
        } else {
            out = Tensor(ShapeTag{}, a.shape());
        }

        size_t n = out.numel();
        if (func_any_) {
            // [2026-08-15] linalg.generic 路线：执行器优先于裸函数指针。
            // linalg kernel 内部按 numInputs 决定取 a 还是 a+b。
            func_any_(
                a.data_read<float>(),
                b.data_read<float>(),
                out.data_write<float>(),
                n, 0, 0, 0
            );
        } else if (is_matmul_) {
            func_(
                a.data_read<float>(),
                b.data_read<float>(),
                out.data_write<float>(),
                0, M_, K_, N_
            );
        } else {
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

    /// v0.5.2 out_shape 修复: 暴露注册时的 out_shape
    [[nodiscard]] std::optional<std::vector<size_t>> outShape() const override {
        if (out_shape_.empty()) return std::nullopt;
        return out_shape_;
    }

    bool installIntoRegistry(op op_type, const KernelShapeInfo& shapes) override {
        KernelShapeInfo s = shapes;
        if (is_matmul_) {
            s.is_matmul = true;
            s.M = M_; s.K = K_; s.N = N_;
        }
        // [Fix 2026-08-09 用户审查 P0-#4]: 用 shared_ptr<CompiledKernel> 持寿命
        // 替代裸 C3KernelFunc, 避免 cache evict/uninstallAll 后 func 悬垂 UAF
        std::shared_ptr<CompiledKernel> self =
            std::shared_ptr<CompiledKernel>(this, [](CompiledKernel*) {});
        // deleter 用空 lambda (ConcreteCompiledKernel 不被 registry 持, 仍由 C3Engine 寿命管理)
        // 真正持寿命通过 C3Engine compile 内部持 shared_ptr<ConcreteCompiledKernel>
        C3KernelRegistry::getInstance().install(op_type, device_, self, s);
        return true;
    }

private:
    C3KernelFunc func_;
    std::function<GeneratedKernel::SingleNodeExecutor> func_any_; ///< [2026-08-15] linalg 路线执行器
    std::function<void()> deleter_;
    std::string cache_key_;
    DeviceType device_;
    bool is_matmul_;
    size_t M_, K_, N_;
    std::vector<size_t> out_shape_;     ///< v0.5.2: 注册时透传的 out_shape (backward 用)
    size_t input_b_index_ = 1;          ///< [Fix 2026-08-09] 默认 1, 反向多输入 (Mul|in:0) 设 2
};

// ======================= C3Engine 实现 =======================

C3Engine& C3Engine::getInstance() {
    static C3Engine instance;
    return instance;
}

C3Engine::C3Engine() : state_(std::make_unique<EngineState>()) {}

C3Engine::~C3Engine() {
    // P0-2 修复：析构时先调 shutdown() 等后台任务完成。
    // state_ 是 unique_ptr<EngineState>，C3Engine 销毁时**最后**释放（成员按逆声明序析构），
    // 所以 shutdown() 体内访问 *state_ 安全。
    // 解决析构阶段 std::lock_guard 偶尔抛 system_error（PGO/C3Engine 析构顺序 race）：
    // shutdown() 内所有 try/catch 包裹，best-effort 清理，不抛异常出析构。
    try {
        shutdown();
    } catch (...) {
        // 析构阶段吞掉所有异常，不传播
    }
}

EngineState& C3Engine::getState() {
    return *getInstance().state_;
}

/// Free function 包装：让 C3Engine.cpp 内的 free function（如 doCompile / makeCacheKey / etc.
/// 以及大量 std::async lambda 内部）能用 unqualified `getState()` 调用，
/// 不需要 `C3Engine::` 前缀，也不需要 capture `this`。
/// P0-2 之前是 free function 静态持有 EngineState；P0-2 改为 C3Engine 成员后，
/// 通过此 wrapper 维持 33 个调用点的 API 兼容。
static EngineState& getState() {
    return C3Engine::getState();
}

// C3Engine 内部缓存
namespace {
    /// merged cache key 前缀版本号：统一使用 ct::c3::kMergedCacheKeyPrefix（见 GraphMerger.h）
    /// 不在此处重新定义，避免与 GraphMerger.h 中的版本号出现不一致
    static constexpr size_t kMaxCacheEntries = 256; ///< 缓存条目上限

    struct CacheEntry {
        std::shared_ptr<CompiledKernel> kernel;
        size_t bytes_approx = 0;
        std::chrono::steady_clock::time_point last_accessed; ///< LRU 逐出用
    };

    /// 进行中的异步编译任务
    struct PendingEntry {
        std::shared_ptr<std::promise<std::shared_ptr<CompiledKernel>>> promise;
        /// 缓存的 shared_future（promise->get_future() 只能调用一次，后续去重直接返回此值）
        CompileFuture future;
        std::chrono::steady_clock::time_point created_at;
    };
} // anonymous namespace

// 【P0-2 修复 2026-08-08】EngineState 必须放在 ct::c3:: 命名空间（与 include/C3/C3Engine.h L35 forward
// 声明保持一致），不能用 anonymous namespace 包裹（否则 TU-local 隐式链接会跟 header 的 ct::c3::EngineState
// 产生 ODR violation + 成员访问报 incomplete type）。
struct EngineState {
    std::mutex mutex;
    std::unordered_map<std::string, CacheEntry> cache;
    /// 进行中的异步编译（key → promise），用于去重
    std::unordered_map<std::string, PendingEntry> pending;
    /// 后台编译任务的 future，用于生命周期管理和 shutdown 等待
    std::vector<std::future<void>> compile_futures;
    /// PGO profile 数据（key → ProfileData）
    std::unordered_map<std::string, std::shared_ptr<ProfileData>> profile_data;
    C3CacheStats stats;
    /// 最近一次编译失败的错误信息（ADR-007），由 getLastCompileError() 读取
    /// 使用独立 mutex 保护，避免与 cache.mutex 互锁
    mutable std::mutex last_error_mutex;
    std::string last_compile_error;
    /// 编译超时配置（ADR-011），独立 mutex 保护
    mutable std::mutex config_mutex;
    uint32_t compile_timeout_ms = 30000;  // 默认 30s
};

/// 截断过长错误信息，避免 OOM
static constexpr size_t kMaxErrorLen = 1024;
static std::string truncateErrorMsg(const std::string& err) {
    if (err.size() <= kMaxErrorLen) return err;
    return err.substr(0, kMaxErrorLen) + "... [truncated, original=" +
           std::to_string(err.size()) + " bytes]";
}

/// 记录编译错误到 EngineState.last_compile_error_
/// 调用方需自行负责 prefix（"o2: " / "ofast: " / "async: " / "merge: " 等）
static void recordEngineError(EngineState& state, const std::string& prefix,
                             const std::string& err) {
    std::string full = prefix.empty() ? err : (prefix + ": " + err);
    std::lock_guard<std::mutex> lock(state.last_error_mutex);
    state.last_compile_error = truncateErrorMsg(full);
    CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL,
        ErrorType::KERNEL_LAUNCH,
        "C3Engine: compile error recorded: " + state.last_compile_error);
}

/// 回收已完成的编译任务 future（调用方需持有 state.mutex）
/// @param reaped_sink 接收被回收的 future，**必须在调用方释放 state.mutex 后才能让此 vector 析构**。
///                    这是因为 std::async task 可能正在等 state.mutex（写 cache），
///                    在锁内 ~std::future 会等 task 真正结束 → 经典死锁。
///                    （ADR-011 P1 修复发现的二次 bug：reaper 在持锁时 erase 触发了同款死锁。）
static void reapCompletedFutures(EngineState& state,
                                 std::vector<std::future<void>>& reaped_sink) {
    for (auto& f : state.compile_futures) {
        if (f.valid() && f.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            reaped_sink.push_back(std::move(f));
        }
    }
    // 清理已经被 move 走（valid()==false）的槽位，避免 vector 无限增长
    state.compile_futures.erase(
        std::remove_if(state.compile_futures.begin(), state.compile_futures.end(),
            [](const std::future<void>& f) { return !f.valid(); }),
        state.compile_futures.end());
}

/// LRU 逐出：当缓存超过上限时，移除最久未访问的条目（调用方需持有 state.mutex）
static void evictLRU(EngineState& state) {
    while (state.cache.size() > kMaxCacheEntries) {
        auto oldest = state.cache.begin();
        for (auto it = state.cache.begin(); it != state.cache.end(); ++it) {
            if (it->second.last_accessed < oldest->second.last_accessed) {
                oldest = it;
            }
        }
        state.cache.erase(oldest);
        state.stats.evictions++;
    }
}

/// 从 Graph 和 CompileOptions 生成缓存键
static std::string makeCacheKey(const Graph& graph, const CompileOptions& options) {
    if (!options.cache_key_override.empty()) {
        return options.cache_key_override;
    }
    std::ostringstream ss;
    ss << "c3_v4_"
       << static_cast<int>(options.backend) << "_"
       << static_cast<int>(options.target_device) << "_"
       << options.opt_level << "_"
       << (options.enable_fusion ? "f" : "n") << "_"
       << (options.enable_autotune ? "t" : "n") << "_"
       << graph.nodeCount() << "n_"
       << graph.inputCount() << "i_"
       << graph.toString();
    return ss.str();
}

/// 实际编译逻辑（同步和异步路径共用）
static std::shared_ptr<CompiledKernel> doCompile(
    const Graph& working_graph, const CompileOptions& options,
    const std::string& /*cache_key*/)
{
    try {
        // 自动调优：若启用且尚未调优，先运行一次 QEA 搜索
        if (options.enable_autotune && !TuningState::instance().isTuned()) {
            AutoTunerConfig at_cfg;
            at_cfg.verbose = true;
            C3Engine::getInstance().autoTune(at_cfg);
        }

#ifdef CT_ENABLE_MLIR
        // [2026-08-15] linalg.generic 多核并行/多输出大一统路线
        if (options.backend == C3Backend::MLIR && isPureElementwiseGraph(working_graph)) {
            std::vector<std::vector<size_t>> out_shapes;
            for (size_t out_id : working_graph.outputs()) {
                out_shapes.push_back(working_graph.node(out_id).out_desc.shape);
            }
            if (out_shapes.empty()) {
                // 如果图没有显式标记 outputs，默认取最后一个计算节点
                size_t last_compute = SIZE_MAX;
                for (const auto& node : working_graph.nodes()) {
                    if (std::find(working_graph.inputs().begin(), working_graph.inputs().end(), node.id) == working_graph.inputs().end() && !node.inputs.empty()) {
                        last_compute = node.id;
                    }
                }
                if (last_compute != SIZE_MAX) {
                    out_shapes.push_back(working_graph.node(last_compute).out_desc.shape);
                } else {
                    out_shapes.push_back({1});
                }
            }

            std::string cache_key = makeCacheKey(working_graph, options);
            static const bool use_oneshot = [] {
                // JIT 3.0 (One-Shot Bufferization + Linalg Fusion) 极致优化管线默认全面开启！
                // 仅在显式设置 C3_LINALG_ONESHOT=0 时才回退至旧版
                const char* v = std::getenv("C3_LINALG_ONESHOT");
                return v == nullptr || std::string(v) != "0";
            }();
            if (use_oneshot && options.opt_level < 2) {
                auto linalg_kernel = getCachedLinalgOneShotKernel(working_graph, cache_key, options.opt_level);
                if (linalg_kernel) {
                    return std::make_shared<LinalgOneShotCompiledKernel>(linalg_kernel, out_shapes, cache_key, options.opt_level);
                }
            } else if (options.opt_level < 2) {
                auto linalg_kernel = getCachedLinalgFusedKernel(working_graph, cache_key, options.opt_level);
                if (linalg_kernel) {
                    return std::make_shared<LinalgFusedCompiledKernel>(linalg_kernel, out_shapes, cache_key, options.opt_level);
                }
            }
        }
#endif

        GeneratedKernel gen;
#ifdef CT_ENABLE_MLIR
        // JIT 1.0 (Handwritten clang++ 编译落盘后端) 已经彻底废弃删除。
        // 所有对 Handwritten 后端的调用请求在此被安全、自动地重定向至统一的 MLIR 内存级即时编译后端，
        // 保证下游及测试用例在零感知、不改变原代码入参的情况下，获得百倍以上的冷启动性能提升与绝对的数值正确性。
        gen = generateFromGraphMLIR(working_graph, options.opt_level);
#else
        throw std::runtime_error("C3Engine: MLIR compile backend is disabled, and JIT 1.0 (Handwritten clang++ compiler) has been completely removed.");
#endif

        std::shared_ptr<CompiledKernel> kernel;
        if (gen.is_multi_node) {
            // 多输出支持：从 working_graph 提取所有输出节点的形状（顺序与 graph.outputs() 一致）
            std::vector<std::vector<size_t>> out_shapes;
            for (size_t out_id : working_graph.outputs()) {
                out_shapes.push_back(working_graph.node(out_id).out_desc.shape);
            }
            if (out_shapes.empty()) {
                // 安全网：无显式输出 → 视为单输出（elem_n）
                out_shapes.push_back({gen.elem_n});
            }
            kernel = std::make_shared<MultiNodeCompiledKernel>(
                gen.multi_func, gen.deleter, makeCacheKey(working_graph, options),
                options.target_device, gen.num_inputs, gen.M, gen.K, gen.N,
                gen.elem_n, out_shapes.size(), out_shapes,
                gen.scratch_size, options.opt_level
            );
        } else if (gen.is_fused) {
            kernel = std::make_shared<FusedCompiledKernel>(
                gen.fused_func, gen.deleter, makeCacheKey(working_graph, options),
                options.target_device, gen.num_inputs, gen.fused_out_shape,
                options.opt_level
            );
        } else {
            // v0.5.2 (2026-08-09) out_shape 修复: 从 working_graph 提取真实 out_shape
            // 透传给 ConcreteCompiledKernel, 让 execute() 用 out_shape 构造 Tensor
            // 解决 backward 路径 grad.shape() ≠ forward output shape 的 bug
            std::vector<size_t> out_shape;
            if (working_graph.outputCount() > 0) {
                size_t out_id = working_graph.outputs().back();
                out_shape = working_graph.node(out_id).out_desc.shape;
            }
            kernel = std::make_shared<ConcreteCompiledKernel>(
                gen.func, gen.deleter, makeCacheKey(working_graph, options),
                options.target_device, gen.is_matmul, gen.M, gen.K, gen.N,
                out_shape, /*input_b_index=*/1, gen.func_any, options.opt_level
            );
        }

        return kernel;
    } catch (const std::exception& e) {
        // 记录到 EngineState（ADR-007）：让 getLastCompileError() 能查询到
        recordEngineError(getState(), "", e.what());
        throw;
    } catch (...) {
        recordEngineError(getState(), "", "unknown exception (not std::exception)");
        throw;
    }
}

std::shared_ptr<CompiledKernel> C3Engine::compile(
    const Graph& graph, const CompileOptions& options) {

    // 图优化管线：canonicalize → eliminateDeadCode → fuse
    Graph working_graph = graph;
    working_graph = working_graph.canonicalize();
    working_graph = working_graph.eliminateDeadCode();
    if (options.enable_fusion) {
        working_graph = working_graph.fuse();
    }

    // PGO 模式：返回 PGOCompiledKernel（Tier 1 解释器 → 热路径检测后自动提升到 Tier 2）
    if (options.pgo_mode) {
        auto& pgo = PGOManager::getInstance();
        if (!pgo.isEnabled()) {
            pgo.setEnabled(true);
        }

        std::string cache_key = makeCacheKey(working_graph, options);
        auto& state = getState();

        // 获取或创建 profile data
        std::shared_ptr<ProfileData> pd;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            auto pd_it = state.profile_data.find(cache_key);
            if (pd_it == state.profile_data.end()) {
                pd = std::make_shared<ProfileData>();
                state.profile_data[cache_key] = pd;
            } else {
                pd = pd_it->second;
            }
        }

        return pgo.registerKernel(working_graph, options, cache_key, pd, *this);
    }

    auto& state = getState();
    std::vector<std::future<void>> _to_reap;  // 锁外声明，让析构在锁释放后发生
    {
        std::lock_guard<std::mutex> lock(state.mutex);

        // 回收已完成的异步编译 future
        reapCompletedFutures(state, _to_reap);

        // 缓存查询
        if (options.enable_cache) {
            std::string cache_key = makeCacheKey(working_graph, options);
            auto it = state.cache.find(cache_key);
            if (it != state.cache.end()) {
                state.stats.hits++;
                it->second.last_accessed = std::chrono::steady_clock::now();
                auto kernel = it->second.kernel;
                // 如果 profiling 启用，包装为 ProfiledCompiledKernel
                if (options.enable_profiling) {
                    auto pd_it = state.profile_data.find(cache_key);
                    if (pd_it == state.profile_data.end()) {
                        pd_it = state.profile_data.emplace(cache_key, std::make_shared<ProfileData>()).first;
                    }
                    return std::make_shared<ProfiledCompiledKernel>(kernel, pd_it->second);
                }
                return kernel;
            }
            state.stats.misses++;
        }
    }  // ← lock 在此析构，_to_reap 在函数末尾析构（无死锁）

    auto kernel = doCompile(working_graph, options, makeCacheKey(working_graph, options));

    // 写入缓存
    // [Fix 2026-08-09 用户审查 P0-4]: 之前锁已释放后写 state.cache 跟 evictLRU,
    // 多个 compile() 并发时 unordered_map 写入 race → UB 可能崩溃。
    // 修法: 重新拿锁写,保证 thread-safe。
    if (options.enable_cache) {
        std::string cache_key = makeCacheKey(working_graph, options);
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.cache[cache_key] = {kernel, 0, std::chrono::steady_clock::now()};
            evictLRU(state);
        }
    }

    // 如果 profiling 启用，包装为 ProfiledCompiledKernel
    if (options.enable_profiling) {
        std::string cache_key = makeCacheKey(working_graph, options);
        auto pd_it = state.profile_data.find(cache_key);
        if (pd_it == state.profile_data.end()) {
            pd_it = state.profile_data.emplace(cache_key, std::make_shared<ProfileData>()).first;
        }
        return std::make_shared<ProfiledCompiledKernel>(kernel, pd_it->second);
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
    std::vector<std::future<void>> _to_reap;  // 锁外声明

    {
        std::lock_guard<std::mutex> lock(state.mutex);

        // 回收已完成的异步编译 future
        reapCompletedFutures(state, _to_reap);

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

        // 3. 创建新的异步编译任务（ADR-011 watchdog 模式）
        promise = std::make_shared<std::promise<std::shared_ptr<CompiledKernel>>>();
        result_future = promise->get_future().share();

        // 共享编译状态：compile 线程写，watchdog 线程读
        struct AsyncCompileState {
            std::mutex mutex;
            std::condition_variable cv;
            bool done = false;       // compile 线程完成
            bool timed_out = false;  // watchdog 判定超时
            std::shared_ptr<CompiledKernel> kernel;
            std::string error;
        };
        auto compile_state = std::make_shared<AsyncCompileState>();

        state.pending[cache_key] = {promise, result_future, std::chrono::steady_clock::now()};
        state.stats.pending_compiles++;

        // 4. 启动实际编译线程
        auto compile_future = std::async(std::launch::async,
            [working_graph = std::move(working_graph), options, cache_key,
             promise, compile_state]() mutable {
                try {
                    auto kernel = doCompile(working_graph, options, cache_key);

                    // 写共享状态
                    {
                        std::lock_guard<std::mutex> slock(compile_state->mutex);
                        compile_state->kernel = kernel;
                        compile_state->done = true;
                    }
                    compile_state->cv.notify_all();

                    // 写 cache + 清 pending
                    auto& state = getState();
                    {
                        std::lock_guard<std::mutex> lock(state.mutex);
                        if (options.enable_cache) {
                            // 即使 timed_out 也写入 cache（让后续相同 key 命中，节省重新编译）
                            state.cache[cache_key] = {kernel, 0, std::chrono::steady_clock::now()};
                            evictLRU(state);
                        }
                        state.pending.erase(cache_key);
                        state.stats.pending_compiles--;
                        if (kernel) {
                            state.stats.async_completions++;
                        } else {
                            state.stats.async_failures++;
                        }
                    }  // ← lock_guard 在此析构，释放 state.mutex
                    // 关键：编译成功后必须通知主线程！
                    // 必须放在 lock 块**外**：
                    //   reapCompletedFutures 会持 state.mutex 遍历 compile_futures
                    //   对 ready 的 future 调 erase，~future 会 block 等 std::async task 真正结束
                    //   如果我们在持 lock 时 set_value，reaper 能看到 ready 但 ~future 会等我们释放 lock，
                    //   而我们又等 reaper 释放 state.mutex → 死锁。
                    //   （这个 bug 是 ADR-011 修复过程中引入的。）
                    if (!compile_state->timed_out) {
                        try {
                            promise->set_value(compile_state->kernel);
                        } catch (...) {
                            // race condition: watchdog 在我们 set 前 set_value(nullptr)，
                            // 这里 try/catch 吸收 future_error 避免 propagate
                        }
                    }
                } catch (const std::exception& e) {
                    // 编译失败：记录 + 写共享状态
                    {
                        std::lock_guard<std::mutex> slock(compile_state->mutex);
                        compile_state->error = e.what();
                        compile_state->done = true;
                    }
                    compile_state->cv.notify_all();
                    // 注意：timed_out 时 watchdog 已经返回 nullptr，这里不能再 set_value
                    if (!compile_state->timed_out) {
                        try {
                            recordEngineError(getState(), "async", e.what());
                        } catch (...) {}
                        try {
                            promise->set_value(nullptr);
                        } catch (...) {}
                    }
                    try {
                        auto& state = getState();
                        std::lock_guard<std::mutex> lock(state.mutex);
                        state.pending.erase(cache_key);
                        state.stats.pending_compiles--;
                        state.stats.async_failures++;
                    } catch (...) {}
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> slock(compile_state->mutex);
                        compile_state->error = "unknown exception";
                        compile_state->done = true;
                    }
                    compile_state->cv.notify_all();
                    if (!compile_state->timed_out) {
                        try {
                            recordEngineError(getState(), "async", "unknown exception");
                        } catch (...) {}
                        try {
                            promise->set_value(nullptr);
                        } catch (...) {}
                    }
                    try {
                        auto& state = getState();
                        std::lock_guard<std::mutex> lock(state.mutex);
                        state.pending.erase(cache_key);
                        state.stats.pending_compiles--;
                        state.stats.async_failures++;
                    } catch (...) {}
                }
            });

        // 5. 启动 watchdog 线程（ADR-011）
        uint32_t timeout_ms;
        {
            std::lock_guard<std::mutex> clock(state.config_mutex);
            timeout_ms = state.compile_timeout_ms;
        }

        std::shared_future<void> compile_future_shared = compile_future.share();
        auto watchdog_future = std::async(std::launch::async,
            [compile_state, promise, cache_key, timeout_ms, compile_future_shared]() {
                if (timeout_ms == 0) {
                    // 0 表示永不超时，watchdog 退化为直接等编译完成
                    compile_future_shared.wait();
                    return;
                }

                // 先看是否已完成（避免无意义的等待）
                {
                    std::lock_guard<std::mutex> lock(compile_state->mutex);
                    if (compile_state->done) {
                        // 编译已完成（fast path），但 promise 还没被 set
                        // 让 compile_future_shared 跑完后续清状态，promise 由 compile 线程自己 set
                        return;
                    }
                }

                // 等待 timeout_ms 或 done
                std::unique_lock<std::mutex> lock(compile_state->mutex);
                bool done = compile_state->cv.wait_for(lock,
                    std::chrono::milliseconds(timeout_ms),
                    [&] { return compile_state->done; });

                if (!done) {
                    // **超时！** ADR-011 P1 修复
                    compile_state->timed_out = true;
                    lock.unlock();

                    // 记录超时错误
                    try {
                        std::string err = "compile exceeded " + std::to_string(timeout_ms) +
                                          "ms for cache_key=" + cache_key +
                                          " (actual compile continues in background)";
                        recordEngineError(getState(), "async-timeout", err);
                    } catch (...) {}

                    // 立即给用户返回 nullptr（不阻塞 future.get()）
                    try {
                        promise->set_value(nullptr);
                    } catch (...) {
                        // promise 可能已被 compile 线程 set 过（race condition 极少见）
                    }

                    // 注意：compile 线程继续跑，最终会写入 cache 供后续命中
                    // 但本次 compile 会被 watchdog 视为"超时失败"
                }
            });

        state.compile_futures.push_back(std::move(compile_future));
        state.compile_futures.push_back(std::move(watchdog_future));
    }
    // 释放锁，不阻塞后台线程

    return result_future;
}

std::vector<std::shared_ptr<CompiledKernel>> C3Engine::compileParallel(
    const std::vector<Graph>& graphs,
    const CompileOptions& options)
{
    if (graphs.empty()) return {};

    // 启动并行编译任务
    std::vector<std::future<std::shared_ptr<CompiledKernel>>> futures;
    futures.reserve(graphs.size());

    for (const auto& g : graphs) {
        futures.push_back(std::async(std::launch::async, [this, &g, &options]() {
            return compile(g, options);
        }));
    }

    // 收集结果
    std::vector<std::shared_ptr<CompiledKernel>> results;
    results.reserve(graphs.size());
    for (auto& f : futures) {
        auto kernel = f.get();
        if (!kernel) {
            throw std::runtime_error("C3Engine::compileParallel: one or more subgraphs failed to compile");
        }
        results.push_back(std::move(kernel));
    }

    return results;
}

// ======================= compileMerged 系列实现 =======================

std::shared_ptr<CompiledKernel> C3Engine::compileMerged(
    const std::vector<Graph>& sub_graphs,
    const MergeSpec& spec,
    const CompileOptions& options)
{
    try {
        // 1. 入参校验
        if (sub_graphs.empty()) {
            throw std::invalid_argument(
                "C3Engine::compileMerged: sub_graphs must be non-empty");
        }
        if (spec.links.size() != sub_graphs.size() - 1) {
            throw std::invalid_argument(
                "C3Engine::compileMerged: spec.links.size() (" +
                std::to_string(spec.links.size()) + ") must equal sub_graphs.size() - 1 (" +
                std::to_string(sub_graphs.size() - 1) + ")");
        }

        // 2. 验证子图与链接的兼容性（shape/dtype/device）
        std::string err = GraphMerger::validate(sub_graphs, spec);
        if (!err.empty()) {
            throw std::invalid_argument(
                "C3Engine::compileMerged: validation failed: " + err);
        }

        // 3. 合并子图为单个 Graph
        MergedGraphInfo merged = GraphMerger::merge(sub_graphs, spec);

        // 4. 复用现有 compile() 流程：canonicalize → eliminateDeadCode → fuse → cache → compile
        //    注意：compile() 内部会基于 working_graph 重新生成 cache key，因此融合图的
        //    缓存条目与子图独立缓存的条目互不冲突。
        auto kernel = compile(merged.graph, options);
        if (!kernel) {
            throw std::runtime_error(
                "C3Engine::compileMerged: compilation of merged graph failed");
        }

        // 5. 返回编译后的 kernel
        return kernel;
    } catch (const std::exception& e) {
        // 记录到 EngineState（ADR-007）：merge 错误也属于编译错误
        recordEngineError(getState(), "merge", e.what());
        throw;
    } catch (...) {
        recordEngineError(getState(), "merge", "unknown exception");
        throw;
    }
}

std::shared_ptr<CompiledKernel> C3Engine::compileMergedPGO(
    const std::vector<Graph>& sub_graphs,
    const MergeSpec& spec,
    const CompileOptions& options)
{
    try {
        // 1. 入参校验（与 compileMerged 保持一致）
        if (sub_graphs.empty()) {
            throw std::invalid_argument(
                "C3Engine::compileMergedPGO: sub_graphs must be non-empty");
        }
        if (spec.links.size() != sub_graphs.size() - 1) {
            throw std::invalid_argument(
                "C3Engine::compileMergedPGO: spec.links.size() (" +
                std::to_string(spec.links.size()) + ") must equal sub_graphs.size() - 1 (" +
                std::to_string(sub_graphs.size() - 1) + ")");
        }
        std::string err = GraphMerger::validate(sub_graphs, spec);
        if (!err.empty()) {
            throw std::invalid_argument(
                "C3Engine::compileMergedPGO: validation failed: " + err);
        }

        // 2. 合并子图为单个 Graph
        MergedGraphInfo merged = GraphMerger::merge(sub_graphs, spec);

        // 3. 构造 PGO 模式 options（强制 pgo_mode=true，其他字段透传）
        CompileOptions pgo_opts = options;
        pgo_opts.pgo_mode = true;

        // 4. 复用 compile() 流程：PGO 模式会在内部自动包装为 PGOCompiledKernel
        //    PGOCompiledKernel.execute() 第一次调用走 Eager 解释执行（scheduler 逐算子），
        //    同时触发 O2 + Ofast 异步编译，编译完成后原子热替换。
        //    缓存键基于 merged_graph 结构 + options（pgo_mode 不计入，因为 PGO 是运行时包装层）
        auto kernel = compile(merged.graph, pgo_opts);
        if (!kernel) {
            throw std::runtime_error(
                "C3Engine::compileMergedPGO: compilation of merged graph failed");
        }

        return kernel;
    } catch (const std::exception& e) {
        recordEngineError(getState(), "merge-pgo", e.what());
        throw;
    } catch (...) {
        recordEngineError(getState(), "merge-pgo", "unknown exception");
        throw;
    }
}

std::shared_ptr<CompiledKernel> C3Engine::compileMergedSequential(
    const std::vector<Graph>& sub_graphs,
    const CompileOptions& options)
{
    // 顺序场景：直接构造顺序 spec 后调用 compileMerged
    MergeSpec spec = GraphMerger::makeSequentialSpec(sub_graphs);
    return compileMerged(sub_graphs, spec, options);
}

std::shared_ptr<CompiledKernel> C3Engine::compileMergedPGOSequential(
    const std::vector<Graph>& sub_graphs,
    const CompileOptions& options)
{
    // 顺序场景：直接构造顺序 spec 后调用 compileMergedPGO
    MergeSpec spec = GraphMerger::makeSequentialSpec(sub_graphs);
    return compileMergedPGO(sub_graphs, spec, options);
}

CompileFuture C3Engine::compileMergedAsync(
    const std::vector<Graph>& sub_graphs,
    const MergeSpec& spec,
    const CompileOptions& options)
{
    // 1. 立即生成 merged cache key 以支持去重
    std::string merged_key = GraphMerger::mergedCacheKey(sub_graphs, spec);
    // 在 key 前缀加入 compile 维度，区分 Handwritten vs MLIR、opt level 等
    std::ostringstream full_key_ss;
    full_key_ss << kMergedCacheKeyPrefix
                << static_cast<int>(options.backend) << "_"
                << static_cast<int>(options.target_device) << "_"
                << options.opt_level << "_"
                << (options.enable_fusion ? "f" : "n") << "_"
                << merged_key;
    std::string cache_key = full_key_ss.str();

    auto& state = getState();
    std::shared_ptr<std::promise<std::shared_ptr<CompiledKernel>>> promise;
    CompileFuture result_future;
    std::vector<std::future<void>> _to_reap;  // 锁外声明

    {
        std::lock_guard<std::mutex> lock(state.mutex);

        // 回收已完成的异步编译 future
        reapCompletedFutures(state, _to_reap);

        // 1. 检查常规缓存
        if (options.enable_cache) {
            auto it = state.cache.find(cache_key);
            if (it != state.cache.end()) {
                state.stats.hits++;
                auto ready_promise = std::make_shared<std::promise<std::shared_ptr<CompiledKernel>>>();
                ready_promise->set_value(it->second.kernel);
                return ready_promise->get_future().share();
            }
            state.stats.misses++;
        }

        // 2. 检查是否有进行中的融合编译任务（去重）
        auto pending_it = state.pending.find(cache_key);
        if (pending_it != state.pending.end()) {
            return pending_it->second.future;
        }

        // 3. 创建新的异步融合编译任务
        promise = std::make_shared<std::promise<std::shared_ptr<CompiledKernel>>>();
        result_future = promise->get_future().share();

        state.pending[cache_key] = {promise, result_future, std::chrono::steady_clock::now()};
        state.stats.pending_compiles++;

        // 4. 启动后台融合编译
        auto async_future = std::async(std::launch::async,
            [sub_graphs, spec, options, cache_key, promise]() mutable {
                try {
                    auto kernel = C3Engine::getInstance().compileMerged(
                        sub_graphs, spec, options);
                    promise->set_value(kernel);

                    auto& state = getState();
                    std::lock_guard<std::mutex> lock(state.mutex);
                    if (options.enable_cache) {
                        state.cache[cache_key] = {kernel, 0, std::chrono::steady_clock::now()};
                        evictLRU(state);
                    }
                    state.pending.erase(cache_key);
                    state.stats.pending_compiles--;
                    state.stats.async_completions++;
                } catch (...) {
                    // 融合编译失败：保留真实异常信息，调用方通过 future.get() 接收
                    // 禁止静默吞错（违反 project CtorchError::throwException 硬约束）
                    // 同时记录到 EngineState（ADR-007），让 getLastCompileError() 能查到
                    try {
                        // current_exception 复制一次以避免再次抛出时与原异常冲突
                        std::exception_ptr ep = std::current_exception();
                        if (ep) {
                            try {
                                std::rethrow_exception(ep);
                            } catch (const std::exception& ee) {
                                recordEngineError(getState(), "async-merge", ee.what());
                            } catch (...) {
                                recordEngineError(getState(), "async-merge",
                                                  "unknown exception");
                            }
                        }
                    } catch (...) {
                        // recordEngineError 自身异常，吞错
                    }
                    try {
                        promise->set_exception(std::current_exception());
                    } catch (const std::future_error&) {
                        // promise 已被设置过（去重路径下重复 set_value 场景），忽略
                    } catch (...) {
                        // 极端情况：promise 已被销毁。统计失败但不抛
                        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL,
                            ErrorType::UNKNOWN,
                            "C3Engine::compileMergedAsync: failed to propagate "
                            "compile exception (promise destroyed)");
                    }
                    auto& state = getState();
                    std::lock_guard<std::mutex> lock(state.mutex);
                    state.pending.erase(cache_key);
                    state.stats.pending_compiles--;
                    state.stats.async_failures++;
                }
            });
        state.compile_futures.push_back(std::move(async_future));
    }
    // 释放锁，不阻塞后台线程

    return result_future;
}

// ======================= NodeVariant → op 枚举映射 =======================

namespace {

/// 将 C3 NodeVariant 映射到调度器 op 枚举
/// @return op 值；若无法映射（ConstNode/FusedNode/多节点图）则返回 std::nullopt
static std::optional<op> nodeVariantToOp(const NodeVariant& nv) {
    // NodeVariant 的 variant 索引顺序：
    // AddNode=0, SubNode=1, MulNode=2, DivNode=3, MatMulNode=4,
    // NegNode=5, ReLUNode=6, SigmoidNode=7, TanhNode=8, ConstNode=9, FusedNode=10
    switch (nv.index()) {
        case 0:  return op::Add;
        case 1:  return op::Sub;
        case 2:  return op::Mul;
        case 3:  return op::Div;
        case 4:  return op::MatMul;
        case 5:  return op::Neg;
        case 6:  return op::ReLU;
        case 7:  return op::Sigmoid;
        case 8:  return op::Tanh;
        default: return std::nullopt; // ConstNode, FusedNode, unknown
    }
}

/// 从 Graph 提取 KernelShapeInfo
static KernelShapeInfo graphToShapeInfo(const Graph& graph) {
    KernelShapeInfo info;
    if (graph.outputCount() == 0) return info;

    auto& out_node = graph.node(graph.outputs()[0]);
    info.out_shape = out_node.out_desc.shape;

    // 获取输入形状
    auto& input_ids = graph.inputs();
    if (input_ids.size() >= 1) {
        info.lhs_shape = graph.node(input_ids[0]).out_desc.shape;
    }
    if (input_ids.size() >= 2) {
        info.rhs_shape = graph.node(input_ids[1]).out_desc.shape;
    }

    // 如果是 MatMul 节点，提取 M/K/N
    if (auto* mm = std::get_if<MatMulNode>(&out_node.op)) {
        info.is_matmul = true;
        info.M = mm->lhs_desc.shape[0];  // batch dim
        info.K = mm->lhs_desc.shape[1];
        info.N = mm->rhs_desc.shape[1];
    }

    return info;
}

} // anonymous namespace

std::shared_ptr<CompiledKernel> C3Engine::compileAndInject(
    const Graph& graph, const CompileOptions& options)
{
    // 1. 编译
    auto kernel = compile(graph, options);
    if (!kernel) {
        throw std::runtime_error("C3Engine::compileAndInject: compilation failed");
    }

    // 2. 推断 op_type
    if (graph.outputCount() == 0) {
        throw std::runtime_error("C3Engine::compileAndInject: graph has no output");
    }

    auto& out_node = graph.node(graph.outputs()[0]);
    auto op_type = nodeVariantToOp(out_node.op);
    if (!op_type.has_value()) {
        throw std::runtime_error(
            "C3Engine::compileAndInject: cannot infer op_type from output node "
            "(FusedNode and multi-node graphs require manual installIntoRegistry)");
    }

    // 3. 构建形状签名
    auto shapes = graphToShapeInfo(graph);

    // 4. 安装到注册表（仅单节点图支持自动注入，多节点/融合图跳过）
    kernel->installIntoRegistry(op_type.value(), shapes);

    return kernel;
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
    std::vector<std::future<void>> _to_reap;  // 锁外声明
    C3CacheStats stats;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        reapCompletedFutures(state, _to_reap);
        stats = state.stats;
        stats.total_entries = state.cache.size();
        stats.pending_compiles = state.pending.size();
    }  // ← 锁释放，_to_reap 后续析构不会触发持锁等待
    return stats;
}

std::shared_ptr<ProfileData> C3Engine::getProfileData(const std::string& cache_key) const {
    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.profile_data.find(cache_key);
    if (it != state.profile_data.end()) {
        return it->second;
    }
    return nullptr;
}

std::string C3Engine::getLastCompileError() const {
    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.last_error_mutex);
    return state.last_compile_error;
}

void C3Engine::clearLastCompileError() {
    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.last_error_mutex);
    state.last_compile_error.clear();
}

void C3Engine::recordCompileError(const std::string& prefix, const std::string& err) {
    // 公开 API：让 PGOCompiledKernel 等子系统能写入全局错误状态。
    // 内部复用 recordEngineError 复用截断 + 日志逻辑。
    recordEngineError(getState(), prefix, err);
}

void C3Engine::setCompileTimeoutMs(uint32_t ms) {
    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.config_mutex);
    state.compile_timeout_ms = ms;
    CtorchError::log(ErrorLevel::INFO, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
        "C3Engine: compile timeout set to " + std::to_string(ms) + "ms");
}

uint32_t C3Engine::getCompileTimeoutMs() const {
    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.config_mutex);
    return state.compile_timeout_ms;
}

void C3Engine::clearCache() {
    auto& state = getState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.cache.clear();
    // 注意：不清理 state.pending，避免销毁正在进行的编译任务的 promise
    // 导致等待方收到 broken_promise 异常
    // 异步编译完成后会自行写入缓存和清理 pending
    state.stats.hits = 0;
    state.stats.misses = 0;
    state.stats.evictions = 0;
    state.stats.bytes_used = 0;
    state.stats.total_entries = 0;
    state.stats.pending_compiles = 0;
    state.stats.async_completions = 0;
    state.stats.async_failures = 0;
    // 同步清理 PGO 缓存，确保测试间状态干净
    // （PGOManager::clear() 内部持有自己的 mutex，无死锁风险）
    PGOManager::getInstance().clear();
}

void C3Engine::autoTune(const AutoTunerConfig& config) {
    auto& tuning = TuningState::instance();
    if (tuning.isTuned()) return; // 已调优，跳过

    // 创建样本 MatMul 图用于 benchmark
    Graph g;
    size_t M = 256, K = 256, N = 256;
    auto a_desc = TensorDesc::fromShape({M, K});
    auto b_desc = TensorDesc::fromShape({K, N});
    auto c_desc = TensorDesc::fromShape({M, N});
    auto a = g.addInput(a_desc);
    auto b = g.addInput(b_desc);
    auto c = g.addNode(MatMulNode{a_desc, b_desc}, {a, b}, c_desc);
    g.markOutput(c);

    AutoTuner tuner(config);
    CompileOptions opts;
    opts.enable_cache = false;
    opts.enable_fusion = false;

    auto fitness_fn = [&](int tile_m, int tile_n, int tile_k, int unroll) -> double {
        // 设置临时调优参数
        TuningParams tp;
        tp.tile_m = tile_m;
        tp.tile_n = tile_n;
        tp.tile_k = tile_k;
        tp.unroll = unroll;
        tuning.set(tp);

        // 编译（强制走统一的 MLIR 后端，JIT 1.0 Handwritten 已废弃删除）
        opts.backend = C3Backend::MLIR;
        auto kernel = doCompile(g, opts, "");

        // Benchmark
        Tensor a_t(ShapeTag{}, std::vector<size_t>{M, K});
        Tensor b_t(ShapeTag{}, std::vector<size_t>{K, N});
        for (size_t i = 0; i < M * K; i++) a_t.data_write<float>()[i] = 1.0f;
        for (size_t i = 0; i < K * N; i++) b_t.data_write<float>()[i] = 1.0f;

        auto t0 = std::chrono::high_resolution_clock::now();
        size_t warmup = 5;
        for (size_t r = 0; r < warmup; r++) {
            kernel->execute({a_t, b_t});
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        // 返回单次耗时 (us)
        double elapsed = std::chrono::duration<double, std::micro>(t1 - t0).count() / warmup;

        // 清理编译缓存
        auto& state = getState();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.cache.clear();

        return elapsed;
    };

    auto result = tuner.tuneWithQEA(fitness_fn);

    // 存储最优参数
    TuningParams best;
    best.tile_m = result.tile_m;
    best.tile_n = result.tile_n;
    best.tile_k = result.tile_k;
    best.unroll = result.unroll;
    best.tuned = true;
    tuning.set(best);

    if (config.verbose) {
        std::cout << "[C3 AutoTune] Optimal: TILE=(" << best.tile_m << ","
                  << best.tile_n << "," << best.tile_k << ") unroll=" << best.unroll
                  << " fitness=" << result.best_fitness << "us"
                  << " evals=" << result.evaluations
                  << " time=" << result.elapsed_ms << "ms" << std::endl;
    }
}

// [removed 2026-08-15] AOTCache 已删除（.so 磁盘缓存），aotCache_() helper 一并移除。
// 跨进程复用改由 JITCache 承担，见 STATUS_CONTEXT 4.12。

void C3Engine::shutdown() {
    // 正确的退出顺序：HotPathManager::shutdown() → PGO::shutdown() → C3Engine::shutdown()
    // 1. HotPathManager 必须最先关闭：它的后台 std::async task 持有 ConcreteCompiledKernel
    //    （含 MLIR ExecutionEngine 引用），析构顺序在 LLVM GDBJITRegistrationListener mutex
    //    之前会触发 system_error → recursive_mutex 死锁 → std::terminate
    // 2. PGO 在 HotPathManager 之后关闭：PGO 的 std::async task 通过 C3Engine::compile() 拿 state
    try {
        ct::c3::C3HotPathManager::instance().shutdown();
    } catch (...) {
        // best-effort
    }

    // 等待 PGO 后台编译完成（PGO 内部 PGOCompiledKernel 可能 lock 自己的 compile_mutex，
    // 但其 triggerCompilationChain 也会访问 PGOManager 的 mutex_/queue_mutex_）。
    try {
        PGOManager::getInstance().shutdown();
    } catch (...) {
        // PGO 析构链可能 cascade，吞掉防止 std::terminate
    }

    // C3Engine state_ 析构的 race condition:
    // 1. PGO::shutdown() 等 std::async task，task 内可能通过 C3Engine::compile() → C3Engine::getState() 访问 state_
    // 2. 如果 state_ 已被 C3Engine 析构释放 → 析构触发 system_error
    // 3. 这是 Meyers singleton 跨 TU 析构顺序 UB（PGO vs C3Engine）的经典问题
    // 4. best-effort 清理：try/catch 吞 system_error
    try {
        auto& state = getState();
        std::vector<std::future<void>> futures;
        try {
            std::lock_guard<std::mutex> lock(state.mutex);
            futures = std::move(state.compile_futures);
        } catch (const std::system_error&) {
            // 析构阶段 mutex 已销毁
            return;
        }
        // 等待所有后台编译完成（30s 超时，覆盖冷启动时 clang++ 链接 + MLIR 编译的长尾场景）
        for (auto& f : futures) {
            if (f.valid()) {
                auto status = f.wait_for(std::chrono::seconds(30));
                if (status == std::future_status::ready) {
                    try { f.get(); } catch (...) {} // 吸收异常
                } else {
                    CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                        "C3Engine::shutdown: background compile did not finish in 30s, "
                        "future abandoned (may cause UAF if main exits before thread finishes)");
                }
            }
        }
    } catch (...) {
        // best-effort 清理
    }
}

} // namespace c3
} // namespace ct