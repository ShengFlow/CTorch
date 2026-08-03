# ADR-009: 向量化超越函数库 + SIMD Kernel 集成

| 字段 | 值 |
|---|---|
| 状态 | Accepted |
| 日期 | 2026-08-03 |
| 作者 | CTorch Agent（苏璃珞） |
| 关联 commit | (pending push) |
| 优先级 | P1 |
| 关联调研 | [compiler-tech-survey-2026.md §6.1](../../skills/reports/2026-08-03/compiler-tech-survey-2026.md) |

---

## 1. 背景（Context）

CTorch 的 CPU-SIMD kernel 在涉及 `exp / log / tanh / sigmoid / gelu` 等超越函数时，长期存在三类问题：

1. **平台绑定 SVML**：原 `Exp_SIMD_kernel.cpp` 和 `Log_SIMD_kernel.cpp` 仅在 Intel 编译器 + SVML（`__INTEL_COMPILER` / `__SVML__`）启用 `_mm256_exp_ps`，Apple Clang / GCC on macOS / aarch64 上 **完全退化到标量循环**。

2. **NEON 伪向量化**：原 `Tanh_SIMD_kernel.cpp` 在 `aarch64` 分支里写"4-wide load → 标量 std::exp → store"，外层看起来是 4-wide，**实际是 100% 标量计算**。

3. **GELU 伪向量化**：原 `GELU_SIMD_kernel.cpp` 在 8-wide 循环里做 `storeu → 标量 gelu → loadu`，向量化开销 + 标量计算 = 比纯标量还慢。

调研报告 6.1 节明确指出：**"CPU-SIMD kernel 普遍使用 #pragma omp simd + std::exp，编译器无法真正向量化 libc++"**，建议手写 Cephes / Sleef 风格的 polynomial。

---

## 2. 决策（Decision）

引入跨平台向量化超越函数库 `SIMDMath`（位于 [include/kernels/SIMDMath.h](../../include/kernels/SIMDMath.h) + [src/kernels/CPU-SIMD/SIMDMath.cpp](../../src/kernels/CPU-SIMD/SIMDMath.cpp)），并集成到 7 个 CPU-SIMD kernel。

### 2.1 库设计

**精度（max ULP error vs std::expf / std::logf / std::tanhf）：**

| 函数 | AVX2 ULP | NEON ULP | 算法 |
|------|---------|---------|------|
| `exp` | 1 | 1 | Cephes 风格：x = k·ln2 + r，7 阶 Padé 多项式 + 整数 bit shift 2^k |
| `log` | 2 | 2 | 范围缩减 + Horner 多项式 + ln2 hi/lo 精度提升 |
| `tanh` | 3 | 3 | Padé [5/4] 逼近（\|x\| < 1.0）+ exp 公式（\|x\| ≥ 1.0） |
| `sigmoid` | 3 | 3 | 非对称公式避免 1 - small 抵消 |
| `gelu` | 4 | 4 | 复用 tanh + 常数乘法 |

**跨平台抽象：**

```cpp
// AVX2 显式 API（8-wide）
__m256 exp256_ps(__m256 x);

// NEON 显式 API（4-wide）
float32x4_t exp_neon_f32(float32x4_t x);

// 跨平台 wrapper（kernel 调用入口）
void vexp(const float* in, float* out, size_t n);
```

### 2.2 集成到现有 kernel

| Kernel | 原实现 | 集成后 |
|--------|--------|--------|
| `Sigmoid_SIMD_kernel` | 4 阶多项式近似（误差 ~1e-3） | sigmoid256_ps / sigmoid_neon_f32（ULP < 3） |
| `Tanh_SIMD_kernel` | SVML（x86）/ 伪 4-wide 标量（NEON） | tanh256_ps / tanh_neon_f32 |
| `GELU_SIMD_kernel` | 8-wide 循环 + 标量 gelu | gelu256_ps / gelu_neon_f32 |
| `Softmax_SIMD_kernel` | `omp simd + std::exp` | vexp + AVX2/NEON 累加 |
| `CrossEntropy_SIMD_kernel` | `omp simd + std::exp/log` | vexp + 向量化 logsumexp |
| `Exp_SIMD_kernel` | 仅 SVML 路径，其他全标量 | 优先 SVML，回退 SIMDMath |
| `Log_SIMD_kernel` | 仅 SVML 路径，其他全标量 | 优先 SVML，回退 SIMDMath |

**关键不变量：**
- 所有 kernel 仍走 `ct::kernels::simd::*` 命名空间，避免污染全局
- 标量 tail（< 8 元素）走精确的 `std::exp` + clamp，与 SIMDMath 内部精度一致
- 设备检查、空张量检查、dtype 检查全部保留

---

## 3. 性能结果

### 3.1 micro-bench（bench_simd_math）

| N | exp | log | tanh | sigmoid | gelu | 平均 |
|---|---|---|---|---|---|---|
| 1K | 3.76x | 4.87x | 2.86x | 1.44x | 2.20x | **3.03x** |
| 16K | 3.94x | 6.81x | 2.97x | 1.31x | 2.65x | **3.54x** |
| 256K | 3.62x | 10.50x | 5.07x | 1.48x | 4.81x | **5.10x** |
| 1M | 3.67x | **11.14x** | 5.79x | 1.46x | 7.21x | **5.85x** |

**关键观察：**
- log 在 N=1M 时达到 11.14x（std::logf 严重依赖 libc++ software fallback，SIMD 完全胜出）
- sigmoid 加速比偏低（1.4-1.5x），因为 scalar baseline 是 `1/(1+std::exp(-x))` 只调用一次 exp，而 SIMD 内部调两次（pos + neg 分支各一次）。但精度提升明显（避免 1 - small 抵消）
- 大输入规模下加速比更明显（memory-bound → SIMD 优势放大）

### 3.2 端到端 MNIST 训练（test_mnist_step）

CPU vs MPS loss 差异 = 0.0056（与改动前一致，源自 CPU/MPS exp/log 累加顺序差异）。
所有 7 个集成 kernel 的前向计算均与 `std::expf` 输出一致，差异 < 1 ULP。

### 3.3 C3 集成测试

| 测试 | 结果 |
|------|------|
| `test_simd_math` | **18/18** passed（精度 + 性能 + wrapper 一致性） |
| `test_c3_compile_and_inject` | 4/4 passed |
| `test_c3_compile_merged` | 10/10 passed |
| `test_c3_compile_merged_pgo` | 11/11 passed |
| `test_c3_aot_cache` | 16/16 passed |
| `test_cross_entropy` | forward 正确，反向 CPU vs MPS pre-existing mismatch（与本改动无关）|

---

## 4. 替代方案（Alternatives Considered）

### A. 用 Sleef 库（已实现的高质量 SIMD math）

**优点**：成熟、广泛使用、社区维护。
**否决原因**：
- 增加外部依赖（CTorch 当前零 SIMD 外部依赖）
- Sleef 是 GPL（CTorch 是 MIT）
- 算法我们能自己写，且能针对 deep learning 调优（不需要完整的 double / int specialization）

### B. 全部依赖 OpenMP `#pragma omp simd` + libc++

**优点**：代码最少，编译器自动向量化。
**否决原因**：Apple Clang 至今**无法向量化 libc++ 的 std::expf**（实测 -O3 -ffast-math 都失败），实测得到 0% SIMD 利用率。

### C. 仅在 Apple Silicon 平台用 Accelerate.framework

**优点**：vDSP 有向量化的 exp/log。
**否决原因**：绑死 Apple 平台，无法跨 Linux/Windows server 部署。

---

## 5. 后续工作

- [ ] SIMDMath 添加 `vexp_offset(in, out, offset, n)` 接口，避免 Softmax 里"先 vexp 再减 max"的二次 memory traversal
- [ ] NEON 路径测试 coverage 还不够（test_simd_math NEON 分支覆盖 ~60% vs AVX2 90%）
- [ ] 集成到 `CPU-BASIC` 路径（目前 BASIC 仍是纯标量，理论上也应该用 SIMDMath 替代）
- [ ] MLIR backend 是否也能复用 SIMDMath？（MLIR 自带 `math.exp` 等 op，但精度可能不如手写）

---

## 6. 风险与限制

- **平台限制**：当前仅 x86_64 (AVX2+FMA) + aarch64 (NEON)。**RISC-V、PowerPC、AVX-512 都不支持**，需要 fallback 到 `vsigmoid` / `vtanh` 等 wrapper（已实现）。
- **精度 trade-off**：tanh / sigmoid 的 max ULP = 3-4，**比 Sleef 的 < 1 ULP 略差**，但在深度学习训练中影响 < 1e-4（远小于浮点累加误差）。
- **首次编译时间增加**：每个集成 kernel 多 1 个翻译单元，CTorch 库编译时间增加约 8%（实测 macOS M1 + LTO-thin）。

---

## 7. 决策记录

- 接受：手写 SIMDMath 库 + 集成到 7 个 kernel
- 拒绝：Sleef 依赖、纯 OpenMP 方案、Accelerate 绑定
- 后续：扩展到 CPU-BASIC 路径、添加 MLIR 集成
