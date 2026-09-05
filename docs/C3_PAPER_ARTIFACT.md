# C3 论文复现说明（Artifact）

> 更新日期：2026-09-04
> 对应：paper/c3_paper_zh.tex（《C3：面向自动微分深度学习框架的异步非阻塞 JIT 编译优化器》）
> 代码仓库：https://github.com/ShengFlow/CTorch（分支 feature-DCU，C3 位于子模块 c3/）

## 1. 环境

- macOS Apple Silicon（M3 Pro, AArch64），Sequoia
- Xcode Command Line Tools（clang）
- MLIR/LLVM 22.1.8（Homebrew：`brew install llvm`，需 18+）
- CMake 3.16+

## 2. 构建

```bash
# C3 启用构建（论文主实验）
git clone --recurse-submodules git@github.com:ShengFlow/CTorch.git
cd CTorch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCT_ENABLE_MLIR=ON
cmake --build build -j8

# 纯 Eager 对照构建（用于 C3 ON vs Eager 对比）
cmake -S . -B build-eager -DCMAKE_BUILD_TYPE=Release -DCT_ENABLE_MLIR=ON -DCT_DISABLE_C3=ON
cmake --build build-eager -j8
```

> 构建选项：`-O3 -ffast-math -march=native -flto=thin`（Release 默认）。

## 3. 正确性复现（测试全绿）

```bash
cd build
./test_c3_backward              # backward 正确性 12/12 PASS（含 CrossEntropy/Softmax/MIMO）
./test_c3_graph                 # 图正确性 115/115 PASS
./test_c3_compile_merged        # 10/10 PASS
./test_c3_compile_merged_pgo    # 11/11 PASS
```

## 4. 性能复现（口径与命令）

> 性能测量须在**无并发构建的干净机器**上进行（避免后台编译污染计时）。

### 端到端训练（MNIST-MLP 型）
```bash
# C3 ON（build/） vs 纯 Eager（build-eager/）
C3_AOT_CACHE_DIR=/tmp/c3c ./build/bench_mlp_ce_train          # 读 median=...
C3_AOT_CACHE_DIR=/tmp/c3c ./build-eager/bench_mlp_ce_train     # 读 median=...
```

### 反向融合（backward 主场）
```bash
# C3（build/） vs Eager backward（C3_DISABLE_BACKWARD=1）
C3_AOT_CACHE_DIR=/tmp/c3c ./build/bench_c3_backward_perf_clean                 # C3，稳态 p50 ≈1.0ms
C3_AOT_CACHE_DIR=/tmp/c3c C3_DISABLE_BACKWARD=1 ./build/bench_c3_backward_perf_clean  # Eager，p50 ≈16ms
```

### 端到端前向（宽 matmul 密集，eager 优势区）
```bash
./build/bench_wide_mlp_e2e 64 4096 4 20       # C3
./build-eager/bench_wide_mlp_e2e 64 4096 4 20 # Eager
```

## 5. 已整理的实测数据

- `docs/C3_PERF_UNIFIED_MATRIX.md`：三口径赢面地图（backward ~10-16x / 端到端≈Eager / 前向 matmul eager 占优）
- `docs/C3_BACKWARD_FIRST_CALL_BUG_REPORT.md`：backward 首访正确性修复（论文 correctness 背书）

## 6. 论文源

- `paper/c3_paper_zh.tex`（中文版）、`paper/c3_paper.tex`（英文版）
- 图：`paper/figures/*.png`（fig_arch/pipeline/e2e/ablation/prewalk/elementwise/mimo）
- 编译：`cd paper && latexmk -xelatex c3_paper_zh.tex`
