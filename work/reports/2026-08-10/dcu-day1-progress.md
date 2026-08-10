# DCU 接入 Day 1 进度 (2026-08-10)

## 今日成果总览

**8 个 commit push 到 `feature-DCU` 分支**,总投入 ~ 5 元 / 3.5 小时机时 (曙光智算 b02r1n05 节点)。

### Commit 列表 (按时间顺序)

| Commit | 标题 | 类别 | 影响 |
|---|---|---|---|
| `15d4c90` | DCU CMakeLists DTK path auto-detect + GCVMBridge v0.5.1 | Dev | DTK 26.04 路径自动探测 (3 fallback), GCVM 真实 API 14 个函数 |
| `acdc142` | Add 99 untracked source/header files | Dev | 节点 clone 拿到全部 99 源文件 |
| `70e2176` | MPS_flush_wait 调用点加 #ifdef __APPLE__ 包裹 (Linux build) | Fix | Tensor.cpp + ReLUNode.cpp 加 ifdef |
| `d505287` | MLIRToLLVMIR helper for DCU Plan A/B/C (v0.5 基础设施) | Dev | 13/13 单测 PASS, 新 API 抽离 |
| `4d19891` | AOTCache 解耦: 抽 IAOTCache 接口 (P1 风险) | Refactor | 接口+DI, P1 风险消除, 精度 97.1755% 保持 |
| `58a6d46` | MLIRKernelGen 抽公开 API + mlirToLLVMIRFromGraph 实装 | Dev | 真 pipeline 跑通 21/21 单测, 5ms 出 857 chars LLVM IR |
| `afb56ae` | Arena.h 显式 include <cstddef> (Linux build, std::max_align_t) | Fix | DTK clang 17 严格不 transitive |
| `57ad383` | Metal/MPS 文件 macOS-only 守卫 (Linux build) | Fix | MetalDevice.h + CMakeLists 加 ifdef |
| `6dc40bd` | CMakeLists 真正删 L89-90 .mm 引用 (Linux build) | Fix | macOS build 不破, Linux build 不编 .mm |
| `e22eb61` | if(APPLE) 改 CMAKE_SYSTEM_NAME + CT-MPS-OPs 顺序 + Graph.h <functional> (Linux+macOS build) | Fix | cmake 跨平台真坑 (APPLE 永远 false) |
| `1709778` | structured binding (auto [a, b]) 改 named 变量 (Linux build, OpenMP lambda capture) | Fix | OpenMP 严格模式 5 处 |
| `97bd266` | 5 类 Linux build 错误 (2026-08-10) | Fix | GCVMBridge comment + kDCU enum + MPS_flush_wait 2 处 + AMX 包 ifdef + filesystem link |

### 核心产出

#### 1. DCU 接入基础设施 (~ 1100 行)
- `include/C3/GCVMBridge.h` (54 行) + `src/C3/GCVMBridge.cpp` (~ 170 行, v0.5.1 真实 API)
- `include/C3/DCUCompiledKernel.h` (98 行) + `src/C3/DCUCompiledKernel.cpp` (263 行)
- `include/C3/MLIRToLLVMIR.h` (100 行) + `src/C3/MLIRToLLVMIR.cpp` (170 行, 13/13 单测)
- `src/C3/MLIRKernelGen.h/cpp` 抽公开 `buildMLIRModule` + `applyLoweringPipeline` (58a6d46)
- `src/tests/standalone/test_mlir_to_llvm_ir.cpp` (170 行, 4 类 test PASS)
- `src/tests/standalone/test_c3_dcu_hello.cpp` (169 行, Phase 2a-2f 流程)

#### 2. CMake 跨平台修复
- `if(APPLE)` 永远 false → `if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")`
- `CT-MPS-OPs` 引用顺序 (set 必须在 if APPLE 之前)
- `set(CT-Core ...)` 列表里删 L89-90 .mm (避免 macOS/Linux 重复添加)
- macOS-only 文件 (.mm) 全部用 `if(APPLE)` 包裹
- `find_library(stdc++fs)` 兜底 Linux libstdc++-12 < 12.3

#### 3. Linux build 适配 (~ 13 处)
- 头文件显式 include (`<cstddef>`, `<functional>`, `<optional>`)
- MPS 代码 `#ifdef __APPLE__` 守卫 (5 处: Tensor.cpp ×3, ReLUNode.cpp, GELUNode.cpp, GradAccumulator.cpp)
- AMX kernels `if(APPLE) list APPEND` + CtorchScheduler AMX registration `#ifdef __APPLE__`
- `DeviceType` enum 加 `kDCU = 7`, `kCount` 改 8, 同步 `CtorchScheduler.h` static_assert
- OpenMP structured binding 改 named 变量 (5 处)

### macOS 端回归 (验证精度不变)

| 测试 | 结果 |
|---|---|
| AOTCache 16/16 | PASS |
| MNIST 5 epoch 精度 | 97.1755% (= 基线 97.1755%) |
| test_c3_graph | 跑过 |
| test_mlir_to_llvm_ir 21/21 | PASS |

### 教训沉淀 (5 条 Agent Memory)

1. **MLIR 22.x 头路径 API 变化** (`mlir/IR/Module.h` 不存在, Bitcode 拆分, 翻译接口显式 include)
2. **C++ Interface 抽象时 shared type 位置** (避免循环 include, 放 interface 头)
3. **per-ctx 状态不要用 static once_flag** (MLIR 翻译接口 per-ctx 注册)
4. **CMake if(APPLE) 永远 false 陷阱** (裸 cmake 项目 APPLE 变量没 set)
5. **OpenMP lambda 不能 capture structured binding** (DTK clang 17 libomp 严格)
6. **macOS Apple Clang vs Linux clang 17 严格模式差异** (5 大点: transitive include, OpenMP, __APPLE__, CMake, variable 顺序)

## 跨 session 接力棒 (明早任务)

### 主线: 节点 b02r1n05 Plan A/B/C 真链路验证

1. **reset + pull + 增量 build** (97bd266 修复应该过)
   ```bash
   cd ~/CTorch
   git fetch origin && git reset --hard origin/feature-DCU
   cd build-dcu
   rm -rf CMakeCache.txt CMakeFiles/  # 仅清 cmake cache, 保留 .o
   module load compiler/dtk/26.04
   ~/.local/bin/cmake -DCT_ENABLE_DCU=ON -DCT_ENABLE_MLIR=OFF -DTEST_ENABLED=OFF \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
   make -k -j8 test_c3_dcu_hello 2>&1 | tee /tmp/build.log
   ```
2. **跑 test_c3_dcu_hello** (Phase 2a-2f)
   - 预期: Phase 2a/d/f ✅
   - 预期: Phase 2b ⚠️ (C3Engine 不知道 kDCU, MLIR backend 走默认)
   - 预期: Phase 2c 真跑 mlirToLLVMIRFromGraph (改动待今晚)
   - 预期: Phase 2e Plan A GCVM `IR_VERSION_MISMATCH` (R1 HIGH 风险)
3. **Plan B 备选** (IR_VERSION_MISMATCH 后切 dcc bitcode)
4. **Plan C 兜底** (CPU baseline)

### 跨 session follow-up 清单

- AOTCache 解耦 refactor (跨 session, 已 commit)
- 49 个 test/bench main 接 C3Cleanup (跨 session, 已修 1/49)
- 5d 接线 (协议层歧义调研, 跨 session)
- Handwritten DEPRECATED (M3 范畴, 跟 MLIR 完整度绑定)
- M2 节点覆盖 (跟 5d 配套, 跨 session)
- MLIRKernelGen refactor (已完成, 跨 session label 解除)
- MLIRToLLVMIR 实装 (已完成, 跨 session label 解除)

### 节点资源情况 (今晚实测)

- **节点**: b02r1n05 (128C-512G-8*BW-1 DCU, 华中一区)
- **DTK**: 26.04 + clang 17.0.0 + GCVM 1.6 (= LLVM 7.0.1)
- **MLIR/LLVM**: 节点缺 mlir-translate, 仅 dcc 自带 LLVM 17 (跟 C3 MLIR 22.1.8 输出 LLVM 14 IR 不兼容)
- **DCU 设备**: Hygon C86 7285 + ZIFANG C878180 (gfx906, 16G VRAM, 64CU×4SIMD)
- **价格**: 1.4 元/小时, 实测 3.5h/5 元

### 机时预算 (user memory 已存)

- 余额 1477+ 元, **可以放心跑节点实验**
- 不用"省着用", DCU 实装 / benchmark / build 验证都可以多跑

## 明日清单 (按 ROI 排)

1. **节点 build 验证 + Plan A/B/C 试错** (主)
2. **如果 Plan A 挂**: Plan B (dcc bitcode) 实装 + 验证
3. **如果 Plan B 挂**: Plan C (CPU baseline) + 性能对比
4. **节点 mpirun 多卡 / 性能 benchmark** (机时充足可以激进)
5. **跑 bench-pytorch-dcu-baseline.py** (拿 PyTorch-DCU 对比数字)
