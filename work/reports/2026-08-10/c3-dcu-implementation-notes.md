# C3 → DCU 接入实装笔记 (v0.5, 2026-08-10 16:08)

> **重要声明**: 本实装是**stub 骨架**，**不能 build 验证** (macOS 无 GCVM 库)。
> 真实 API 签名/函数名待 `probe-dcu-dtk24.sh` 探针回来后调整 (标 [TODO: probe-adjust] 处)。
> 探针成功 → 按 gcvm.h 实际签名调 API → 在 DCU 节点 build + 跑通。

---

## 1. 实装清单 (本 session 完成)

| # | 文件 | 行数 | 状态 |
|---|---|---|---|
| 1 | `include/C3/GCVMBridge.h` | 58 | ✅ 头文件 stub |
| 2 | `src/C3/GCVMBridge.cpp` | 105 | ✅ 实装 stub (API 函数名是推测) |
| 3 | `include/C3/DCUCompiledKernel.h` | 100 | ✅ 头文件 stub |
| 4 | `src/C3/DCUCompiledKernel.cpp` | 200 | ✅ 实装 stub (hipModule/hipMemcpy/launch 框架) |
| 5 | `src/tests/standalone/test_c3_dcu_hello.cpp` | 190 | ✅ 6 阶段 Hello World 测试 |
| 6 | `CMakeLists.txt` (改, +55 行) | 55 | ✅ WITH_DCU option + GCVM/hipBLAS 链接 |
| 7 | `scripts/probe-dcu-dtk24.sh` (前 session) | 264 | ✅ 探针 |
| 8 | `scripts/bench-pytorch-dcu-baseline.py` (前 session) | 206 | ✅ baseline 脚本 |

**总代码量**: ~914 行 (新文件 + CMake 改 + scripts)
**未 commit** (按 user 偏好 "NO push, NO tag")：等你探针后实测验证再决定 commit 策略

---

## 2. 关键风险 + 缓解

### 2.1 API 推测风险 (HIGH)

**问题**: GCVM C API 函数名是按 IREE 接入示例 (0003_6.4.3_IREE_接入_GCVM.md) 推断的，
实际函数名以 `/opt/dtk/llvm/gcvm/include/gcvm.h` 为准。

**当前推测**:
```cpp
gcvmCreateProgram(void** prog)  // IREE 接入示例有
gcvmGetCompiledResult(void* prog, const char** result)  // IREE 接入示例有
gcvmDestroyProgram(void* prog)  // IREE 接入示例有
gcvmAddLLVMIR(void* prog, const char* ir)  // 推测
gcvmSetTargetTriple(void* prog, const char* triple)  // 推测
gcvmSetOptLevel(void* prog, int opt_level)  // 推测
gcvmCompile(void* prog)  // 推测
```

**缓解**:
1. 探针 `probe-dcu-dtk24.sh` 跑 `grep -hE "^[A-Z][a-zA-Z]*\s+gcvm[A-Z][a-zA-Z]*" gcvm.h` 拿实际 API
2. 实装按真实 API 调
3. 如果 API 完全不同（如 gcvmCompileModule 或其他）→ 改 GCVMBridge.cpp

### 2.2 hip/hipBLAS API 风险 (MEDIUM)

**问题**: hipModule/hipMalloc/hipMemcpy/hipModuleLaunchKernel 真实签名跟 CUDA 类似但有差异。

**当前实装**: 标 [TODO: probe-adjust]，按 CUDA 习惯写，探针后根据 `/opt/dtk/include/hip/hip_runtime.h` 调

**缓解**:
1. 探针报告后看 hip 头文件
2. 如果 hipModuleLoadData 不支持 (需要 file path) → 写 tmp file 用 hipModuleLoad
3. 如果 grid/block dims 算错 → 按 actual workload 实测

### 2.3 C3Engine target_device 扩展风险 (MEDIUM)

**问题**: C3Engine.cpp 当前不识别 `DeviceType::kDCU`，test_c3_dcu_hello 设了 opts.target_device = kDCU 但 C3Engine 会忽略。

**当前实装**: DCUCompiledKernel 不依赖 C3Engine，直接用 GCVMCompileResult.code_object 构造

**缓解**:
1. 后续给 C3Engine 加 DCU path (Phase 2 实装)
2. 或者 test 手动构造 DCUCompiledKernel (当前做法)

### 2.4 build 风险 (LOW, 探针前)

**macOS build (默认 WITH_DCU=OFF)**: 
- GCVMBridge.cpp / DCUCompiledKernel.cpp 加了但 #ifdef WITH_DCU 包裹 → 不编译内容
- 不会破坏现有 macOS build
- 验证: `make -C build-debug` 应该 OK

**DCU 节点 build (WITH_DCU=ON)**:
- 必须 libgcvm.so 存在 (DTK 24.04+) → CMakeLists.txt 检查
- 必须 hipBLAS 存在 → CMakeLists.txt 检查
- hip/GCVM 头文件 API 跟推测可能不同 → build 可能 fail 但有清晰错误

---

## 3. 探针前实装的限制 (写清楚)

### 3.1 我做了的 (本 session)

- ✅ 完整代码骨架 (~914 行)
- ✅ CMakeLists 跨编译配置 (WITH_DCU option, libgcvm 链接)
- ✅ 6 阶段 Hello World 测试 (Graph 构造 → C3 compile → MLIR/LLVM → GCVM → DCU execute → correctness verify)
- ✅ probe 脚本 (9 步自动检查)
- ✅ PyTorch baseline 脚本 (4 场景)

### 3.2 我**没做**的 (探针后才能做)

- ❌ **GCVM 真实 API 调通** (需要 `gcvm.h` 头文件)
- ❌ **DCU 节点 build 验证** (无 DCU 节点环境)
- ❌ **DCU 节点跑通 Hello World** (无 DCU 节点)
- ❌ **C3 Engine target_device=kDCU 路径实装** (等 C3Engine 接 DCU 路径)
- ❌ **PyTorch-DCU baseline 数字** (无 DCU 节点)
- ❌ **C3-DCU vs PyTorch-DCU 性能对比** (无 DCU 节点)

### 3.3 探针回来后的调整步骤

```bash
# 1. 洛锦 ssh 到 b02r4n13 跑探针
ssh b02r4n13
module load compiler/dtk/24.04
bash scripts/probe-dcu-dtk24.sh
# 输出: work/reports/2026-08-10/dcu-probe-dtk24-b02r4n13.md

# 2. 洛锦传探针报告 (copy 到本地)

# 3. 我读 gcvm.h 实际 API + 调整 GCVMBridge.cpp
# 4. 我读 hip/hip_module.h 实际 API + 调整 DCUCompiledKernel.cpp
# 5. 洛锦在 DCU 节点 build + 跑 test_c3_dcu_hello
# 6. 跑通后 commit + 跑 PyTorch baseline + 性能对比
```

---

## 4. 探针后实装路线 (时间线)

### Week 1 (8-10 ~ 17): Phase 0-1
- 洛锦: ssh 跑 probe-dcu-dtk24.sh
- 洛锦: 跑 bench-pytorch-dcu-baseline.py (Phase 1)
- 拿到 9 步探针报告 + 4 场景 PyTorch baseline

### Week 1 末 - Week 2 初: 实装调整
- 我: 读 gcvm.h + 改 GCVMBridge.cpp (~30 行核心)
- 我: 读 hip/hip_module.h + 改 DCUCompiledKernel.cpp (~50 行)
- 洛锦: DCU 节点 build + 跑 test_c3_dcu_hello

### Week 2 末 (8-24 ~ 31): Phase 3 性能对比
- 跑 C3-DCU MLP inference
- 跟 PyTorch-DCU baseline 对比
- 出 Phase 3 报告

### Week 3-4 (8-31 ~ 9-10): Phase 4 优化
- 优化 fused kernel
- 3B FP32 推理 (LLM)

---

## 5. Switch 条件 (实装期间)

| 触发 | 动作 |
|---|---|
| GCVM C API 函数名跟推测完全不一样 | 改 GCVMBridge.cpp, 实装时间 +1-2 天 |
| hipModuleLoadData 不支持 (需 file) | 写 tmp file path, 改用 hipModuleLoad |
| hipModuleLaunchKernel 真实 signature 不同 | 改 launchKernel 内部, 实装时间 +1 天 |
| C3Engine compile 失败 (DCU backend 未接) | 跳过 C3Engine, test 直接用 GCVMCompileResult |
| 整体跑不通 | 切 Plan B (hipcc 直接编译 LLVM IR, ~100 行) |

---

## 6. 关键文件位置

| 文件 | 路径 |
|---|---|
| 接入设计文档 | `work/reports/2026-08-10/c3-dcu-integration-design.md` (前 session 出) |
| 实装笔记 (本文件) | `work/reports/2026-08-10/c3-dcu-implementation-notes.md` |
| 探针脚本 | `scripts/probe-dcu-dtk24.sh` |
| PyTorch baseline 脚本 | `scripts/bench-pytorch-dcu-baseline.py` |
| 头文件 (新) | `include/C3/GCVMBridge.h` + `include/C3/DCUCompiledKernel.h` |
| 实装 (新) | `src/C3/GCVMBridge.cpp` + `src/C3/DCUCompiledKernel.cpp` |
| 测试 (新) | `src/tests/standalone/test_c3_dcu_hello.cpp` |
| CMakeLists 改 | `WITH_DCU` option + GCVM/hipBLAS 链接 |
| DCU 上下文 | `STATUS_DCU_ADAPT.md` v0.5 |

---

## 7. 当前 commit 状态

**未 commit** (按 user 偏好"NO push, NO tag"):
- 6 个新文件 (~914 行) untracked
- CMakeLists.txt modified (+55 行 WITH_DCU option)

**Commit 策略** (待你拍板):
- **方案 A**: 探针后 commit (一次 commit "C3 → DCU Phase 0 接入 + Phase 1 实装")
- **方案 B**: 分 2 commit ("Design 文档 + Stub 实装" + "探针后真实 API 实装")
- **方案 C**: 不 commit (本 session 探索性, 探针后决定)

---

## 8. 下一步行动

### 你 (洛锦) 晚上做的事

1. ssh 到 b02r4n13
2. `module load compiler/dtk/24.04` (或 25.04/26.04)
3. `bash scripts/probe-dcu-dtk24.sh`
4. `python3 scripts/bench-pytorch-dcu-baseline.py`
5. 上传 2 份报告 (dcu-probe-dtk24-*.md + pytorch-dcu-baseline-*.md) 到 `work/reports/2026-08-10/`

### 我等探针回来后做的事

1. 读探针报告 + gcvm.h 实际 API
2. 调 GCVMBridge.cpp (调 ~10 行 API 名/签名)
3. 调 DCUCompiledKernel.cpp (调 ~20 行 hip API)
4. 写实装调整报告 (Phase 2 实装完成)
5. 配合你 build + 跑通

---

*实装笔记写于 2026-08-10 16:08 CST*
*作者: mavis*
*Status: stub 骨架, 探针后实装调整*
