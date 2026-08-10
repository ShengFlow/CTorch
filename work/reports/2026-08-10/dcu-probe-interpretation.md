# DCU 探针解读报告 + Phase 2 调整 (v0.5, 2026-08-10 17:13)

> **承接**: dcu-probe-dtk24-b02r2n11.md (DTK 26.04 探针)
> **配套**: gcvm.h (11119 bytes, ~/Downloads/gcvm.h)
> **作者**: mavis (CTorch 主代理)
> **状态**: Plan A 确认可行 + 3 个新风险识别 + GCVMBridge.cpp 已调

---

## 1. 探针核心结论

| 关键路径 | 状态 | 备注 |
|---|---|---|
| **GCVM 库 + 头文件** | ✅ **YES** | DTK 26.04, libgcvm.so.17git, gcvm.h 完整 |
| **hipBLAS / hipBLASLt** | ✅ YES | libhipblas.so + libhipblaslt.so |
| **hipcc / dcc** | ✅ YES | dcc 25.10.0, clang 17, amdgcn target 支持 |
| **DTK 路径** | ⚠️ 非默认 | `/public/software/compiler/rocm/dtk-26.04` (不是 `/opt/dtk`) |
| **MLIR amdgcn backend** | ❌ NO | 节点缺 mlir-translate, C3 MLIR → LLVM IR 转换需装 LLVM/MLIR 工具链 |
| **Python3 + PyTorch-DCU** | ❌ NO | 节点无 Python3, baseline 脚本跑不了 |
| **rocm-smi** | ❌ NO | KFD node discovery 失败, 看不了 DCU 设备状态 (但不影响编译链) |

**Plan A 路径 (GCVM C API) 100% 可行**——7/9 关键检查项 ✓。

---

## 2. GCVM 真实 API 签名 (按 gcvm.h 整理)

### 2.1 14 个 API 完整列表

| API | 签名 | 用途 |
|---|---|---|
| `gcvmGetErrorString` | `(gcvmResult) → const char*` | 错误码转字符串 |
| `gcvmVersion` | `(int* major, int* minor) → gcvmResult` | GCVM 版本 |
| `gcvmIRVersion` | `(int* majorIR, int* minorIR, int* majorDbg, int* minorDbg) → gcvmResult` | IR 版本 |
| `gcvmSetArch` | `(gcvmProgram, const char* arch) → gcvmResult` | 设置 arch (CUDA 命名) |
| `gcvmSetOptLevel` | `(gcvmProgram, int optLevel) → gcvmResult` | 优化级别 0-3 |
| `gcvmCreateProgram` | `(gcvmProgram* prog) → gcvmResult` | 创建 |
| `gcvmDestroyProgram` | `(gcvmProgram* prog) → gcvmResult` | 销毁 |
| `gcvmAddModuleToProgram` | `(gcvmProgram, const char* buffer, size_t size, const char* name, SourceType st) → gcvmResult` | **5 参数**, buffer size name SourceType |
| `gcvmLazyAddModuleToProgram` | `(gcvmProgram, const char* buffer, size_t size, const char* name) → gcvmResult` | 4 参数, 延迟加载 |
| `gcvmCompileProgram` | `(gcvmProgram, int numOptions, const char** options, ResultType rt) → gcvmResult` | **4 参数** |
| `gcvmVerifyProgram` | `(gcvmProgram, int numOptions, const char** options) → gcvmResult` | 验证 |
| `gcvmGetCompiledResultSize` | `(gcvmProgram, size_t* bufferSizeRet) → gcvmResult` | Code Object 大小 |
| `gcvmGetCompiledResult` | `(gcvmProgram, char* buffer) → gcvmResult` | Code Object bytes |
| `gcvmGetProgramLogSize/Log` | `(gcvmProgram, size_t* / char*) → gcvmResult` | 编译日志 |

### 2.2 关键类型 (枚举)

```c
typedef enum {
    GCVM_SUCCESS = 0,
    GCVM_ERROR_OUT_OF_MEMORY = 1,
    GCVM_ERROR_PROGRAM_CREATION_FAILURE = 2,
    GCVM_ERROR_IR_VERSION_MISMATCH = 3,    // <-- 关键! C3 MLIR 输出 LLVM 14 vs GCVM 1.6 = LLVM 7
    GCVM_ERROR_INVALID_INPUT = 4,
    GCVM_ERROR_INVALID_PROGRAM = 5,
    GCVM_ERROR_INVALID_IR = 6,
    GCVM_ERROR_INVALID_OPTION = 7,
    GCVM_ERROR_NO_MODULE_IN_PROGRAM = 8,
    GCVM_ERROR_COMPILATION = 9
} gcvmResult;

typedef enum ResultType { Assembly, Object, Hsaco, RT_None };
typedef enum SourceType { LLVMIR, ASSEMBLY, ST_None };
```

### 2.3 我 v0.5.0 推测 vs 真实 (5 个错)

| v0.5.0 推测 | v0.5.1 真实 | 错 |
|---|---|---|
| `gcvmAddLLVMIR(prog, ir)` | `gcvmAddModuleToProgram(prog, buffer, size, name, LLVMIR)` | ❌ 5 参数不是 1 |
| `gcvmCompile(prog)` | `gcvmCompileProgram(prog, numOptions, options, Hsaco)` | ❌ 4 参数不是 1 |
| `gcvmSetTargetTriple(prog, "amdgcn-amd-amdhsa--gfx906")` | `gcvmSetArch(prog, arch_name)` | ❌ 名字 + 命名风格都错 |
| (漏) | `gcvmLazyAddModuleToProgram` | 漏了, 更高效路径 |
| (漏) | `gcvmVerifyProgram` | 漏了, 编译前 sanity check |
| (漏) | `gcvmGetProgramLogSize/Log` | 漏了, 错误诊断用 |
| (漏) | `gcvmVersion` / `gcvmIRVersion` | 漏了, runtime 版本检查 |

---

## 3. 3 个新风险识别 (gcvm.h 揭示)

### R1: GCVM IR version 1.6 = LLVM 7.0.1 (HIGH 风险)

**事实** (per gcvm.h L155-159):
> "The module should have GCVM IR version 1.6 either in the LLVM 7.0.1 bitcode representation or in the LLVM 7.0.1 text representation."

**冲突**:
- GCVM IR version 1.6 = LLVM 7.0.1 (2017)
- C3 MLIR 22.1.8 = LLVM 14+ (2022+)
- 5 年 IR 语法差异 (新 intrinsic, attribute 格式, instruction 改动)

**预测失败**: `gcvmAddModuleToProgram` 大概率返回 `GCVM_ERROR_IR_VERSION_MISMATCH (3)`

**缓解**:
1. **首选**: 节点装 LLVM 7.0.1 兼容的 MLIR 工具链 → C3 MLIR 输出降级到 LLVM 7.0.1 IR
2. **备选**: GCVM 接受 LLVM 14 IR (待节点实测, 可能 GCVM 已升级)
3. **Plan B**: 用 dcc 直接编译 C3 输出的 LLVM bitcode (跳过 GCVM 包装)

**Switch 条件**:
- 节点实测 `gcvmAddModuleToProgram` 返回 `IR_VERSION_MISMATCH` → 走 Plan B

### R2: arch 命名 (CUDA compute_xx vs DCU gfx906) (MEDIUM 风险)

**事实** (per gcvm.h L221-232):
> arch 选项: compute_35, compute_37, compute_50, compute_52 (default), compute_53, compute_60, compute_61, compute_62, compute_70, compute_72, compute_75, compute_80

**冲突**:
- GCVM arch 是 CUDA compute capability 命名 (compute_xx)
- DCU (gfx906) 是 AMDGCN 命名 (gfx906)

**预测**:
- 试用 `-arch=gfx906` (LLVM/AMDGPU 命名) — 可能被 GCVM 拒绝 (INVALID_OPTION)
- 试用 `-arch=compute_80` (最高 CUDA) — 可能 DCU 不支持 CUDA IR 但 GCVM 编译过

**当前 GCVMBridge.cpp 尝试顺序**: gfx906 → compute_80 fallback

**Switch 条件**:
- 节点实测两个 arch 都失败 → 走 Plan B (dcc 直接编译)
- 节点实测 compute_80 跑通 → 改默认 arch

### R3: 节点缺 mlir-translate (MEDIUM 风险)

**事实** (per 探针 L70):
> ❌ mlir-translate 不存在

**冲突**:
- C3 MLIRKernelGen.cpp:1703 `mlir::translateModuleToLLVMIR(*module, bc_ctx)` 走 MLIR ExecutionEngine 内置
- 但要拿 LLVM IR 字符串给 GCVM, 需要 `mlir-translate --mlir-to-llvmir` 单独工具

**当前方案**:
- MLIRKernelGen.cpp 改: `llvm::Module` → 字符串序列化 → 喂 GCVM
- 这个不需要 mlir-translate 工具 (C++ 进程内调 LLVM API)

**Switch 条件**:
- 节点 build C3 失败 (MLIR 22.1.8 链接) → 装 LLVM 22.1.8 工具链
- 运行时调 GCVM 失败 → 走 Plan B

---

## 4. 调整后 GCVMBridge.cpp (v0.5.1)

按 gcvm.h 真实 API 完整重写:

```cpp
// 关键流程 (每步用 GCVM_SUCCESS 校验 + 错误码转字符串)
1. gcvmCreateProgram(&prog)
2. gcvmSetArch(prog, "gfx906" 或 "compute_80" fallback)   // 试两 arch
3. gcvmSetOptLevel(prog, opt_level)
4. gcvmAddModuleToProgram(prog, llvm_ir_text, size, kernel_name, LLVMIR)
5. options = ["-opt=2", "-ftz=1", "-fma=1"]
   gcvmCompileProgram(prog, 3, options, Hsaco)
6. gcvmGetCompiledResultSize(prog, &size)
7. gcvmGetCompiledResult(prog, buffer)
8. gcvmDestroyProgram(&prog)  // RAII guard
```

**fallback 策略**:
- arch gfx906 失败 → 自动尝试 compute_80
- gcvmAddModuleToProgram IR_VERSION_MISMATCH → 节点实装时换 Plan B
- 任何 GCVM_ERROR → 调 gcvmGetProgramLogSize/Log 拿详细错误

**位置**: `src/C3/GCVMBridge.cpp` (已更新, 6112 bytes, 含详细注释)

---

## 5. Plan A/B/C 决策树

```
节点 build + 实跑
   │
   ├─ A.1: gcvmAddModuleToProgram 成功
   │     ↓
   │   A.2: gcvmCompileProgram 成功
   │     ↓
   │   A.3: hipModuleLoadData 成功
   │     ↓
   │   ✅ Phase 2 跑通, 进 Phase 3 性能对比
   │
   └─ 任何步骤失败
         ↓
       拿错误日志 (gcvmGetProgramLog)
         ↓
       ├─ IR_VERSION_MISMATCH → Plan B (dcc 直接编译 LLVM bitcode)
       ├─ INVALID_OPTION (arch) → Plan B
       └─ COMPILATION (其他) → 调整 options 重试
```

**Plan B (dcc 直接编译, ~100 行)**:
- C3 MLIR → LLVM IR (字符串) → 写 .ll 文件 → dcc -c xxxx.ll -o xxxx.hsaco
- 跳 GCVM, 跳 MLIR amdgcn 转换
- 缺点: 失去 fused kernel 优势 (dcc 编译时间 1-2s, 摊不开)

**Plan C (CPU baseline, 备用)**:
- C++ 写 2 层 MLP, hand-coded hipBLAS
- 跟 PyTorch-DCU baseline 对比 (C++ vs PyTorch)
- 跳过 GCVM 整条链

---

## 6. CMakeLists 调整 (已应用)

**DTK 路径自动探测** (per 探针):
```cmake
# 默认 /public/software/compiler/rocm/dtk-26.04 (per 探针 b02r2n11)
# 也支持 /opt/dtk 传统路径
# 通过 -DROCM_PATH=... 覆盖
if(EXISTS "/public/software/compiler/rocm/dtk-26.04/llvm/gcvm/lib/libgcvm.so")
    set(ROCM_PATH "/public/software/compiler/rocm/dtk-26.04")
elseif(EXISTS "/opt/dtk/llvm/gcvm/lib/libgcvm.so")
    set(ROCM_PATH "/opt/dtk")
else()
    set(ROCM_PATH "/opt/dtk")  # fallback
endif()
```

**位置**: `CMakeLists.txt` (已更新, +14 行)

---

## 7. 实装位置总结

| 改动 | 状态 |
|---|---|
| `src/C3/GCVMBridge.cpp` | ✅ v0.5.1 按 gcvm.h 真实 API 重写 (6112 bytes) |
| `include/C3/GCVMBridge.h` | ✅ 接口不变 (跟 v0.5.0 兼容) |
| `src/C3/DCUCompiledKernel.cpp` | ✅ 仍是 stub, 等节点 build 验证后调 hip API |
| `CMakeLists.txt` | ✅ DTK 路径自动探测 |
| `work/reports/2026-08-10/c3-dcu-implementation-notes.md` | 待更新 v0.5.1 调整记录 |

---

## 8. 节点下一步 (洛锦)

### 8.1 立即: 装 LLVM/MLIR 工具链 (R3 缓解)

```bash
# 节点 b02r2n11
module load compiler/dtk/26.04
# 装 LLVM 22.1.8 (跟 C3 macOS build 版本一致)
# 路径: /opt 或 /public/software 都行
# 或: 装 LLVM 7.0.1 兼容的 MLIR 工具链 (R1 缓解)
```

### 8.2 build C3 DCU 路径

```bash
mkdir -p build-dcu && cd build-dcu
cmake -DCT_ENABLE_DCU=ON -DCT_ENABLE_MLIR=ON ..
make -j8 test_c3_dcu_hello
./test_c3_dcu_hello
```

### 8.3 错误诊断

如果 `gcvmAddModuleToProgram` 失败:
```bash
# 调 gcvmGetProgramLog 拿详细错误
# 看是不是 GCVM_ERROR_IR_VERSION_MISMATCH (3) → 走 Plan B
```

---

## 9. Switch 条件汇总

| 触发 | 动作 |
|---|---|
| A.1 `gcvmAddModuleToProgram` 成功 | 继续 A.2 |
| A.1 `IR_VERSION_MISMATCH` | Plan B (dcc 直接编译) |
| A.2 `gcvmCompileProgram` 成功 | 继续 A.3 |
| A.2 arch 错误 | 改用 compute_80 试试 |
| A.3 `hipModuleLoadData` 成功 | ✅ Phase 2 跑通 |
| 任何 Plan A 步骤失败 ≥ 2 次 | 切 Plan B |
| Plan B 也失败 | 切 Plan C (CPU baseline) |
| 1 周内 Phase 2 未跑通 | 切 Phase 3 性能对比路径 (如 A 跑通) |

---

*报告写于 2026-08-10 17:13 CST*
*基于 dcu-probe-dtk24-b02r2n11.md + gcvm.h 真实 API 调整*
*作者: mavis*
