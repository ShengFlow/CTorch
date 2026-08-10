# C3 → 海光 DCU 接入设计文档

> **v0.5 主线** — C3 在海光 DCU (gfx906) 上跑通 + 比 PyTorch-DCU 快
> **日期**：2026-08-10 16:00
> **作者**：mavis (CTorch 主代理)
> **配套**：`STATUS_DCU_ADAPT.md` v0.5 / `c3-perf-report-1546.md` / `scripts/probe-dcu-dtk24.sh`

---

## 0. 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                       C3 (用户级 Python/C++ API)                  │
│                          Graph IR (高层算子)                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                 C3Engine::compile (MLIR backend)                  │
│  - C3 Graph → MLIR Module (buildMultiNodeMLIR)                    │
│  - MLIR Module → LLVM IR (translateModuleToLLVMIR) ← 已有          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼  ⭐ 新接入层
┌──────────────────────────────────────────────────────────────────┐
│              GCVM C API 桥接 (新 ~30 行)                          │
│  - LLVM IR (string) → GCVM Program (gcvmCreateProgram)            │
│  - GCVM Program → Object Code (gcvmGetCompiledResult)              │
│  - Object Code → Memory buffer (load to memory)                    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│           DCC/LingShu 后端 (海光自家, 在 GCVM 内部)                │
│  - Target Machine init / CodeGen Pipeline                          │
│  - 指令选择 / 调度 / Fast-Math / Atomic                            │
│  - 输出 HSACO (gfx906 Code Object)                                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                DCU 执行 (kernel launch via hipBLAS / 裸)          │
│  - dlopen HSACO → 解析 symbol table                                │
│  - 拷贝 input 到 device memory                                     │
│  - 启动 kernel (类似 cuLaunchKernel)                                │
│  - 同步回 host memory                                              │
└──────────────────────────────────────────────────────────────────┘
```

**关键认知**：
- **GCVM IR = LLVM IR + AMDGPU/DCU 目标约束**（per `0001_1._简介.md`）
- **C3 已有 LLVM IR 输出路径**（MLIRKernelGen.cpp:1703 `translateModuleToLLVMIR`）
- **新增 30 行桥接代码**走 GCVM C API，无需重新实现 LLVM → DCU 编译

---

## 1. 核心 API 桥接代码（~30 行）

```cpp
// 新文件: src/C3/GCVMBridge.cpp (本 session 设计，未实装)
// 路径: include/C3/GCVMBridge.h 头文件 + 头文件路径 /opt/dtk/llvm/gcvm/include

#include <gcvm.h>  // GCVM C API 头文件 (DTK 24.04+ /opt/dtk/llvm/gcvm/include)

namespace ct::c3 {

/// 把 LLVM Module 编译成 gfx906 Code Object (HSACO 等价)
/// @param module MLIR/LLVM Module (from translateModuleToLLVMIR)
/// @param opt_level 0=None, 1=Less, 2=Default, 3=Aggressive
/// @return Code Object bytes, 空字符串 = 失败
std::string compileLLVMToDCUObject(llvm::Module& module, int opt_level = 2) {
    // 1. 序列化 LLVM IR 到 string
    std::string ir_source;
    llvm::raw_string_ostream ir_stream(ir_source);
    module.print(ir_stream, nullptr);
    ir_stream.flush();
    
    // 2. 创建 GCVM Program
    gcvmProgram gcvm_prog;
    if (GCVM_SUCCESS != gcvmCreateProgram(&gcvm_prog)) {
        return "";  // 失败
    }
    
    // 3. 喂 LLVM IR
    if (GCVM_SUCCESS != gcvmAddLLVMIR(gcvm_prog, ir_source.c_str())) {
        gcvmDestroyProgram(&gcvm_prog);
        return "";
    }
    
    // 4. 设置 target triple (gfx906 = Hygon C86 7285)
    if (GCVM_SUCCESS != gcvmSetTargetTriple(gcvm_prog, "amdgcn-amd-amdhsa--gfx906")) {
        gcvmDestroyProgram(&gcvm_prog);
        return "";
    }
    
    // 5. 设置优化级别
    gcvmSetOptLevel(gcvm_prog, opt_level);
    
    // 6. 编译
    if (GCVM_SUCCESS != gcvmCompile(gcvm_prog)) {
        gcvmDestroyProgram(&gcvm_prog);
        return "";
    }
    
    // 7. 拿编译结果 (Code Object bytes)
    const char* result = nullptr;
    size_t result_size = 0;
    if (GCVM_SUCCESS != gcvmGetCompiledResult(gcvm_prog, &result, &result_size)) {
        gcvmDestroyProgram(&gcvm_prog);
        return "";
    }
    
    // 8. 复制到 std::string (own copy)
    std::string code_object(result, result_size);
    
    // 9. 清理
    gcvmDestroyProgram(&gcvm_prog);
    
    return code_object;
}

/// 把 Code Object bytes 加载到 host 内存，解析 symbol 拿到 kernel 函数指针
/// @param code_object compileLLVMToDCUObject 返回的 Code Object
/// @param kernel_name kernel symbol 名字
/// @return kernel 函数指针, nullptr = 失败
void* loadDCUKernelFromCodeObject(const std::string& code_object, 
                                   const std::string& kernel_name) {
    // 写 tmp file (HSACO 需要 file path 给 hipModuleLoad)
    // 或: 使用 hipModuleLoadData 直接从 memory load
    hipModule_t hip_module;
    if (hipSuccess != hipModuleLoadData(&hip_module, code_object.data())) {
        return nullptr;
    }
    hipFunction_t hip_func;
    if (hipSuccess != hipModuleGetFunction(&hip_func, hip_module, kernel_name.c_str())) {
        return nullptr;
    }
    return reinterpret_cast<void*>(hip_func);
}

}  // namespace ct::c3
```

**注意**：上面 GCVM C API 是**示意** — 真实 API 需要查 `/opt/dtk/llvm/gcvm/include/gcvm.h` 头文件。IREE 接入示例给出 11 个核心 API（`gcvmCreateProgram` + `gcvmAddLLVMIR` + `gcvmCompile` + `gcvmGetCompiledResult` + `gcvmDestroyProgram` 等），实际签名以头文件为准。

---

## 2. CMakeLists 跨编译配置

```cmake
# 在 CMakeLists.txt 添加: WITH_DCU 选项
option(WITH_DCU "Build with Hygon DCU (gfx906) backend support" OFF)

if(WITH_DCU)
    # 1. 路径默认 DTK 24.04+ 路径
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/dtk" CACHE STRING "Default ROCM installation directory.")
    endif()
    
    set(GCVM_INCLUDE_DIR "${ROCM_PATH}/llvm/gcvm/include")
    set(GCVM_LIBRARY "${ROCM_PATH}/llvm/gcvm/lib/libgcvm.so")
    
    # 2. 验证 GCVM 库存在
    if(NOT EXISTS "${GCVM_INCLUDE_DIR}")
        message(FATAL_ERROR "GCVM include directory not found: ${GCVM_INCLUDE_DIR}. DTK 24.04+ required.")
    endif()
    if(NOT EXISTS "${GCVM_LIBRARY}")
        message(FATAL_ERROR "GCVM library not found: ${GCVM_LIBRARY}. DTK 24.04+ required.")
    endif()
    
    # 3. 导入 gcvm + hip 库
    add_library(gcvm SHARED IMPORTED)
    set_target_properties(gcvm PROPERTIES
        IMPORTED_LOCATION "${GCVM_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GCVM_INCLUDE_DIR}"
    )
    
    find_package(hip REQUIRED)
    
    # 4. 编译选项: amdgcn target
    add_definitions(-DWITH_DCU=1)
    
    # 5. 加 GCVMBridge.cpp 到 CTorch 静态库
    set(CT-DCUSUPPORT
        src/C3/GCVMBridge.cpp
        src/C3/DCUExecution.cpp
    )
    
    # 6. 链接 gcvm + hip + hipblas
    target_link_libraries(CTorch PRIVATE gcvm hip::host hip::device hipblas)
    
    message(STATUS "DCU backend enabled: GCVM at ${GCVM_LIBRARY}, hipBLAS at ${ROCM_PATH}/lib")
endif()
```

**关键约束**:
- `WITH_DCU=ON` 必须在 **DCU 节点** build（macOS 上 GCVM 库不存在，build 失败）
- 跨编译暂时不支持（必须在 DCU 节点 native build）

---

## 3. 集成点改动

### 3.1 C3Engine.cpp - 加 DCU backend 选项

```cpp
// 在 C3Engine::compile (L462 附近) 添加 backend 选择逻辑
if (options.target_device == DeviceType::kDCU) {
    // DCU 路径: MLIR → LLVM IR → GCVM → Code Object → hipModule load
    auto llvm_module = mlir::translateModuleToLLVMIR(*mlir_module, llvm_ctx);
    if (!llvm_module) {
        throw std::runtime_error("MLIR → LLVM IR translation failed");
    }
    
    std::string code_object = ct::c3::compileLLVMToDCUObject(*llvm_module, 2);
    if (code_object.empty()) {
        throw std::runtime_error("GCVM compilation failed (DTK 24.04+ required)");
    }
    
    // 加载 kernel + 包装成 DCUCompiledKernel
    auto* kernel = new DCUCompiledKernel(code_object, graph, options);
    return std::shared_ptr<CompiledKernel>(kernel);
}
```

### 3.2 DCUCompiledKernel 类 (新文件 src/C3/DCUCompiledKernel.cpp)

```cpp
// 类似 ConcreteCompiledKernel, 但:
// - execute() 走 hipModuleLaunchKernel 而非函数指针
// - 输入/输出 device memory 走 hipMalloc/hipMemcpy
// - 同步走 hipDeviceSynchronize 或 hipStreamSynchronize

class DCUCompiledKernel : public CompiledKernel {
public:
    DCUCompiledKernel(const std::string& code_object, 
                      const Graph& graph, 
                      const CompileOptions& opts);
    
    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        // 1. host → device memcpy
        // 2. 设置 kernel args (hipFunctionLaunch)
        // 3. hipModuleLaunchKernel
        // 4. device → host memcpy
        // 5. 返回 output Tensor
    }
    
    ~DCUCompiledKernel() override {
        // hipModuleUnload + free device memory
    }
};
```

---

## 4. 测试矩阵 (Phase 顺序)

### Phase 0: 探针 (立即, 洛锦做)

```bash
# 节点 b02r4n13 上跑
module load compiler/dtk/24.04  # 或 25.04/26.04
bash scripts/probe-dcu-dtk24.sh
```

**关键探针**:
- libgcvm.so 存在？
- `/opt/dtk/llvm/gcvm/include/gcvm.h` 头文件存在？
- hipcc 可用？
- PyTorch-DCU 已装？（`python3 -c "import torch; print(torch.cuda.is_available())"` 改成 `torch.dcu.is_available()`）

### Phase 1: PyTorch-DCU baseline (Phase 0 完成后)

```bash
# DCU 节点跑
python3 scripts/bench-pytorch-dcu-baseline.py
```

**输出**:
- 3 层 MLP (784→256→128→10) forward 延迟
- ResNet50 forward 延迟
- 内存占用

### Phase 2: C3 "Hello World" (Phase 1 后, 我实装 + 洛锦 build 跑)

- 实装 GCVMBridge.cpp + DCUCompiledKernel.cpp
- 写 `test_c3_dcu_hello.cpp`: 1 个 Add kernel 跑通
- 洛锦 build `WITH_DCU=ON` + 跑通

### Phase 3: 性能对比 v1 (Phase 2 后)

- C3-DCU 跑 MLP forward
- 对比 PyTorch-DCU baseline
- 5 维 perf: latency / throughput / memory / compile time

---

## 5. 风险与备选

### 5.1 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| DTK 24.04+ 仍未提供 GCVM | 30% | 切到 DTK 25.04/26.04, 都没就 fallback CPU |
| C3 MLIR → LLVM IR 不能直接接 GCVM | 20% | MLIR 已是 LLVM 兼容, 实装验证 |
| hipModuleLoadData 失败 (Code Object 兼容) | 10% | 试 hipModuleLoad (file path) |
| GCVMBridge 编译时间拖慢 C3 整体 build | 5% | 默认 WITH_DCU=OFF, 仅 DCU 节点开启 |

### 5.2 备选方案

**Plan B: 不走 GCVM, 走 hipcc 直接编译**
- 把 C3 MLIR 输出 dump 到 .ll
- hipcc -c xxxx.ll -o xxxx.hsaco
- 缺点: 编译开销大, 失去运行时 fused 优势

**Plan C: 走 Triton 风格 (C3 不直接调 GCVM)**
- 把 C3 fused kernel 输出为 Triton kernel
- 走 Triton → GCVM → DCU
- 缺点: 引入 Triton 依赖, 不符合 C3 轻量级定位

**Plan A (推荐)**: 直接 GCVM C API ~30 行, 最低侵入性

---

## 6. 时间线 (1 个月冲刺)

| 周 | 任务 | 负责人 |
|---|---|---|
| **Week 1 (8-10 ~ 17)** | Phase 0 探针 + Phase 1 PyTorch baseline | 洛锦跑 |
| **Week 2 (8-17 ~ 24)** | Phase 2 C3 Hello World + GCVMBridge 实装 | 洛锦 build, 我写 |
| **Week 3 (8-24 ~ 31)** | Phase 3 性能对比 + 优化 fused | 双方 |
| **Week 4 (8-31 ~ 9-10)** | Phase 4 LLM 推理 (3B FP32, gfx906 FP16 缺) | 双方 |
| **缓冲期** | Switch 条件触发 → 切 Plan B/C | — |

---

## 7. 关键文件清单

### 7.1 新增 (本 session 设计)

- `work/reports/2026-08-10/c3-dcu-integration-design.md` (本文件)
- `scripts/probe-dcu-dtk24.sh` (升级版探针, GCVM 验证)
- `scripts/bench-pytorch-dcu-baseline.py` (PyTorch-DCU baseline)

### 7.2 待实装 (Phase 2 阶段)

- `include/C3/GCVMBridge.h` (新)
- `src/C3/GCVMBridge.cpp` (新, ~30 行)
- `include/C3/DCUCompiledKernel.h` (新)
- `src/C3/DCUCompiledKernel.cpp` (新, ~150 行)
- `src/C3/DCUExecution.cpp` (新, host-device memcpy + kernel launch)
- `src/tests/standalone/test_c3_dcu_hello.cpp` (新, Hello World)
- `CMakeLists.txt` (改, +WITH_DCU option)

### 7.3 配套文档 (本 session 出)

- `work/reports/2026-08-10/c3-perf-report-1546.md` (C3 性能基线)
- `work/reports/2026-08-10/auto-code-review-001910.md` (代码审计)
- `STATUS_DCU_ADAPT.md` (DCU 适配上下文)

---

## 8. Switch 条件

**继续 MLP 目标** if:
- Phase 0 探针 1 周内完成 (libgcvm.so 存在)
- Phase 2 Hello World 2 周内跑通
- Phase 3 性能差距 ≥ 0%

**切到 Plan B** if:
- GCVM C API 不可用 → 走 hipcc 直接编译 LLVM IR
- 性能 ≥ PyTorch-DCU 但 < 1.25x → 继续优化 fused kernel

**切到 Plan C / 暂停** if:
- C3 MLIR → GCVM 接入卡 2 周以上
- Phase 3 性能 < PyTorch-DCU
- 时间不够 1 个月

---

## 9. 实装检查清单 (Phase 2 详细)

- [ ] `git submodule add` 或 `git pull` 拉 IREE 接入示例（不必要, 本地 dcu-docs 已有）
- [ ] 实装 `include/C3/GCVMBridge.h` 接口定义
- [ ] 实装 `src/C3/GCVMBridge.cpp` (~30 行核心)
- [ ] 实装 `DCUCompiledKernel.cpp` 包装 hipModule launch
- [ ] 实装 `DCUExecution.cpp` 处理 host-device 同步
- [ ] CMakeLists 加 `WITH_DCU=ON` option
- [ ] 写 `test_c3_dcu_hello.cpp` 测试 Add kernel
- [ ] 洛锦在 b02r4n13 build + 跑通 "Hello World"
- [ ] 跑 3 层 MLP 跟 PyTorch-DCU baseline 对比
- [ ] 出 Phase 3 性能对比报告

---

*设计稿写于 2026-08-10 16:00 CST*
*基于本地 dcu-docs/knowledge_base/ DCC + GCVM IR 完整文档 + IREE 接入示例*
*作者：mavis*
