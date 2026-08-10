cmake_minimum_required(VERSION 3.31)
project(CTorch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# 最激进的编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -march=native")

# LTO (链接时优化) - Thin LTO (平衡编译速度和运行时性能)
# 通过 CT_ENABLE_LTO 开关控制，默认开启；关闭后可做 no-LTO 基线/PGO 对照实验
option(CT_ENABLE_LTO "Enable Link-Time Optimization (Thin LTO)" ON)
if(CT_ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=thin")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto=thin")
endif()

include(FetchContent)

# ---- C3 开关 ----
option(CT_DISABLE_C3 "Disable C3 JIT compilation and fusion optimizations" OFF)
if(CT_DISABLE_C3)
    add_definitions(-DCT_DISABLE_C3)
    message(STATUS "C3 optimizations DISABLED")
else()
    message(STATUS "C3 optimizations enabled")
endif()

# ---- C3 子功能开关（编译期硬开关，运行时仍可用对应 env 变量关闭）----
# 每项生成对应宏，供 include/C3/C3Config.h 的查询接口短路判断。
option(CT_C3_DISABLE_SINGLE_KERNEL "Disable C3 single-kernel hotpath injection" OFF)
option(CT_C3_DISABLE_REGION_FUSION "Disable C3 region fusion" OFF)
option(CT_C3_DISABLE_BACKWARD      "Disable C3 backward fusion" OFF)
option(CT_C3_DISABLE_AOT           "Disable C3 AOT disk cache" OFF)
option(CT_C3_DISABLE_HOTPATH       "Disable C3 hotpath tracking/compile trigger" OFF)

foreach(_C3_FEAT SINGLE_KERNEL REGION_FUSION BACKWARD AOT HOTPATH)
    if(CT_C3_DISABLE_${_C3_FEAT})
        add_definitions(-DCT_C3_DISABLE_${_C3_FEAT})
        message(STATUS "C3 sub-feature ${_C3_FEAT} DISABLED (compile-time)")
    else()
        message(STATUS "C3 sub-feature ${_C3_FEAT} enabled (runtime env can override)")
    endif()
endforeach()

# ---- LLVM / MLIR (Phase 1: JIT 后端) ----
option(CT_ENABLE_MLIR "Enable MLIR/LLVM JIT backend for C3" ON)
if(CT_ENABLE_MLIR)
    set(LLVM_DIR /opt/homebrew/opt/llvm/lib/cmake/llvm)
    set(MLIR_DIR /opt/homebrew/opt/llvm/lib/cmake/mlir)
    find_package(LLVM REQUIRED CONFIG)
    find_package(MLIR REQUIRED CONFIG)
    message(STATUS "MLIR backend enabled (LLVM ${LLVM_PACKAGE_VERSION})")
endif()

option(FETCH_GOOGLETEST "Fetch googletest from remote" OFF)
enable_testing()

if(FETCH_GOOGLETEST)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://gh-proxy.org/https://github.com/google/googletest.git
            GIT_TAG        v1.17.0
    )
    FetchContent_MakeAvailable(googletest)
else()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "googletest not found and FETCH_GOOGLETEST=OFF, skipping tests")
        set(TEST_ENABLED OFF CACHE BOOL "Disable tests due to missing gtest" FORCE)
    endif()
endif()
include(GoogleTest)

set(CT-Core
        src/Tensor.cpp
        src/Storage.cpp
        src/Ctools.cpp
        src/ThreadPool.cpp
        src/CtorchScheduler.cpp
        src/AutoGrad/ComputeCore.cpp
        src/Arena.cpp
        src/Assertion.cpp
        src/AutoGrad.cpp
        # [Fix] v0.5.2 Linux build: Metal/MPS .mm 文件 macOS 专属, 下面 if(APPLE) 块条件添加
        # (DTK clang 17 不支持 Objective-C runtime, 编 .mm 会失败)
        src/DeviceAllocator/AllocatorManager.cpp
        src/C3/Graph.cpp
        src/C3/HandwrittenKernelGen.cpp
        src/C3/C3Engine.cpp
        src/C3/GraphMerger.cpp
        src/C3/PGOManager.cpp
        src/C3/AOTCache.cpp
        src/C3/C3BackwardCapture.cpp
        src/C3/C3KernelRegistry.cpp
        src/C3/RollingHash.cpp
        src/C3/RegionFusionRegistry.cpp
)

# [Fix] v0.5.2 Linux build: Metal/MPS .mm 文件 macOS 专属, 下面 if APPLE 块条件添加
# (DTK clang 17 不支持 Objective-C runtime, 编 .mm 文件会失败)
# 注: 完整 if 块 (含 ${CT-MPS-OPs} 引用) 必须在 set(CT-MPS-OPs) 之后
#     L106 这里只 list APPEND MetalDevice.mm + MPSAllocator.mm (单独 set)

# MLIR kernel 生成器：仅在 CT_ENABLE_MLIR 时编译
if(CT_ENABLE_MLIR)
    list(APPEND CT-Core src/C3/MLIRKernelGen.cpp)
    list(APPEND CT-Core src/C3/JITCache.cpp)
    list(APPEND CT-Core src/C3/MLIRToLLVMIR.cpp)  # [Dev] v0.5 DCU 接入: MLIR → LLVM IR helper
endif()

# ---- 海光 DCU 后端 (v0.5 DCU 接入, 2026-08-10) ----
# 仅在 DCU 节点 build (需要 DTK 24.04+ + libgcvm.so + hipBLAS)
# macOS 上 WITH_DCU 默认 OFF (build 不会失败)
option(CT_ENABLE_DCU "Enable Hygon DCU (gfx906) backend for C3" OFF)
if(CT_ENABLE_DCU)
    # 1. ROCM 路径 (DTK 24.04+)
    # 默认 /public/software/compiler/rocm/dtk-26.04 (per dcu-probe-dtk24-b02r2n11.md 探针)
    # 也支持 /opt/dtk 传统路径, 通过 -DROCM_PATH=... 覆盖
    if(NOT DEFINED ROCM_PATH)
        if(EXISTS "/public/software/compiler/rocm/dtk-26.04/llvm/gcvm/lib/libgcvm.so")
            set(ROCM_PATH "/public/software/compiler/rocm/dtk-26.04" CACHE STRING "Default ROCM installation directory (DTK 26.04).")
        elseif(EXISTS "/opt/dtk/llvm/gcvm/lib/libgcvm.so")
            set(ROCM_PATH "/opt/dtk" CACHE STRING "Default ROCM installation directory (DTK /opt/dtk).")
        else()
            set(ROCM_PATH "/opt/dtk" CACHE STRING "Default ROCM installation directory (assumed; verify with scripts/probe-dcu-dtk24.sh).")
        endif()
    endif()

    set(GCVM_INCLUDE_DIR "${ROCM_PATH}/llvm/gcvm/include")
    set(GCVM_LIBRARY "${ROCM_PATH}/llvm/gcvm/lib/libgcvm.so")
    set(HIPBLAS_LIBRARY "${ROCM_PATH}/lib/libhipblas.so")

    # 2. 验证 GCVM 库存在
    if(NOT EXISTS "${GCVM_INCLUDE_DIR}")
        message(FATAL_ERROR "GCVM include directory not found: ${GCVM_INCLUDE_DIR}. DTK 24.04+ required.")
    endif()
    if(NOT EXISTS "${GCVM_LIBRARY}")
        message(FATAL_ERROR "GCVM library not found: ${GCVM_LIBRARY}. DTK 24.04+ required.")
    endif()
    if(NOT EXISTS "${HIPBLAS_LIBRARY}")
        message(FATAL_ERROR "hipBLAS library not found: ${HIPBLAS_LIBRARY}. DTK 24.04+ required.")
    endif()

    # 3. 导入 GCVM + hip 库
    add_library(gcvm SHARED IMPORTED)
    set_target_properties(gcvm PROPERTIES
        IMPORTED_LOCATION "${GCVM_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GCVM_INCLUDE_DIR}"
    )

    find_package(hip REQUIRED)

    # 4. DCU 编译宏 + 源文件
    add_definitions(-DCT_ENABLE_DCU=1 -DWITH_DCU=1)
    list(APPEND CT-Core src/C3/GCVMBridge.cpp)
    list(APPEND CT-Core src/C3/DCUCompiledKernel.cpp)

    message(STATUS "DCU backend enabled: GCVM at ${GCVM_LIBRARY}, hipBLAS at ${HIPBLAS_LIBRARY}")
endif()

set(CT-Core-Headers
        include/Tensor.h
        include/Storage.h
        include/Ctools.h
        include/CtorchScheduler.h
        include/CtorchError.h
        include/AutoGrad/DataCore.h
        include/ThreadPool.h
        include/AutoGrad/ComputeCore.h
        include/Arena.h
        include/AutoGrad.h
        include/DeviceAllocator.h
        include/MetalDevice.h
        include/ctQALS/ctQALS.h
        include/C3/C3Engine.h
        include/C3/C3Config.h
        include/C3/Graph.h
        include/C3/GraphMerger.h
        include/C3/Tracer.h
        include/C3/C3KernelRegistry.h
        include/C3/AOTCache.h
        include/C3/C3BackwardCapture.h
)

set(CT-Nodes-Headers
        include/AutoGrad/Node.h
        include/AutoGrad/Nodes/AddNode.h
        include/AutoGrad/Nodes/SubNode.h
        include/AutoGrad/Nodes/MulNode.h
        include/AutoGrad/Nodes/DivNode.h
        include/AutoGrad/Nodes/NegNode.h
        include/AutoGrad/Nodes/SinNode.h
        include/AutoGrad/Nodes/CosNode.h
        include/AutoGrad/Nodes/TanhNode.h
        include/AutoGrad/Nodes/SigmoidNode.h
        include/AutoGrad/Nodes/GELUNode.h
        include/AutoGrad/Nodes/ReLUNode.h
        include/AutoGrad/Nodes/LReLUNode.h
        include/AutoGrad/Nodes/MatMulNode.h
        include/AutoGrad/Nodes/CrossEntropyNode.h
        include/AutoGrad/Nodes/SoftmaxNode.h
        include/AutoGrad/Nodes/GradAccumulator.h
        include/AutoGrad/Nodes/LogNode.h
        include/AutoGrad/Nodes/ExpNode.h
        include/AutoGrad/Nodes/AbsNode.h
        include/AutoGrad/Nodes/MinNode.h
        include/AutoGrad/Nodes/MaxNode.h
)

set(CT-Nodes
        src/AutoGrad/Node.cpp
        src/AutoGrad/Nodes/AddNode.cpp
        src/AutoGrad/Nodes/SubNode.cpp
        src/AutoGrad/Nodes/MulNode.cpp
        src/AutoGrad/Nodes/DivNode.cpp
        src/AutoGrad/Nodes/NegNode.cpp
        src/AutoGrad/Nodes/SinNode.cpp
        src/AutoGrad/Nodes/CosNode.cpp
        src/AutoGrad/Nodes/TanhNode.cpp
        src/AutoGrad/Nodes/SigmoidNode.cpp
        src/AutoGrad/Nodes/GELUNode.cpp
        src/AutoGrad/Nodes/ReLUNode.cpp
        src/AutoGrad/Nodes/LReLUNode.cpp
        src/AutoGrad/Nodes/MatMulNode.cpp
        src/AutoGrad/Nodes/CrossEntropyNode.cpp
        src/AutoGrad/Nodes/SoftmaxNode.cpp
        src/AutoGrad/Nodes/GradAccumulator.cpp
        src/AutoGrad/Nodes/LogNode.cpp
        src/AutoGrad/Nodes/ExpNode.cpp
        src/AutoGrad/Nodes/AbsNode.cpp
        src/AutoGrad/Nodes/MinNode.cpp
        src/AutoGrad/Nodes/MaxNode.cpp
)

set(CT-CPU-BASIC-OPs
        src/kernels/CPU-BASIC/Add_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Sub_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Mul_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Div_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Neg_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/ReLU_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Cos_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Sin_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Tanh_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Sigmoid_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/GELU_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/LReLU_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/LReLU_Grad_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Softmax_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/MatMul_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Dot_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/MSE_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/CrossEntropy_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/MAE_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Log_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Exp_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Abs_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Min_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Max_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/unary_inplace_BASIC_kernels.cpp
)
set(CT-AMX-OPs
        src/kernels/AMX/MatMul_AMX_kernel.cpp
        src/kernels/AMX/GELU_AMX_kernel.cpp
        src/kernels/AMX/Rot_AMX_kernel.cpp
        src/kernels/AMX/ApplyHk_AMX_kernel.cpp
)

# [Fix] v0.5.2 Linux build: AMX kernels 用 Accelerate framework, macOS 专属
# (AMX = Apple Matrix coprocessor + Accelerate BLAS, Linux 都没)
# Linux 上 CT-AMX-OPs 跳过, CtorchScheduler.cpp AMX registration 也用 #ifdef __APPLE__ 守
if(APPLE)
    list(APPEND CT-Core ${CT-AMX-OPs})
endif()

set(CT-MPS-OPs
        src/kernels/MPS/MPS_kernel_dispatch.mm
)

# [Fix] v0.5.2 Linux build: Metal/MPS 文件 macOS 专属, Linux/DCU 节点不编译
# (DTK clang 17 不支持 Objective-C runtime, 编 .mm 文件会失败)
# 必须放在 set(CT-MPS-OPs) 之后, add_library 之前
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    list(APPEND CT-Core src/MetalDevice.mm)
    list(APPEND CT-Core src/DeviceAllocator/MPSAllocator.mm)
    list(APPEND CT-Core ${CT-MPS-OPs})
endif()

# 暂时禁用CPU-SIMD相关文件，因为需要AVX指令集支持
# set(CT-CPU-SIMD-OPs
#         src/kernels/CPU-SIMD/Add_SIMD_kernel.cpp
#         src/kernels/CPU-SIMD/Div_SIMD_kernel.cpp
#         src/kernels/CPU-SIMD/Mul_SIMD_kernel.cpp
#         src/kernels/CPU-SIMD/Sub_SIMD_kernel.cpp
#         src/kernels/CPU-SIMD/Tanh_SIMD_kernel.cpp
# )

set(CT-CPU-SIMD-OPs
        src/kernels/CPU-SIMD/Add_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Sub_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Mul_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Div_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Neg_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/ReLU_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Tanh_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Sigmoid_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/GELU_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Log_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Exp_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Abs_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Min_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Max_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/MatMul_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Dot_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/MSE_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/MAE_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/CrossEntropy_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Softmax_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Sin_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Cos_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/LReLU_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/LReLU_Grad_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/SIMDMath.cpp
        src/kernels/CPU-SIMD/SIMDWrapper.cpp
)

set(CT-Distributed
        include/Distributed/DeviceBackend.h
        include/Distributed/BackendManager.h
        include/Distributed/DeviceMigration.h
        include/Distributed/CDTF.h
        include/Distributed/CommEngine.h
        include/Distributed/GradientAggregator.h
        include/Distributed/DistributedOptimizer.h
        include/Distributed/GTCScheduler.h
        include/Distributed/EntropyAwareCompressor.h
        include/Distributed/TopologyManager.h
        include/Distributed/QuorumManager.h
        include/Distributed/NodeDiscovery.h
        include/Distributed/FaultTolerance.h
        include/Distributed/CheckpointManager.h
        include/Distributed/Transport.h
        include/Distributed/CPUBackend.h
        include/Distributed/MPSBackend.h
        include/Distributed/DistributedTrainer.h
        src/Distributed/BackendManager.cpp
        src/Distributed/CDTF.cpp
        src/Distributed/CommEngine.cpp
        src/Distributed/GradientAggregator.cpp
        src/Distributed/DistributedOptimizer.cpp
        src/Distributed/GTCScheduler.cpp
        src/Distributed/EntropyAwareCompressor.cpp
        src/Distributed/TopologyManager.cpp
        src/Distributed/QuorumManager.cpp
        src/Distributed/NodeDiscovery.cpp
        src/Distributed/FaultTolerance.cpp
        src/Distributed/CheckpointManager.cpp
        src/Distributed/Transport.cpp
        src/Distributed/CPUBackend.cpp
        src/Distributed/MPSBackend.mm
        src/Distributed/DistributedTrainer.cpp
)

set(ctQALS
        include/ctQALS/ctQALS.h
        include/ctQALS/Random.h
        src/ctQALS/xoshiro.cpp
        src/ctQALS/ziggurat.cpp
)

add_library(CTorch OBJECT
        ${CT-Core}
        ${CT-Core-Headers}
        ${CT-CPU-BASIC-OPs}
        ${CT-CPU-SIMD-OPs}
        ${CT-AMX-OPs}
        ${CT-Nodes}
        ${CT-Nodes-Headers}
        ${CT-Distributed}
        ${ctQALS}
)

set_target_properties(CTorch PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION ${CT_ENABLE_LTO}
)

# 包含头文件目录
target_include_directories(CTorch
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# MLIR/LLVM 头文件与链接
if(CT_ENABLE_MLIR)
    target_include_directories(CTorch SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
    target_include_directories(CTorch SYSTEM PUBLIC ${MLIR_INCLUDE_DIRS})
    target_compile_definitions(CTorch PUBLIC CT_ENABLE_MLIR=1)
    target_link_libraries(CTorch PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransforms
        MLIRSupport
        MLIRArithDialect
        MLIRSCFDialect
        MLIRFuncDialect
        MLIRMemRefDialect
        MLIRLLVMDialect
        MLIRArithToLLVM
        MLIRArithAttrToLLVMConversion
        MLIRSCFToControlFlow
        MLIRControlFlowToLLVM
        MLIRMemRefToLLVM
        MLIRFuncToLLVM
        MLIRReconcileUnrealizedCasts
        MLIRLLVMToLLVMIRTranslation
        MLIRBuiltinToLLVMIRTranslation
        MLIRExecutionEngine
    )
endif()

# 海光 DCU 链接 (v0.5 DCU 接入, 2026-08-10)
# 仅 WITH_DCU=ON 时生效, 在 DCU 节点 build
if(CT_ENABLE_DCU)
    target_link_libraries(CTorch PUBLIC gcvm hip::host hip::device hipblas ${HIPBLAS_LIBRARY})
    target_include_directories(CTorch SYSTEM PUBLIC ${GCVM_INCLUDE_DIR})
    target_compile_definitions(CTorch PUBLIC CT_ENABLE_DCU=1 WITH_DCU=1)
endif()

add_compile_options(-march=native)

# 链接 Accelerate 框架（用于 AMX BLAS 加速）
target_link_libraries(CTorch PUBLIC "-framework Accelerate")
target_link_libraries(CTorch PUBLIC "-framework Metal")
target_link_libraries(CTorch PUBLIC "-framework Foundation")

# [Fix] v0.5.2 Linux build: <filesystem> 在 libstdc++ 早期版本 (12.3 之前) 需要
# link -lstdc++fs 兜底. macOS libc++ 自带 filesystem, 不需要.
# find_library 失败时 stdc++fs 库不存在, link_libraries noop 跳过
find_library(STDCPPEFS_LIB NAMES stdc++fs PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib64)
if(STDCPPEFS_LIB)
    target_link_libraries(CTorch PUBLIC ${STDCPPEFS_LIB})
    message(STATUS "Linked stdc++fs for <filesystem> support (Linux libstdc++-12 < 12.3)")
endif()

add_executable(MnistTest mnist/mnist.cpp mnist/mnist_loader.cpp mnist/mnist_loader.h)

# Standalone test executables (migrated from repo root to src/tests/standalone)
add_executable(test_matmul_stride src/tests/standalone/test_matmul_stride.cpp)
target_link_libraries(test_matmul_stride PRIVATE CTorch)
target_include_directories(test_matmul_stride PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_rsvd_tensor src/tests/standalone/test_rsvd_tensor.cpp)
target_link_libraries(test_rsvd_tensor PRIVATE CTorch)
target_include_directories(test_rsvd_tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# DCU Hello World test (v0.5 DCU 接入, 2026-08-10)
# 仅在 WITH_DCU=ON 时 build (DCU 节点)
if(CT_ENABLE_DCU)
    add_executable(test_c3_dcu_hello src/tests/standalone/test_c3_dcu_hello.cpp)
    target_link_libraries(test_c3_dcu_hello PRIVATE CTorch)
    target_include_directories(test_c3_dcu_hello PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # [Dev] v0.5.2: test_dcu_hello_v2 完整链路 (MLIR→GCVM/dcc/host CPU Plan A/B/C fallback)
    # 跟 test_c3_dcu_hello 区别: 走真 mlirToLLVMIRFromGraph pipeline + Plan A/B/C 自动 fallback
    add_executable(test_dcu_hello_v2 src/tests/standalone/test_dcu_hello_v2.cpp)
    target_link_libraries(test_dcu_hello_v2 PRIVATE CTorch)
    target_include_directories(test_dcu_hello_v2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# MLIRToLLVMIR helper 单测 (v0.5 DCU 接入基础设施, 2026-08-10)
# 只需 CT_ENABLE_MLIR=ON, 不依赖 DCU
if(CT_ENABLE_MLIR)
    add_executable(test_mlir_to_llvm_ir src/tests/standalone/test_mlir_to_llvm_ir.cpp)
    target_link_libraries(test_mlir_to_llvm_ir PRIVATE CTorch)
    target_include_directories(test_mlir_to_llvm_ir PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

add_executable(test_bidiag_svd_accuracy src/tests/standalone/test_bidiag_svd_accuracy.cpp)
target_link_libraries(test_bidiag_svd_accuracy PRIVATE CTorch)
target_include_directories(test_bidiag_svd_accuracy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_cross_entropy src/tests/standalone/test_cross_entropy.cpp)
target_link_libraries(test_cross_entropy PRIVATE CTorch)
target_include_directories(test_cross_entropy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_grad_accum src/tests/standalone/test_grad_accum.cpp src/tests/standalone/test_mps_flush.mm)
target_link_libraries(test_grad_accum PRIVATE CTorch)
target_include_directories(test_grad_accum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_sgd_sync src/tests/standalone/test_sgd_sync.cpp)
target_link_libraries(test_sgd_sync PRIVATE CTorch)
target_include_directories(test_sgd_sync PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_mnist_step src/tests/standalone/test_mnist_step.cpp mnist/mnist_loader.cpp mnist/mnist_loader.h)
target_link_libraries(test_mnist_step PRIVATE CTorch)
target_include_directories(test_mnist_step PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_relu_backward src/tests/standalone/test_relu_backward.cpp)
target_link_libraries(test_relu_backward PRIVATE CTorch)
target_include_directories(test_relu_backward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_mnist_perf src/tests/standalone/test_mnist_perf.cpp mnist/mnist_loader.cpp mnist/mnist_loader.h)
target_link_libraries(test_mnist_perf PRIVATE CTorch)
target_include_directories(test_mnist_perf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(exp_mnist_cold_start src/tests/standalone/exp_mnist_cold_start.cpp mnist/mnist_loader.cpp mnist/mnist_loader.h)
target_link_libraries(exp_mnist_cold_start PRIVATE CTorch)
target_include_directories(exp_mnist_cold_start PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_c3_matmul_blas src/tests/standalone/test_c3_matmul_blas.cpp)
target_link_libraries(test_c3_matmul_blas PRIVATE CTorch)
target_include_directories(test_c3_matmul_blas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_h2_matmul_precision src/tests/standalone/test_h2_matmul_precision.cpp)
target_link_libraries(test_h2_matmul_precision PRIVATE CTorch)
target_include_directories(test_h2_matmul_precision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_h2_registry_override src/tests/standalone/test_h2_registry_override.cpp)
target_link_libraries(test_h2_registry_override PRIVATE CTorch)
target_include_directories(test_h2_registry_override PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_c3_mnist_step src/tests/standalone/test_c3_mnist_step.cpp mnist/mnist_loader.cpp mnist/mnist_loader.h)
target_link_libraries(test_c3_mnist_step PRIVATE CTorch)
target_include_directories(test_c3_mnist_step PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_c3_compile_time src/tests/standalone/test_c3_compile_time.cpp)
target_link_libraries(test_c3_compile_time PRIVATE CTorch)
target_include_directories(test_c3_compile_time PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(diag_mini_train src/tests/standalone/diag_mini_train.cpp)
target_link_libraries(diag_mini_train PRIVATE CTorch)
target_include_directories(diag_mini_train PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_c3_mnist_train src/tests/standalone/test_c3_mnist_train.cpp mnist/mnist_loader.cpp mnist/mnist_loader.h)
target_link_libraries(test_c3_mnist_train PRIVATE CTorch)
target_include_directories(test_c3_mnist_train PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_c3_compile_and_inject src/tests/standalone/test_c3_compile_and_inject.cpp)
target_link_libraries(test_c3_compile_and_inject PRIVATE CTorch)
target_include_directories(test_c3_compile_and_inject PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_c3_compile_merged src/tests/standalone/test_c3_compile_merged.cpp)
target_link_libraries(test_c3_compile_merged PRIVATE CTorch)
target_include_directories(test_c3_compile_merged PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_c3_compile_merged_pgo src/tests/standalone/test_c3_compile_merged_pgo.cpp)
target_link_libraries(test_c3_compile_merged_pgo PRIVATE CTorch)
target_include_directories(test_c3_compile_merged_pgo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_c3_pgo_deopt src/tests/standalone/test_c3_pgo_deopt.cpp)
target_link_libraries(test_c3_pgo_deopt PRIVATE CTorch)
target_include_directories(test_c3_pgo_deopt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_c3_compile_error src/tests/standalone/test_c3_compile_error.cpp)
target_link_libraries(test_c3_compile_error PRIVATE CTorch)
target_include_directories(test_c3_compile_error PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_c3_compile_timeout src/tests/standalone/test_c3_compile_timeout.cpp)
target_link_libraries(test_c3_compile_timeout PRIVATE CTorch)
target_include_directories(test_c3_compile_timeout PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 临时调试目标（需要时手动创建源文件）
# add_executable(test_c3_compile_timeout_minimal /tmp/test_c3_compile_timeout_minimal.cpp)
# target_link_libraries(test_c3_compile_timeout_minimal PRIVATE CTorch)
# target_include_directories(test_c3_compile_timeout_minimal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
#
# add_executable(test_timeout_simple /tmp/test_timeout_simple.cpp)
# target_link_libraries(test_timeout_simple PRIVATE CTorch)
# target_include_directories(test_timeout_simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_c3_aot_cache src/tests/standalone/test_c3_aot_cache.cpp)
target_link_libraries(test_c3_aot_cache PRIVATE CTorch)
target_include_directories(test_c3_aot_cache PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bench_aot_speedup src/tests/standalone/bench_aot_speedup.cpp)
target_link_libraries(bench_aot_speedup PRIVATE CTorch)
target_include_directories(bench_aot_speedup PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_mlir_simd_integration src/tests/standalone/test_mlir_simd_integration.cpp)
target_link_libraries(test_mlir_simd_integration PRIVATE CTorch)
target_include_directories(test_mlir_simd_integration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# -rdynamic 导出主程序符号到动态符号表，MLIR JIT 通过 dlsym 查找
# 自定义 SIMD 函数（ct_simd_vsigmoid 等）需要被 JIT 可见
target_link_options(test_mlir_simd_integration PRIVATE -rdynamic)

add_executable(bench_mlir_simd src/tests/standalone/bench_mlir_simd.cpp)
target_link_libraries(bench_mlir_simd PRIVATE CTorch)
target_include_directories(bench_mlir_simd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(bench_mlir_simd PRIVATE -rdynamic)

add_executable(bench_subgraph_roi src/tests/standalone/bench_subgraph_roi.cpp)
target_link_libraries(bench_subgraph_roi PRIVATE CTorch)
target_include_directories(bench_subgraph_roi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bench_auto_c3 src/tests/standalone/bench_auto_c3.cpp)
target_link_libraries(bench_auto_c3 PRIVATE CTorch)
target_include_directories(bench_auto_c3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_fusion_dispatch src/tests/standalone/test_fusion_dispatch.cpp)
target_link_libraries(test_fusion_dispatch PRIVATE CTorch)
target_include_directories(test_fusion_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_region_fusion src/tests/standalone/test_region_fusion.cpp)
target_link_libraries(test_region_fusion PRIVATE CTorch)
target_include_directories(test_region_fusion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# C3_DISPATCH_TIMING disabled: per-dispatch fprintf overhead (~20-50us) skews perf measurements

add_executable(test_region_fusion_auto src/tests/standalone/test_region_fusion_auto.cpp)
target_link_libraries(test_region_fusion_auto PRIVATE CTorch)
target_include_directories(test_region_fusion_auto PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_debug_fused src/tests/standalone/test_debug_fused.cpp)
target_link_libraries(test_debug_fused PRIVATE CTorch)
target_include_directories(test_debug_fused PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_c3_backward src/tests/standalone/test_c3_backward.cpp)
target_link_libraries(test_c3_backward PRIVATE CTorch)
target_include_directories(test_c3_backward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bench_backward_fusion src/tests/standalone/bench_backward_fusion.cpp)
target_link_libraries(bench_backward_fusion PRIVATE CTorch)
target_include_directories(bench_backward_fusion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bench_c3_backward_perf src/tests/standalone/bench_c3_backward_perf.cpp)
target_link_libraries(bench_c3_backward_perf PRIVATE CTorch)
target_include_directories(bench_c3_backward_perf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bench_c3_backward_perf_clean src/tests/standalone/bench_c3_backward_perf_clean.cpp)
target_link_libraries(bench_c3_backward_perf_clean PRIVATE CTorch)
target_include_directories(bench_c3_backward_perf_clean PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(split_fwd_bwd src/tests/standalone/split_fwd_bwd.cpp)
target_link_libraries(split_fwd_bwd PRIVATE CTorch)
target_include_directories(split_fwd_bwd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(exp_c3_warmup src/tests/standalone/exp_c3_warmup.cpp)
target_link_libraries(exp_c3_warmup PRIVATE CTorch)
target_include_directories(exp_c3_warmup PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_simd_math src/tests/standalone/test_simd_math.cpp)
target_link_libraries(test_simd_math PRIVATE CTorch)
target_include_directories(test_simd_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bench_simd_math src/tests/standalone/bench_simd_math.cpp)
target_link_libraries(bench_simd_math PRIVATE CTorch)
target_include_directories(bench_simd_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bench_amx_startup src/tests/standalone/bench_amx_startup.cpp)
target_link_libraries(bench_amx_startup PRIVATE CTorch)
target_include_directories(bench_amx_startup PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# (SECURITY_SENSITIVE) MNIST Loader security regression tests
add_executable(test_mnist_loader_security src/tests/standalone/test_mnist_loader_security.cpp mnist/mnist_loader.cpp mnist/mnist_loader.h)
target_link_libraries(test_mnist_loader_security PRIVATE CTorch)
target_include_directories(test_mnist_loader_security PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# (SECURITY_SENSITIVE) Red-team PoC: Storage integer overflow
add_executable(poc_storage_integer_overflow src/tests/standalone/poc_storage_integer_overflow.cpp)
target_link_libraries(poc_storage_integer_overflow PRIVATE CTorch)
target_include_directories(poc_storage_integer_overflow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# (SECURITY_SENSITIVE) Red-team PoC: Tensor numel()/computeStrides() integer overflow
add_executable(poc_tensor_numel_overflow src/tests/standalone/poc_tensor_numel_overflow.cpp)
target_link_libraries(poc_tensor_numel_overflow PRIVATE CTorch)
target_include_directories(poc_tensor_numel_overflow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# (SECURITY_SENSITIVE) Red-team PoC: MPS CrossEntropy target out-of-bounds read
add_executable(poc_mps_crossentropy_target_oob src/tests/standalone/poc_mps_crossentropy_target_oob.cpp)
target_link_libraries(poc_mps_crossentropy_target_oob PRIVATE CTorch)
target_include_directories(poc_mps_crossentropy_target_oob PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# (SECURITY_SENSITIVE) Red-team PoC: MPS view storage_offset buffer lookup failure
add_executable(poc_mps_storage_offset_lookup src/tests/standalone/poc_mps_storage_offset_lookup.cpp)
target_link_libraries(poc_mps_storage_offset_lookup PRIVATE CTorch)
target_include_directories(poc_mps_storage_offset_lookup PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# (SECURITY_SENSITIVE) Red-team PoC: CPU kernel int elem_count narrowing
add_executable(poc_cpu_kernel_int_narrowing src/tests/standalone/poc_cpu_kernel_int_narrowing.cpp)
target_link_libraries(poc_cpu_kernel_int_narrowing PRIVATE CTorch)
target_include_directories(poc_cpu_kernel_int_narrowing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# (SECURITY_SENSITIVE) Red-team PoC: Storage::Deleter exit-time UAF
add_executable(poc_storage_deleter_uaf src/tests/standalone/poc_storage_deleter_uaf.cpp)
target_link_libraries(poc_storage_deleter_uaf PRIVATE CTorch)
target_include_directories(poc_storage_deleter_uaf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Gen 2 分布式系统单元测试
add_executable(test_distributed_gen2 src/tests/standalone/test_distributed_gen2.cpp)
target_link_libraries(test_distributed_gen2 PRIVATE CTorch)
target_include_directories(test_distributed_gen2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Gen 2 端到端集成测试
add_executable(test_gen2_e2e src/tests/standalone/test_gen2_e2e.cpp)
target_link_libraries(test_gen2_e2e PRIVATE CTorch)
target_include_directories(test_gen2_e2e PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Gen 2 性能基准测试
add_executable(bench_gen2 src/tests/standalone/bench_gen2.cpp)
target_link_libraries(bench_gen2 PRIVATE CTorch)
target_include_directories(bench_gen2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_compile_options(-Wno-ignored-attributes)

# Discover all GDB pretty-printer Python modules in scripts/
# (evaluated at configure time — re-run cmake if you add new .py files)
file(GLOB _CT_GDB_SCRIPTS "${CMAKE_SOURCE_DIR}/scripts/*.py")

function(add_multiarch_executable NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs FILES ARCH)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Default to native architecture
    if(NOT ARG_ARCH)
        set(ARG_ARCH "native")
    endif()

    foreach(ARCH ${ARG_ARCH})
        set(TARGET_NAME "${NAME}-${ARCH}")
        add_executable(${TARGET_NAME} ${ARG_FILES})
        target_compile_options(${TARGET_NAME} PRIVATE -march=${ARCH})
        target_link_libraries(${TARGET_NAME} PRIVATE CTorch gtest gtest_main)
        gtest_discover_tests(${TARGET_NAME})

        # --- GDB pretty-printer auto-load: generate <target>-gdb.py ---
        list(LENGTH _CT_GDB_SCRIPTS _CT_NUM_SCRIPTS)
        if(_CT_NUM_SCRIPTS GREATER 0)
            set(_CT_IMPORTS "")
            foreach(_CT_MOD ${_CT_GDB_SCRIPTS})
                get_filename_component(_CT_MNAME "${_CT_MOD}" NAME_WE)
                string(APPEND _CT_IMPORTS "try:\n    import ${_CT_MNAME}\nexcept Exception:\n    pass\n")
            endforeach()

            # Write a loader script that imports every module in scripts/
            set(_CT_LOADER "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-gdb.py")
            file(WRITE "${_CT_LOADER}"
"import sys
sys.path.insert(0, '${CMAKE_SOURCE_DIR}/scripts')

${_CT_IMPORTS}")

            # Copy the loader next to the executable after each build
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_CT_LOADER}"
                    "$<TARGET_FILE_DIR:${TARGET_NAME}>/${TARGET_NAME}-gdb.py"
                COMMENT "Installing GDB pretty-printer for ${TARGET_NAME}"
            )
        endif()

    endforeach()
endfunction()

# tl-ArrayTest
add_executable(tl-ArrayTest src/tests/tl/ArrayTest.cpp)
target_link_libraries(tl-ArrayTest PRIVATE CTorch gtest gtest_main)  # Note: build with CT_DEBUG defined
gtest_discover_tests(tl-ArrayTest)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64")
    # tl-VecBaseTest
    add_multiarch_executable(tl-VecBaseTest
            FILES src/tests/tl/vec/VecBaseTest.cpp
            ARCH x86-64
    )

    # tl-VecScalarTest
    add_multiarch_executable(tl-VecScalarTest
            FILES src/tests/tl/vec/VecScalarTest.cpp
            ARCH x86-64
    )

    # tl-VecLoadStoreTest
    add_multiarch_executable(tl-VecLoadStoreTest
            FILES src/tests/tl/vec/VecLoadStoreTest.cpp
            ARCH x86-64 corei7-avx core-avx2 native
    )
    # tl-VecArithTest
    add_multiarch_executable(tl-VecArithTest
            FILES src/tests/tl/vec/VecArithTest.cpp
            ARCH x86-64 corei7-avx core-avx2 native
    )
    # tl-VecConversionTest
    add_multiarch_executable(tl-VecConversionTest
            FILES src/tests/tl/vec/VecConversionTest.cpp
            ARCH x86-64 corei7-avx core-avx2 native
    )
    # tl-VecShuffleTest
    add_multiarch_executable(tl-VecShuffleTest
            FILES src/tests/tl/vec/VecShuffleTest.cpp
            ARCH x86-64 corei7-avx core-avx2 native
    )
endif()

# target_link_libraries(PerformanceTest PRIVATE CTorch)
# target_link_libraries(StdTest PRIVATE CTorch)
# target_link_libraries(ActivationLossTest PRIVATE CTorch)
# target_link_libraries(TensorGradTest PRIVATE CTorch)
target_link_libraries(MnistTest PRIVATE CTorch)
target_link_libraries(MnistTest PRIVATE "-framework Accelerate")

# 添加编译选项
target_compile_options(CTorch PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -O3
)


# 检测平台并添加相应的编译选项
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # macOS 特定选项
    target_compile_options(CTorch PRIVATE
            -march=native
    )
elseif(UNIX AND NOT APPLE)
    # Linux 特定选项
    target_compile_options(CTorch PRIVATE
            -march=native
            -fopenmp
    )
    target_link_libraries(CTorch PRIVATE
            -fopenmp
    )
endif()

option(TEST_ENABLED "Enable tests?" ON)
if (${TEST_ENABLED})
    # Homebrew gtest include path (macOS)
    if(APPLE AND EXISTS /opt/homebrew/include/gtest/gtest.h)
        include_directories(/opt/homebrew/include)
        link_directories(/opt/homebrew/lib)
    endif()
    set(TEST_GROUP_Storage src/tests/units/Storage/*.cpp)
    set(TEST_GROUP_Tensor src/tests/units/Tensor/*.cpp)
    set(TEST_GROUP_C3 src/tests/units/C3/*.cpp)
    file(GLOB_RECURSE ALL_TEST src/tests/units/*.cpp)

    if (NOT ALL_TEST)
        message(STATUS "No tests found.Skip.")
        set(TEST_ENABLED OFF)
        return()
    endif ()

    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testConfig.cfg)
        file(READ ${CMAKE_CURRENT_SOURCE_DIR}/testConfig.cfg TEST_CONFIG_CONTENT)
        string(REPLACE "\n" ";" TEST_CONFIG "${TEST_CONFIG_CONTENT}")
    else ()
        message(FATAL_ERROR "Tests enabled but no testConfig.cfg found,did you put it in the same directory with CMakeLists.txt?")
    endif ()

    set(TESTS_TO_BUILD "")
    set(ALL OFF)

    foreach (LINE ${TEST_CONFIG})
        string(REGEX REPLACE "^[\t]*#.*" "" LINE ${LINE})
        string(STRIP "${LINE}" LINE)

        if (NOT LINE)
            continue()
        endif ()

        if (LINE STREQUAL "all")
            set(ALL ON)
            set(TESTS_TO_BUILD ${ALL_TEST})
            break()
        endif ()

        if(LINE MATCHES "^group:([a-zA-Z0-9_]+)$")
            set(GROUP_NAME ${CMAKE_MATCH_1})
            if (DEFINED TEST_GROUP_${GROUP_NAME})
                file(GLOB GROUP_TESTS ${TEST_GROUP_${GROUP_NAME}})
                list(APPEND TESTS_TO_BUILD ${GROUP_TESTS})
                message(STATUS "Test group added:${GROUP_NAME}")
            else ()
                message(WARNING "Unknown test group:${GROUP_NAME}")
            endif ()
        elseif (LINE MATCHES "^single:([a-zA-Z0-9_-]+)$")
            set(TEST_NAME ${CMAKE_MATCH_1})
            set(FOUND OFF)
            foreach (FILE ${ALL_TEST})
                get_filename_component(FILE_NAME ${FILE} NAME_WE)
                if (FILE_NAME STREQUAL TEST_NAME)
                    list(APPEND TESTS_TO_BUILD ${FILE})
                    set(FOUND ON)
                    message(STATUS "Test added:${TEST_NAME}")
                    break()
                endif ()
            endforeach ()
            if (NOT FOUND)
                message(WARNING "No single test matches:${TEST_NAME}")
            endif ()
        else ()
            message(WARNING "Cannot resolve configuration line:${LINE}")
        endif ()
    endforeach ()

    if (ALL)
        message(STATUS "Compile all the tests")
        message(STATUS "Chosen tests:")
        foreach (TEST ${TESTS_TO_BUILD})
            message(STATUS "${TEST}\n")
        endforeach ()
    elseif (TESTS_TO_BUILD)
        list(REMOVE_DUPLICATES TESTS_TO_BUILD)
        message(STATUS "Chosen tests:")
        foreach (TEST ${TESTS_TO_BUILD})
            message(STATUS "${TEST}\n")
        endforeach ()
    else ()
        message(STATUS "No tests chosen")
    endif ()

    foreach (TEST ${TESTS_TO_BUILD})
        get_filename_component(TEST_NAME ${TEST} NAME_WE)
        # 每个测试文件自带 main()，不再向其注入 test_rng.cpp（曾导致 _main 重复）
        add_executable(${TEST_NAME} ${TEST})
        target_link_libraries(${TEST_NAME} PRIVATE CTorch gtest gtest_main)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach ()

    enable_testing()

    add_custom_target(showTests
            COMMAND ${CMAKE_COMMAND} -E echo "Current configured tests"
            COMMAND ${CMAKE_COMMAND} -E echo "${TESTS_TO_BUILD}"
            COMMENT "To show the chosen tests")
endif ()
