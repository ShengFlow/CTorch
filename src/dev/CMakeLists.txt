cmake_minimum_required(VERSION 3.10)
project(CTorchDev LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 当本项目作为子目录被 add_subdirectory() 引入时，默认不构建这些可执行文件（避免“被依赖方”顺带构建一堆 demo）。
set(_CTORCHDEV_TOP_LEVEL OFF)
if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(_CTORCHDEV_TOP_LEVEL ON)
endif()
option(CTORCHDEV_BUILD_EXECUTABLES "Build dev executables (Ctorch_test, performance_test, etc.)" ${_CTORCHDEV_TOP_LEVEL})

# 编译器 UTF-8 源码选项（只对支持的编译器生效）
if (MSVC)
    add_compile_options("/source-charset:utf-8")
else()
    add_compile_options(-finput-charset=UTF-8)
endif()

# -------------------------
# CTorch 静态库（核心代码只编译一次，供所有可执行文件/demo 复用）
# -------------------------
add_library(CTorch STATIC)

target_sources(CTorch
    PRIVATE
        Ctools.cpp
        AutoDiff.cpp
        Tensor.cpp
        Storage.cpp
        kernels/CPU-BASIC/Add_BASIC_kernel.cpp
        kernels/CPU-BASIC/Sub_BASIC_kernel.cpp
        kernels/CPU-BASIC/Mul_BASIC_kernel.cpp
        kernels/CPU-BASIC/Div_BASIC_kernel.cpp
        kernels/CPU-BASIC/MatMul_BASIC_kernel.cpp
        kernels/CPU-BASIC/Dot_BASIC_kernel.cpp
        kernels/CPU-BASIC/Neg_BASIC_kernel.cpp
        kernels/CPU-BASIC/ReLU_BASIC_kernel.cpp
        kernels/CPU-BASIC/Cos_BASIC_kernel.cpp
        kernels/CPU-BASIC/Sin_BASIC_kernel.cpp
        kernels/CPU-BASIC/Tanh_BASIC_kernel.cpp
        kernels/CPU-BASIC/Sigmoid_BASIC_kernel.cpp
        kernels/CPU-BASIC/Softmax_BASIC_kernel.cpp
        kernels/CPU-BASIC/MSE_BASIC_kernel.cpp
        kernels/CPU-BASIC/CrossEntropy_BASIC_kernel.cpp
        kernels/CPU-BASIC/MAE_BASIC_kernel.cpp
)

target_include_directories(CTorch
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 更“系统级”的隐藏问题，往往会被 warning 暗示；这里统一打开。
target_compile_options(CTorch PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# Release 优化（不强制覆盖 Debug）
target_compile_options(CTorch PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-ffast-math>
)

# 尽量只在非 MSVC 下开启 -march=native；AppleClang 也支持，但保守起见只在 Release 使用。
if (NOT MSVC)
    target_compile_options(CTorch PRIVATE $<$<CONFIG:Release>:-march=native>)
endif()

# Apple: 使用 Accelerate (BLAS) 时，链接到框架更稳；避免仅靠头文件 include。
if (APPLE)
    target_link_libraries(CTorch PUBLIC "-framework Accelerate")
endif()

# -------------------------
# 可执行文件（dev 测试/基准）
# -------------------------
if (CTORCHDEV_BUILD_EXECUTABLES)
    add_executable(Ctorch_test main.cpp)
    target_link_libraries(Ctorch_test PRIVATE CTorch)

    add_executable(test_activation_loss test_activation_loss.cpp)
    target_link_libraries(test_activation_loss PRIVATE CTorch)

    add_executable(performance_test performance_test.cpp)
    target_link_libraries(performance_test PRIVATE CTorch)

    add_executable(test_autodiff_comprehensive test_autodiff_comprehensive.cpp)
    target_link_libraries(test_autodiff_comprehensive PRIVATE CTorch)

    add_executable(test_linear_regression test_linear_regression.cpp)
    target_link_libraries(test_linear_regression PRIVATE CTorch)
endif()

# mkdir build && cd build && cmake .. && cmake --build . -j