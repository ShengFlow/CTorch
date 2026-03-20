cmake_minimum_required(VERSION 3.31)
project(CTorch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

include(FetchContent)

enable_testing()
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://gh-proxy.org/https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

set(CT-Core
        src/Tensor.cpp
        src/AutoDiff.cpp
        src/Storage.cpp
        src/Ctools.cpp
        src/Assertion.cpp
)

set(CT-Core-Headers
        include/Tensor.h
        include/AutoDiff.h
        include/Storage.h
        include/Ctools.h
        include/Ctorch_Scheduler.h
        include/Ctorch_Error.h

)

set(CT-OPs
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
        src/kernels/CPU-BASIC/Softmax_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/MatMul_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Dot_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/MSE_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/CrossEntropy_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/MAE_BASIC_kernel.cpp
)

add_library(CTorch STATIC ${CT-Core} ${CT-Core-Headers} ${CT-OPs})

# 包含头文件目录
target_include_directories(CTorch
        PUBLIC
         ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_compile_options(-march=native)

add_executable(PerformanceTest src/tests/performance_test.cpp)
add_executable(StdTest src/tests/test.cpp)
add_executable(ActivationLossTest src/tests/test_activation_loss.cpp)
add_executable(TensorGradTest src/tests/test_tensor_grad.cpp)
add_executable(MnistTest mnist/mnist.cpp mnist/mnist_loader.cpp mnist/mnist_loader.h)

# Ignore warnings
add_compile_options(-Wno-ignored-attributes)

function(add_x86_multiarch_executable NAME)
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
    endforeach()
endfunction()

# tl-ArrayTest
add_executable(tl-ArrayTest src/tests/tl/ArrayTest.cpp)
target_link_libraries(tl-ArrayTest PRIVATE CTorch gtest gtest_main)  # Note: build with CT_DEBUG defined
gtest_discover_tests(tl-ArrayTest)

# tl-VecTest
add_executable(tl-VecTest src/tests/tl/VecTest.cpp)
target_compile_options(tl-VecTest PRIVATE -O3)
target_link_libraries(tl-VecTest PRIVATE CTorch gtest gtest_main)
gtest_discover_tests(tl-VecTest)

# tl-VecBaseTest
add_executable(tl-VecBaseTest src/tests/tl/vec/VecBaseTest.cpp)
target_link_libraries(tl-VecBaseTest PRIVATE CTorch gtest gtest_main)
gtest_discover_tests(tl-VecBaseTest)

# tl-VecScalarTest
add_executable(tl-VecScalarTest src/tests/tl/vec/VecScalarTest.cpp)
target_compile_options(tl-VecScalarTest PRIVATE -O3)
target_link_libraries(tl-VecScalarTest PRIVATE CTorch gtest gtest_main)
gtest_discover_tests(tl-VecScalarTest)

# tl-X86_128Test
add_x86_multiarch_executable(tl-X86_128Test
        FILES src/tests/tl/vec/X86_128Test.cpp
        ARCH core-avx-i core-avx2 native
)

target_link_libraries(PerformanceTest PRIVATE CTorch)
target_link_libraries(StdTest PRIVATE CTorch)
target_link_libraries(ActivationLossTest PRIVATE CTorch)
target_link_libraries(TensorGradTest PRIVATE CTorch)
target_link_libraries(MnistTest PRIVATE CTorch)

# 添加编译选项
target_compile_options(CTorch PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -O3
)


# 检测平台并添加相应的编译选项
if(APPLE)
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