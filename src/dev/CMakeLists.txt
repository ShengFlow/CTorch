cmake_minimum_required(VERSION 3.10)
project(Ctorch)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)/
# 确保编译器使用UTF-8编码处理源文件
if(MSVC)
    # Visual Studio编译器
    add_compile_options("/source-charset:utf-8")
else()
    # GCC/Clang编译器
    add_compile_options(-finput-charset=UTF-8)
endif()

set(ct-SRC
        Ctools.cpp
        Ctools.h
        AutoDiff.cpp
        AutoDiff.h
        Tensor.cpp
        Tensor.h
        Storage.cpp
        Storage.h
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

add_library(ct STATIC ${ct-SRC})

# 添加包含目录
target_include_directories(ct
    PUBLIC
    ${CMAKE_SOURCE_DIR}
)

add_executable(StdTest main.cpp)

target_link_libraries(StdTest PRIVATE ct)

# 测试激活函数和损失函数的可执行文件
add_executable(ActivationLossTest test_activation_loss.cpp)

target_link_libraries(ActivationLossTest PRIVATE ct)

# 性能测试可执行文件
add_executable(PerformanceTest performance_test.cpp)

target_link_libraries(PerformanceTest PRIVATE ct)
