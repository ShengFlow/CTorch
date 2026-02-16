cmake_minimum_required(VERSION 3.31)
project(CTorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CT-Core
        src/Tensor.cpp
        src/AutoDiff.cpp
        src/Storage.cpp
        src/Ctools.cpp
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

add_executable(PerformanceTest src/tests/performance_test.cpp)
add_executable(StdTest src/tests/test.cpp)
add_executable(ActivationLossTest src/tests/test_activation_loss.cpp)
add_executable(TensorGradTest src/tests/test_tensor_grad.cpp)
add_executable(MnistTest mnist/mnist.cpp mnist/mnist_loader.cpp mnist/mnist_loader.h)

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