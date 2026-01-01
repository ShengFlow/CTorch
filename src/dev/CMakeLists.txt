cmake_minimum_required(VERSION 3.10)
project(Ctorch)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)/
# 确保编译器使用UTF-8编码处理源文件
if (MSVC)
    # Visual Studio编译器
    add_compile_options("/source-charset:utf-8")
else()
    # GCC/Clang编译器
    add_compile_options(-finput-charset=UTF-8)
endif()


add_executable(Ctorch_test
        kernels/CPU-BASIC/MatMul_BASIC_kernel.cpp)
target_sources(Ctorch_test PUBLIC main.cpp Tensor.h AutoDiff.h Tensor.cpp AutoDiff.cpp kernels/CPU-BASIC/Add_BASIC_kernel.cpp kernels/CPU-BASIC/Sub_BASIC_kernel.cpp kernels/CPU-BASIC/Mul_BASIC_kernel.cpp kernels/CPU-BASIC/Div_BASIC_kernel.cpp kernels/CPU-BASIC/Neg_BASIC_kernel.cpp Ctools.h Ctools.cpp)
# mkdir build && cd build && cmake .. && make