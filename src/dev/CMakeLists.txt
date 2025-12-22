cmake_minimum_required(VERSION 3.10)
project(Ctorch)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
add_executable(Ctorch_test)
target_sources(Ctorch_test PUBLIC main.cpp Tensor.h AutoDiff.h Tensor.cpp AutoDiff.cpp kernels/CPU-BASIC/Add_BASIC_kernel.cpp kernels/CPU-BASIC/Sub_BASIC_kernel.cpp kernels/CPU-BASIC/Mul_BASIC_kernel.cpp kernels/CPU-BASIC/Div_BASIC_kernel.cpp kernels/CPU-BASIC/Neg_BASIC_kernel.cpp Ctools.h Ctools.cpp)
# mkdir build && cd build && cmake .. && make