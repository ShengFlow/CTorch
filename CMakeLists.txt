cmake_minimum_required(VERSION 3.31)
project(CTorch)

set(CMAKE_CXX_STANDARD 20)
add_library(CTorch STATIC)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
# 使用LLVM的lld链接器
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")

add_compile_options(
        -stdlib=libc++
        -std=c++20
)

add_link_options(
        -stdlib=libc++
)

target_sources(CTorch
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES
        BASE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES
#        include/dev/Tensor.cppm
        include/dev/NeuralNet.cppm
        include/dev/Functional.cppm
        include/dev/Tensor_dev.cppm

        PRIVATE
#        src/dev/Tensor.cpp
        src/dev/NeuralNet.cpp
        src/dev/Functional.cpp
        src/dev/Tensor_dev.cpp
)

add_executable(test src/dev/test/test.cpp)
target_link_libraries(test PRIVATE CTorch)