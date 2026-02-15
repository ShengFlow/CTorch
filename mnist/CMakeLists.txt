cmake_minimum_required(VERSION 3.10)
project(MNIST)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")

# 添加Ctorch静态库的路径
set(CTORCH_LIBRARY_DIR ${CMAKE_SOURCE_DIR})
set(CTORCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../src/dev)

# 添加可执行文件
add_executable(mnist
        mnist.cpp
        mnist_loader.cpp
)

# 添加包含目录
target_include_directories(mnist
    PRIVATE
    ${CTORCH_INCLUDE_DIR}
)

# 链接Ctorch静态库
target_link_libraries(mnist
    PRIVATE
    ${CTORCH_LIBRARY_DIR}/libCtorch.a
)
