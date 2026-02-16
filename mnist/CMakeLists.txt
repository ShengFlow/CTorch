cmake_minimum_required(VERSION 3.10)
project(MNIST)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")

# 直接引入 CTorch（使用 src/dev 的 CMakeLists.txt 作为单一权威来源）
# 这样可以避免硬编码 lib 路径、也避免 libCtorch.a / libCTorch.a 大小写导致的跨平台链接失败。
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../src/dev ${CMAKE_BINARY_DIR}/ctdev)

# 添加可执行文件
add_executable(mnist
        mnist.cpp
        mnist_loader.cpp
)

# 添加简化版MNIST可执行文件
add_executable(mnist_simple
        mnist_simple.cpp
)

# 添加重写版MNIST可执行文件
add_executable(mnist_simple_rewritten
        mnist_simple_rewritten.cpp
)

# 添加包含目录
target_include_directories(mnist
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_include_directories(mnist_simple
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_include_directories(mnist_simple_rewritten
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

# 链接 CTorch 静态库 target（由 add_subdirectory 引入）
target_link_libraries(mnist
    PRIVATE
    CTorch
)

target_link_libraries(mnist_simple
    PRIVATE
    CTorch
)

target_link_libraries(mnist_simple_rewritten
    PRIVATE
    CTorch
)
