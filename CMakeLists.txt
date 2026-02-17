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
        src/kernels/CPU-BASIC/Softmax_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/MatMul_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/Dot_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/MSE_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/CrossEntropy_BASIC_kernel.cpp
        src/kernels/CPU-BASIC/MAE_BASIC_kernel.cpp
)

set(CT-CPU-SIMD-OPs
        src/kernels/CPU-SIMD/Add_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Div_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Mul_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Sub_SIMD_kernel.cpp
        src/kernels/CPU-SIMD/Tanh_SIMD_kernel.cpp
)

add_library(CTorch STATIC ${CT-Core} ${CT-Core-Headers} ${CT-CPU-BASIC-OPs} ${CT-CPU-SIMD-OPs})

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

option(TEST_ENABLED "Enable tests?" ON)
if (${TEST_ENABLED})
    set(TEST_GROUP_Storage src/tests/Storage/*.cpp)
    set(TEST_GROUP_Tensor src/tests/Tensor/*.cpp)
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
        add_executable(${TEST_NAME} ${TEST})
        target_link_libraries(${TEST_NAME} PRIVATE CTorch)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach ()

    enable_testing()

    add_custom_target(showTests
            COMMAND ${CMAKE_COMMAND} -E echo "Current configured tests"
            COMMAND ${CMAKE_COMMAND} -E echo "${TESTS_TO_BUILD}"
            COMMENT "To show the chosen tests")
endif ()