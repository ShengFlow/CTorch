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
        src/Storage.cpp
        src/Ctools.cpp
        src/ThreadPool.cpp
        src/CtorchScheduler.cpp
        src/AutoGrad/ComputeCore.cpp
        src/Arena.cpp
        src/Assertion.cpp
        src/AutoGrad.cpp
)

set(CT-Core-Headers
        include/Tensor.h
        include/Storage.h
        include/Ctools.h
        include/CtorchScheduler.h
        include/CtorchError.h
        include/AutoGrad/DataCore.h
        include/ThreadPool.h
        include/AutoGrad/ComputeCore.h
        include/Arena.h
        include/AutoGrad.h
        include/ctQALS/ctQALS.h
)

set(CT-Nodes-Headers
        include/AutoGrad/Node.h
        include/AutoGrad/Nodes/AddNode.h
        include/AutoGrad/Nodes/SubNode.h
        include/AutoGrad/Nodes/MulNode.h
        include/AutoGrad/Nodes/DivNode.h
        include/AutoGrad/Nodes/NegNode.h
        include/AutoGrad/Nodes/SinNode.h
        include/AutoGrad/Nodes/CosNode.h
        include/AutoGrad/Nodes/TanhNode.h
        include/AutoGrad/Nodes/SigmoidNode.h
        include/AutoGrad/Nodes/ReLUNode.h
        include/AutoGrad/Nodes/MatMulNode.h
        include/AutoGrad/Nodes/CrossEntropyNode.h
        include/AutoGrad/Nodes/SoftmaxNode.h
        include/AutoGrad/Nodes/GradAccumulator.h
)

set(CT-Nodes
        src/AutoGrad/Node.cpp
        src/AutoGrad/Nodes/AddNode.cpp
        src/AutoGrad/Nodes/SubNode.cpp
        src/AutoGrad/Nodes/MulNode.cpp
        src/AutoGrad/Nodes/DivNode.cpp
        src/AutoGrad/Nodes/NegNode.cpp
        src/AutoGrad/Nodes/SinNode.cpp
        src/AutoGrad/Nodes/CosNode.cpp
        src/AutoGrad/Nodes/TanhNode.cpp
        src/AutoGrad/Nodes/SigmoidNode.cpp
        src/AutoGrad/Nodes/ReLUNode.cpp
        src/AutoGrad/Nodes/MatMulNode.cpp
        src/AutoGrad/Nodes/CrossEntropyNode.cpp
        src/AutoGrad/Nodes/SoftmaxNode.cpp
        src/AutoGrad/Nodes/GradAccumulator.cpp
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
set(CT-AMX-OPs
        src/kernels/AMX/MatMul_AMX_kernel.cpp
)

# 暂时禁用CPU-SIMD相关文件，因为需要AVX指令集支持
# set(CT-CPU-SIMD-OPs
#         src/kernels/CPU-SIMD/Add_SIMD_kernel.cpp
#         src/kernels/CPU-SIMD/Div_SIMD_kernel.cpp
#         src/kernels/CPU-SIMD/Mul_SIMD_kernel.cpp
#         src/kernels/CPU-SIMD/Sub_SIMD_kernel.cpp
#         src/kernels/CPU-SIMD/Tanh_SIMD_kernel.cpp
# )

set(CT-CPU-SIMD-OPs
)

set(ctQALS
        include/ctQALS/ctQALS.h
        include/ctQALS/Random.h
        src/ctQALS/xoshiro.cpp
        src/ctQALS/ziggurat.cpp
)

add_library(CTorch STATIC
        ${CT-Core}
        ${CT-Core-Headers}
        ${CT-CPU-BASIC-OPs}
        ${CT-CPU-SIMD-OPs}
        ${CT-AMX-OPs}
        ${CT-Nodes}
        ${CT-Nodes-Headers}
        ${ctQALS}
)

# 包含头文件目录
target_include_directories(CTorch
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_compile_options(-march=native)

# 链接 Accelerate 框架（用于 AMX BLAS 加速）
target_link_libraries(CTorch PUBLIC "-framework Accelerate")

add_executable(MnistTest mnist/mnist.cpp mnist/mnist_loader.cpp mnist/mnist_loader.h)

add_compile_options(-Wno-ignored-attributes)

# Discover all GDB pretty-printer Python modules in scripts/
# (evaluated at configure time — re-run cmake if you add new .py files)
file(GLOB _CT_GDB_SCRIPTS "${CMAKE_SOURCE_DIR}/scripts/*.py")

function(add_multiarch_executable NAME)
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

        # --- GDB pretty-printer auto-load: generate <target>-gdb.py ---
        list(LENGTH _CT_GDB_SCRIPTS _CT_NUM_SCRIPTS)
        if(_CT_NUM_SCRIPTS GREATER 0)
            set(_CT_IMPORTS "")
            foreach(_CT_MOD ${_CT_GDB_SCRIPTS})
                get_filename_component(_CT_MNAME "${_CT_MOD}" NAME_WE)
                string(APPEND _CT_IMPORTS "try:\n    import ${_CT_MNAME}\nexcept Exception:\n    pass\n")
            endforeach()

            # Write a loader script that imports every module in scripts/
            set(_CT_LOADER "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-gdb.py")
            file(WRITE "${_CT_LOADER}"
"import sys
sys.path.insert(0, '${CMAKE_SOURCE_DIR}/scripts')

${_CT_IMPORTS}")

            # Copy the loader next to the executable after each build
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_CT_LOADER}"
                    "$<TARGET_FILE_DIR:${TARGET_NAME}>/${TARGET_NAME}-gdb.py"
                COMMENT "Installing GDB pretty-printer for ${TARGET_NAME}"
            )
        endif()

    endforeach()
endfunction()

# tl-ArrayTest
add_executable(tl-ArrayTest src/tests/tl/ArrayTest.cpp)
target_link_libraries(tl-ArrayTest PRIVATE CTorch gtest gtest_main)  # Note: build with CT_DEBUG defined
gtest_discover_tests(tl-ArrayTest)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64")
    # tl-VecBaseTest
    add_multiarch_executable(tl-VecBaseTest
            FILES src/tests/tl/vec/VecBaseTest.cpp
            ARCH x86-64
    )

    # tl-VecScalarTest
    add_multiarch_executable(tl-VecScalarTest
            FILES src/tests/tl/vec/VecScalarTest.cpp
            ARCH x86-64
    )

    # tl-VecLoadStoreTest
    add_multiarch_executable(tl-VecLoadStoreTest
            FILES src/tests/tl/vec/VecLoadStoreTest.cpp
            ARCH x86-64 corei7-avx core-avx2 native
    )
    # tl-VecArithTest
    add_multiarch_executable(tl-VecArithTest
            FILES src/tests/tl/vec/VecArithTest.cpp
            ARCH x86-64 corei7-avx core-avx2 native
    )
    # tl-VecConversionTest
    add_multiarch_executable(tl-VecConversionTest
            FILES src/tests/tl/vec/VecConversionTest.cpp
            ARCH x86-64 corei7-avx core-avx2 native
    )
    # tl-VecShuffleTest
    add_multiarch_executable(tl-VecShuffleTest
            FILES src/tests/tl/vec/VecShuffleTest.cpp
            ARCH x86-64 corei7-avx core-avx2 native
    )
endif()

# target_link_libraries(PerformanceTest PRIVATE CTorch)
# target_link_libraries(StdTest PRIVATE CTorch)
# target_link_libraries(ActivationLossTest PRIVATE CTorch)
# target_link_libraries(TensorGradTest PRIVATE CTorch)
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
    set(TEST_GROUP_Storage src/tests/units/Storage/*.cpp)
    set(TEST_GROUP_Tensor src/tests/units/Tensor/*.cpp)
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
        # 每个测试文件自带 main()，不再向其注入 test_rng.cpp（曾导致 _main 重复）
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
