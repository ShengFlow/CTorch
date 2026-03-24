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