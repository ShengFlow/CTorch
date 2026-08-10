#include "Tensor.h"
#include "CtorchError.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

static void print_matrix(const Tensor& t, const std::string& name) {
    std::cout << name << " shape [" << t.shape()[0] << ", " << t.shape()[1] << "]" << std::endl;
    const float* data = t.data_read<float>();
    size_t s0 = t.strides()[0];
    size_t s1 = t.strides()[1];
    for (size_t i = 0; i < t.shape()[0]; ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < t.shape()[1]; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2)
                      << data[i * s0 + j * s1];
            if (j + 1 < t.shape()[1]) std::cout << ", ";
        }
        std::cout << " ]" << std::endl;
    }
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::FULL);

    std::cout << "=== CPU matmul stride test ===" << std::endl;

    // A: 2x3, row-major values 1..6
    Tensor A(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
    float* a_ptr = A.data_write<float>();
    for (size_t i = 0; i < 6; ++i) a_ptr[i] = static_cast<float>(i + 1);

    // B_cont: 3x4 contiguous, row-major values 1..12
    Tensor B_cont(ShapeTag{}, {3, 4}, DType::kFloat, DeviceType::kCPU);
    float* b_ptr = B_cont.data_write<float>();
    for (size_t i = 0; i < 12; ++i) b_ptr[i] = static_cast<float>(i + 1);

    std::cout << "\n-- Test 1: A(2,3) * B(3,4), both contiguous --" << std::endl;
    print_matrix(A, "A");
    print_matrix(B_cont, "B");

    Tensor C_cont = A.matmul(B_cont);
    print_matrix(C_cont, "C_cont = A * B");

    // Expected result for A * B_cont
    // A = [[1,2,3],[4,5,6]], B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    float expected[8] = {
        38.0f, 44.0f, 50.0f, 56.0f,
        83.0f, 98.0f, 113.0f, 128.0f
    };

    bool expected_ok = true;
    const float* c_cont_ptr = C_cont.data_read<float>();
    for (size_t i = 0; i < 8; ++i) {
        if (std::fabs(c_cont_ptr[i] - expected[i]) > 1e-5f) {
            std::cout << "  mismatch at " << i << ": expected " << expected[i]
                      << ", got " << c_cont_ptr[i] << std::endl;
            expected_ok = false;
        }
    }
    std::cout << "  contiguous result correct: " << (expected_ok ? "true" : "false") << std::endl;

    // B_nc_source: 4x3 contiguous. Fill it so that transpose(0,1) is logically equal to B_cont.
    Tensor B_nc_source(ShapeTag{}, {4, 3}, DType::kFloat, DeviceType::kCPU);
    float* bs_ptr = B_nc_source.data_write<float>();
    for (size_t i = 0; i < 3; ++i) {       // row of B_cont / col of B_nc_source
        for (size_t j = 0; j < 4; ++j) {   // col of B_cont / row of B_nc_source
            bs_ptr[j * 3 + i] = static_cast<float>(i * 4 + j + 1);
        }
    }

    Tensor B_t = B_nc_source.transpose(0, 1);  // shape (3,4), non-contiguous strides [1, 3]

    std::cout << "\n-- Test 2: A(2,3) * B.transpose(0,1), B non-contiguous --" << std::endl;
    std::cout << "B_nc_source shape [" << B_nc_source.shape()[0] << ", " << B_nc_source.shape()[1] << "]"
              << ", strides [" << B_nc_source.strides()[0] << ", " << B_nc_source.strides()[1] << "]"
              << std::endl;
    std::cout << "B_t shape [" << B_t.shape()[0] << ", " << B_t.shape()[1] << "]"
              << ", strides [" << B_t.strides()[0] << ", " << B_t.strides()[1] << "]"
              << std::endl;

    bool strides_ok = (B_t.strides()[0] == 1 && B_t.strides()[1] == 3);
    std::cout << "  transpose strides correct (1, 3): " << (strides_ok ? "true" : "false") << std::endl;

    Tensor C_nc = A.matmul(B_t);
    print_matrix(C_nc, "C_nc = A * B_t");

    bool match = true;
    const float* c_nc_ptr = C_nc.data_read<float>();
    for (size_t i = 0; i < 8; ++i) {
        if (std::fabs(c_cont_ptr[i] - c_nc_ptr[i]) > 1e-5f) {
            std::cout << "  mismatch at " << i << ": contiguous " << c_cont_ptr[i]
                      << ", non-contiguous " << c_nc_ptr[i] << std::endl;
            match = false;
        }
    }
    std::cout << "  non-contiguous matches contiguous: " << (match ? "true" : "false") << std::endl;

    std::cout << "\n=== MPS matmul stride test ===" << std::endl;

    // 4. MPS 上连续矩阵 A(2,3) 和 B(3,4) 的 matmul 结果
    Tensor A_mps = A.to(DeviceType::kMPS);
    Tensor B_cont_mps = B_cont.to(DeviceType::kMPS);

    std::cout << "\n-- Test 4: MPS A(2,3) * B(3,4), both contiguous --" << std::endl;
    Tensor C_cont_mps = A_mps.matmul(B_cont_mps);
    print_matrix(C_cont_mps, "C_cont_mps = A_mps * B_cont_mps");

    bool mps_cont_match_cpu = true;
    const float* c_cont_mps_ptr = C_cont_mps.data_read<float>();
    for (size_t i = 0; i < 8; ++i) {
        if (std::fabs(c_cont_mps_ptr[i] - expected[i]) > 1e-4f) {
            std::cout << "  MPS contiguous mismatch vs CPU expected at " << i
                      << ": expected " << expected[i]
                      << ", got " << c_cont_mps_ptr[i] << std::endl;
            mps_cont_match_cpu = false;
        }
    }
    std::cout << "  MPS contiguous matches CPU expected: " << (mps_cont_match_cpu ? "true" : "false") << std::endl;

    // 5. MPS 上 B.transpose(0,1) 后（非连续）的 matmul 结果
    Tensor B_nc_source_mps = B_nc_source.to(DeviceType::kMPS);
    Tensor B_t_mps = B_nc_source_mps.transpose(0, 1);

    std::cout << "\n-- Test 5: MPS A(2,3) * B.transpose(0,1), B non-contiguous --" << std::endl;
    std::cout << "B_t_mps shape [" << B_t_mps.shape()[0] << ", " << B_t_mps.shape()[1] << "]"
              << ", strides [" << B_t_mps.strides()[0] << ", " << B_t_mps.strides()[1] << "]"
              << std::endl;

    bool mps_t_strides_ok = (B_t_mps.strides()[0] == 1 && B_t_mps.strides()[1] == 3);
    std::cout << "  transpose strides correct (1, 3): " << (mps_t_strides_ok ? "true" : "false") << std::endl;

    Tensor C_nc_mps = A_mps.matmul(B_t_mps);
    print_matrix(C_nc_mps, "C_nc_mps = A_mps * B_t_mps");

    // 6. 比较 MPS 连续与非连续结果是否一致，以及是否与 CPU 结果一致
    bool mps_nc_match_cont = true;
    const float* c_nc_mps_ptr = C_nc_mps.data_read<float>();
    for (size_t i = 0; i < 8; ++i) {
        if (std::fabs(c_cont_mps_ptr[i] - c_nc_mps_ptr[i]) > 1e-4f) {
            std::cout << "  MPS non-contiguous mismatch vs contiguous at " << i
                      << ": contiguous " << c_cont_mps_ptr[i]
                      << ", non-contiguous " << c_nc_mps_ptr[i] << std::endl;
            mps_nc_match_cont = false;
        }
    }
    std::cout << "  MPS non-contiguous matches contiguous: " << (mps_nc_match_cont ? "true" : "false") << std::endl;

    bool mps_nc_match_cpu = true;
    for (size_t i = 0; i < 8; ++i) {
        if (std::fabs(c_nc_mps_ptr[i] - expected[i]) > 1e-4f) {
            std::cout << "  MPS non-contiguous mismatch vs CPU expected at " << i
                      << ": expected " << expected[i]
                      << ", got " << c_nc_mps_ptr[i] << std::endl;
            mps_nc_match_cpu = false;
        }
    }
    std::cout << "  MPS non-contiguous matches CPU expected: " << (mps_nc_match_cpu ? "true" : "false") << std::endl;

    std::cout << "\n=== Summary ===" << std::endl;
    if (expected_ok && strides_ok && match &&
        mps_cont_match_cpu && mps_t_strides_ok && mps_nc_match_cont && mps_nc_match_cpu) {
        std::cout << "All checks passed." << std::endl;
        return 0;
    } else {
        std::cout << "Some checks failed." << std::endl;
        return 1;
    }
}
