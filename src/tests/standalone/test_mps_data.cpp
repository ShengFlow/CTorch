#include "Tensor.h"
#include "DeviceAllocator.h"
#include <iostream>

int main() {
    std::cout << "Testing MPS data loading..." << std::endl;
    
    Tensor mps_tensor(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kMPS);
    
    std::cout << "MPS tensor created: " << mps_tensor.shape()[0] << "x" << mps_tensor.shape()[1] << std::endl;
    
    float* data = mps_tensor.data_write<float>();
    std::cout << "Data pointer: " << data << std::endl;
    
    if (data) {
        std::cout << "Writing data to MPS tensor..." << std::endl;
        for (size_t i = 0; i < mps_tensor.numel(); ++i) {
            data[i] = static_cast<float>(i + 1);
        }
        
        std::cout << "Reading data from MPS tensor: ";
        for (size_t i = 0; i < mps_tensor.numel(); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "ERROR: Data pointer is null!" << std::endl;
        return 1;
    }
    
    std::cout << "\nTesting MPS matrix multiplication..." << std::endl;
    Tensor W(ShapeTag{}, {3, 2}, DType::kFloat, DeviceType::kMPS);
    float* w_data = W.data_write<float>();
    for (size_t i = 0; i < W.numel(); ++i) {
        w_data[i] = static_cast<float>(i + 1);
    }
    
    Tensor result = mps_tensor.matmul(W);
    std::cout << "Result shape: " << result.shape()[0] << "x" << result.shape()[1] << std::endl;
    
    float* r_data = result.data_write<float>();
    std::cout << "Result data: ";
    for (size_t i = 0; i < result.numel(); ++i) {
        std::cout << r_data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nMPS test passed!" << std::endl;
    return 0;
}
