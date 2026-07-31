#include "Tensor.h"
#include "CtorchError.h"

int main() {
    CtorchError::setPrintLevel(PrintLevel::FULL);
    
    // 创建一个 2x3 的张量
    Tensor a(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
    Tensor b(ShapeTag{}, {3, 2}, DType::kFloat, DeviceType::kCPU);
    
    // 填充一些值
    for (int i = 0; i < 6; i++) {
        a.data_write<float>()[i] = i + 1;
        b.data_write<float>()[i] = i + 1;
    }
    
    // 打印形状和 strides
    std::cout << "a shape: [" << a.shape()[0] << ", " << a.shape()[1] << "]" << std::endl;
    std::cout << "a strides: [" << a.strides()[0] << ", " << a.strides()[1] << "]" << std::endl;
    std::cout << "b shape: [" << b.shape()[0] << ", " << b.shape()[1] << "]" << std::endl;
    std::cout << "b strides: [" << b.strides()[0] << ", " << b.strides()[1] << "]" << std::endl;
    
    // 检查是否连续
    bool a_contiguous = (a.strides()[0] == a.shape()[1] && a.strides()[1] == 1);
    bool b_contiguous = (b.strides()[0] == b.shape()[1] && b.strides()[1] == 1);
    
    std::cout << "a is contiguous: " << a_contiguous << std::endl;
    std::cout << "b is contiguous: " << b_contiguous << std::endl;
    
    // 执行矩阵乘法
    Tensor c = a.matmul(b);
    
    return 0;
}