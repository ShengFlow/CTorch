#include "AutoGrad.h"
#include "Tensor.h"
#include "CtorchError.h"
#include <iostream>

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    
    // 简单测试：2 个输入，3 个输出的线性层 + CrossEntropy
    // batch_size = 2, input_size = 3, output_size = 2
    Tensor W(ShapeTag{}, {3, 2}, DType::kFloat, DeviceType::kMPS);
    Tensor b(ShapeTag{}, {2}, DType::kFloat, DeviceType::kMPS);
    
    float* w_data = W.data_write<float>();
    w_data[0] = 0.1f; w_data[1] = 0.2f;
    w_data[2] = 0.3f; w_data[3] = 0.4f;
    w_data[4] = 0.5f; w_data[5] = 0.6f;
    
    float* b_data = b.data_write<float>();
    b_data[0] = 0.1f; b_data[1] = 0.2f;
    
    W.requires_grad(true);
    b.requires_grad(true);
    
    Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kMPS);
    float* x_data = x.data_write<float>();
    x_data[0] = 1.0f; x_data[1] = 2.0f; x_data[2] = 3.0f;
    x_data[3] = 4.0f; x_data[4] = 5.0f; x_data[5] = 6.0f;
    
    Tensor target(ShapeTag{}, {2, 2}, DType::kFloat, DeviceType::kMPS);
    float* t_data = target.data_write<float>();
    t_data[0] = 1.0f; t_data[1] = 0.0f;  // 类 0
    t_data[2] = 0.0f; t_data[3] = 1.0f;  // 类 1
    
    AutoGrad::EnableGrad = true;
    
    Tensor logits = x.matmul(W) + b;
    Tensor loss = logits.cross_entropy(target);
    
    std::cout << "Loss: " << loss.item<float>() << std::endl;
    
    AutoGrad::backward(loss.getRelatedNode(), false);
    
    Tensor grad_W = W.grad();
    Tensor grad_b = b.grad();
    
    std::cout << "grad_W shape: " << grad_W.shape()[0] << "x" << grad_W.shape()[1] << std::endl;
    const float* gw = grad_W.data_read<float>();
    std::cout << "grad_W: ";
    for (int i = 0; i < 6; ++i) std::cout << gw[i] << " ";
    std::cout << std::endl;
    
    std::cout << "grad_b shape: " << grad_b.shape()[0] << std::endl;
    const float* gb = grad_b.data_read<float>();
    std::cout << "grad_b: ";
    for (int i = 0; i < 2; ++i) std::cout << gb[i] << " ";
    std::cout << std::endl;
    
    return 0;
}
