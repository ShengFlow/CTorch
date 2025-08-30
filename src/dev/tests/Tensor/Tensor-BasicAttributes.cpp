//
// Created by Beapoe on 2025/8/30.
//
import Tensor_dev;
#include <cassert>
#include <vector>

int main() {
    Tensor a({1.0f,2.0f,4.0f,9.0f},{2,2});

    // 获取形状
    assert(a.shape() == std::vector<size_t>({2,2}));

    // 获取步长
    assert(a.strides() == std::vector<size_t>({2,1}));

    // 获取维度数
    assert(a.dim() == 2);

    // 获取元素个数
    assert(a.numel() == 4);

    // 获取dtype
    assert(a.dtype() == DType::kFloat);

    // 获取device
    assert(a.device() == DeviceType::kCPU);

    // 获取data
    assert(*(a.data()) == 1.0f);

    // 获取连续性
    assert(a.is_contiguous() == true);

    // 获取自动微分设置情况
    assert(a.isGradRequired() == false);

    // 获取梯度(需等待反向传播后)
    // auto* ptr = new AutoGrad;
    // AutoGradContext::Guard guard(ptr);
    // assert(a.grad() == Tensor());

    // 获取储存偏移量
    assert(a.storageOffset() == 0);

    // 设置形状
    a.setShape(std::vector<size_t>({1,1,1,1}));
    assert(a.shape() == std::vector<size_t>({1,1,1,1}));

    // 设置步长
    std::vector<size_t> strides = a.strides();
    a.setStrides(std::vector<size_t>({1}));
    assert(a.strides() == std::vector<size_t>({1}));
    a.setStrides(strides);

    // 设置自动微分标志
    a.requires_grad(true);
    assert(a.isGradRequired());

    // 设置dtype
    a.setDtype(DType::kInt);
    assert(a.dtype() == DType::kInt);
    a.setDtype(DType::kFloat);

    return 0;
}