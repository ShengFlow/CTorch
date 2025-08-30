//
// Created by Beapoe on 2025/8/30.
//
import Tensor_dev;
#include <cassert>
#include <initializer_list>

int main() {
    std::initializer_list<int> data = {1,2,3};
    Storage a(data.begin(),data.size(),DType::kInt,DeviceType::kCPU);

    // 获取原始数据的类型化指针
    assert(*(a.data<int>()) == 1);

    // 获取元素个数
    assert(a.size() == 3);

    // 获取dtype
    assert(a.dtype() == DType::kInt);

    // 获取device
    assert(a.device() == DeviceType::kCPU);

    // 克隆
    Storage b = a.clone();
    assert(*(b.data<int>()) == *(a.data<int>()));

    return 0;
}