//
// Created by beapoe on 2025/8/30.
//
import Tensor_dev;
#include <cassert>
#include <initializer_list>

int main() {
    std::initializer_list<int> data = {1,2,3};
    Storage a(data.begin(),data.size(),DType::kInt,DeviceType::kCPU);
    assert(*(a.data<int>()) == 1);
}