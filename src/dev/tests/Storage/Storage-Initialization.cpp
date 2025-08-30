//
// Created by beapoe on 2025/8/30.
//
import Tensor_dev;
#include <cassert>

int main() {
    Storage a(0,DType::kInt,DeviceType::kCPU);
    assert(a.data<int>() == nullptr);
}