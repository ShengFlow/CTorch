//
// Created by Beapoe on 2025/8/30.
//
import Tensor_dev;
#include <cassert>

int main() {
    Tensor a({1,2,3,4},{4});

    // 1D张量方括号访问
    assert(a[2] == 3);

    // 多维张量索引访问
    Tensor b({1,2,3,4},{2,2});
    assert(b({1,1}) == 4);

    // 标量访问
    Tensor c(1);
    assert(c.item() == 1);
    return 0;
}