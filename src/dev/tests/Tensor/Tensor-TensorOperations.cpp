//
// Created by beapoe on 2025/8/30.
//
import Tensor_dev;
#include <cassert>
#include <vector>

int main() {
    Tensor a({1,2,3,4},{4});

    // 克隆
    Tensor b = a.clone();
    assert(a == b);

    // 获取视图
    Tensor c = b.view({2,2});
    assert(c.shape() == std::vector<size_t>({2,2}));

    // 降维
    Tensor d = c.sum();
    int result{0};
    for (int* ptr = d.data<int>();ptr != nullptr;ptr++)
        result += *ptr;
    assert(d.item() == result);
    return 0;
}