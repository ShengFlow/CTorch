//
// Created by Beapoe on 2025/8/30.
//
import Tensor_dev;
#include <cassert>
#include <utility>

int main() {
    // 分配空内存
    Storage a(0,DType::kInt,DeviceType::kCPU);
    assert(a.data<int>() == nullptr);

    // 分配指定空间内存
    Storage b(4,DType::kInt,DeviceType::kCPU);
    assert(a.data<int>()+4 != nullptr);

    // 初始化构造
    int data[] = {1,2,3,4};
    Storage c(data,sizeof(data),DType::kInt,DeviceType::kCPU);
    assert(*(c.data<int>()+3) == 4);

    // 拷贝语义构造
    Storage d(c);
    assert(*(d.data<int>()+3) == 4);

    Storage e = d;
    assert(*(e.data<int>()+3) == 4);

    // 移动语义构造
    Storage f(std::move(e));
    assert(*(f.data<int>()+3) == 4);

    Storage g = std::move(f);
    assert(*(g.data<int>()+3) == 4);

    return 0;
}