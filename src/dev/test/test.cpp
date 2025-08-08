import Tensor_dev;
import nn;
import functional;
#include <iostream>

int main() {
    Tensor a = Tensor<float>({1.0,2.0,3.0,4.0},{2,2});
    Tensor b = Tensor<float>({5.0,6.0,7.0,8.0},{2,2});

    std::cout<<a<<std::endl<<b;

    return 0;
}