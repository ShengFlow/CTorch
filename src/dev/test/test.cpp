import Tensor_dev;
import nn;
import functional;
#include <iostream>

int main() {
    TensorINIT<float> param1{
        {1.0,2.0,3.0,4.0},
        {2,2}
    };
    TensorINIT<float> param2{
        {5.0,6.0,7.0,8.0},
        {2,2}
    };
    Tensor a(param1);
    Tensor b(param2);

    std::cout<<a<<std::endl<<b;

    return 0;
}