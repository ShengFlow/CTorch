import Tensor_dev;
import nn;
import functional;
#include <iostream>

class nn:public Module<nn> {
public:
    std::string className() const override {return "nn(Module)";}

};

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

    std::cout<<"Tensor a:"<<a<<std::endl<<"Tensor b:"<<b<<std::endl;

    // Tensor c = a*b;
    //
    // std::cout<<"Tensor c:"<<c<<std::endl;
    //
    // Tensor d = c * 2.0f;
    //
    // std::cout<<"Tensor d:"<<d<<std::endl;
    //
    // Tensor e = matMul(a,b);
    //
    // std::cout<<"Tensor e:"<<e<<std::endl;

    // TensorINIT<float> param3{
    //     {1.0,3.0},
    //     {2,1}
    // };
    //
    // Tensor f(param3);
    //
    // std::cout<<"Tensor f:"<<f<<std::endl;
    //
    // Tensor g = matMul(a,f);
    //
    // std::cout<<"Tensor g:"<<g<<std::endl;


    return 0;
}