//
// Created by Beapoe on 25-7-28.
//
module;

import Tensor_dev;
#include <functional>
export module functional;

// 函数指针别名
export using COS = Tensor(Tensor);
export using SIN = Tensor(Tensor);
export using RELU = Tensor(Tensor);
export using SIGMOID = Tensor(Tensor);
export using TANH = Tensor(Tensor);
export using SOFTMAX = Tensor(Tensor,int);

// 包装函数
template <typename TargetSignature,typename Func,typename... T0> //这里这个TargetSignature填上面的宏
std::function<TargetSignature> make_function(Func f,T0&&... args) {
    return [f,...args = std::forward<T0>(args)]() -> typename std::function<TargetSignature>::result_type {
        // 使用完美转发调用原始函数
        return f(args...);
    };
}

// 逐元素余弦
Tensor cos(Tensor x,AutoGrad& ctx);

// 逐元素正弦
Tensor sin(Tensor x,AutoGrad& ctx);

// ReLU激活函数
Tensor relu(Tensor x,AutoGrad& ctx);

// Sigmoid激活函数
Tensor sigmoid(Tensor x,AutoGrad& ctx);

// Tanh激活函数
Tensor tanh(Tensor x,AutoGrad& ctx);

// Softmax激活函数
Tensor softmax(Tensor x,AutoGrad& ctx,int dim = -1);