//
// Created by Beapoe on 25-7-28.
//
module;

#include "../../src/dev/Tensor.h"
#include <type_traits>
#include <functional>
export module functional;

// 函数指针宏
#define COS Tensor(Tensor)
#define SIN Tensor(Tensor)
#define RELU Tensor(Tensor)
#define SIGMOID Tensor(Tensor)
#define TANH Tensor(Tensor)
#define SOFTMAX Tensor(Tensor,int)

// 包装函数
template <typename TargetSignature,typename Func,typename... T0> //这里这个TargetSignature填上面的宏
std::function<TargetSignature> make_function(Func f,T0&&... args) {
    return [f, args]() -> typename std::function<TargetSignature>::result_type {
        // 使用完美转发调用原始函数
        return f(std::forward<T0>(args)...);
    };
}

// 逐元素余弦
Tensor cos(Tensor x);

// 逐元素正弦
Tensor sin(Tensor x);

// ReLU激活函数
Tensor relu(Tensor x);

// Sigmoid激活函数
Tensor sigmoid(Tensor x);

// Tanh激活函数
Tensor tanh(Tensor x);

// Softmax激活函数
Tensor softmax(Tensor x,int dim = -1);