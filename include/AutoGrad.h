/**
 *@file AutoGrad.h
 *@author Beapoe
 *@brief 自动微分类接口
 *@date 2026/4/4
 **/

#ifndef CTORCH_AUTOGRAD_H
#define CTORCH_AUTOGRAD_H

// 前向声明
class Tensor;
class Node;

// 包含必要的头文件
#include "AutoGrad/DataCore.h"
#include "AutoGrad/ComputeCore.h"

namespace AutoGrad {

    //线程本地的全局变量，用于控制是否记录计算图
    inline thread_local bool EnableGrad{true};
    
    template <typename T>
    void registerNode(const std::vector<Tensor>& inputs, std::weak_ptr<Tensor> result) {
        if (EnableGrad) {
            // 对于单输入操作
            if (inputs.size() == 1) {
                // 这里可以添加单输入操作的处理逻辑
            }
            // 对于双输入操作
            else if (inputs.size() == 2) {
                DataCore::registerNode<T>(inputs[0], inputs[1], result);
            }
        }
    }
    
    template <typename T>
    void registerNode(const Tensor& a, const Tensor& b, std::weak_ptr<Tensor> result) {
        if (EnableGrad) {
            DataCore::registerNode<T>(a, b, result);
        }
    }
    
    void backward(std::shared_ptr<Node> root, bool retainGraph);
};

#endif // CTORCH_AUTOGRAD_H
