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
    void registerNode(std::vector<Tensor> inputs, std::weak_ptr<Tensor>& result) {
        if (EnableGrad) {
            DataCore::registerNode<T>(inputs,result);
        }
    }
    
    void backward(std::shared_ptr<Node> root, bool retainGraph);
};

#endif // CTORCH_AUTOGRAD_H
