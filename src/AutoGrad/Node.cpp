/**
 *@file Node.cpp
 *@author Beapoe
 *@brief 节点实现
 *@date 2026/2/17
 **/

#include "../include/AutoGrad/Node.h"

Node::Node(const std::vector<std::weak_ptr<Node>> &upStreamNodes,
           const std::vector<Tensor> &inputs)
               :_upStreamNodes(upStreamNodes),_inputs(inputs)
{}

Node::Node(const std::vector<std::weak_ptr<Node>> &upStreamNodes, const std::vector<Tensor> &inputs,
           const std::weak_ptr<Tensor> &result)
               :_upStreamNodes(upStreamNodes),_inputs(inputs),_result(result)
{}