/**
*@file AutoGrad.cpp
 *@author Beapoe
 *@brief 自动微分类接口
 *@date 2026/4/4
 **/

#include <utility>

#include "../include/AutoGrad.h"

void AutoGrad::backward(std::shared_ptr<Node> root, bool retainGraph) {
    ComputeCore::getInstance().backward(std::move(root),retainGraph);
}
