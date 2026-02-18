/**
 *@file GradFn.h
 *@author Beapoe
 *@brief 反向传播函数基类
 *@date 2026/2/17
 **/

#ifndef CTORCH_GRADFN_H
#define CTORCH_GRADFN_H

#include <vector>
#include <memory>
#include "../Tensor.h"

/**
 *@class GradFn
 *@brief 反向传播函数基类
 **/
class GradFn {
  public:
    /**
     *@brief 默认析构函数
     **/
    virtual ~GradFn() = default;

    /**
     * @brief 计算梯度函数
     * @param gradInput 从下游函数传入的梯度
     * @return 传递给上游函数的梯度
     */
    virtual std::vector<Tensor> apply(const Tensor &gradInput) = 0;

    /**
     * @brief 获取上游函数
     * @return 上游函数
     */
    virtual std::vector<std::weak_ptr<GradFn>> nextFuncs() const = 0;
};

#endif // CTORCH_GRADFN_H
