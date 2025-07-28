//
// Created by Beapoe on 25-7-27.
//
module;

#include "../../src/dev/Tensor.h"
#include <vector>
import functional;

export module nn;

// 神经元结构体
struct neuron {
    size_t stage;
    AutoDiff ctx;
};

// 神经网络基类
export class Module {
  private:
    bool _train{true};
    Module* _parent{nullptr};
    std::vector<Module*> _children{nullptr};
    std::vector<neuron*> _neurons{nullptr};

  protected:
    virtual Tensor forward(Tensor &input) = 0;

  public:
    // 构造&析构
    virtual ~Module() = default;

    // 运算符重载
    Tensor operator()(Tensor &input);

    // 基本属性
    void setTrain(bool train,bool recur);
    void addChild(Module* child);
    void addChildren(std::vector<Module*> children);
    template<typename...Args>
    void Module::apply(Module* root,auto func,Args...args) {
        auto recursive = [&func, &recursive, &args](Module* root)->void {
            std::vector<Module *> children = root->_children;
            for (size_t i = 0; i < children.size(); ++i) {
                if (children[i]->_children.size() > 0) recursive(children[i]);
                else func(args);
            }
        }(root);
    }

    // 梯度相关
    void zero_grad() const;
};

// 线性层
export class Linear {
  private:
    size_t _input_size{0};
    size_t _output_size{0};
    bool _bias{true};

  public:
    Linear(size_t input_s, size_t output_s, bool bias);
};

