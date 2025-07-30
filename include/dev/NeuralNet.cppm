//
// Created by Beapoe on 25-7-27.
//
module;

#include "../../src/dev/Tensor.h"
#include <vector>
#include <unordered_map>
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
    std::unordered_map<std::string,Module*> _children;
    std::vector<neuron*> _neurons{nullptr};

  protected:
    virtual Tensor forward(Tensor &input) = 0;

  public:
    // 构造&析构
    virtual ~Module() = default;

    // 运算符重载
    Tensor operator()(Tensor &input);

    // 基本属性
    void train(bool recur);

    void eval(bool recur);

    void addChild(std::string name,Module* child);

    void addChildren(std::unordered_map<std::string,Module*> children);

    std::unordered_map<std::string,Module*> children() const;

    std::vector<Module*> childrenRecur(Module* root) const;

    template<typename...Args>
    void Module::apply(Module* root,auto func,Args...args) {
        auto recursive = [&func, &recursive, &args](Module* root)->void {
            std::unordered_map<std::string,Module*> children = root->_children;
            for (auto [_,child]:children) {
                if (child->_children.size()) recursive(child);
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

