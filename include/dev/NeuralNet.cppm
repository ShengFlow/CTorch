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

export class Parameter : public Tensor{
private:
    Tensor _data;
    bool _initialized{false};
public:
    // 构造
    Parameter(Tensor data,bool requiresGrad);

    //基本属性
    bool isInitialized() const;

    Tensor data() const;
};

// 神经网络基类
export class Module {
  private:
    bool _train{true};
    Module* _parent{nullptr};
    std::unordered_map<std::string,Module*> _children;
    std::unordered_map<std::string,Parameter*> _parameters;
    std::unordered_map<std::string,Tensor*> _buffers;
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

    void registerParameter(std::string name,Parameter* parameter);

    void registerParameters(std::unordered_map<std::string,Parameter*> parameters);

    Parameter parameter(std::string name) const;

    std::vector<Parameter*> parameters(std::initializer_list<std::string> names) const;

    // 梯度相关
    void zero_grad() const;
};

