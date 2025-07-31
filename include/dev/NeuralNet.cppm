//
// Created by Beapoe on 25-7-27.
//
module;

#include "../../src/dev/Tensor.h"
#include <vector>
#include <unordered_map>
#include <fstream>
import functional;

export module nn;

// 神经元结构体
struct neuron {
    size_t stage;
    AutoDiff ctx;

    void save(std::ofstream& os) const;

    void load(std::ifstream& is);
};

export class Parameter : public Tensor{
private:
    Tensor _data;
    bool _initialized{false};
public:
    // 构造
    Parameter(Tensor data,bool requiresGrad);

    // 基本属性
    bool isInitialized() const;

    void setInitialized(bool status);

    Tensor data() const;

    void setData(Tensor data);
};

export class Buffer : public Tensor {
private:
    Tensor _data;
    bool _initialized{false};
public:
    //构造
    explicit Buffer(Tensor data);

    //基本属性
    bool isInitialized() const;

    void setInitialized(bool status);

    Tensor data() const;

    void setData(Tensor data);
};

// 神经网络基类
export class Module {
  private:
    bool _train{true};
    Module* _parent{nullptr};
    std::unordered_map<std::string,Module*> _children;
    std::unordered_map<std::string,Parameter*> _parameters;
    std::unordered_map<std::string,Buffer*> _buffers;
    std::vector<neuron*> _neurons{nullptr};

    Module* findModule(const std::string& path);

    void save_impl(std::ofstream& os) const;

    void load_impl(std::ifstream& is);

    static void save_tensor(std::ofstream& os, const Tensor& tensor);

    static void load_tensor(std::ifstream& is, Tensor& tensor);

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

    std::vector<Parameter> parameters(std::initializer_list<std::string> names) const;

    void registerBuffer(std::string name,Buffer* buffer);

    void registerBuffers(std::unordered_map<std::string,Buffer*> buffers);

    Buffer buffer(std::string name) const;

    std::vector<Buffer> buffers(std::initializer_list<std::string> names) const;

    template <typename T>
    void setType(Module* root) {
        auto root      = this;
        auto recursive = [&train, &recursive](const Module *root) -> void {
            std::unordered_map<std::string,Module*> children = root->_children;
            for (auto& [_,child]:children) {
                if (child->_children.size() > 0)
                    recursive(child);
                else {
                    for (auto& [_,param]:child->_parameters) param->data().setDtype(cpp2DType<T>());
                    for (auto& [_,buffer]:child->_buffers) buffer->data().setDtype(cpp2DType<T>());
                }
            }
        }(root);
    }

    virtual std::string className() const =0;

    // IO
    std::vector<std::unordered_map<std::string,Tensor*>> state(std::string prefix,bool keepVars) const;

    void loadState(std::vector<std::unordered_map<std::string,Tensor*>> state,bool strict);

    virtual std::string extra_expr() const;

    void operator<<(Module* content) const;

    void save(const std::string& filename) const;

    void load(const std::string& filename);

    virtual void setExtraState(std::vector<std::unordered_map<std::string,Tensor*>> state);

    virtual std::vector<std::unordered_map<std::string,Tensor*>> getExtraState() const;

    // 梯度
    void zero_grad() const;
};

