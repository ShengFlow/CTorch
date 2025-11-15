//
// Created by Beapoe on 25-7-27.
//
module;

import Tensor_dev;
import functional;
#include <vector>
#include <unordered_map>
#include <functional>
#include <optional>
#include <cstring>
#include <unordered_set>
#include <iostream>
#include <ranges>
#include <memory>
#include <stdexcept>
#include <type_traits>

export module nn;

export class ModuleBase {};
template <typename Derived> class Module;
export using HOOK_RET = std::optional<Tensor>;
export using BACKWARD_HOOK_RET = std::optional<std::vector<std::optional<Tensor>>>;
export using ForwardPreHook         = HOOK_RET (*)(const ModuleBase *self,
                                        const std::vector<std::optional<Tensor>> input);
export using ForwardHook            = HOOK_RET (*)(const ModuleBase *self,
                                 const std::vector<std::optional<Tensor>> grad_input,
                                 std::vector<std::optional<Tensor>> grad_output);
export using FullModuleBackwardHook = BACKWARD_HOOK_RET (*)(
    const ModuleBase *self, const std::vector<std::optional<Tensor>> grad_input,
    std::vector<std::optional<Tensor>> grad_output);

// ======================= Parameter =======================
export class Parameter : public Tensor {
  private:
    Tensor _data;
    bool _initialized{false};

  public:
    // 构造
    explicit Parameter(const Tensor &data, bool requiresGrad);

    // 基本属性
    [[nodiscard]] bool isInitialized() const;

    void setInitialized(bool status);

    [[nodiscard]] Tensor data() const;

    void setData(Tensor data);
};

// ======================= Buffer =======================
export class Buffer : public Tensor {
  private:
    Tensor _data;
    bool _initialized{false};

  public:
    // 构造
    explicit Buffer(const Tensor &data);

    // 基本属性
    [[nodiscard]] bool isInitialized() const;

    void setInitialized(bool status);

    [[nodiscard]] Tensor data() const;

    void setData(Tensor data);
};

// ======================= Module =======================
template <typename Derived> class Module : public ModuleBase {
  private:
    bool _train{true};
    Module *_parent{nullptr};
    std::unordered_map<std::string, std::unique_ptr<ModuleBase>> _children;
    std::unordered_map<std::string, std::unique_ptr<Parameter>> _parameters;
    std::unordered_map<std::string, std::unique_ptr<Buffer>> _buffers;
    AutoGrad _ctx;
    std::vector<std::optional<Tensor>> _argsGrad;
    std::vector<std::optional<Tensor>> _outputsGrad;
    std::vector<ForwardPreHook> _forwardPreHooks;
    std::vector<ForwardHook> _forwardHooks;
    std::vector<FullModuleBackwardHook> _fullModuleBackwardHooks;

    // ======================= 私有IO =======================
    void save_impl(std::ofstream &os) const {
        // 保存基本状态
        os.write(reinterpret_cast<const char *>(&_train), sizeof(_train));

        // 保存子模块数量及子模块
        size_t children_count = _children.size();
        os.write(reinterpret_cast<const char *>(&children_count), sizeof(children_count));

        for (const auto &pair : _children) {
            // 保存名称长度和名称
            size_t name_len = pair.first.length();
            os.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
            os.write(pair.first.c_str(), name_len);

            // 递归保存子模块
            static_cast<Module *>(pair.second.get())->save_impl(os);
        }

        // 保存参数数量及参数
        size_t params_count = _parameters.size();
        os.write(reinterpret_cast<const char *>(&params_count), sizeof(params_count));

        for (const auto &pair : _parameters) {
            // 保存名称
            size_t name_len = pair.first.length();
            os.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
            os.write(pair.first.c_str(), name_len);

            // 保存参数是否初始化
            bool initializeStatus = pair.second->isInitialized();
            os.write(reinterpret_cast<const char *>(&initializeStatus), sizeof(bool));

            // 如果已初始化，保存数据
            if (pair.second->isInitialized()) {
                save_tensor(os, pair.second->data());
            }
        }

        // 保存缓冲区数量及缓冲区
        size_t buffers_count = _buffers.size();
        os.write(reinterpret_cast<const char *>(&buffers_count), sizeof(buffers_count));

        for (const auto &pair : _buffers) {
            // 保存名称
            size_t name_len = pair.first.length();
            os.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
            os.write(pair.first.c_str(), name_len);

            // 保存缓冲区是否初始化
            bool initializeStatus = pair.second->isInitialized();
            os.write(reinterpret_cast<const char *>(&initializeStatus), sizeof(bool));

            // 如果已初始化，保存数据
            if (pair.second->isInitialized()) {
                save_tensor(os, pair.second->data());
            }
        }
    }

    void load_impl(std::ifstream &is) {
        // 加载基本状态
        is.read(reinterpret_cast<char *>(&_train), sizeof(_train));

        // 加载子模块
        size_t children_count;
        is.read(reinterpret_cast<char *>(&children_count), sizeof(children_count));

        for (size_t i = 0; i < children_count; ++i) {
            // 读取名称
            size_t name_len;
            is.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));

            std::string name(name_len, '\0');
            is.read(&name[0], name_len);

            // 假设子模块已存在且类型匹配
            auto it = _children.find(name);
            if (it == _children.end()) {
                throw std::runtime_error("加载模型时找不到子模块: " + name);
            }

            // 递归加载子模块
            static_cast<Module *>(it->second.get())->load_impl(is);
        }

        // 加载参数
        size_t params_count;
        is.read(reinterpret_cast<char *>(&params_count), sizeof(params_count));

        for (size_t i = 0; i < params_count; ++i) {
            // 读取名称
            size_t name_len;
            is.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));

            std::string name(name_len, '\0');
            is.read(&name[0], name_len);

            // 查找参数
            auto it = _parameters.find(name);
            if (it == _parameters.end()) {
                throw std::runtime_error("加载模型时找不到参数: " + name);
            }

            // 读取初始化状态
            bool initialized;
            is.read(reinterpret_cast<char *>(&initialized), sizeof(bool));

            // 如果已初始化，加载数据
            if (initialized) {
                Tensor data;
                load_tensor(is, data);
                it->second->setData(data);
                it->second->setInitialized(true);
            } else {
                it->second->setInitialized(false);
            }
        }

        // 加载缓冲区
        size_t buffers_count;
        is.read(reinterpret_cast<char *>(&buffers_count), sizeof(buffers_count));

        for (size_t i = 0; i < buffers_count; ++i) {
            // 读取名称
            size_t name_len;
            is.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));

            std::string name(name_len, '\0');
            is.read(&name[0], name_len);

            // 查找缓冲区
            auto it = _buffers.find(name);
            if (it == _buffers.end()) {
                throw std::runtime_error("加载模型时找不到缓冲区: " + name);
            }

            // 读取初始化状态
            bool initialized;
            is.read(reinterpret_cast<char *>(&initialized), sizeof(bool));

            // 如果已初始化，加载数据
            if (initialized) {
                Tensor data;
                load_tensor(is, data);
                it->second->setData(data);
                it->second->setInitialized(true);
            } else {
                it->second->setInitialized(false);
            }
        }
    }

    static void save_tensor(std::ofstream &os, const Tensor &tensor) {
        // 保存requires_grad
        bool gradRequired = tensor.isGradRequired();
        os.write(reinterpret_cast<const char *>(&gradRequired), sizeof(bool));

        // 保存维度大小
        size_t strides_size = tensor.strides().size();
        os.write(reinterpret_cast<const char *>(&strides_size), sizeof(strides_size));
        os.write(reinterpret_cast<const char *>(tensor.strides().data()),
                 strides_size * sizeof(size_t));

        // 保存存储偏移量
        size_t offset = tensor.storageOffset();
        os.write(reinterpret_cast<const char *>(&offset), sizeof(size_t));

        // 保存设备类型
        DeviceType device = tensor.device();
        os.write(reinterpret_cast<const char *>(&device), sizeof(DeviceType));

        // 保存数据类型
        DType dtype = tensor.dtype();
        os.write(reinterpret_cast<const char *>(&dtype), sizeof(DType));

        // 保存存储数据
        // 假设Storage类有serialize方法
        tensor.serialize(os);

        // 自动微分上下文不需要保存，加载时重新创建
    }

    static void load_tensor(std::ifstream &is, Tensor &tensor) {
        // 加载requires_grad
        bool gradRequired = tensor.isGradRequired();
        is.read(reinterpret_cast<char *>(&gradRequired), sizeof(bool));

        // 加载维度大小
        size_t strides_size;
        is.read(reinterpret_cast<char *>(&strides_size), sizeof(strides_size));
        std::vector strides(tensor.strides());
        is.read(reinterpret_cast<char *>(strides.data()), strides_size * sizeof(size_t));
        tensor.setStrides(strides);

        // 加载存储偏移量
        size_t offset = tensor.storageOffset();
        is.read(reinterpret_cast<char *>(&offset), sizeof(size_t));

        // 加载设备类型
        DeviceType device = tensor.device();
        is.read(reinterpret_cast<char *>(&device), sizeof(DeviceType));

        // 加载数据类型
        DType dtype = tensor.dtype();
        is.read(reinterpret_cast<char *>(&dtype), sizeof(DType));

        // 加载存储数据
        // 假设Storage类有deserialize方法
        tensor.deserialize(is);

        // 重置自动微分上下文
        if (!tensor.grad().empty()) {
            tensor.zeroGrad();
        }
    }

    template <typename... Args,typename ret> ret forward(Args &&...args) {
        return static_cast<Derived *>(this)->forward(std::forward<Args>(args)...);
    }

  public:
    // ======================= 构造 =======================
    Module<Derived>() {
        _argsGrad = std::vector<std::optional<Tensor>>();
        _buffers = std::unordered_map<std::string, std::unique_ptr<Buffer>>();
        _parameters = std::unordered_map<std::string, std::unique_ptr<Parameter>>();
        _children = std::unordered_map<std::string,std::unique_ptr<ModuleBase>>();
        _ctx = *AutoGradContext::current();
        _forwardPreHooks = std::vector<ForwardPreHook>();
        _forwardHooks = std::vector<ForwardHook>();
        _fullModuleBackwardHooks = std::vector<FullModuleBackwardHook>();
        _outputsGrad = std::vector<std::optional<Tensor>>();
        _parent = nullptr;
        _train = true;
    }

    // ======================= 析构 =======================
    virtual ~Module() = default;

    // // ======================= 辅助函数 =======================
    // template <typename... Args>
    // void forwardInit(Args... args) {
    //     _args = std::make_tuple(std::forward<Args>(args)...);
    // }

    // ======================= 功能性函数 =======================
    void zero_grad() {
        Tensor root = _ctx.rootPtr()->tensor;
        _ctx.zeroGrad(&root);
    }

    void backward() {
        Tensor top = _ctx.rootPtr()->tensor;
        _ctx.backward(top);
        for (auto fn : _fullModuleBackwardHooks) {
            std::vector<std::optional<Tensor>> grads = _argsGrad;
            for (const auto &param : _parameters) {
                Tensor grad = param.second->grad();
                grads.push_back(std::make_optional<Tensor>(grad));
            }
            BACKWARD_HOOK_RET val = fn(this, grads, _outputsGrad);
            if (val.has_value()) {
                grads.clear();
                grads = val.value();
                for (const auto &param : _parameters) {
                    Tensor grad = param.second->grad();
                    grads.push_back(std::make_optional<Tensor>(grad));
                }
                _argsGrad = grads;
            }
        }
    }

    // ======================= 运算符重载 =======================
    template <typename... Args> auto operator()(Args... args) {

        (
            [&args, this]() {
                if constexpr (!(std::is_fundamental_v<decltype(args)> && std::is_pointer(args)) &&
                              std::is_same_v<decltype(args), Tensor>)
                    _argsGrad.push_back(std::optional<Tensor>(args.grad()));
            }(),
            ...);
        for (auto fn : _forwardPreHooks)
            (
                [this, &args, &fn]() {
                    if (fn(this, std::forward<Args>(args)...).has_value())
                        args = fn(this, std::forward<Args>(args)...).value();
                }(),
                ...);
        auto result = forward(std::forward<Args>(args)...);
        for (auto fn : _forwardHooks)
            if (fn(this, std::forward<Args>(args)..., result).has_value())
                result = fn(this, std::forward<Args>(args)..., result).value();
        for (auto elem : result) {
            if (!(std::is_fundamental_v<decltype(elem)> && std::is_pointer(elem)) &&
                std::is_same_v<decltype(elem), Tensor>)
                _outputsGrad.push_back(std::optional<Tensor>(elem.grad()));
        }
        return result;
    }

    // ======================= 基本属性 =======================
    void train(bool recur) {
        if (recur) {
            auto root                                    = this;
            std::function<void(Module * root)> recursive = [&recursive,
                                                            this](Module *root) -> void {
                root->_train = true;
                std::unordered_map<std::string, Module *> children;
                for (const auto &[name, ptr] : _children)
                    children.emplace(name, static_cast<Module *>(ptr.get()));
                for (auto &[_, child] : *children)
                    recursive(child);
            };
            recursive(root);
        } else
            _train = true;
    }

    void eval(bool recur) {
        if (recur) {
            auto root                                    = this;
            std::function<void(Module * root)> recursive = [&recursive,
                                                            this](Module *root) -> void {
                root->_train = false;
                std::unordered_map<std::string, Module *> children;
                for (const auto &[name, ptr] : _children)
                    children.emplace(name, static_cast<Module *>(ptr.get()));
                for (auto &[_, child] : *children)
                    recursive(child);
            };
            recursive(root);
        } else
            _train = false;
    }

    void addChild(const std::string &name, std::unique_ptr<ModuleBase> child) {
        if (!child)
            if (!_children.contains(name))
                _children.emplace(name, std::move(child));
            else
                throw std::runtime_error("Child " + name + " already exists");
        else
            throw std::invalid_argument("addChild");
    }

    void addChildren(std::unordered_map<std::string, std::unique_ptr<ModuleBase>> children) {
        _children.reserve(_children.size() + children.size());
        for (auto it = children.begin(); it != children.end();) {
            auto node = children.extract(it++);
            if (!_children.insert(std::move(node)).inserted) {
                children.insert(std::move(node));
                for (auto rit = it; rit != children.end();) {
                    auto prev = _children.extract(rit->first);
                    if (!prev.empty())
                        children.insert(std::move(prev));
                    ++rit;
                }
                throw std::runtime_error("Duplicate key found:" + node.key());
            }
        }
    }

    std::unordered_map<std::string, Module *> children() const {
        std::unordered_map<std::string, Module *> result;
        for (const auto &[name, child] : _children) {
            Module *ch = static_cast<Module *>(child.get());
            if (!ch)
                result.emplace(name, child);
            else
                throw std::runtime_error("Unsupported child type " + name);
        }
        return result;
    }

    std::vector<Module *> childrenRecur(Module *root) const {
        std::vector<Module *> result;
        std::function<void(Module *)> recursive = [&result, &recursive,
                                                   this](Module *re_root) -> void {
            result.push_back(re_root);
            std::unordered_map<std::string, Module *> children;
            for (const auto &[name, ptr] : _children)
                children.emplace(name, static_cast<Module *>(ptr.get()));
            for (const auto &[_, child] : children)
                recursive(child);
        };
        recursive(root);
        return result;
    }

    template <typename... Args> void apply(Module *root, auto func, Args... args) {
        // ([&func,&args,&root]() {
        //     std::function<void(Module*)> recursive = [&func, &recursive, &args](Module*
        //     root)->void {
        //         std::unordered_map<std::string,Module*> children;
        //         for (auto [name,ptr]:root->_children)
        //         children.emplace(name,static_cast<Module*>(ptr.get())); for (auto
        //         [_,child]:children) {
        //             if (child->_children.size()) recursive(child);
        //             else func(args);
        //         }
        //     };
        //     recursive(root);
        // }(),...);
        std::function<void(Module *)> recursive = [&func, &recursive, &args...](Module *root) -> void {
            std::unordered_map<std::string, Module *> children;
            for (auto [name, ptr] : root->_children)
                children.emplace(name, static_cast<Module *>(ptr.get()));
            for (auto [_, child] : children) {
                if (child->_children.size())
                    recursive(child);
                else
                    func(std::forward<Args>(args)...);
            }
        };
        recursive(root);
    }

    void registerParameter(const std::string &name, Parameter *parameter) {
        if (parameter->isInitialized())
            _parameters.emplace(name, std::make_unique<Parameter>(*parameter));
        else
            throw std::runtime_error("Parameter '" + name + "' is not initialized");
    }

    void
    registerParameters(std::unordered_map<std::string, std::unique_ptr<Parameter>> &parameters) {
        _parameters.reserve(_parameters.size() + parameters.size());
        for (auto it = parameters.begin(); it != parameters.end();) {
            if (it->second->isInitialized()) {
                auto node = parameters.extract(it++);
                if (!_parameters.insert(std::move(node)).inserted) {
                    for (auto rit = it; rit != parameters.end();) {
                        auto prev = parameters.extract(it->first);
                        if (!prev.empty())
                            _parameters.insert(std::move(prev));
                        ++rit;
                    }
                    throw std::runtime_error("Duplicate key found:" + node.key());
                }
            } else
                throw std::runtime_error("Parameter '" + it->first + "' is not initialized");
        }
    }

    [[nodiscard]] Parameter parameter(const std::string &name) const {
        return *_parameters.at(name);
    }

    [[nodiscard]] std::vector<Parameter>
    parameters(std::initializer_list<std::string> names) const {
        std::vector<Parameter> result;
        for (std::string name : names)
            result.push_back(*(_parameters.at(name)));
        return result;
    }

    void registerBuffer(const std::string &name, Buffer *buffer) {
        if (buffer->isInitialized())
            _buffers.emplace(name, buffer);
        else
            throw std::runtime_error("Buffer '" + name + "' is not initialized");
    }

    void registerBuffers(std::unordered_map<std::string, std::unique_ptr<Buffer>> &buffers) {
        _parameters.reserve(_buffers.size() + buffers.size());
        for (auto it = buffers.begin(); it != buffers.end();) {
            if (it->second->isInitialized()) {
                auto node = buffers.extract(it++);
                if (!_buffers.insert(std::move(node)).inserted) {
                    for (auto rit = it; rit != buffers.end();) {
                        auto prev = buffers.extract(it->first);
                        if (!prev.empty())
                            _buffers.insert(std::move(prev));
                        ++rit;
                    }
                    throw std::runtime_error("Duplicate key found:" + node.key());
                }
            } else
                throw std::runtime_error("Buffer '" + it->first + "' is not initialized");
        }
    }

    [[nodiscard]] Buffer buffer(const std::string &name) const { return *_buffers.at(name); }

    [[nodiscard]] std::vector<Buffer>
    buffers(const std::initializer_list<std::string> names) const {
        std::vector<Buffer> result;
        for (std::string name : names)
            result.push_back(*(_buffers.at(name)));
        return result;
    }

    template <typename T> void setType() {
        auto root      = this;
        std::function<void(const Module*)> recursive = [&recursive](const Module *root) -> void {
            auto children = static_cast<std::unordered_map<std::string, Module *>>(root->_children);
            for (auto &[_, child] : children) {
                if (child->_children.size() > 0)
                    recursive(child);
                else {
                    for (auto &[_, param] : child->_parameters)
                        param->data().setDtype(cpp2DType<T>());
                    for (auto &[_, buffer] : child->_buffers)
                        buffer->data().setDtype(cpp2DType<T>());
                }
            }
        }(root);
    }

    [[nodiscard]] virtual std::string className() const = 0;

    void registerForwardPreHook(ForwardPreHook func) { _forwardPreHooks.push_back(func); }

    void registerForwardHook(ForwardHook func) { _forwardHooks.push_back(func); }

    void registerFullModuleBackwardHook(FullModuleBackwardHook func) {
        _fullModuleBackwardHooks.push_back(func);
    }

    void removeForwardPreHook(size_t idx) {
        if (idx < _forwardPreHooks.size())
            _forwardPreHooks.erase(_forwardPreHooks.begin() + idx);
        else
            throw std::runtime_error("removeForwardPreHook index out of range");
    }

    void removeForwardHook(size_t idx) {
        if (idx < _forwardHooks.size())
            _forwardHooks.erase(_forwardHooks.begin() + idx);
        else
            throw std::runtime_error("removeForwardHook index out of range");
    }

    void removeFullModuleBackwardHook(size_t idx) {
        if (idx < _fullModuleBackwardHooks.size())
            _fullModuleBackwardHooks.erase(_fullModuleBackwardHooks.begin() + idx);
        else
            throw std::runtime_error("removeFullModuleBackwardHook index out of range");
    }

    void removeAllForwardPreHooks() { _forwardPreHooks.clear(); }

    void removeAllForwardHooks() { _fullModuleBackwardHooks.clear(); }

    void removeAllFullModuleBackwardHooks() { _fullModuleBackwardHooks.clear(); }

    std::vector<ForwardPreHook> forwardPreHooks() const { return _forwardPreHooks; }

    std::vector<ForwardHook> forwardHooks() const { return _forwardHooks; }

    std::vector<FullModuleBackwardHook> fullModuleBackwardHooks() const {
        return _fullModuleBackwardHooks;
    }

    ForwardPreHook forwardPreHook(size_t idx) const {
        if (idx < _forwardPreHooks.size())
            return _forwardPreHooks[idx];
        throw std::runtime_error("forwardPreHook index out of range");
    }

    ForwardHook forwardHook(size_t idx) const {
        if (idx < _forwardHooks.size())
            return _forwardHooks[idx];
        throw std::runtime_error("forwardHook index out of range");
    }

    FullModuleBackwardHook fullModuleBackwardHook(size_t idx) const {
        if (idx < _fullModuleBackwardHooks.size())
            return _fullModuleBackwardHooks[idx];
        throw std::runtime_error("fullModuleBackwardHook index out of range");
    }

    Module *findModule(const std::string &path) {
        // 分割路径为组件
        std::vector<std::string> components;
        size_t start = 0, end = 0;
        while ((end = path.find('.', start)) != std::string::npos) {
            components.push_back(path.substr(start, end - start));
            start = end + 1;
        }
        components.push_back(path.substr(start));

        // 从当前模块开始遍历
        Module *current = this;
        for (const auto &comp : components) {
            auto it = current->_children.find(comp);
            if (it == current->_children.end()) {
                throw std::runtime_error("Module:" + comp + " not found."); // 未找到
            }
            current = static_cast<Module *>(it->second);
        }
        return current;
    }

    Parameter *findParameter(std::string &path) {
        // 分割路径为组件
        std::vector<std::string> components;
        size_t start = 0, end = 0;
        while ((end = path.find('.', start)) != std::string::npos) {
            components.push_back(path.substr(start, end - start));
            start = end + 1;
        }
        components.push_back(path.substr(start));

        // 从当前模块开始遍历
        Module *parent = this;
        for (size_t i{0}; i < components.size() - 1; i++) {
            auto it = parent->_children.find(components[i]);
            if (it == parent->_children.end()) {
                throw std::runtime_error("Module:" + components[i] + " not found."); // 未找到
            }
            parent = static_cast<Module *>(it->second);
        }
        return parent->_parameters.at(components.back());
    }

    Buffer *findBuffer(std::string &path) {
        // 分割路径为组件
        std::vector<std::string> components;
        size_t start = 0, end = 0;
        while ((end = path.find('.', start)) != std::string::npos) {
            components.push_back(path.substr(start, end - start));
            start = end + 1;
        }
        components.push_back(path.substr(start));

        // 从当前模块开始遍历
        Module *parent = this;
        for (size_t i{0}; i < components.size() - 1; i++) {
            auto it = parent->_children.find(components[i]);
            if (it == parent->_children.end()) {
                throw std::runtime_error("Module:" + components[i] + " not found."); // 未找到
            }
            parent = static_cast<Module *>(it->second);
        }
        return parent->_buffers.at(components.back());
    }

    // ======================= 公开IO =======================
    [[nodiscard]] std::vector<std::unordered_map<std::string, Tensor *>>
    state(std::string prefix, bool keepVars) {
        std::unordered_map<std::string, Tensor *> parameters;
        std::unordered_map<std::string, Tensor *> buffers;

        std::function<void(const std::string, const Module *)> recursive;
        recursive = [&recursive, &prefix, &keepVars, &parameters, &buffers,
                     this](const std::string &name, const Module *root) {
            for (auto [key, val] : root->_parameters) {
                if (!val->isInitialized())
                    throw std::runtime_error("Parameter '" + name + "' is not initialized");
                Tensor data = val->data().clone();
                if (keepVars) {
                    data.requires_grad(false);
                    data.zeroGrad();
                }
                parameters.emplace(prefix + name + key, &data);
            }
            for (auto [key, val] : root->_buffers) {
                if (!val->isInitialized())
                    throw std::runtime_error("Buffer '" + name + "' is not initialized");
                Tensor data = val->data().clone();
                if (keepVars) {
                    data.requires_grad(false);
                    data.zeroGrad();
                    _ctx.set_retain_graph(false);
                }
                buffers.emplace(prefix + name + key, &data);
            }
            for (auto &[mo_name, child] : root->_children) {
                // 添加分隔符构建新前缀: parent_prefix + child_name + "."
                std::string new_prefix = prefix + mo_name + ".";
                recursive(new_prefix, static_cast<Module *>(child));
            }
        };
        recursive(prefix, this);
        return {std::move(parameters), std::move(buffers)};
    }

    void loadState(std::vector<std::unordered_map<std::string, Tensor *>> state, bool strict) {
        // 提取参数和缓冲区状态
        const auto &param_state  = state[0]; // 参数状态
        const auto &buffer_state = state[1]; // 缓冲区状态

        // 存储未匹配的键
        std::unordered_set<std::string> unmatched_keys;

        // 1. 加载参数
        for (const auto &[key, tensor] : param_state) {
            // 分割键为模块路径和参数名
            size_t pos = key.find_last_of('.');
            if (pos == std::string::npos) {
                // 根节点参数
                auto it = _parameters.find(key);
                if (it != _parameters.end()) {
                    // 复制张量值
                    it->second->setData(*tensor);
                } else if (strict) {
                    throw std::runtime_error("Parameter '" + key + "' not found in model");
                } else {
                    unmatched_keys.insert(key);
                }
            } else {
                // 递归查找模块
                std::string module_path = key.substr(0, pos);
                std::string param_name  = key.substr(pos + 1);

                if (Module *module = findModule(module_path)) {
                    auto it = module->_parameters.find(param_name);
                    if (it != module->_parameters.end()) {
                        // 复制张量值
                        it->second->setData(*tensor);
                    } else if (strict) {
                        throw std::runtime_error("Parameter '" + key + "' not found in model");
                    } else {
                        unmatched_keys.insert(key);
                    }
                } else if (strict) {
                    throw std::runtime_error("Module '" + module_path +
                                             "' not found for parameter '" + key + "'");
                } else {
                    unmatched_keys.insert(key);
                }
            }
        }

        // 2. 加载缓冲区
        for (const auto &[key, tensor] : buffer_state) {
            // 分割键为模块路径和缓冲区名
            size_t pos = key.find_last_of('.');
            if (pos == std::string::npos) {
                // 根节点缓冲区
                auto it = _buffers.find(key);
                if (it != _buffers.end()) {
                    // 复制张量值
                    it->second->setData(*tensor);
                } else if (strict) {
                    throw std::runtime_error("Buffer '" + key + "' not found in model");
                } else {
                    unmatched_keys.insert(key);
                }
            } else {
                // 递归查找模块
                std::string module_path = key.substr(0, pos);
                std::string buffer_name = key.substr(pos + 1);

                if (Module *module = findModule(module_path)) {
                    auto it = module->_buffers.find(buffer_name);
                    if (it != module->_buffers.end()) {
                        // 复制张量值
                        it->second->setData(*tensor);
                    } else if (strict) {
                        throw std::runtime_error("Buffer '" + key + "' not found in model");
                    } else {
                        unmatched_keys.insert(key);
                    }
                } else if (strict) {
                    throw std::runtime_error("Module '" + module_path + "' not found for buffer '" +
                                             key + "'");
                } else {
                    unmatched_keys.insert(key);
                }
            }
        }

        // 3. 在strict模式下检查未匹配的键
        if (strict && !unmatched_keys.empty()) {
            std::string error_msg = "Unmatched keys in state dict:";
            for (const auto &key : unmatched_keys) {
                error_msg += "\n  " + key;
            }
            throw std::runtime_error(error_msg);
        }
    }

    [[nodiscard]] virtual std::string extra_expr() const { return ""; }

    void operator<<(Module *content) const {
        std::cout << extra_expr() << std::endl;

        std::cout << className() << "(\n" << "   Submodules(\n";
        std::function<void(std::string,Module*)> recursive = [&recursive](std::string name, Module *root) -> void {
            std::unordered_map<std::string, Module *> children;
            for (auto [name, child] : root->_children)
                children.emplace(name, child);
            for (auto &[key, val] : children) {
                if (val->_children.size())
                    recursive(key, val);
                else
                    std::cout << "      Submodule Name:" << key
                              << " Submodule ClassName:" << val->className() << " Parent:" << name
                              << std::endl;
            }
        };
        for (auto &[key, val] : _children)
            recursive(key, static_cast<Module *>(val.get()));
        std::cout << "   )\n";
        std::cout << "   Parameters(\n";

        for (const auto &key : _parameters | std::views::keys)
            std::cout << "Parameter Name:" << key << std::endl;
        std::cout << "   )\n";
        std::cout << "   Buffers(\n";
        for (const auto &key : _buffers | std::views::keys)
            std::cout << "Buffer Name:" << key << std::endl;
        std::cout << "   )\n" << ")";
    }

    void save(const std::string &filename) const {
        std::ofstream os(filename, std::ios::binary | std::ios::trunc);
        if (!os.is_open()) {
            throw std::runtime_error("无法打开文件进行写入: " + filename);
        }

        // 写入文件标识和版本信息
        const char *magic = "PTH1.0";
        os.write(magic, 6);

        // 递归保存模块数据
        save_impl(os);
    }

    void load(const std::string &filename) {
        std::ifstream is(filename, std::ios::binary);
        if (!is.is_open()) {
            throw std::runtime_error("无法打开文件进行读取: " + filename);
        }

        // 验证文件标识和版本
        char magic[6];
        is.read(magic, 6);
        if (std::strncmp(magic, "PTH1.0", 6) != 0) {
            throw std::runtime_error("无效的pth文件格式");
        }

        // 递归加载模块数据
        load_impl(is);
    }

    virtual void
    setExtraState(const std::vector<std::unordered_map<std::string, Tensor *>> &state) {}

    [[nodiscard]] virtual std::vector<std::unordered_map<std::string, Tensor *>>
    getExtraState() const {
        return {};
    }
};
