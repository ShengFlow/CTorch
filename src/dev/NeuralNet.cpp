//
// Created by Beapoe on 25-7-27.
//
module;

#include "../../src/dev/Tensor.h"
#include <vector>
#include <functional>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <typeinfo>

module nn;

// Parameter
 Parameter::Parameter(Tensor data, bool requiresGrad = true) {
     _data = data;
     _data.requires_grad(requiresGrad);
     _initialized = true;
 }

bool Parameter::isInitialized() const {
     return _initialized;
 }

void Parameter::setInitialized(bool status) {_initialized = status;}

Tensor Parameter::data() const {
     return _data;
 }

void Parameter::setData(Tensor data) {
     _data = std::move(data);
 }

// Buffer
 Buffer::Buffer(Tensor data) {
     _data = data;
     _initialized = true;
 }

bool Buffer::isInitialized() const {
     return _initialized;
 }

void Buffer::setInitialized(bool status) { _initialized = status; }


Tensor Buffer::data() const {
     return _data;
 }

void Buffer::setData(Tensor data) {
     _data = std::move(data);
 }


// Module
using ForwardPreHook = HOOK_RET(*)(const Module* self,const std::vector<Tensor&> input);
using ForwardHook = HOOK_RET(*)(const Module* self,const std::vector<Tensor&> grad_input, std::vector<Tensor&> grad_output);
using FullModuleBackwardHook = HOOK_RET(*)(const Module& self,const std::vector<Tensor&> grad_input, std::vector<Tensor&> grad_output);

void Module::zero_grad() {ctx.zero_grad(ctx.rootPtr());}

void Module::backward() {
     ctx.backward((ctx.rootPtr())->tensor);
     for (auto fn:_fullModuleBackwardHooks) fn(*this,ctx.inputGrad(),ctx.outputGrad()); // TODO：这里要等奕帆修完才行，应该多输出对应多梯度张量
 }

Module* Module::findModule(const std::string& path) {
     // 分割路径为组件
     std::vector<std::string> components;
     size_t start = 0, end = 0;
     while ((end = path.find('.', start)) != std::string::npos) {
         components.push_back(path.substr(start, end - start));
         start = end + 1;
     }
     components.push_back(path.substr(start));

     // 从当前模块开始遍历
     Module* current = this;
     for (const auto& comp : components) {
         auto it = current->_children.find(comp);
         if (it == current->_children.end()) {
             return nullptr; // 未找到
         }
         current = it->second;
     }
     return current;
 }

void Module::save_impl(std::ofstream& os) const {
    // 保存基本状态
    os.write(reinterpret_cast<const char*>(&_train), sizeof(_train));

    // 保存子模块数量及子模块
    size_t children_count = _children.size();
    os.write(reinterpret_cast<const char*>(&children_count), sizeof(children_count));

    for (const auto& pair : _children) {
        // 保存名称长度和名称
        size_t name_len = pair.first.length();
        os.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        os.write(pair.first.c_str(), name_len);

        // 递归保存子模块
        pair.second->save_impl(os);
    }

    // 保存参数数量及参数
    size_t params_count = _parameters.size();
    os.write(reinterpret_cast<const char*>(&params_count), sizeof(params_count));

    for (const auto& pair : _parameters) {
        // 保存名称
        size_t name_len = pair.first.length();
        os.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        os.write(pair.first.c_str(), name_len);

        // 保存参数是否初始化
        bool initializeStatus = pair.second->isInitialized();
        os.write(reinterpret_cast<const char*>(&initializeStatus), sizeof(bool));

        // 如果已初始化，保存数据
        if (pair.second->isInitialized()) {
            save_tensor(os, pair.second->data());
        }
    }

    // 保存缓冲区数量及缓冲区
    size_t buffers_count = _buffers.size();
    os.write(reinterpret_cast<const char*>(&buffers_count), sizeof(buffers_count));

    for (const auto& pair : _buffers) {
        // 保存名称
        size_t name_len = pair.first.length();
        os.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        os.write(pair.first.c_str(), name_len);

        // 保存缓冲区是否初始化
        bool initializeStatus = pair.second->isInitialized();
        os.write(reinterpret_cast<const char*>(&initializeStatus), sizeof(bool));

        // 如果已初始化，保存数据
        if (pair.second->isInitialized()) {
            save_tensor(os, pair.second->data());
        }
    }
}

void Module::load_impl(std::ifstream& is) {
    // 加载基本状态
    is.read(reinterpret_cast<char*>(&_train), sizeof(_train));

    // 加载子模块
    size_t children_count;
    is.read(reinterpret_cast<char*>(&children_count), sizeof(children_count));

    for (size_t i = 0; i < children_count; ++i) {
        // 读取名称
        size_t name_len;
        is.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

        std::string name(name_len, '\0');
        is.read(&name[0], name_len);

        // 假设子模块已存在且类型匹配
        auto it = _children.find(name);
        if (it == _children.end()) {
            throw std::runtime_error("加载模型时找不到子模块: " + name);
        }

        // 递归加载子模块
        it->second->load_impl(is);
    }

    // 加载参数
    size_t params_count;
    is.read(reinterpret_cast<char*>(&params_count), sizeof(params_count));

    for (size_t i = 0; i < params_count; ++i) {
        // 读取名称
        size_t name_len;
        is.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

        std::string name(name_len, '\0');
        is.read(&name[0], name_len);

        // 查找参数
        auto it = _parameters.find(name);
        if (it == _parameters.end()) {
            throw std::runtime_error("加载模型时找不到参数: " + name);
        }

        // 读取初始化状态
        bool initialized;
        is.read(reinterpret_cast<char*>(&initialized), sizeof(bool));

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
    is.read(reinterpret_cast<char*>(&buffers_count), sizeof(buffers_count));

    for (size_t i = 0; i < buffers_count; ++i) {
        // 读取名称
        size_t name_len;
        is.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

        std::string name(name_len, '\0');
        is.read(&name[0], name_len);

        // 查找缓冲区
        auto it = _buffers.find(name);
        if (it == _buffers.end()) {
            throw std::runtime_error("加载模型时找不到缓冲区: " + name);
        }

        // 读取初始化状态
        bool initialized;
        is.read(reinterpret_cast<char*>(&initialized), sizeof(bool));

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

void Module::save_tensor(std::ofstream& os, const Tensor& tensor) {
     // 保存requires_grad
    bool gradRequired = tensor.isGradRequired();
     os.write(reinterpret_cast<const char*>(&gradRequired), sizeof(bool));

     // 保存维度大小
     size_t strides_size = tensor.strides().size();
     os.write(reinterpret_cast<const char*>(&strides_size), sizeof(strides_size));
     os.write(reinterpret_cast<const char*>(tensor.strides().data()),
              strides_size * sizeof(size_t));

     // 保存存储偏移量
    size_t offset = tensor.storageOffset();
     os.write(reinterpret_cast<const char*>(&offset), sizeof(size_t));

     // 保存设备类型
    DeviceType device = tensor.device();
     os.write(reinterpret_cast<const char*>(&device), sizeof(DeviceType));

     // 保存数据类型
    DType dtype = tensor.dtype();
     os.write(reinterpret_cast<const char*>(&dtype), sizeof(DType));

     // 保存存储数据
     // 假设Storage类有serialize方法
     tensor.serialize(os);

     // 自动微分上下文不需要保存，加载时重新创建
 }

void Module::load_tensor(std::ifstream& is, Tensor& tensor) {
     // 加载requires_grad
    bool gradRequired = tensor.isGradRequired();
     is.read(reinterpret_cast<char*>(&gradRequired), sizeof(bool));

     // 加载维度大小
     size_t strides_size;
     is.read(reinterpret_cast<char*>(&strides_size), sizeof(strides_size));
     tensor.strides().resize(strides_size);
     is.read(reinterpret_cast<char*>(tensor.strides().data()),
             strides_size * sizeof(size_t));

     // 加载存储偏移量
    size_t offset = tensor.storageOffset();
     is.read(reinterpret_cast<char*>(&offset), sizeof(size_t));

     // 加载设备类型
    DeviceType device = tensor.device();
     is.read(reinterpret_cast<char*>(&device), sizeof(DeviceType));

     // 加载数据类型
    DType dtype = tensor.dtype();
     is.read(reinterpret_cast<char*>(&dtype), sizeof(DType));

     // 加载存储数据
     // 假设Storage类有deserialize方法
     tensor.deserialize(is);

     // 重置自动微分上下文
     if (tensor.hasGrad()) {
         tensor.clearCtx();
     }
 }

Tensor Module::operator()(Tensor &input) {
     for (auto fn:_forwardPreHooks)
         if (fn(this,input).has_value()) input = fn(this, input).value();
    Tensor result = forward(input);
    for (auto fn:_forwardHooks)
        if (fn(this,input,result).has_value()) result = fn(this, input, result).value();
    return result;
 }

void Module::train(bool recur) {
    if (recur) {
        auto root      = this;
        auto recursive = [&recursive](Module *root) -> void {
            root->_train = true;
            std::unordered_map<std::string,Module*>* children = &root->_children;
            for (auto& [_,child]:*children) recursive(child);
        };
        recursive(root);
    } else
        _train = true;
}

void Module::eval(bool recur) {
    if (recur) {
        auto root      = this;
        auto recursive = [&recursive](Module *root) -> void {
            root->_train = false;
            std::unordered_map<std::string,Module*>* children = &root->_children;
            for (auto& [_,child]:*children) recursive(child);
        };
        recursive(root);
    } else
        _train = false;
}

void Module::addChild(std::string name,Module *child) {
    _children.emplace(name,child);
}

void Module::addChildren(std::unordered_map<std::string,Module*> children) {
    _children.reserve(_children.size()+children.size());
    for (auto it = children.begin();it!=children.end();) {
        auto node = children.extract(it++);
        if (!_children.insert(std::move(node)).inserted) {
            children.insert(std::move(node));
            for (auto rit = it;rit!=children.end();) {
                auto prev = _children.extract(rit->first);
                if (!prev.empty()) children.insert(std::move(prev));
                ++rit;
            }
            throw std::runtime_error("Duplicate key found:"+node.key());
        }
    }
}

std::unordered_map<std::string,Module*> Module::children() const{
    return _children;
}

std::vector<Module*> Module::childrenRecur(Module* root) const {
    std::vector<Module*> result;
    auto recursive = [&result, &recursive](Module *root) -> void {
        result.push_back(root);
        std::unordered_map<std::string, Module *>* children = &root->_children;
        for (auto &[_, child] : children) recursive(child);
    };
     recursive(root);
    return result;
}

void Module::registerParameter(std::string name, Parameter *parameter) {
     if (parameter->isInitialized()) _parameters.emplace(name,parameter);
     else throw std::runtime_error("Parameter '"+name+"' is not initialized");
 }

void Module::registerParameters(std::unordered_map<std::string,Parameter*> parameters) {
     _parameters.reserve(_parameters.size()+parameters.size());
     for (auto it = parameters.begin();it != parameters.end();) {
         if (it->second->isInitialized()) {
             auto node = parameters.extract(it++);
             if (!_parameters.insert(std::move(node)).inserted) {
                 for (auto rit = it;rit!=parameters.end();) {
                     auto prev = parameters.extract(it->first);
                     if (!prev.empty()) _parameters.insert(std::move(prev));
                     ++rit;
                 }
                 throw std::runtime_error("Duplicate key found:"+node.key());
             }
         }else throw std::runtime_error("Parameter '"+it->first+"' is not initialized");
     }
 }

Parameter Module::parameter(std::string name) const {
     return *_parameters.at(name);
 }

std::vector<Parameter> Module::parameters(std::initializer_list<std::string> names) const {
     std::vector<Parameter> result;
     for (std::string name:names) result.push_back(*(_parameters.at(name)));
     return result;
 }

void Module::registerBuffer(std::string name,Buffer *buffer) {
     if (buffer->isInitialized()) _buffers.emplace(name,buffer);
     else throw std::runtime_error("Buffer '"+name+"' is not initialized");
 }

void Module::registerBuffers(std::unordered_map<std::string,Buffer*> buffers) {
     _parameters.reserve(_buffers.size()+buffers.size());
     for (auto it = buffers.begin();it != buffers.end();) {
         if (it->second->isInitialized()) {
             auto node = buffers.extract(it++);
             if (!_buffers.insert(std::move(node)).inserted) {
                 for (auto rit = it;rit!=buffers.end();) {
                     auto prev = buffers.extract(it->first);
                     if (!prev.empty()) _buffers.insert(std::move(prev));
                     ++rit;
                 }
                 throw std::runtime_error("Duplicate key found:"+node.key());
             }
         }else throw std::runtime_error("Buffer '"+it->first+"' is not initialized");
     }
 }

Buffer Module::buffer(std::string name) const {
     return *_buffers.at(name);
}

std::vector<Buffer> Module::buffers(std::initializer_list<std::string> names) const {
     std::vector<Buffer> result;
     for (std::string name:names) result.push_back(*(_buffers.at(name)));
     return result;
 }

void Module::registerForwardPreHook(const ForwardPreHook func) { _forwardPreHooks.push_back(func); }

void Module::registerForwardHook(const ForwardHook func) { _forwardHooks.push_back(func); }

void Module::registerFullModuleBackwardHook(const FullModuleBackwardHook func) { _fullModuleBackwardHooks.push_back(func); }

void Module::removeForwardPreHook(const size_t idx) {
     if (idx<_forwardPreHooks.size()) _forwardPreHooks.erase(_forwardPreHooks.begin()+idx);
     else throw std::runtime_error("removeForwardPreHook index out of range");
 }

void Module::removeForwardHook(const size_t idx) {
     if (idx<_forwardHooks.size()) _forwardHooks.erase(_forwardHooks.begin()+idx);
     else throw std::runtime_error("removeForwardHook index out of range");
 }

void Module::removeFullModuleBackwardHook(const size_t idx) {
     if (idx<_fullModuleBackwardHooks.size()) _fullModuleBackwardHooks.erase(_fullModuleBackwardHooks.begin()+idx);
     else throw std::runtime_error("removeFullModuleBackwardHook index out of range");
 }

void Module::removeAllForwardPreHooks() { _forwardPreHooks.clear(); }

void Module::removeAllForwardHooks() { _fullModuleBackwardHooks.clear(); }

void Module::removeAllFullModuleBackwardHooks() { _fullModuleBackwardHooks.clear(); }

std::vector<ForwardPreHook> Module::forwardPreHooks() const {return _forwardPreHooks;}

std::vector<ForwardHook> Module::forwardHooks() const {return _forwardHooks;}

std::vector<FullModuleBackwardHook> Module::fullModuleBackwardHooks() const { return _fullModuleBackwardHooks; }

ForwardPreHook Module::forwardPreHook(size_t idx) const {
    if (idx < _forwardPreHooks.size())
        return _forwardPreHooks[idx];
    throw std::runtime_error("forwardPreHook index out of range");
}

ForwardHook Module::forwardHook(size_t idx) const {
    if (idx < _forwardHooks.size())
        return _forwardHooks[idx];
    throw std::runtime_error("forwardHook index out of range");
}

FullModuleBackwardHook Module::fullModuleBackwardHook(size_t idx) const {
    if (idx < _fullModuleBackwardHooks.size())
        return _fullModuleBackwardHooks[idx];
    throw std::runtime_error("fullModuleBackwardHook index out of range");
}

std::vector<std::unordered_map<std::string,Tensor*>>  Module::state(std::string prefix="",bool keepVars=false) const {
     std::unordered_map<std::string,Tensor*> parameters;
     std::unordered_map<std::string,Tensor*> buffers;

     std::function<void(const std::string,const Module*)> recursive;
     recursive = [&recursive,&prefix,&keepVars,&parameters,&buffers](const std::string name,const Module* root) {
         for (auto [key,val]:root->_parameters) {
             if (!val->isInitialized()) throw std::runtime_error("Parameter '"+name+"' is not initialized");
             Tensor data = val->data().clone();
             if (keepVars) {
                 data.requires_grad(false);
                 data.zeroGrad();
             }
             parameters.emplace(prefix+name+key,&data);
         }
         for (auto [key,val]:root->_buffers) {
             if (!val->isInitialized()) throw std::runtime_error("Buffer '"+name+"' is not initialized");
             Tensor data = val->data().clone();
             if (keepVars) {
                 data.requires_grad(false);
                 data.zeroGrad();
                 data.setRetainGraph(false);
             }
             buffers.emplace(prefix+name+key,&data);
         }
         for (auto& [mo_name,child] : root->_children) {
             // 添加分隔符构建新前缀: parent_prefix + child_name + "."
             std::string new_prefix = prefix + mo_name + ".";
             recursive(new_prefix, child);
         }
     };
     recursive(prefix,this);
     return {std::move(parameters), std::move(buffers)};
}

void Module::loadState(std::vector<std::unordered_map<std::string,Tensor*>> state,bool strict){
    // 提取参数和缓冲区状态
    const auto& param_state = state[0];  // 参数状态
    const auto& buffer_state = state[1]; // 缓冲区状态

    // 存储未匹配的键
    std::unordered_set<std::string> unmatched_keys;

    // 1. 加载参数
    for (const auto& [key, tensor] : param_state) {
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
            std::string param_name = key.substr(pos + 1);

            Module* module = findModule(module_path);
            if (module) {
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
                throw std::runtime_error("Module '" + module_path + "' not found for parameter '" + key + "'");
            } else {
                unmatched_keys.insert(key);
            }
        }
    }

    // 2. 加载缓冲区
    for (const auto& [key, tensor] : buffer_state) {
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

            Module* module = findModule(module_path);
            if (module) {
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
                throw std::runtime_error("Module '" + module_path + "' not found for buffer '" + key + "'");
            } else {
                unmatched_keys.insert(key);
            }
        }
    }

    // 3. 在strict模式下检查未匹配的键
    if (strict && !unmatched_keys.empty()) {
        std::string error_msg = "Unmatched keys in state dict:";
        for (const auto& key : unmatched_keys) {
            error_msg += "\n  " + key;
        }
        throw std::runtime_error(error_msg);
    }
}


std::string Module::extra_expr() const {
     return "";
 }

void Module::operator<<(Module *content) const {
     std::cout<<extra_expr()<<std::endl;

     std::cout<<className()<<"(\n"<<"   Submodules(\n";
     auto recursive = [&recursive](std::string name,Module* root)->void {
         auto children = root->_children;
         for (auto& [key,val]:children) {
             if (val->_children.size()) recursive(key,val);
             else std::cout<<"      Submodule Name:"<<key<<" Submodule ClassName:"<<val->className()<<" Parent:"<<name<<std::endl;
         }
     };
     for (auto& [key,val]:_children) recursive(key,val);
     std::cout<<"   )\n";
     std::cout<<"   Parameters(\n";

     for (auto& [key,_]:_parameters) std::cout<<"Parameter Name:"<<key<<std::endl;
     std::cout<<"   )\n";
     std::cout<<"   Buffers(\n";
     for (auto& [key,_]:_buffers) std::cout<<"Buffer Name:"<<key<<std::endl;
     std::cout<<"   )\n"<<")";
 }

void Module::save(const std::string& filename) const {
     std::ofstream os(filename, std::ios::binary | std::ios::trunc);
     if (!os.is_open()) {
         throw std::runtime_error("无法打开文件进行写入: " + filename);
     }

     // 写入文件标识和版本信息
     const char* magic = "PTH1.0";
     os.write(magic, 6);

     // 递归保存模块数据
     save_impl(os);
 }

void Module::load(const std::string& filename) {
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

void Module::setExtraState(std::vector<std::unordered_map<std::string, Tensor *>> state) {}

std::vector<std::unordered_map<std::string, Tensor *>> Module::getExtraState() const { return std::vector<std::unordered_map<std::string, Tensor *>>();}

