//
// Created by beapoe on 2025/8/14.
//

#ifndef CTORCH_BASICS_H
#define CTORCH_BASICS_H

#include <optional>
import Tensor_dev;
import nn;

#define HOOK_RET std::optional<Tensor>
#define BACKWARD_HOOK_RET std::optional<std::vector<std::optional<Tensor>>>

using Hook = HOOK_RET(*)(Tensor& self);
using ForwardPreHook         = HOOK_RET (*)(const ModuleBase *self,
                                        const std::vector<std::optional<Tensor>> input);
using ForwardHook            = HOOK_RET (*)(const ModuleBase *self,
                                 const std::vector<std::optional<Tensor>> grad_input,
                                 std::vector<std::optional<Tensor>> grad_output);
using FullModuleBackwardHook = BACKWARD_HOOK_RET (*)(
    const ModuleBase &self, const std::vector<std::optional<Tensor>> grad_input,
    std::vector<std::optional<Tensor>> grad_output);

#endif // CTORCH_BASICS_H
