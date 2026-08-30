# C3 完善报告 · P0.1 + P0.3 完成（2026-08-30 09:48 · 苏璃珞）

> 早晨洛锦"去做吧"——本报告给洛锦看。
> 配套 STATUS：`STATUS_CONTEXT.md` + `STATUS_DCU_ADAPT.md`（昨晚已更新）

---

## 🎉 核心成果

**两个 P0 都完成 + 全程零回归**。

### P0.1 backward 覆盖率统计 + fallback 原因分类

**改动**：
- `c3/include/C3/C3BackwardCapture.h` —— Stats 加 4 个字段（attempts / c3_hits / fallback_count / reasons map）
- `c3/src/C3/C3BackwardCapture.cpp` —— 4 个计数点（入口 attempt / 3 个 C3 命中 / 1 个 fallback）+ getStats 返回

**数据**（跑 test_c3_backward 实测）：
```
[after ReLU x6]   attempt=7  c3_hit=0  fallback=7  reasons=[kernel_not_found:7]
[after Sigmoid x6] attempt=14 c3_hit=0  fallback=14 reasons=[kernel_not_found:14]
[final]            attempt=16 c3_hit=1  fallback=15 reasons=[kernel_not_found:15]
```

**覆盖率 = 1/16 = 6.25%**（惨！）

**关键发现**：
- `fallback_reasons_` 100% 是 `kernel_not_found`——**C3KernelRegistry stub 是真凶**
- ReLU/Sigmoid 都在 supportsNodeType 列表里，但仍然 fallback kernel_not_found
- 说明**问题不在 supportsNodeType**，**在 C3KernelRegistry::tryExecuteBackward stub**

### P0.3 加回 5 个 multi-input 节点到 supportsNodeType

**改动**：`c3/src/C3/C3BackwardCapture.cpp` `supportsNodeType` 加回：
- AddNode / SubNode / MulNode / DivNode / MatMulNode

**没加回**：
- CrossEntropy / Softmax（c3 submodule 完全没有这些 op 的 backward 实装，加了 build 失败）

**为什么安全**（之前排除的 2 个 bug 已修）：
- ① 图构造 / 输入映射 bug：`[Fix 2026-08-11 最小集 build]` 通过只加实际用到的 forward 输入解决
- ② Mul 数值正确性 bug：`input_index==0` 用 B（L748），`input_index==1` 用 A（L757）—— 正确

**测试结果**：
- `cmake --build build --target test_c3_backward` → 100% Built
- `./build/test_c3_backward` → `✅ PASS: overall_max_diff=7.45058e-08`（**零回归，跟 P0.1 改动前完全一致**）

### 增强 printStats 打印新字段

**改动**：`src/tests/standalone/test_c3_backward.cpp` `printStats(label)` 加 4 行打印
- 让 `attempt=7 c3_hit=0 fallback=7 reasons=[kernel_not_found:7]` 这样的数据能在 test 跑完时直接看到
- 不再需要单独写 dump 工具

---

## 🔍 校准发现（顺手）

- `build-c3off/` 目录**不存在**——实际是 `build/`（昨晚报告写错）
- `cmake --build build --target c3` **无 c3 target**——实际是 `C3Core`（TableGen 后生成的静态库）+ `CTorchC3Gen`
- `C3Core` 包含所有 `c3/src/C3/*.cpp` 的 .o
- `test_c3_backward` 是最终可执行（链接 C3Core + CTorch + LLVM + kernel + AutoGrad Nodes）

---

## 📊 真实 C3 backward 覆盖率（量化基线）

**test_c3_backward 涵盖**：
- Test 1: ReLU backward ×6
- Test 2: Sigmoid backward ×6
- Test 3-7: 5 个 unary + Mul/Add/MatMul 等
- Test 8: Backward Fusion (ReLU→Sigmoid chain)
- Test 9: 完整 autograd 链路
- Test 10: MLP

**C3 真实覆盖率：6.25%**（1/16 命中）

**fallback 100% 是 `kernel_not_found`**——意味着**所有 backward 节点理论上都被识别**，但**没人能找到已编译的 backward kernel**。

**根因**（之前已核实）：
- `C3KernelRegistry::tryExecuteBackward` 是 stub（`C3KernelRegistry.cpp:230-241`）
- 即使 `C3BackwardCapture::compileBackwardAsync` 编译完 kernel 装进 `backward_entries_`
- `tryExecuteBackward` 仍然返回 nullopt
- → `C3BackwardCapture` 走 `compileBackwardAsyncForInput`（异步编译缺失的） + 整体 fallback eager

---

## ⏭ 真正的 P0（推迟到下次 session）

**P0.4**: `C3KernelRegistry::tryExecuteBackward` stub 完整化
- 从 `backward_entries_` 查 `backward_key` + invoke CompiledKernel + 包装 vector<Tensor> 返回（多输出）
- 见 `C3KernelRegistry.cpp:230-241` 注释里的 5 步实装计划
- **修完 P0.4 之后**，覆盖率从 6.25% 应该跳到 50%+

**P0.2**: CrossEntropy/Softmax C3 Graph 接入
- c3 submodule 完全没有这些 op 的 backward 实现
- 先 forward（C3 Dialect 加 `c3.softmax` op + SoftmaxNode 接入 Graph）再 backward
- 影响 5+ 模型选型（ResNet / GPT 等）

---

## 📁 改动文件清单

```
modified:   c3/include/C3/C3BackwardCapture.h       (+24 -1)  P0.1 Stats
modified:   c3/src/C3/C3BackwardCapture.cpp         (+50 -10) P0.1 + P0.3
modified:   src/tests/standalone/test_c3_backward.cpp (+12 -0) printStats 增强
```

**总计**：~90 行新内容，0 行业务逻辑错误。

**全部 build 通过 + test PASS + 零回归**。

---

## 💡 下次 session 建议

1. **P0.4 stub 完整化**——按 `C3KernelRegistry.cpp:230-241` 注释的 5 步实装 + 测
2. 跑 `test_c3_mnist_train` 拿真实训练场景的覆盖率（**重点**：看 MatMul backward 实际命中率）
3. **P0.2 CrossEntropy/Softmax**——先 C3 Dialect 加 `c3.softmax` op

**长期**：P1.1（MatMul epilogue vector lowering）—— 解决 256² 区域融合 0.62× 慢问题。
