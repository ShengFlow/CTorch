# P0.2 step 5: Softmax backward graph 实装完成

**日期**: 2026-08-30 11:49
**作者**: 苏璃珞
**关联**: ASPLOS 2027 · C3 完善优先

## 目标

把 SoftmaxNode 接入 C3 backward 编译路径，构造反向子图
`grad_x = y * (grad - sum(grad*y, dim=1, keepdim=true))`（axis=1 行 softmax）。

## 改动

### 1. Softmax backward graph 构造（C3BackwardCapture.cpp）

- 新增 `buildSoftmaxBackwardGraph(grad_desc, input_desc)` 函数，7 op 分解：
  1. 重算 `y = softmax(x, dim=1)`：`Exp(x)` + `SumReduce[axis=1, keepdim=true]` + `Div`（广播）
  2. `Mul(grad, y)`（梯度 × 输出）
  3. `SumReduce[axis=1, keepdim=true]`（行内聚合）
  4. `Sub(grad, sum_grad_y)`（广播 `[M,1] → [M,N]`）
  5. `Mul(y, diff)`（最终梯度）
- 图输入 `[grad, x]`，x 对应 `forward_inputs[0]`，输出 1 个 `grad_x`。

### 2. 分发与白名单（C3BackwardCapture.cpp）

- `buildBackwardGraphForTypeAndIndex` 新增 `SoftmaxNode` 分支（input_index==0 校验）。
- `supportsNodeType` 新增 `SoftmaxNode` 匹配（它是单输入节点，dim 是 attribute）。
- `C3BackwardCapture.h` 新增 `buildSoftmaxBackwardGraph` 声明 + 文档注释。

### 3. SumReduceNode 加 `keepdim` 字段（核心 primitives 扩展）

Softmax backward 需要 `keepdim=true` 让 `[M, N] axis=1 → [M, 1]`（而非去掉维度），便于后续 Sub/Mul 广播到 `[M, N]`。

| 文件 | 改动 |
|------|------|
| `c3/include/C3/Graph.h` | `SumReduceNode` 加 `bool keepdim = false`（默认 false 保持向后兼容）|
| `c3/include/C3/C3Ops.td` | `C3_SumReduceOp` 加 `I32Attr:$keepdim`，builder 同步加参数 |
| `c3/include/C3/C3BackwardCapture.h` | 加 `buildSoftmaxBackwardGraph` 声明 |
| `c3/src/C3/C3Dialect.cpp` | `SumReduceOp::build` 手写实现加 keepdim 属性 |
| `c3/src/C3/C3DialectLowering.cpp` | `SumReduceOpLowering` 取 `op.getKeepdim()`（注：仅 shape 描述，循环体不变）|
| `c3/src/C3/MLIRKernelGen.cpp` | 两处 `SumReduceOp` 创建调用（标量/向量化路径）补 `sr.keepdim` |

`keepdim` 不改输出 buffer 元素数（仍 M 个 float），只改 Graph 层的 shape 描述——这正是 Softmax 广播需要的元数据。

## 验证

```
cd build && cmake --build . --target test_c3_backward -j8
[100%] Built target test_c3_backward ✅

./test_c3_backward
✅ PASS: C3 backward 结果正确 (overall_max_diff=7.45058e-08)
```

零回归。Test 1~10 全部通过，max_diff 与基线 7.45e-08 一致。

## 已知限制（留给 step 6 / 后续）

- **axis 暂固定 1**：行 softmax 写死，axis=0 不支持（编译时降维 axis 广播在现有 linalg pipeline 缺支持）。
- **数值稳定版未做**：公式是朴素 `exp(x)`，没先减行 max。后续可加 `Neg + Max + Sub` 改造成稳定版（linalg.softmax 在 forward 端已经稳定，但 backward 重算 y 用的是朴素 c3.softmax op，**不**经过 linalg.softmax 路径，所以会重蹈覆辙）。**建议**：直接把 c3.softmax op 在 backward 子图里复用时改走 linalg.softmax 路径，跟 forward 对齐。
- **MLIRKernelGen broadcast 仅支持 numel 判定**：`[M, 1]` 广播到 `[M, N]` 的现有逻辑是 `idx % node_numel`，对 M=4 N=8 这种尺寸返回错位（`(i*N + j) % M ≠ i`）。当前 buildSoftmaxBackwardGraph 的 Sub/Div 节点会触发这条路径——**实际跑 Softmax backward 时会数值错误**。修复路径：broadcast 改成 shape-based 索引（用 shape 算 `idx / N`），后续单独立项。
- **test_c3_backward.cpp 还没加 Softmax 测试 case**：当前测试只覆盖 ReLU/Sigmoid/Tanh/Add/Sub/Div/MatMul/MLP。Softmax 数值正确性需要等 broadcast 修了再加 case 验证。

## 下一步

- **step 6**：CrossEntropy 完整接入（`c3.cross_entropy` op + Node + lowering + forward + backward）。CrossEntropy 内部就是 Softmax + NLL，所以会复用 Softmax 的 graph + broadcast 修复。
- **broadcast shape-based 修复**：立专项 P0.2.1，Softmax/广播相关 backward 都依赖这个。

## Commit

待 c3 submodule 提交（改动 8 个文件：Graph.h / C3Ops.td / C3BackwardCapture.h+cpp / C3Dialect.cpp / C3DialectLowering.cpp / MLIRKernelGen.cpp x2）。
