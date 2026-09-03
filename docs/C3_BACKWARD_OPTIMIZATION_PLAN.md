# C3 Backward 优化执行方案

> 更新日期：2026-08-30  
> 项目：CTorch-optimize-AutoDiff  
> 目标：在保持 backward 数值正确性的前提下，逐步恢复 C3 backward 性能，并最终达到不劣于 eager 的端到端性能。

## 1. 当前状态

### 1.1 正确性

当前 `test_c3_backward` 已达到：

```text
12/12 test PASS
```

已覆盖并通过：

- ReLU backward
- Sigmoid backward
- ReLU → Sigmoid
- ReLU → ReLU
- MLP backward
- Softmax backward
- CrossEntropy backward

其他回归测试：

```text
test_c3_graph                 115/115 PASS
test_c3_compile_merged        10/10 PASS
test_c3_compile_merged_pgo    11/11 PASS
```

### 1.2 默认开关

backward C3 当前默认开启。

显式关闭方式：

```bash
C3_DISABLE_BACKWARD=1 ./build/test_c3_backward
```

或：

```bash
C3_ENABLE_BACKWARD=0 ./build/test_c3_backward
```

`C3_ENABLE_MIMO_BACKWARD=1` 仍为 MIMO 多输出实验子开关。

### 1.3 性能基线

端到端 benchmark：`bench_c3_backward_perf_clean`

测试配置：

- 输入：`[512 x 512]`，约 0.25M elements
- 计算链：`x → Tanh → Sigmoid → ReLU → backward`
- 测量：120 次，无预热

| 模式 | 稳态 mean | 稳态 p50 | 吞吐 | 数值 guard |
|---|---:|---:|---:|---:|
| Eager（`C3_DISABLE_BACKWARD=1`） | 11.855 ms | 11.230 ms | 89.05 iter/s | 0 |
| C3 默认开启 | 2.090 ms | 1.068 ms | 936.51 iter/s | 8.9407e-08 |

按上述 benchmark，C3 当前约为 eager 的 **0.095x 单次 p50 时间**，即约 **10.51x 加速**。

> 注意：用户提供的另一组端到端训练数据为 `C3=2854us`、`Eager=325us`，对应另一 benchmark/工作负载，结论为 C3 慢约 8.8x。两组数据不能直接混用，后续性能优化必须固定 benchmark 口径。

### 1.4 已知问题

benchmark 在测量和数值 guard 完成后，退出清理阶段仍可能触发：

```text
recursive_mutex lock failed: Invalid argument
```

因此性能与数值数据可用，但 benchmark 进程可能以 `134` 退出。该问题需要单独处理，不能与 kernel 数值或性能结果混为一谈。

## 2. 已完成的稳定性修复

### 2.1 `Gt` 标量尾循环条件修复

`MLIRKernelGen.cpp` 的标量 tail path 原先将 `x > 0` 错误映射为 `0`、将 `x <= 0` 映射为 `1`。

修复为：

```cpp
select(cmp, one, zero)
```

这解决了小张量 ReLU backward 结果反转问题。

### 2.2 backward 分支 DAG 避免误用线性链向量化

Sigmoid backward 的重算图包含共享依赖和分支，不满足当前向量化 builder 的严格线性链假设。

当前策略：

- backward 多节点图走标量 DAG 生成器；
- 不使用不满足输入语义的线性链向量化路径；
- 后续完成 live range 与 DAG 输入建模后，再按图属性精确恢复向量化。

### 2.3 scratch buffer 保守独立分配

此前尝试使用两个 scratch buffer 交替复用，导致 Sigmoid backward 分支图中间值被覆盖。

当前策略为每个 intermediate 独占槽位，优先确保正确性。代价是 scratch 容量和内存流量增加。

## 3. 优先级路线

## P0.2.4：Pool buffer live range 分析

### 目标

在不重新引入中间值覆盖的前提下，减少 scratch slot 数量和内存开销。

性能目标：

```text
恢复当前保守实现造成的性能回归：2854us → 约 1663us
```

### 关键判断

当前存在多套编号体系：

```text
Graph node id
compute_nodes 顺序
node_to_buffer logical index
logical_to_pool physical slot
```

同时还叠加：

```text
node_buffer_reuse
显式 output segment
FusedNode
ConstNode
scratchpad offset
```

因此不能仅依据 compute node 的拓扑位置做简单着色。

### 实施步骤

#### P0.2.4a：统一 allocation plan

先抽出纯分析层结构：

```cpp
struct BufferLiveRange {
    size_t node_id;
    size_t logical_buffer;
    size_t start;
    size_t end;
    size_t numel;
    size_t slot;
};

struct BufferAllocationPlan {
    std::vector<BufferLiveRange> ranges;
    size_t slot_count;
    size_t scratch_size;
};
```

分析结果必须同时驱动：

1. `tmp_buffers` 创建；
2. `logical_to_pool` 映射；
3. `getInputPtr()` 的物理 buffer 查找；
4. `result.scratch_size` 计算。

禁止生成阶段和 runtime size 阶段各自实现一套算法。

#### P0.2.4b：计算真实 live range

对每个 intermediate：

```text
start = 定义节点的 compute 序号
end   = 所有实际消费者中最大 compute 序号
```

必须遵守：

- 显式 output 不参与 scratch slot coloring；
- 隐式最后输出不参与 scratch slot coloring；
- `ConstNode` 使用专属常量槽位；
- 共享依赖的最后消费者决定 `end`；
- 仅当 `old.end < new.start` 时允许复用；
- slot 容量按该 slot 内最大 `numel` 计算，而不是所有槽位统一使用全局最大值。

#### P0.2.4c：诊断与验收

增加临时诊断开关：

```bash
C3_TRACE_BUFFER_PLAN=1
```

输出每个节点的：

```text
node_id
logical buffer
[start, end]
numel
physical slot
slot capacity
scratch_size
```

验收条件：

```text
12/12 backward tests PASS
scratch slot 数量少于独立槽位方案
scratch_size 不超过独立槽位方案
性能不劣于当前保守实现
```

如果任一条件失败，立即回退到独立槽位方案。

## P0.2.5：恢复 Sigmoid backward C3

依赖：P0.2.4b 通过。

恢复 `SigmoidNode` 的 C3 支持，并验证：

```text
Test 2 max_diff=0
Test 8 max_diff < 1e-4
```

必须覆盖：

- 小张量尾循环；
- 向量主循环与标量尾循环；
- 分支 DAG 中间值；
- 重复执行和异步编译完成后的 cache hit。

## P0.2.6：恢复 Softmax backward C3

依赖：P0.2.4b 通过，且 P0.2.5 不回归。

Softmax backward 包含：

- 两次 `SumReduce`；
- 广播；
- 多个 intermediate；
- 归约临时 buffer。

必须单独验证：

```text
Test 11 max_diff=0
```

并检查：

- reduce 输出生命周期；
- `[M,1] → [M,N]` 广播索引；
- 两次 reduce 的临时 buffer 是否互相覆盖；
- 多归约图是否错误进入线性链向量化路径。

## P0.2.7：MIMO multi-output 重启

当前 `tryExecuteFusedBackward()` 的部分实验入口仍保持保守隔离。

该项需要处理：

- upstream gradient 追踪；
- 多输出 segment 布局；
- preAct 输出；
- 多 Tensor 共享 storage；
- output offset；
- scratch 与 output 生命周期。

建议使用：

```bash
C3_ENABLE_MIMO_BACKWARD=1
```

作为 opt-in 验证入口。

MIMO 必须在 P0.2.4～P0.2.6 稳定后推进，避免同时引入多个生命周期变量。

## P0.3：C3 backward 性能冲刺

目标：

```text
C3 ≤ Eager
```

当前 C3 与 eager 的根本差异是：

- eager 使用手写 SIMD/AMX kernel；
- C3 backward 主要使用 MLIR 生成的标量 DAG kernel。

候选路线：

1. 完整支持安全的 backward vectorization；
2. 为 unary backward 生成专用 SIMD kernel；
3. 使用 fused SIMD 实现 ReLU/Sigmoid/Tanh 导数；
4. 减少逐节点临时写回；
5. 降低 Tensor/storage 构造和 dispatch 开销；
6. 完善 broadcast vectorization，而不只支持 scalar broadcast；
7. 对适合的 MatMul 路径使用现有 CBLAS/AMX，而不是标量循环。

## 4. 统一验收矩阵

每次修改至少运行：

```bash
cmake --build build --target \
  test_c3_backward \
  test_c3_graph \
  test_c3_compile_merged \
  test_c3_compile_merged_pgo -j2
```

正确性要求：

```text
test_c3_backward              12/12 PASS
test_c3_graph                 115/115 PASS
test_c3_compile_merged        10/10 PASS
test_c3_compile_merged_pgo    11/11 PASS
```

性能测试至少记录：

```text
cold mean
cold p50
steady mean
steady p50
p95
throughput
数值 guard
进程退出码
```

性能对照至少包括：

```bash
# C3 默认开启
./build/bench_c3_backward_perf_clean

# eager 基线
C3_DISABLE_BACKWARD=1 ./build/bench_c3_backward_perf_clean
```

必要时增加：

```bash
C3_MLIR_NO_VECTORIZE=1 ./build/bench_c3_backward_perf_clean
```

## 5. 当前决策

当前不应为了减少 scratch 内存而直接采用未经验证的简单 live-range coloring。

推荐执行顺序：

```text
P0.2.4a 统一 allocation plan
→ P0.2.4b 正确 live-range coloring
→ P0.2.5 恢复 Sigmoid C3
→ P0.2.6 恢复 Softmax C3
→ P0.2.7 MIMO multi-output
→ P0.3 SIMD/AMX 性能冲刺
```

核心原则：

> 正确性优先；任何 buffer 复用优化都必须以 12/12 backward 正确性和完整回归通过为前提。
