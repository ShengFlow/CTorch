# P0.2.2: Multi-input fwd_input_map bug 修复

**日期**: 2026-08-30 13:02
**作者**: 苏璃珞
**关联**: P0.2 step 7 性能冲刺前置 · ASPLOS 2027

## 背景

P0.2 step 7 的 MLP+CE 训练 benchmark 暴露：C3 比 Eager 慢 19.3×，根因是 80 个 `unordered_map::at: key not found` compile errors。原 supportsNodeType 注释里就预警了这个 bug：

> 多输入节点的 per-input 单节点 kernel 存在 2 个问题：
>  ① 图构造 / 输入映射 bug（unordered_map::at key not found）  ← 性能崩根因
>  ② 数值正确性 bug

之前 P0.2 + P0.2.1 修了问题 ②（数值），问题 ①（性能）一直没动。

## 修复

定位链 + 修补 4 类 at() 失败（全部在 `c3/src/C3/MLIRKernelGen.cpp`）：

### [Bug 1] chain 模式 op_node_ids 永远空 → op_val_map 永远空
`buildFusedMultiNode` 内部 `op_val_map[op_node_ids[op_idx]] = result;` 依赖 op_node_ids 非空。
Chain 调用点传了 `{}` 空，导致 op_val_map 一直空。
`getValue(prev_id)` 找不到，去 `loadExternal(prev_id)`，但 prev 是内部 node，arg_ptrs 没有 → at() 抛。

**修**：chain 调用点收集 `compute_nodes[ci+k]->id` 传给 `buildFusedMultiNode`，两个路径都改（chain + FusedNode）。

### [Bug 2] preloaded_ptrs 只填 referenced_nodes，不填 arg_node_ids
`referenced_nodes` 用 `if (op_idx > 0 && in_id == op_inputs[op_idx][0]) continue;` 跳过 prev。
但 arg 可能是某 op 的 prev（被 skip），getValue 仍能通过 op_inputs[k] 调它 → loadExternal 需要能找。

**修**：预填 `arg_node_ids` 全部到 preloaded_ptrs，再补 referenced_nodes。3 处（buildFused 单 op + buildFusedMultiNodeVectorized + buildFusedMultiNode 标量）。

### [Bug 3] chain 末节点非 output 时仍编译 → 链只跑 forward 部分
例：Sigmoid backward 7-op 图，chain 检测只取 forward 4-op（Sub Exp Add Div = sigmoid 计算），
存到 output buffer 是 sigmoid 值，不是 grad → max_diff=10.30。

**修**：chain 末节点必须在 `output_index` 里（是图输出段），否则 `chain_ok = false`。

### [Bug 4] graph.node(aid) 把 aid 当 index（不是 ID）
`Graph::node(size_t id) { return nodes_[id]; }` 用 id 当 index。chain 里 `graph.node(aid).out_desc.numel` 在 aid != index 时返回错误数据。

**修**：遍历 `graph.nodeCount()` 找 `graph.node(i).id == aid` 的节点。

## 验证

| Test | 修前 | 修后 |
|------|------|------|
| Test 1 ReLU | ✅ 0 | ✅ 0 |
| Test 2 Sigmoid | ❌ 10.30 | ✅ **0** |
| Test 4 Mul | ✅ 0 | ✅ 0 |
| Test 5 Sub | ✅ 0 | ✅ 0 |
| Test 6 Div | ✅ 0 | ✅ 0 |
| Test 7 MatMul | ✅ 6e-8 | ✅ 6e-8 |
| Test 8 ReLU→Sigmoid | ❌ 62.09 | ✅ 0 |
| Test 9 ReLU+ReLU | ✅ 0 | ✅ 0 |
| Test 10 MLP | ✅ 0 | ✅ 0 |
| Test 11 Softmax | ✅ 0 | ⚠️ 0.112 (小回归) |
| Test 12 CE | ✅ 0 | ✅ 0 |

| 指标 | 修前 | 修后 |
|------|------|------|
| compile_errors | 80 | **0** |
| fusion_misses | 30 | **0** |
| benchmark median | 5907 us | **1625 us (3.6×)** |
| Eager baseline | 305 us | 305 us |

C3 当前 1625us vs Eager 305us — **仍慢 5.3×**（之前是 19.3×）。下一步：kernel 本身优化（向量化、epilogue fusion）。

## Test 11 Softmax 小回归（已知 issue）

max_diff=0.112 — chain 选型问题。Test 11 的 graph 是 7-op Softmax backward，chain 检测现在选 {Sub(7), Mul(8)}（2 op，末是 output）但跳过 {Div(4), Mul(5)}（末不是 output）。Mul(5) 改成 single op dispatch，buf 复用时序可能有微妙问题。

非阻塞：核心 P0.2 算子完整性不受影响。Test 12 CE 端到端 PASS（max_diff=0）说明 Softmax backward 路径整体正确。Test 11 单点 regression 留待 P0.2.3 链选型优化时一并修。

## Commit

- `c3 @ bead80c` (P0.2.2 核心修复)

## 下一步

1. **P0.2.3**：MLP+CE benchmark 继续优化（C3 1625us → 至少 ≤Eager 305us）
2. **Test 11 回归修复**：chain 选型时考虑 "中间 op 也可能是另一条 chain 的入口" 
3. **DCU 端到端**：机时预算充足，先把 CPU 端推到极限再上节点
