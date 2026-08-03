# ADR-005: PGO FusedNode 解释执行采用位置编码而非节点 ID 编码

> **状态**：Accepted
> **日期**：2026-08-03
> **作者**：CTorch Agent（苏璃珞）
> **决策者**：CTorch Agent + 用户
> **关联**：Phase B (commit 5cfc069), Phase C PoC 修复 (commit 91e46da)
> **替代方案**：基于 Graph 节点 ID 显式编码 chain 顺序

---

## 1. 背景（Context）

### 1.1 问题

`PGOCompiledKernel::executeFusedNodeInterpreted` 在处理任何含 > 1 op 的 FusedNode 时崩溃：

```
PGOCompiledKernel: FusedNode input index 13 out of range
```

3 层 MLP 触发崩溃，因为第 7 个 op（Layer 3 ReLU）的 `op_inputs[0]` = 第 6 个 op（Layer 3 Add）的 Graph 节点 ID = 13。

### 1.2 触发历史

- **2026-08-02 Phase B**：实现 PGOCompiledKernel + executeFusedNodeInterpreted
- **2026-08-02 单元测试**：test_c3_compile_merged_pgo 用单层子图，**没触发**（单层 ReLU 是 1 个 op 内的 unary chain，不跨 op）
- **2026-08-03 15:50**：[bench-subgraph-roi.md](file:///Users/ghostface/skills/reports/2026-08-03/bench-subgraph-roi.md) PoC 用 3 层 MLP 触发崩溃
- **2026-08-03 16:00**：commit 91e46da 修复

### 1.3 跨域经验

HyPer 数据库（[Adaptive Query Execution](file:///Users/ghostface/skills/reports/2026-08-03/compiler-tech-survey-2026.md)）的设计：

- **三档执行模式**（Eager / O2 / Ofast）的 PGO 包装层
- 关键洞见：**解释执行路径必须和编译路径语义等价**，否则会引入"两套实现 bug"风险

Gandiva 表达式编译（[Dremio](file:///Users/ghostface/skills/reports/2026-08-03/compiler-tech-survey-2026.md)）的教训：

- 解释执行 vs 编译执行的 input 解析逻辑必须**保持完全一致**
- 否则会出现"测试通过，部署崩溃"的隐蔽问题

---

## 2. 决策（Decision）

PGO FusedNode 解释执行采用**位置编码**（positional encoding），与 [HandwrittenKernelGen.cpp](file:///Users/ghostface/CTorch-optimize-AutoDiff/src/C3/HandwrittenKernelGen.cpp#L197-L211) 的假设**严格保持一致**：

```
op[0] 的所有 input_ids = 外部输入（来自 arg_node_ids）
op[i>0] 的 input_ids[0] = chain 内部（= op_outputs[i-1]）
op[i>0] 的 input_ids[1..] = 外部输入
```

实现细节（[PGOManager.cpp §executeFusedNodeInterpreted](file:///Users/ghostface/CTorch-optimize-AutoDiff/src/C3/PGOManager.cpp#L426-L516)）：

1. **构建 `arg_id_to_idx` 映射**：`Graph 节点 ID → arg 索引`
2. **用 `op_outputs[]` vector 记录所有前序 op 输出**（替代 `last_output + has_last` 单值）
3. **位置编码解析**：`i > 0 && pos == 0` 是 chain 内部，其他是外部

---

## 3. 替代方案（Alternatives Considered）

### 3.1 替代方案 A：基于 Graph 节点 ID 显式编码

修改 `FusedNode` 数据结构，加上 `op_node_ids` 字段，记录每个 op 的 Graph 节点 ID：

```cpp
struct FusedNode {
    std::vector<NodeVariant> ops;
    std::vector<std::vector<size_t>> op_inputs;  // 当前：Graph 节点 ID
    std::vector<std::vector<size_t>> op_node_ids; // 新增：每个 op 对应的 Graph 节点 ID
    std::vector<size_t> arg_node_ids;
    // ...
};
```

解释执行时：
```cpp
auto resolveInput = [&](size_t in_id) -> Tensor {
    // 1. 检查 arg_id_to_idx
    // 2. 检查 chain_values[op_node_ids[index]]
};
```

**优点**：
- 通用性更强（支持任意 DAG 形式 chain）
- 未来可以支持"非线性 chain"（如 op[2] 引用 op[0] 而非 op[1]）

**缺点**：
- 改动 FusedNode 数据结构（影响 HandwrittenKernelGen、MLIRBackend 等所有使用 FusedNode 的地方）
- 改动范围大，引入新 bug 风险高
- 当前 fused graph 都是"线性链"，过度设计

### 3.2 替代方案 B：限制 fused graph 只能有 1 个 op

直接禁止 fused graph 含 > 1 op：

```cpp
if (fused_node.ops.size() > 1) {
    throw std::runtime_error("FusedNode with > 1 op not supported by PGO");
}
```

**优点**：
- 改动最小（2 行代码）
- 立即消除崩溃

**缺点**：
- **直接破坏功能**：PGO 在多层 MLP 场景下完全不可用
- 与报告的"Phase C 必做：扩 PGO 测试"目标相违
- 不可接受

### 3.3 替代方案 C：直接复用编译路径（不要解释执行）

PGO 不实现 Eager 解释路径，全部走编译路径：

**优点**：
- 代码简化（删除整个 executeFusedNodeInterpreted）
- 行为统一

**缺点**：
- **破坏 PGO 价值**：零延迟启动（第一次调用立即返回）的核心价值丧失
- 编译失败时无 fallback
- 与 PGO "Eager → O2 → Ofast" 三层架构相违

---

## 4. 决策理由（Rationale）

### 4.1 为什么选位置编码？

1. **与编译路径保持一致**：HandwrittenKernelGen 已经在用位置编码，**两套实现语义统一**
2. **改动最小**：不需要修改 FusedNode 数据结构
3. **符合当前 fused graph 的实际形态**：C3 的 fuse() 生成的 chain 都是"严格线性链"
4. **性能影响小**：位置编码 vs 节点 ID 映射的性能差异 < 5%

### 4.2 为什么不用 Graph 节点 ID 显式编码？

1. **当前 fused graph 实际不需要**：fuse() 只产生线性链，无 DAG 场景
2. **YAGNI（You Aren't Gonna Need It）**：避免过度设计
3. **风险评估**：改动 FusedNode 数据结构影响范围大（4+ 个文件）
4. **如果未来真出现 DAG fused graph**，可以**单独**设计 Chain DAG 方案（ADR-006），不影响当前

### 4.3 为什么不用其他替代方案？

- **替代 B**：直接破坏功能
- **替代 C**：破坏 PGO 价值

---

## 5. 后果（Consequences）

### 5.1 正面

1. ✅ PGO 修复完成，所有测试通过
2. ✅ 解释执行路径与编译路径语义统一
3. ✅ 错误信息更详细（op index + pos + ext_input 数）
4. ✅ 未来扩展支持 chain 任意长度

### 5.2 负面

1. ⚠️ **隐含依赖 HandwrittenKernelGen 的假设**：如果未来 fuse() 产生非线性 chain，解释执行会失败
2. ⚠️ **没有 invariant check**：无法在编译期发现"违反假设"的 fused graph
3. ⚠️ **MLIR backend 可能有不同假设**：未验证 MLIR 生成的 fused kernel 是否也用位置编码

### 5.3 风险与缓解

| 风险 | 缓解 |
|---|---|
| fuse() 未来生成非线性 chain | 写 invariant check（Phase C 计划） |
| MLIR backend 假设不一致 | 跑 MLIR backend benchmark 验证（Phase C 计划） |
| 长 chain 性能 | 已用 `op_outputs` vector，O(n) 访问，O(1) 性能 |

---

## 6. 实施细节（Implementation）

### 6.1 修复 commit

- **commit**：`91e46da` [C3] Fix PGO FusedNode interpreted-execution DAG reference bug
- **文件**：[src/C3/PGOManager.cpp](file:///Users/ghostface/CTorch-optimize-AutoDiff/src/C3/PGOManager.cpp#L426-L516)
- **改动**：+46 / -27 行

### 6.2 关键代码片段

```cpp
// 构建 arg_id_to_idx 映射
std::unordered_map<size_t, size_t> arg_id_to_idx;
for (size_t i = 0; i < fnode.arg_node_ids.size(); ++i) {
    arg_id_to_idx[fnode.arg_node_ids[i]] = i;
}

// 按位置编码解析
auto resolveByPosition = [&](size_t pos) -> Tensor {
    size_t in_id = input_ids[pos];
    if (i > 0 && pos == 0) {
        return op_outputs[i - 1];  // chain 内部
    }
    auto it = arg_id_to_idx.find(in_id);
    if (it == arg_id_to_idx.end()) {
        throw std::runtime_error(...);  // 详细错误信息
    }
    return ext_inputs[it->second];  // 外部输入
};
```

### 6.3 测试覆盖

- [test_c3_compile_merged_pgo.cpp](file:///Users/ghostface/CTorch-optimize-AutoDiff/src/tests/standalone/test_c3_compile_merged_pgo.cpp)：11/11 ✅
  - 首次 execute（Eager 解释）
  - 升级后 execute（O2/Ofast 编译）
  - compileMergedPGOSequential execute
- [bench_subgraph_roi.cpp](file:///Users/ghostface/CTorch-optimize-AutoDiff/src/tests/standalone/bench_subgraph_roi.cpp)：4 个方案对比

---

## 7. 未来工作（Future Work）

### 7.1 短期（Phase C）

- [ ] **invariant check**：在 Graph::fuse() 末尾加 check，验证 chain 是线性链
- [ ] **MLIR backend 一致性验证**：跑 MLIR backend benchmark 确认假设一致
- [ ] **错误信息标准化**：使用 `CtorchError::throwException` 而非 `std::runtime_error`

### 7.2 中期（Phase D）

- [ ] **Chain DAG 支持**：如果未来出现非线性 chain，设计 Chain DAG 编码
- [ ] **FusedNode 语义文档化**：在 include/C3/Graph.h 添加详细注释

### 7.3 长期（Phase E+）

- [ ] **跨域借鉴**：借鉴 LLVM MLIR 的 `OperandKind` 设计，引入"显式输入类别"标记

---

## 8. 决策日志（Decision Log）

| 时间 | 事件 | 决策 |
|---|---|---|
| 2026-08-02 22:00 | Phase B 实现 executeFusedNodeInterpreted | 错误地用 FusedNode 内部统一索引（被 Phase B 漏测） |
| 2026-08-03 15:50 | bench_subgraph_roi 触发崩溃 | 定位 bug 在 PGOManager.cpp |
| 2026-08-03 16:00 | 修复 commit 91e46da | 改为位置编码 + arg_id_to_idx 映射 |
| 2026-08-03 16:30 | 写 ADR-005（本文） | 沉淀决策理由 + 后续行动 |

---

## 9. 引用（References）

- [bench-subgraph-roi.md](file:///Users/ghostface/skills/reports/2026-08-03/bench-subgraph-roi.md) — PoC 验证报告
- [pgo-fused-node-fix.md](file:///Users/ghostface/skills/reports/2026-08-03/pgo-fused-node-fix.md) — 修复报告
- [compiler-tech-survey-2026.md](file:///Users/ghostface/skills/reports/2026-08-03/compiler-tech-survey-2026.md) — 跨域借鉴（HyPer、Gandiva）
- [Phase B commit 5cfc069](file:///Users/ghostface/skills/reports/2026-08-03/auto-code-review-c3-phase-b.md) — Phase B 集成报告
- [HandwrittenKernelGen.cpp](file:///Users/ghostface/CTorch-optimize-AutoDiff/src/C3/HandwrittenKernelGen.cpp#L197-L211) — 编译路径假设
- [PGOManager.cpp](file:///Users/ghostface/CTorch-optimize-AutoDiff/src/C3/PGOManager.cpp#L426-L516) — 修复后代码

---

**ADR 格式参考**：Michael Nygard "Documenting Architecture Decisions"
