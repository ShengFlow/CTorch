# P0.2.1: Shape-based broadcast 修复（实测 PASS）

**日期**: 2026-08-30 12:25
**作者**: 苏璃珞
**关联**: P0.2 step 5/6 阻塞解除 + ASPLOS 2027 · 训练关键路径

## 问题

现有 broadcast 用 **numel-based** `idx % node_numel`，对 numel=1（标量）或 numel==n（全尺寸）正确，
但对**部分广播**（`[M, 1] → [M, N]`，M=4, N=8 这种非平凡尺寸）会返回错位：

```
LHS flat idx = i*N + j,  idx % M = (i*N + j) % M ≠ i  (除非 M|N)
```

Softmax backward 公式 `y * (grad - sum(grad*y, dim, keepdim))` 必须做 `[M, 1] → [M, N]` 广播，
CrossEntropy backward `softmax(logits) - target` 同理。修复前 Softmax 测试 `max_diff = 8782.89` ❌。

## 修复（3 块组合改动）

### 1. `computeBroadcastSourceIdx` —— 按 shape 逐维算

```
for d in 0..rank(out_shape):
  out_stride = product of out_shape[d+1..]    # row-major stride
  in_dim  = padded_in_shape[d]                 # 左边 pad 1 到与 out_shape 同秩
  out_dim = out_shape[d]
  if in_dim == out_dim:  idx_d = (out_idx / out_stride) % out_dim
  elif in_dim == 1:      idx_d = 0             # broadcast
  source_idx += idx_d * in_stride[d]
```

边界：scalar (numel=1) → 直接 0；in_shape == out_shape → 直接 out_idx。

### 2. `buildFusedMultiNode` 接口扩展

新参数 `arg_shapes`（`node_id → shape`）+ `out_shape`。两处调用点（chain + FusedNode）同步传：

```cpp
ew_arg_shapes[aid] = std::vector<int64_t>(in_shape.begin(), in_shape.end());
...
buildFusedMultiNode(..., ew_arg_shapes, ew_out_shape_i64);
```

`loadExternal` lambda 加 shape-based 路径，**老 numel-based 作 fallback**（arg_shapes 缺时）。

### 3. Chain detection 放宽 + 默认开启

原 chain 检测要求 `op[i+1].inputs[0] == op[i].id`（严格线性），但 `Mul(grad, y)` 这类
**链前驱在 inputs[1]** 的图构型会断链。修：

- 检测条件改为 `std::find(op[i+1].inputs, op[i].id) != end()`（任意 input 位置）
- 链构建前 reorder `op_inputs` 把前驱挪到 `[0]`（buildFusedMultiNode 内部依赖此布局）
- `ew_chain_fusion_on` 默认开启（`C3_EW_CHAIN_FUSION=0` 可关）

## 验证

```
test_c3_backward 加 Test 11 (M=4, N=8 故意非平凡尺寸)
修前：Test 11 Softmax max_diff=8782.89  ❌
修后：Test 11 Softmax max_diff=0         ✅
整体：✅ PASS overall_max_diff=5.96e-08
```

Test 11 got 完全等于 ref（按元素 bit-identical）：

```
row 0: [0]=4.62e-9/4.62e-9 [1]=5.23e-9/5.23e-9 ... (完全相等)
row 1: [8]=4.62e-9/4.62e-9 [9]=5.23e-9/5.23e-9 ...
row 2: [16]=4.62e-9/4.62e-9 ...
row 3: [24]=4.62e-9/4.62e-9 ...
```

零回归：Test 1~10 + Test 11 全部通过。

## 改动量

| 文件 | 改动 |
|------|------|
| `c3/src/C3/MLIRKernelGen.cpp` | +135 行 -23 行 |
| `src/tests/standalone/test_c3_backward.cpp` | +74 行 -1 行 |

## 解锁的下游任务

- ✅ **P0.2 step 5 Softmax backward**：现在能跑通（虽然 step 5 commit 时只验证 build PASS）
- ✅ **P0.2 step 6 CrossEntropy backward**：`softmax(logits) - target` 中 `[M, 1] → [M, N]` 广播走通
- ✅ **MIMO region fusion** MLP+CE 端到端训练路径
- ✅ **ASPLOS 2027 论文 C3 dialect opset 章节**："shape-based broadcast" 章节可写

## Commit

- `c3 @ 9a7aa95` (P0.2.1 broadcast 修复主体)
- `main @ c438a46` (Test 11 Softmax test)

## 后续（不阻塞）

- 37 个 "compile threw: unordered_map::at: key not found" 警告（已存在，本次未引入）—— ReLU 单 op 路径遗留，需要独立清理
- 进一步把 broadcast 修复推广到 `getBroadcastMod`（单 op 路径的 c3.add/sub/mul/div 旧 numel-based 兜底）—— 当前测试下未触发，可下个 sprint 清理
- axis 写死 1：行 Softmax/CE only，axis=0 留待后续
