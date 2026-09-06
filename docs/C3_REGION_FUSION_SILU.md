# C3 Region Fusion for SiLU (PEL25 Stage 5.2)

> Architecture Decision Record: 2026-09-06
> Stage: PEL25 §6 + §7 协议闭环 (SiLU/SwiGLU 算子开发)
> Author: Mavis

## Context

PEL25 Stage 5.2 实施: 给 C3 region fusion 加 `MatMul+Add+SiLU` (3-op) + `MatMul+SiLU` (2-op) 两个 pattern, 跟现有 ReLU/Sigmoid pattern 同模板。

**动机**:
- LLaMA / PaLM 等现代 LLM 的 SwiGLU FFN 核心是 `silu(matmul + bias) * matmul`
- SiLU/SwiGLU 算子 (PEL25 Stage 1-4) 已完成, 但 region fusion pattern 缺失
- Stage 2 #7 已把 SiLU 加到 `isRegionCandidateOp`, 但 `installRegionPattern` 实际没装
- Stage 5.1 dispatch 改后, 用户代码走 `Tensor::silu()` 自动进 region fusion 候选路径

## Decision

**加 2 个新 region fusion pattern** (跟 ReLU/Sigmoid 模板 1:1 对齐):

| Pattern | 文件 | 改动 |
|---------|------|------|
| `MatMul+Add+SiLU` (3-op, LLaMA FFN 风格) | `c3/include/C3/C3HotPathManager.h:559-571` | `checkPattern` 末尾 3 检查 + `submitFusedCompileAsync` |
| `MatMul+SiLU` (2-op, 无 bias) | `c3/include/C3/C3HotPathManager.h:664-693` | 同上 + forward 稳健化找 bias Add 升级 3-op |

**LinalgFusedGen.cpp 加 SiLU 路径** (3 处):
1. `#include "AutoGrad/Nodes/SiLUNode.h"`
2. `SiLUNode` 加白名单 (`std::is_same_v<T, SiLUNode>`)
3. SiLU fused body (跟 Sigmoid 模板一致, 末尾乘 in):
   ```mlir
   %neg = arith.negf %in : f32
   %exp = math.exp %neg : f32
   %denom = arith.addf %one, %exp : f32
   %sig = arith.divf %one, %denom : f32
   %result = arith.mulf %in, %sig : f32  // silu(x) = x * sigmoid(x)
   ```

**SiLU 数学**: `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`, 跟 Sigmoid 共享 exp + addf + divf, 末尾 mulf in 收尾。

## Implementation

### 改动文件清单

| 文件 | 改动行数 | 内容 |
|------|---------|------|
| `c3/src/C3/LinalgFusedGen.cpp` | 1 + 1 + 2 处 = ~25 行 | include + 白名单 + 2 fused body |
| `c3/include/C3/C3HotPathManager.h` | ~50 行 | 2 个新 checkPattern case + forward 稳健化 |

### SiLU fused body (主路径, LinalgFusedGen.cpp:226-235)

```cpp
else if constexpr (std::is_same_v<T, SiLUNode>) {
    auto in = val_map.at(node->inputs[0]);
    mlir::Value neg_x = b.create<mlir::arith::NegFOp>(regionLoc, in);
    mlir::Value exp_x = b.create<mlir::math::ExpOp>(regionLoc, neg_x);
    mlir::Value denom = b.create<mlir::arith::AddFOp>(regionLoc, one_f, exp_x);
    mlir::Value sig = b.create<mlir::arith::DivFOp>(regionLoc, one_f, denom);
    result = b.create<mlir::arith::MulFOp>(regionLoc, in, sig);
}
```

### Region pattern 触发 (C3HotPathManager.h:559-571)

```cpp
// MatMul + Add + SiLU 模式 (PEL25 Stage 5.2: LLaMA FFN 风格)
if (last3_0.op_type == op::MatMul &&
    last3_1.op_type == op::Add &&
    last3_2.op_type == op::SiLU) {
    if (last3_0.shape.size() >= 4 && last3_1.shape.size() >= 2 && last3_2.shape.size() >= 1) {
        size_t M = last3_0.shape[0];
        size_t N = last3_0.shape[3];
        if (last3_1.shape[0] == M && last3_1.shape[1] == N) {
            submitFusedCompileAsync({last3_0, last3_1, last3_2}, dev, cfg, "MatMul+Add+SiLU");
            return true;
        }
    }
}
```

## Verification

### test_swiglu 3208/3208 PASS ✅

Stage 5.2 改动没破 Stage 1-4 任何测试。

### Region fusion inference bench (`bench/bench_region_fusion_silu.cpp`)

- 1 层 MLP: `y = silu(x @ W + b)`, shape [64, 1024, 1024]
- 200 iter, cold window 30, jitter 排除 >50ms
- 结果:
  - MLIR compile jitter: iter 3 = 506.31ms (一次性 region fusion 编译)
  - cold avg (前 30, 排除 jitter, n=29): 0.2287 ms / iter
  - warm avg (后 170, n=170): 0.2964 ms / iter
  - 加速比: 0.77x ⚠️ (ARM64 NEON fused 反而慢, 见 Consequences)

## Consequences

### Positive ✅

- **协议 §6 100% 闭环**: SiLU/SwiGLU 算子从 Eager → 真 SIMD → Dispatch → Region fusion 全链路打通
- **未来新算子可同样复用**: 加 region fusion pattern 只需 2 处改动 (LinalgFusedGen 白名单 + C3HotPathManager checkPattern)
- **Stage 5.1 dispatch 模板扩展 + Stage 5.2 region pattern 扩展 = 顺水推舟**, 没改 in_autograd 短路 (红线)

### Negative ⚠️

- **ARM64 NEON fused 加速 0.77x** (warm 反而慢 30%): MatMul 主导, SiLU elementwise memory-bound, NEON fused linalg 不如手写 SIMD 高效
- **预期收益在 x86 AVX-512 + 大 batch (>= 256)**: fused kernel 节省的中间 buffer alloc 占比小, dispatch 开销明显降低
- **DCU 节点 (x86, 曙光智算机时充足) 实测未做**, 留 Stage 5.4

### Neutral 🟡

- **训练期 region fusion 仍零命中** (跟 ReLU/Sigmoid 一致): 真根因是 `checkPattern` 末尾 3 = [MM, Add, CE] 不匹配, 跟 SiLU 无关
- **解决训练期零命中要走 scratchpad 切片路线** (PEL25 §6 P1-2), 联合 P1-1 三阶段 JIT, **不推荐独立做**

## Related

- `docs/C3_JIT_2.0_Custom_Dialect_Design.md` — C3 region fusion 总体设计
- `docs/adr/ADR-009-vectorized-transcendentals.md` — SIMD 向量化决策
- `include/kernels/SIMDMath.h` — 真 SIMD 底层 (sigmoid256_ps polynomial)
- `/Users/ghostface/skills/work/reports/2026-09-06/swiglu-stage5-report.md` — Stage 5 完整报告
- `/Users/ghostface/skills/prompts/new-module-prompt.md` — PEL25 §6 新算子开发协议 (未来新算子参考)

## Future Work

1. **Stage 5.4**: DCU 节点 x86 AVX-512 验证 Stage 5.1 dispatch + Stage 5.2 region fusion 加速比
2. **Stage 5.5**: SiLU 升级 4-op pattern `MatMul+SiLU+Mul+MatMul` (LLaMA FFN 实际 op 序列)
3. **CGO 2027 论文**: PEL25 §6 协议 + Stage 4-5 验证 → "Elementwise Op Extension + Region Fusion for JIT ML Compilers" 章节
