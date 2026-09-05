# C3 端到端 Forward 瓶颈归因（宽 matmul 密集）

> 日期：2026-09-04 · 测量：bench_wide_mlp_e2e（B=64 H=4096 L=4 纯前向 20 steps）

## 现象
- C3 ON：21.6 ms/step；Eager（cblas）：16.7 ms/step → **C3 慢 ~1.29x**
- C3 融合命中高：fused_hit=75/80（接近全命中），compiles=5

## 归因（非单一 bug，组合因素）
1. **kernel 级 matmul+act 融合是快的**（bench_fusion_scale 实测 sigmoid(XW+B) 1.3-1.5x）——C3 此路为赢点；
2. **但端到端宽 MLP 慢 1.29x**：eager 每层 matmul 走 Apple 优化 cblas（AMX），融合省下的激活 epilogue 收益被巨量 matmul 成本稀释，叠加调度税 → 净亏。
3. LinalgFusedGen 注释明确 matmul 不参与其融合；matmul+act 融合走另一条（MLIRKernelGen）路径。

## 结论
Forward matmul 密集场景是与 Apple cblas 对擂，C3 难赢（eager 优势区，论文已诚实标注）。若优化，需**系统性**：
- ① 压调度税（region 检测/prewalk/dispatch；STATUS_CONTEXT 已做过 23.5→11.2µs 一轮）；
- ② 确保端到端融合 kernel 的 matmul 主体保持 cblas 级效率（勿朴素循环降级）。
ROI 有限，适合 NOIP 后集中优化，非当前优先。
