# Session Handoff — 2026-09-06 (sum/mean 断链修复 + LLaMA FFN 反向 MIMO)

> 给下次任何 Agent 的接手指引。详细逐条日志见 `../../STATUS_CONTEXT.md` §4.53-4.59;
> 项目入口/当前状态/红线见 `../../AGENTS.md`(本报告是其最近变更的展开)。

## 本会话完成(按时间序, 均 commit+push)

| 事项 | 主仓 commit | c3 commit | STATUS § |
|---|---|---|---|
| MatMul epilogue 向量化(移除 vector.broadcast) | - | 4b1d459 | §4.53 |
| DEBT-2 降级 superseded(被 MIMO 取代, 不复活) | - | 43d9fbe | §4.54 |
| MNIST 性能画像 + 全量回归矩阵 | ca10477/0da70ae | - | §4.54/4.55 |
| 偷工减料专项审查(3 子代理) + 修复 | 4bff8f7 | c1f7239 | §4.56/4.57 |
| LLaMA-FFN 训练基准 `bench_llama_ffn_train` | cab437e | - | §4.58 |
| **sum()/mean() 反向断链修复** | 99e1fae/dbe6e92/b57a52d | - | §4.58 |
| **FFN 反向 MIMO(无 bias SwiGLU, 单内核 9 输出)** | cae8063 | 12ac4c6 | §4.59 |
| **sum-loss 死分支断链修复(ComputeCore 活跃子图)** | 3085a6b | - | §4.59 |

## 关键 bug / 架构事实(下次别踩)

1. **DotNode 从未实现**: `Tensor::sum()` 注释称经 dot(ones) 自动挂 DotNode, 但仓库无 DotNode
   (仅 Ctools.h 有 op::Dot 枚举) → sum/mean 作 loss 静默不填梯度。已补 SumNode/MeanNode/DimReduceNode。
2. **GraphMerger 不去重外部输入**: 多子图共用张量(grad/x)会各自占一个外部输入 → FFN MIMO 用
   "双输出子图"(buildMMDual) 合并重复引用; 外部输入序必须与编译侧 merge 严格一致。
3. **sum-loss 死分支依赖**: 图里含无关节点(如 sum-loss 下仍注册的 CE 头)会让活跃节点依赖计数
   永远多 1 → ComputeCore 在 backward 入口按"活跃子图"重算 _count(3085a6b)。
4. **纯 eager 对照必须用 `CT_DISABLE_C3` build**(build-eager), 不能用 env `C3_ENABLE_BACKWARD=0`
   (后者 forward 仍走 C3 单 kernel, forward_inputs 会错位 → 假基线)。
5. FFN 反向已天然有 **transpose folding**(MLIRKernelGen 设 cblas tA/tB=112), 无转置物化;
   backward ~120ms 是 6 个 cblas GEMM 硬件时间(~370 GFLOP/s 近 M3 FP32 上限)。

## 验证数据(干净机器)

- MNIST 稳态 epoch ~138-160ms(受热降频 ±15% 波动), acc 97.1421%, loss 0.0985, 数值无回退。
- LLaMA FFN(128×4096×11008): C3 ~180ms/step vs 纯 eager ~190ms; backward MIMO ~8% 快;
  mimo_hit 命中, bw_hit 66→16。梯度与 numpy 逐位一致。
- 全量回归绿: graph 115 / backward max_diff=0 / sum_mean_grad 18 / compile_merged 10 /
  merged_pgo 11 / mnist_step / fused_bw / grad_accum / pgo_deopt 7 / compile_error 11。

## 下一步(详见 AGENTS.md)

1. batched GEMM 合并压 FFN backward(唯一结构性空间)。
2. pre-existing standalone 红: test_relu_backward(MPS 崩溃)、test_region_fusion(性能退化类)。
3. DCU 节点 + x86 AVX-512 实测。4. forward 优化 + RC2 进程级异步(c3d)。

## 附: 新测试 / bench 用法(见 AGENTS.md「构建 & 测试」)

- `test_sum_mean_grad`: sum/mean/dim/dims/DotNode 断链回归(18 断言)
- `bench_llama_ffn_train 128 4096 11008 N`: FFN 训练 MIMO; 对跑 build-eager 同参数取 c3 vs eager
- env: `FFN_CBLAS_PROBE`/`C3_FFN_DUMP`/`FFN_LOSS_SUM`/`FFN_DUMP_GRAD`
