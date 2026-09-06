# CTorch Agent Context

> AI agent onboarding doc for **CTorch** — 笙歌/ShengFlow 团队的轻量级 C++ 深度学习框架。
> Last updated: 2026-09-06 (post PEL25 Stage 5)

## 项目一句话

轻量级 C++ 深度学习框架, 类 PyTorch 接口, 核心是 **C3 JIT 编译器** (MLIR → LLVM IR → ExecutionEngine) + 区域融合 (region fusion) + 多后端 kernel (CPU-BASIC / CPU-SIMD / AMX / MPS)。

## 当前状态 (2026-09-06)

| Stage | 状态 | 关键交付 |
|-------|------|----------|
| **Stage 1-4** | ✅ DONE | SwiGLU/SiLU 算子开发 (PEL25 §6 协议完整闭环) |
| **Stage 5.1** | ✅ DONE | `Tensor::silu()/swiglu()` 走 C3 dispatch, 端到端 1.37-1.48x |
| **Stage 5.2** | ✅ DONE | Region fusion `MatMul+Add+SiLU` (3-op) + `MatMul+SiLU` (2-op) 装上 |
| **Stage 5.3+** | 🟡 PENDING | LLaMA-1B SwiGLU FFN 端到端 bench, DCU 节点验证, CGO 2027 论文 |

**已完成算子数**: 30 个 (kCount=30, Ctools.h:178-221), 包含 8 个 SIMD 真向量化 + 6 个 region fusion pattern (MatMul+Add+ReLU/Sigmoid/SiLU × 2/3-op)。

**当前测试**: test_swiglu 3208/3208 PASS。

## 关键路径速查

| 关注点 | 路径 |
|--------|------|
| op 枚举 (30 个) | `include/Ctools.h:178-221` |
| op 静态断言 | `include/CtorchScheduler.h:229-230` (`kCount==30`) |
| Region candidate 白名单 | `include/CtorchScheduler.h:34-36` (5 pattern: MatMul/Add/ReLU/Sigmoid/SiLU) |
| dispatch 表 | `src/CtorchScheduler.cpp:99, 112, 134, 158, 985` |
| Eager API 入口 | `include/Tensor.h` (~1380 行) |
| AutoGrad dispatch 模板 | `include/AutoGrad.h:113-229` (单/双输入 if constexpr 派发) |
| C3 Engine (MLIR→LLVM) | `c3/src/C3/C3Engine.cpp` |
| C3 region fusion registry | `c3/src/C3/RegionFusionRegistry.cpp` (237 行) |
| C3 region pattern 触发 | `c3/include/C3/C3HotPathManager.h:529-660` (`tryFuseRecentDispatches`) |
| Linalg fused IR gen | `c3/src/C3/LinalgFusedGen.cpp` (SiLU/ReLU/Sigmoid 等 fused body) |
| SIMD 真向量化 | `include/kernels/SIMDMath.h` + `src/kernels/CPU-SIMD/SIMDMath.cpp` |
| Backward graph 捕获 | `c3/src/C3/C3BackwardCapture.cpp` |
| 新算子协议 | `PEL25 §6` + 文档沉淀 → `/Users/ghostface/skills/prompts/new-module-prompt.md` |

## 构建 & 测试

```bash
# 构建
cd /Users/ghostface/CTorch-optimize-AutoDiff/build
cmake --build . --target test_swiglu -j 4  # 或 ctest -j 4 跑全部

# 跑单个 test
./test_swiglu  # 3208 断言 (SiLU/SwiGLU 5+3+1+1 测试)

# 跑全部 (ctest 50+ test)
ctest --output-on-failure -j 4

# Bench harness (Stage 4-5 验证用)
./bench_swiglu_simd 1048576 100    # SiLU/SwiGLU kernel-level + e2e
/tmp/bench_region_fusion_silu 200 30  # Region fusion inference (外部 link)
```

**关键开关** (env):
- `C3_DISABLE_HOTPATH=1` 关闭 C3 hotpath 检测
- `C3_DISABLE_REGION_FUSION=1` 关闭 region fusion
- `C3_DISABLE_SINGLE_KERNEL=1` 关闭单 kernel 编译触发

## PEL25 §6 新算子开发协议 (Stage 1-4 沉淀)

**任何新算子必须按以下 7 步走** (PEL25 §6 协议):
1. **接口契约**: `include/Tensor.h` 加 `Tensor::xxx()` 声明 + `include/ops/Xxx.h` 加 Eager API
2. **Eager CPU (BASIC + SIMD)**: `src/ops/Xxx.cpp` + `src/kernels/CPU-{BASIC,SIMD}/Xxx_*.cpp`
3. **Autograd Node**: `include/AutoGrad/Nodes/XxxNode.h` + `.cpp` (4 构造 + 1 backward 虚函数)
4. **op 枚举扩展**: `include/Ctools.h` + `CtorchScheduler.h:229-230` 静态断言更新
5. **C3 Kernel Registry**: 3 个后端 (kCPU/kSIMD/kAMX) dispatch 表注册
6. **MLIR TableGen**: `c3/include/C3/C3Ops.td` (新 op 定义)
7. **Region fusion pattern** (可选): LinalgFusedGen.cpp 加白名单 + C3HotPathManager.h 加 checkPattern

**Stage 5 简化的进阶** (5.1 协议):
- `Tensor::xxx()` 走 `AutoGrad::dispatch<op::Xxx>(...)` 模板, 跟 gelu() 模式一致
- 避免手写 registerNode 逻辑, dispatch 模板 if constexpr 自动派发

## 🔴 绝对不要碰的红线 (洛锦 2026-08-13 警告)

| 路径 | 风险 | 备注 |
|------|------|------|
| `c3/include/C3/C3HotPathManager.h:236-240` (`in_autograd` 短路) | 触及训练一致性核心, 改错破 parity 97.18% | **2026-08-13 revert 警告**, 改前必须跟洛锦确认 |
| `include/CtorchScheduler.h:229-230` 静态断言 | op 枚举跟 binary 不一致会 segfault | 改 op 枚举必须同步 static_assert |
| `include/Ctools.h:178-221` op 枚举顺序 | C3 dispatch 表按 op 索引, 顺序错了 runtime 行为乱 | 新 op 永远加在末尾 (GELU 后) |

## 已知未解决问题

| 级别 | 问题 | 触发场景 | 建议 |
|------|------|----------|------|
| **P0** | 无 | - | - |
| **P1** | Region fusion 训练期零命中 (跟 ReLU/Sigmoid 一致) | forward 末 3 = [MM, Add, CE] ≠ [MM, Add, ReLU/SiLU] | 走 scratchpad 切片路线 (PEL25 §6 P1-2), 联合 P1-1 三阶段 JIT |
| **P1** | Stage 5.2 ARM64 NEON fused 加速 0.77x (反直觉) | x86 AVX-512 + DCU 节点预期显著加速 | Stage 5.4 DCU 验证 |
| **P1** | x86 AVX-512 实测未做 | 曙光智算节点机时充足 | Stage 5.4 |
| **P2** | Stage 1 伪 SIMD (8-wide + 标量 exp) 比纯标量慢 -2.4% | anti-pattern, ops/SiLU.cpp 仍保留 | Stage 5.1 dispatch 改后, ops 路径降级 fallback |

## Cross-Project Memory (Agent lessons, 跨项目适用)

append 到 `/Users/ghostface/.minimax/agents/mavis/memory/MEMORY.md` 的 CTorch lessons:
- **2026-08-13**: "C3 region fusion 训练期修复走 multi_node 代码层, 不碰 in_autograd 短路"
- **2026-08-13**: "C3 8.3x forward 退步根因 (误判修正) — 训练期走 Eager bypass, MLIR pipeline 不影响"
- **2026-08-13**: "MiniMax Code 必须通过 launchd plist 拉起, 否则 CDP 9341 没人 listen"
- **2026-09-05**: "PEL 候选 prompt 生成必须 cross-check user/agent memory 硬约束"
- **2026-09-05**: "PEL 启动前必须先 cross-check 种子 prompt 本身"

## 报告路径 (PEL25 阶段产物)

```
/Users/ghostface/skills/work/reports/2026-09-05/
  prompt-evolution-summary-PEL23-25.md    # 3 轮 PEL 总结
  pel{23,24,25}-candidate-Seed.md        # Seed 评测
  pel{23,24,25}-candidate-MUT-A/B/C.md   # MUT 候选评测

/Users/ghostface/skills/work/reports/2026-09-06/
  swiglu-stage4-report.md      # Stage 4 真 SIMD (1.52-1.56x)
  swiglu-stage5-report.md      # Stage 5.1+5.2 dispatch + region fusion

/Users/ghostface/skills/memories/2026-09-05/
  prompt-evolution-failures-pel{23,24,25}.md

/Users/ghostface/skills/prompts/
  performance-optimization-prompt.md  # PEL23 沉淀 + §13
  compiler-flags-prompt.md            # PEL24 沉淀 + §5.8/§12
  new-module-prompt.md                # PEL25 沉淀 + §6+§7
```

## Quick reference: 给 agent 的一条精简 workflow

```bash
# 新会话开头:
1. cat ~/skills/main.md  # 洛锦的 AGI 总纲
2. cd /Users/ghostface/CTorch-optimize-AutoDiff
3. git log --oneline -20  # 看最新 commit (洛锦 9bc3361 = C3 8.3x 退步根因 commit)
4. ls build/  # 看 test_* 可跑 binary
5. 跟洛锦确认 scope + 决策门

# 跑测试 sanity:
cd build && cmake --build . --target test_swiglu -j 4 && ./test_swiglu
```

## Test 矩阵 (跑这些保平安)

| 关注点 | 测试 target | 备注 |
|--------|-------------|------|
| SwiGLU/SiLU (Stage 5 核心) | `test_swiglu` | 3208 断言, 必过 |
| GELU (孪生兄弟, dispatch 模式) | `test_gelu` | 改动 if constexpr 必跑 |
| Autograd 通用 | `test_autograd_issues` `test_autograd_v2` | dispatch 模板改动必跑 |
| C3 region fusion | `test_c3_graph` `test_graph_merger` | LinalgFusedGen 改动必跑 |
| C3 compile pipeline | `test_c3_compile_and_inject` `test_c3_compile_merged` | region fusion pattern 改动必跑 |
| Elementwise linalg IR | `test_linalg_elementwise` | Stage 2 #8 LinalgElementwiseGen 改 SiLU 必跑 |
| Inplace op | `test_unary_inplace` | SiLU_BASIC_inplace 必跑 |
| Kernel hot swap | `test_kernel_hot_swap` | dispatch 表改动必跑 |
| 端到端 | `MnistTest` `test_mnist_perf` | 性能回归 baseline |
