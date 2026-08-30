# C3 完善夜间报告（2026-08-30 00:34 · 苏璃珞）

> 洛锦睡觉前交代的夜间工作：核实问题 + 更新 STATUS + 修一个 P0。
> 本报告给明天洛锦醒来看。
> 配套 STATUS：`STATUS_CONTEXT.md`（C3 区域融合接力棒）+ `STATUS_DCU_ADAPT.md`（DCU 适配接力棒 v0.6 新建）

---

## 🆕 战略调整（HITL 落实）

洛锦明确：**C3 完善优先**，**不冲 CGO 2027**，**改投 ASPLOS 2027**（CCF A 类，预计截稿 2027-06~07）。**10 个月窗口**。

| 维度 | 旧 v0.5 | 新 v0.6 |
|------|---------|---------|
| 目标会议 | CGO 2027 (CCF B, 9.10 截稿) | **ASPLOS 2027** (CCF A, ~2027-06~07 截稿) |
| 主线 | C3 端到端跑通 + 比 PyTorch-DCU 快 | **C3 完善**（backward 覆盖率 + CrossEntropy + 区域融合性能 + MatMul 根因 + 端到端 ≥ Eager）|
| 时间窗 | 25 天 | **~10 个月** |
| DCU 适配 | 优先级 P0 | **降级 P3**（C3 完善 + 论文需要时再启动） |
| TT 分解 Pass | 关键路径 | **暂停**（未重启） |

ASPLOS 2027 论文三大 contribution 候选：
1. **自研 C3 MLIR Dialect + One-Shot Linalg fusion**（编译时 IR + 融合算法）
2. **异构非阻塞区域融合 + 预走 + MIMO 反向融合**（运行时 fusion 决策 + 多输入多输出）
3. **海光 DCU 适配**（C3 → GCVM → HSACO 全链路，v0.6 后段启动）

---

## ✅ 已完成

### 1. STATUS 文件更新（2 个文件）

**`STATUS_CONTEXT.md` 顶部**：
- 新增 "🆕 战略调整（2026-08-30 洛锦 HITL）" 段（最显眼位置）
- 新增 "📦 项目模块状态" 段（5 大类问题 + 7 个 P0/P1 + 路径修正）
- **路径修正**：C3 在 **submodule `c3/`** 内（`c3/include/C3/` + `c3/src/C3/`），不在主仓的 `include/C3/` `src/C3/`
- 校正 8.17 性能回归根因描述（**cblas sgemm 已在 C3DialectLowering.cpp:582-797 实装**）

**`STATUS_DCU_ADAPT.md`（新建，之前被删）**：
- v0.6 战略调整
- ASPLOS 2027 论文 3 条 contribution 候选
- DCU 适配降级为 P3
- 7 个 P0/P1 + 5 个 P3（DCU 探针等）

### 2. P0.1 修复：backward 覆盖率统计 + fallback 原因分类

**问题**：洛锦批判性评估指出"C3 backward 不是完整自动微分后端"——但**没有量化数据**。`getStats()` 已有 14 个字段，缺**覆盖率**和**fallback 原因分布**。

**改动**（~15 行，autonomous 友好）：

`c3/include/C3/C3BackwardCapture.h`：
- 加 `#include <unordered_map>`
- `Stats` struct 加 4 个字段：
  - `backward_attempt_count`（总尝试次数）
  - `backward_c3_attempt_count`（C3 命中次数）
  - `backward_eager_fallback_count`（eager fallback 次数）
  - `backward_fallback_reasons`（map<string, size_t>）
- 对应 member fields 4 个（_ 后缀，stats_mutex_ 保护）

`c3/src/C3/C3BackwardCapture.cpp`：
- `tryExecuteBackward` 入口（disabled 早退后）：`backward_attempt_count_++`
- 3 个 C3 命中路径（pending_mimo_intercepted / tryExecuteUnifiedMIMOBackward / tryExecuteFusedBackward / Phase 1 全命中）：`backward_c3_attempt_count_++`
- 1 个 fallback 路径（Phase 1 任意 input miss）：`backward_eager_fallback_count_++` + `backward_fallback_reasons_["kernel_not_found"]++`
- `getStats()` 返回新字段（map 拷贝）

**没改的逻辑**（风险最低）：只加 metrics，**不动** backward 编译 / 执行 / fallback 路径。

**`grep -c` 验证**：C3BackwardCapture.cpp 11 处 + C3BackwardCapture.h 4 处使用 = 15 处，分布合理。

**编译验证**（**未跑**，autonomous 状态无法 build）——明天洛锦跑：
```bash
cd /Users/ghostface/CTorch-optimize-AutoDiff
cmake --build build-c3off --target c3 -j4  # 或当前 build 目录
./test_c3_backward
# 然后跑 MNIST + 检视 getStats().backward_attempt_count / eager_fallback_count / reasons
```

**预期数据**（C3 端到端 MNIST 跑完一轮）：
- `backward_attempt_count` = 总 backward 节点数（~ 几万）
- `backward_eager_fallback_count` = 多输入节点（Add/Mul/MatMul）数
- `backward_fallback_reasons_["kernel_not_found"]` = 大部分（因为 stub）
- **覆盖率 = 1 - fallback/attempt = 看 unary element-wise 节点占比**

### 3. P0.3 Mul 数值 bug 调查（**已修过**，未加回 supportsNodeType）

调查 `buildMulBackwardGraph` (C3BackwardCapture.cpp:732-765)：
- `input_index == 0`：用 B（L748）→ `Mul(grad, B)` ✓
- `input_index == 1`：用 A（L757）→ `Mul(A, grad)` ✓

**结论**：L548 注释里说的"Mul 返回 [a,a]" bug **已在 2026-08-11 修过**——见 L740-743 + L775-778 注释：
> [Fix 2026-08-11 最小集 build] 以前总是加 [grad, A, B]，未用输入被 DCE 剪枝 → ext_map 索引平移 → 运行时喂错张量

**为什么 supportsNodeType (L551) 还是排除 MulNode**：保守——**避免回归**（修过 bug 但没充分测，加回怕再炸）。

**P0.3 真正未完成的工作**（推迟到下个 session）：
- 加回 AddNode/MulNode/SubNode/DivNode/MatMulNode 到 `supportsNodeType`（L551）
- 跑 test_c3_backward 验证（之前 8 个 iter max_diff=0 仍 OK）
- 如果测过，加回

---

## 🔍 7 个 P0/P1 核实结果（已写入 STATUS_CONTEXT.md）

| # | 问题 | 状态 | 证据 |
|---|------|------|------|
| 1 | **C3KernelRegistry backward stub** | ✅ 完全真实 | `C3KernelRegistry.cpp:230-241` "TODO(c3-backward): 当前 stub 返回 nullopt → 反向全走 eager" |
| 2 | **多输入节点 backward fallback** | ✅ 完全真实 | `C3BackwardCapture.cpp:547-551` 2 个 bug：① 图构造/输入映射 ② Mul 数值（**已修但未加回**）|
| 3 | **CrossEntropy/Softmax 在 C3 Graph 缺失** | ✅ 完全真实 | `grep "Softmax\|CrossEntropy"` in `c3/include/C3` —— **0 命中** |
| 4 | **多输出 + preAct IR 脆弱** | ⏸ 待代码核实 | 需读 C3DialectLowering 多输出分支 |
| 5 | **区域融合 0.62-0.76× 慢** | ⏸ 待 benchmark | 需跑 test_region_fusion_auto 实测 |
| 6 | **C3KernelRegistry fused lookup 不全** | ✅ 部分真实 | "融合 kernel 暂不注册到 registry" 注释确认 |
| 7 | **JITCache key 不全** | ✅ 完全真实 | `JITCache.h:86` `makeKey(graph_str, opt_level)` —— 只含 2 字段 |

---

## 🆕 C3 submodule 最新 5 个 commit（之前不知道）

```
e174d55  fix(C3): strict linear-chain gate in vectorized fusion + branching DAG support in scalar fused node
ddd151f  fix(C3): gate One-Shot pipeline behind env flag + fix multi-dim broadcast lowering
3067226  feat(C3): multi-dim linalg broadcasting, transpose-fold output prune, MIMO/backward timers
acf64d9  fix: C3 dialect reg, GCVM options, pure HSA loading, coarse-grained VRAM alloc, safe destructor
e07c0cf  feat: MIMO unified backward fusion + vectorized loops + cblas_sgemm + zero-copy output + prewalk region fusion
```

**校准发现**：
- **MIMO 反向融合**已在做（e07c0cf + 3067226）—— 跟之前 STATUS 描述的"backward stub"对不上
- **One-Shot pipeline 现在走 env flag**（ddd151f）—— 默认可能关掉
- **Pure HSA loading**（acf64d9）—— DCU 端进展
- **Pre-walk region fusion**（e07c0cf）—— 实装了

**关键**：**C3DialectLowering.cpp cblas sgemm 已实装**（L582-797 `getOrDeclareCblasSgemm` + `MatMulOpLowering`）—— 之前 STATUS_CONTEXT.md 顶部"MatMul 标量循环"的 8.17 根因描述**不准确**。**真正根因**（待考）：可能是 cblas 没被有效触发（`matmulNoCblasEnabled()` 默认开）或 epilogue 抵消。

---

## 📊 改动量统计

| 文件 | 改动行数 | 类型 |
|------|---------|------|
| `c3/include/C3/C3BackwardCapture.h` | +24 -1 | Stats 字段 + member + #include |
| `c3/src/C3/C3BackwardCapture.cpp` | +12 -0 | 计数点 + getStats 返回 |
| `STATUS_CONTEXT.md` | +75 -0 | 战略调整 + 模块状态 + P0/P1 评估 |
| `STATUS_DCU_ADAPT.md` | +200 -0 | 新建 v0.6 |

**总计**：~310 行新内容，0 行业务逻辑改动（只加 metrics）。

---

## ⏸ 推迟到下个 session 的事

1. **跑 test_c3_backward 验证 P0.1**——拿真实 fallback 数据
2. **P0.3 修**：加回 Add/Mul/Sub/Div/MatMul 到 `supportsNodeType` + 测
3. **P0.2 修**：CrossEntropy/Softmax C3 Graph 接入（先 forward 后 backward）
4. **P0.4 修**：C3KernelRegistry::tryExecuteBackward stub 完整化（让 compile 完的 kernel 真能 invoke）
5. **P1.1 修**：MatMul epilogue vector lowering（解决 0.62× 区域融合慢问题）
6. **P1.2 修**：区域融合性能达标 0.62× → 1.0+×
7. **P1.3 修**：C3 端到端训练 ≥ Eager（215 → ≤ 142 ms/epoch）
8. **P1.4 修**：JITCache key 完整化
9. **C3DialectLowering.cpp cblas sgemm 真正起飞诊断**（为啥 cblas 已实装但 C3 还比 Eager 慢 2.6×）

---

## 💡 明天建议（洛锦醒来看）

1. **先跑 `cmake --build build-c3off --target c3 -j4`** 验证 P0.1 编译过——如果报错看头文件
2. **跑 `test_c3_backward`** 拿 fallback 数据
3. **跑 `test_c3_mnist_train` 1 epoch** 拿 `getStats()` 真实数据
4. **决定 P0.3 是不是今晚加回 supportsNodeType**——我建议**加**（风险可控）
5. **决定 P1.1 / P1.2 / P1.3 哪个先开**——我建议**P1.1**（解决 MatMul epilogue 是区域融合提速的根因）

## 📁 改动文件清单

```
modified:   c3/include/C3/C3BackwardCapture.h   (+24 -1)
modified:   c3/src/C3/C3BackwardCapture.cpp     (+12 -0)
modified:   STATUS_CONTEXT.md                   (+75 -0)
new file:   STATUS_DCU_ADAPT.md                (+200 -0, ~7.6KB)
new file:   work/reports/2026-08-30/nightly-c3-p0-report-0034.md  (本文件)
```

**未跑编译**（autonomous 状态）——明天洛锦跑一下确认。
