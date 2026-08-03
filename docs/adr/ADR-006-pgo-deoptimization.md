# ADR-006: PGOCompiledKernel 运行时失败自动 Deoptimization（O2/Ofast crash → Eager fallback）

> **状态**：Accepted
> **日期**：2026-08-03
> **作者**：CTorch Agent（苏璃珞）
> **决策者**：CTorch Agent + 用户
> **关联**：Phase C PoC（deopt-poc-design.md），PGO 三层异步编译流水线
> **替代方案**：保持 silent fail（拒绝）/ 进程崩溃（拒绝）/ 仅编译失败时 deopt（部分接受）

---

## 1. 背景（Context）

### 1.1 问题

[C3 完整技术报告 v0.3.0 §6.1](file:///Users/ghostface/CTorch-optimize-AutoDiff/c3-complete-report.pdf) 明确列出 4 种多级熔断机制：

> "若某类表达式连续编译失败，系统自动熔断，强制走回退路径，并标记该表达式避免重复尝试。"

但当前 [PGOCompiledKernel::execute](file:///Users/ghostface/CTorch-optimize-AutoDiff/src/C3/PGOManager.cpp#L109-L143) **只处理编译时失败**，对**运行时失败**直接崩溃：

```cpp
// 现状（deopt 改造前）
if (auto k = ofast_kernel_) {
    auto result = k->execute(inputs);  // ← 抛异常 = 主进程崩溃
    return result;
}
```

**风险**：
- MLIR backend 实验性 pass 在边缘 input 下崩溃 → 主进程无 fallback
- Ofast kernel 在某种 shape 下越界 → 段错误，无任何 trace
- 编译时无 fail-fast（编译能过 = 默认运行时 OK），但用户实际上没机会验证

### 1.2 跨域经验

**V8 TurboFan / HotSpot C2**（[现代编译器调研报告](file:///Users/ghostface/skills/reports/2026-08-03/compiler-tech-survey-2026.md)）：
- 高优化代码假设（hidden class、loop trip count）被违反 → deopt 到低 tier
- 真实工作负载 deopt 率 1-3% 是健康范围
- **deopt 不可怕，可怕的是 deopt 之后继续崩**

**HyPer morsel-level 决策**：
- 每个 morsel 独立决定编译/解释
- 失败兜底：解释执行

**Gandiva 表达式编译**（Dremio）：
- 编译失败时 fallback 到解释（无 deopt，无 disable）
- 缺点：每次都重新尝试编译（浪费资源）

### 1.3 触发历史

- **2026-08-02 Phase B**：PGOCompiledKernel 首次实现，编译失败只 log warn + 设 kernel=nullptr（**部分 deopt 隐式存在**）
- **2026-08-03 17:00**：报告指出 deopt 仅在编译时生效，运行时无 deopt
- **2026-08-03 17:30**：[deopt-poc-design.md](file:///Users/ghostface/skills/work/sessions/2026-08-03/deopt-poc-design.md) 决策：实现运行时 deopt
- **2026-08-03 18:00**：commit 实现 + 7/7 PoC 测试通过

---

## 2. 决策（Decision）

PGOCompiledKernel 三层（Ofast / O2 / Eager）执行时**加 try-catch + disable + 统计**：

```
Ofast 抛异常 → recordDeopt("ofast", e.what()) → ofast_disabled_ = true → fall through
O2 抛异常   → recordDeopt("o2", e.what())    → o2_disabled_ = true     → fall through
Eager 永不 deopt（最后兜底，必须永远成功）
```

### 2.1 核心实现（[PGOManager.cpp §execute](file:///Users/ghostface/CTorch-optimize-AutoDiff/src/C3/PGOManager.cpp#L109-L167)）

```cpp
if (auto k = ofast_kernel_) {
    if (!ofast_disabled_.load(memory_order_acquire)) {
        try {
            auto result = k->execute(inputs);
            return result;  // 成功路径
        } catch (const std::exception& e) {
            recordDeopt("ofast", e.what());
        } catch (...) {
            recordDeopt("ofast", "unknown exception");
        }
        ofast_disabled_.store(true, memory_order_release);
    }
}
```

### 2.2 公开 API（[PGOManager.h](file:///Users/ghostface/CTorch-optimize-AutoDiff/include/C3/PGOManager.h#L122-L143)）

| API | 用途 | 调用方 |
|---|---|---|
| `deoptCount()` | 总 deopt 次数（监控指标） | 性能监控 / SLA 报表 |
| `lastDeoptReason()` | 最近一次 deopt 原因（含 tier 标签） | 调试 / 用户日志 |
| `isOfastDisabled()` | Ofast 是否已被永久 deopt | 用户决策"是否需要 clearCache" |
| `isO2Disabled()` | O2 是否已被永久 deopt | 同上 |

### 2.3 Test-only 注入点

为支持 fuzz / chaos 测试，引入 [PGOCompiledKernelTestAccess](file:///Users/ghostface/CTorch-optimize-AutoDiff/src/tests/standalone/test_c3_pgo_deopt.cpp#L99-L117)：

- `friend class PGOCompiledKernelTestAccess;`（在 [PGOManager.h](file:///Users/ghostface/CTorch-optimize-AutoDiff/include/C3/PGOManager.h#L85-L88)）
- `setO2Kernel()` / `setOfastKernel()` 注入 mock kernel

---

## 3. 替代方案（Alternatives Considered）

### A. 保持 silent fail（拒绝）
- **理由**：用户只能看到"PGO 没工作"，看不到"为什么"
- **违反** C3 报告 §6.1 "连续失败熔断" 原则

### B. 进程崩溃 + 重启（拒绝）
- **理由**：MLIR 实验性 pass 经常崩，但底层 Eager 是好的
- **违反** C3 报告 §2.3 "崩溃隔离" 原则

### C. try-catch + disable + 统计（**采用**）
- 每次 deopt 都记日志，可观察
- disable 后不重试，避免雪崩
- 兜底 Eager 永远成功，程序稳定

### D. 指数退避重试（拒绝）
- 理由：若 Ofast 永远崩，重试浪费资源
- disable + 一次性 deopt 即可

---

## 4. 后果（Consequences）

### 4.1 正面

- **可观察性**：用户能看到 deopt 次数、原因、哪个 tier 失败
- **自动恢复**：Eager 永远兜底，程序不会崩
- **避免雪崩**：disable 防止后续重试崩溃 kernel
- **报告 §6.1 部分对齐**：实现"连续失败熔断"的运行时部分

### 4.2 负面

- **try-catch 性能开销**：成功路径几乎零开销（现代编译器能优化）
- **disable 后无法自动恢复**：需用户主动 `clearCache()` 重新编译
- **段错误无法 catch**：kernel 段错误仍会 crash（属上游 bug，应修而非 deopt）

### 4.3 与 P1 缺陷的关系

本次实现**同时部分解决 P1-2（getLastCompileError）**：
- `lastDeoptReason()` 本质就是"最近的运行时错误"
- 未来 `getLastCompileError()` 可整合 `lastDeoptReason + 编译失败 log`

---

## 5. 不做的事（Out of Scope）

- ❌ **OSR**（on-stack replacement）— 太复杂，PoC 范围外
- ❌ **Deopt recovery**（自动重新尝试编译）— 用户需主动 clearCache
- ❌ **修改 compileO2/compileOfast 编译失败逻辑**— 已正确，只补运行时

---

## 6. 验证（Validation）

| 测试 | 状态 | 覆盖场景 |
|---|---|---|
| [test_c3_pgo_deopt.cpp](file:///Users/ghostface/CTorch-optimize-AutoDiff/src/tests/standalone/test_c3_pgo_deopt.cpp) | 7/7 ✅ | 基线 / Ofast crash / O2 crash / disable 不重试 / reason 完整 |
| [test_c3_compile_merged_pgo.cpp](file:///Users/ghostface/CTorch-optimize-AutoDiff/src/tests/standalone/test_c3_compile_merged_pgo.cpp) | 11/11 ✅ | 既有 PGO 流程不回归 |

---

## 7. 失败模式与缓解

| 失败模式 | 缓解 |
|---|---|
| Ofast 抛异常 | 降级到 O2（若 O2 也 disabled 则 Eager） |
| O2 抛异常 | 降级到 Eager |
| Eager 抛异常 | **预期不会发生**，但若发生主进程仍崩（无第五层） |
| Kernel 段错误 | 不可 catch，**应在上游修复**（PGO / MLIR pass bug） |
| 用户想知道"现在跑的是哪一层" | 通过 `o2Kernel() != nullptr` / `ofastKernel() != nullptr` + `isXxxDisabled()` 推断 |

---

## 8. 跨域映射（Cross-Domain Mapping）

| 概念 | V8 | HotSpot | C3 PGO |
|---|---|---|---|
| 高 tier | TurboFan | C2 | Ofast |
| 低 tier | Ignition | Interpreter | Eager |
| Deopt 触发 | hidden class 改变 | 假设失败 | kernel execute 抛异常 |
| Disable | 不禁用，下次重新编译 | 不禁用 | **禁用**（保守策略） |
| 统计 | IC / deopt count | perf counter | `deopt_count_` + `lastDeoptReason()` |

**C3 选择 disable** 而非"重新编译"，因为：
- MLIR pass 失败很难自动恢复（往往要修代码）
- 重试浪费编译资源（PGO heat score 已高）
- 用户可主动 `clearCache()` 触发重编

---

## 9. 未来工作（Future Work）

1. **compile-time deopt**（编译时也加 disable 计数）— 当前编译失败只 log，未统计
2. **deopt reason classification**（按异常类型分类：device mismatch / shape mismatch / MLIR bug）— 便于自动诊断
3. **per-tier canary 切换**（1% 调用走 Ofast，99% 走 O2，监测 deopt 率）— 实现"金丝雀发布"
4. **getLastCompileError() API**（P1-2 缺陷）— 整合 deopt reason + compile log

---

**最后更新**：2026-08-03
**状态**：Accepted，代码 + 测试已落地
