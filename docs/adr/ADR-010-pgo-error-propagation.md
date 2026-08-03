# ADR-010: PGO 编译错误传播到 C3Engine 全局状态

| 字段 | 值 |
|---|---|
| 状态 | Accepted |
| 日期 | 2026-08-03 |
| 作者 | CTorch Agent（苏璃珞） |
| 关联 commit | (pending push) |
| 优先级 | P1 |
| 关联 | ADR-007（getLastCompileError 初版） |

---

## 1. 背景（Context）

ADR-007 引入了 `C3Engine::getLastCompileError()`，让调用方可以查询最近的编译失败信息。它覆盖了：
- `compile()` 主路径
- `compileAsync()` 后台编译
- `compileMerged()` / `compileMergedPGO()` 链接规格错误
- async-merge 路径

**但是有一个 P1 缺口**：PGO O2/Ofast 编译失败时，错误记录在 `PGOCompiledKernel::last_compile_error_`（per-kernel 字段），**没有回传到 C3Engine 的全局 last_compile_error_**。

这意味着调用方：
```cpp
auto kernel = engine.compileMergedPGO(...);
kernel->execute({x, w1, b1, ...});
// O2/Ofast 编译失败，但调用方不知道，除非保留 kernel 指针
// 调 engine.getLastCompileError() 仍然返回空字符串
```

这违反了 ADR-007 的设计初衷："所有编译路径错误都可通过统一 API 查询"。

---

## 2. 决策（Decision）

**两步修复**：

### 2.1 在 C3Engine 添加 `recordCompileError(prefix, err)` 公开 API

```cpp
// C3Engine.h
void recordCompileError(const std::string& prefix, const std::string& err);
```

这是 PGOCompiledKernel 等子系统写入全局错误状态的 callback hook。内部复用现有的 `recordEngineError` 实现（含 1KB 截断 + 日志）。

### 2.2 在 PGOCompiledKernel::recordCompileError 回调 engine

```cpp
// PGOManager.cpp
void PGOCompiledKernel::recordCompileError(const char* tier, const std::string& reason) {
    // ... 原有 per-kernel 记录逻辑 ...

    // 新增：回传到 C3Engine
    try {
        engine_.recordCompileError(tier, truncated_reason);
    } catch (...) {
        // 双保险吞错：避免回调自身抛异常搞砸 PGO 流程
    }
}
```

**关键设计点**：
- **try-catch 吞错**：`recordCompileError` 理论上不会抛（只有 mutex 锁），但 PGO 路径是热路径，不能因为观测 API 把执行流搞乱
- **latest-wins 语义**：engine 字段被覆盖，但 PGO 编译链中 O2 失败 → Ofast 通常也不会再尝试，所以 latest-wins 合理
- **per-kernel 字段保留**：调用方如果有 kernel 指针，仍然可以查 `kernel.lastCompileError()`，新 API 只是提供**统一观察点**

---

## 3. 测试覆盖

新增 3 个测试到 [test_c3_compile_error.cpp](../../src/tests/standalone/test_c3_compile_error.cpp)：

| 测试 | 验证内容 |
|------|---------|
| **Test 7** | PGOCompiledKernel::recordCompileError 会同步更新 C3Engine::getLastCompileError() |
| **Test 8a** | `recordCompileError(prefix, err)` 公开 API 可独立调用并被 getLastCompileError() 读到 |
| **Test 8b** | 长错误信息（>1KB）正确截断 |

### 测试结果

```
=== test_c3_compile_error ===
  PASS [1]: 编译成功，getLastCompileError() 为空
  PASS [3]: 异步编译成功，getLastCompileError 为空
  PASS [4]: PGOCompiledKernel::lastCompileError() 初始为空
  PASS [4]: clearLastCompileError() 重置成功
  PASS [5]: C3Engine::clearLastCompileError() 初始为空
  PASS [5]: 成功 compile 后 getLastCompileError 仍为空
  PASS [6]: PGO 编译链触发后，lastCompileError 为空（编译成功）
  PASS [7]: PGO 编译错误已传播到 C3Engine 全局状态
  PASS [8]: recordCompileError 公开 API 工作正常
  PASS [8]: 长错误信息截断正常 (size=1060)
=== 总计: 11 passed, 0 failed ===
```

---

## 4. 全量回归（无破坏）

| 测试套件 | 结果 |
|---------|------|
| test_c3_compile_and_inject | 4/4 |
| test_c3_compile_merged | 10/10 |
| test_c3_compile_merged_pgo | 11/11 |
| test_c3_compile_error | **11/11** (was 8/8) |
| test_c3_pgo_deopt | 7/7 |
| test_simd_math | 18/18 |
| test_c3_aot_cache | 16/16 |
| **合计** | **77/77** |

---

## 5. 替代方案（Alternatives Considered）

### A. 把 recordEngineError 改成 public，让 PGOCompiledKernel 直接调用

**否决原因**：
- `recordEngineError` 是 `static` 函数 + 需要 `EngineState&` 参数，公开会暴露内部状态结构
- `recordCompileError` 公开 API 隐藏了 EngineState 细节，更干净

### B. 维护一张 per-tier error map（多错误查询）

```cpp
std::unordered_map<std::string, std::string> last_compile_errors_per_tier_;
std::string getLastCompileError(tier) const;
```

**否决原因**：
- 增加 API 复杂度（调用方需要知道有哪些 tier）
- 现实场景："最近一次失败"足够诊断
- latest-wins 语义已能满足 99% 用例

### C. PGOCompiledKernel 移除自己的 lastCompileError 字段

**否决原因**：
- per-kernel 字段仍有价值：调用方可能在不同 kernel 上调 execute()，需要 per-kernel 状态
- 两个字段**不冲突**，global 是 latest-wins 聚合视图

---

## 6. 后续工作

- [ ] 在 `PGOManager::promoteAll()` 中也加 recordCompileError（如果批量升级时某些 kernel 失败）
- [ ] 考虑给 recordCompileError 加 timestamp 字段（`engine.getLastCompileError() -> {timestamp, msg}`），便于诊断
- [ ] 在 MLIRKernelGen 失败时也加 recordCompileError（目前 MLIR 错误可能直接抛异常走到 C3Engine 的 catch block）

---

## 7. 决策记录

- 接受：两步修复（公开 API + PGO 回调）
- 拒绝：暴露 EngineState、per-tier 错误 map、移除 per-kernel 字段
- 后续：补 promoteAll / MLIR 路径、统一 timestamp API
