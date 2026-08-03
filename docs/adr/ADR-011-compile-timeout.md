# ADR-011: C3Engine 异步编译 watchdog 超时熔断

| 字段 | 值 |
|---|---|
| 状态 | Accepted |
| 日期 | 2026-08-03 |
| 作者 | CTorch Agent（苏璃珞） |
| 优先级 | P0 |
| 关联 | ADR-007（getLastCompileError），ADR-010（PGO 错误传播） |

---

## 1. 背景（Context）

### 1.1 问题

`C3Engine::compileAsync()` 在后台线程中执行实际编译，调用方通过 `future.get()` 阻塞等待结果。**当前实现没有超时机制**，如果某个图触发了 MLIR 复杂优化卡死、clang++ 子进程异常、OS 调度问题，调用方会**无限期阻塞**。

**具体风险**：

1. **冷启动卡死**：MLP 多层同时触发 `compileAsync`，某张图触发 MLIR 长时间优化，主线程 `future.get()` 永远不返回
2. **shutdown 永远等**：`C3Engine::shutdown()` 等所有 background compile 完成，但卡死的 task 让 shutdown 永远不返回 → 进程退出卡住
3. **CI 假阳性**：CI 环境的 clang++ 慢机器 + MLIR 复杂优化 → 测试 flaky，开发者误以为是测试 bug
4. **PGO 升级链卡死**：`PGOCompiledKernel::promote()` 触发 O2 → Ofast 异步编译，如果 Ofast 卡死，PGO 永远停在 O2 状态

### 1.2 为什么不能简单 kill 编译线程

clang++ / MLIR 是子进程 + LLVM threadpool 协作，单纯从 C++ 侧无法干净地：
- 取消 LLVM threadpool 中正在执行的 pass
- 通知 clang++ 子进程停止 codegen
- 释放 child process 的临时文件 (.o / .d / .so)

如果直接 `pthread_cancel`，会留下 leak 的临时文件 + 损坏的 .so。

---

## 2. 决策（Decision）

**Watchdog 线程 + soft timeout 熔断**：

1. **新增** `setCompileTimeoutMs(uint32_t)` / `getCompileTimeoutMs()` 配置 API，默认 30000ms（30s）
2. **compileAsync 启动一个 watchdog 线程**，与 compile 线程并行
3. **watchdog 等待 `timeout_ms`**：若 compile 线程在 `timeout_ms` 内未完成，watchdog 判定超时
4. **超时后 watchdog 立即 `promise->set_value(nullptr)`**，主线程 `future.get()` 立即返回 nullptr
5. **compile 线程继续跑**（不取消），跑完后写入 cache 供后续相同 key 命中
6. **记录到 last_compile_error_**：含 `[async-timeout]` 前缀 + cache_key + timeout 数值

### 2.1 关键 API

```cpp
// C3Engine.h
void setCompileTimeoutMs(uint32_t ms);  // 0 = 永不超时（不推荐）
uint32_t getCompileTimeoutMs() const;   // 默认 30000
```

### 2.2 行为矩阵

| timeout_ms | compile 状态 | 用户 future 行为 | last_compile_error_ | cache 写入 |
|---|---|---|---|---|
| 0 | 完成 | 立即返回 kernel | 无 | 正常 |
| 30000 | < 30s 完成 | 立即返回 kernel | 无 | 正常 |
| 30000 | ≥ 30s | **立即 nullptr**（不阻塞） | `async-timeout: compile exceeded 30000ms for cache_key=...` | **仍写入**（后续命中） |
| 10 | 卡死 | 立即 nullptr | `async-timeout: ...` | 仍写入 |

### 2.3 调优建议

| 环境 | 推荐 timeout | 理由 |
|---|---|---|
| 生产 hot path | 10-15s | 快速失败，依赖 cache 复用 |
| 冷启动（多图并发） | 20-30s | 给 MLIR 复杂优化充足余量 |
| 开发/调试 | 60s | 避免被合法大图误杀 |
| 单元测试 | 2-5s | 让 timeout 路径可以快速验证 |

---

## 3. 实现要点

### 3.1 共享状态机

```cpp
struct AsyncCompileState {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;       // compile 线程完成
    bool timed_out = false;  // watchdog 判定超时
    std::shared_ptr<CompiledKernel> kernel;
    std::string error;
};
```

两个原子标志 `done` + `timed_out` 构成状态机：
- compile 写 `done=true` + `kernel`
- watchdog 写 `timed_out=true`（仅在 done=false 时）
- 双方互斥检查后决定是否 `set_value`

### 3.2 compile 线程的双路径处理

```cpp
// 编译成功
{
    std::lock_guard slock(compile_state->mutex);
    compile_state->kernel = kernel;
    compile_state->done = true;
}
cv.notify_all();

// 写 cache + 清 pending
{
    std::lock_guard lock(state.mutex);  // ← state.mutex
    state.cache[cache_key] = {kernel, ...};
    state.pending.erase(cache_key);
}  // ← 锁释放！

// ⚠️ promise->set_value 必须在 state.mutex 锁外！
if (!compile_state->timed_out) {
    promise->set_value(compile_state->kernel);
}
```

**关键不变量**：`promise->set_value()` 不能在持 `state.mutex` 时调用。

### 3.3 watchdog 线程

```cpp
auto watchdog_future = std::async(std::launch::async, [...]() {
    if (timeout_ms == 0) {
        // 永不超时，watchdog 退化为直等
        compile_future_shared.wait();
        return;
    }

    // fast path: 已完成
    {
        std::lock_guard lock(compile_state->mutex);
        if (compile_state->done) return;
    }

    // 等 timeout 或 done
    std::unique_lock lock(compile_state->mutex);
    bool done = cv.wait_for(lock, ms(timeout_ms),
                             [&] { return done; });
    if (!done) {
        // 超时！
        compile_state->timed_out = true;
        lock.unlock();
        recordEngineError("async-timeout", "compile exceeded Xms for cache_key=Y");
        promise->set_value(nullptr);  // 立即通知主线程
    }
});
```

### 3.4 错误消息格式

```
async-timeout: compile exceeded 10000ms for cache_key=c3_v4_1_0_2_f_n_5n_3i_Graph
(5 nodes, 3 inputs, 1 outputs)
[0] INPUT Const(0) -> [8x16]
[3] MatMul (0, 1) -> [8x32]
[4] Fused(Add -> ReLU) args:[32,8x32] (2, 3) -> [8x32] *OUTPUT*
(actual compile continues in background)
```

包含：
- `[async-timeout]` 前缀（便于 grep）
- 超时毫秒数
- 完整 cache_key
- graph 结构摘要（节点数 + 输入输出 + 关键算子）
- 后台 compile 继续运行的提示

---

## 4. 死锁陷阱（关键 bug fix）

实施过程中遇到 **两个** 互锁 bug，必须同时修复：

### 4.1 Bug #1: compile 线程在持 state.mutex 时 set_value

**症状**：
- 主线程 `future.get()` 阻塞
- 复现 stack:
  - 主线程: `reapCompletedFutures` → `~__async_assoc_state` → `~shared_future` → `__on_zero_shared` → `wait()` → cv wait
  - 另一线程: `__execute` → `std::mutex::lock()` 等
- 死锁

**根因**：
- reapCompletedFutures 在持 `state.mutex` 时调 `erase`，触发 `~std::future<void>`
- `~std::future`（来自 `std::async`）会**阻塞等 task 真正结束**
- compile task 在 `set_value` 之前持 `state.mutex`（写 cache），所以 reap 等它
- 但 reap 也持 `state.mutex`，互相等 → 死锁

**修复**：`promise->set_value()` 必须在 `state.mutex` 锁外（已在 3.2 节代码中标注）。

### 4.2 Bug #2: reapCompletedFutures 在锁内 erase 触发同款死锁

**症状**：即使 Bug #1 修复后，测试卡在主线程调 `reapCompletedFutures`，子线程在 `~std::future` 等。

**根因**：reaper 在 `state.mutex` 锁内调 `it->get()` + `it = erase(it)`，`erase` 触发 `~std::future` 等待 task 结束，task 持 `state.mutex`（等锁），死锁。

**修复**：reaper 改为两阶段：
```cpp
static void reapCompletedFutures(EngineState& state,
                                 std::vector<std::future<void>>& reaped_sink) {
    // 锁内：只 move-out ready future + erase 空槽
    for (auto& f : state.compile_futures) {
        if (f.valid() && f.wait_for(0) == std::future_status::ready) {
            reaped_sink.push_back(std::move(f));
        }
    }
    state.compile_futures.erase(
        std::remove_if(state.compile_futures.begin(), state.compile_futures.end(),
            [](const std::future<void>& f) { return !f.valid(); }),
        state.compile_futures.end());
    // reaped_sink 由 caller 在锁外析构
}
```

所有 4 个 reaper 调用点都要在 caller-scope 声明 `std::vector<std::future<void>> to_reap`，在 `lock_guard` 块**外**声明，确保 `to_reap` 在锁释放后才析构。

---

## 5. 测试设计

`test_c3_compile_timeout.cpp` 覆盖 6 个场景：

| # | 场景 | 期望 |
|---|---|---|
| 1a | 默认 timeout = 30000 | 验证默认配置 |
| 1b/c | set/get 双向同步 | 验证 API |
| 2 | timeout=30s + 简单图 → kernel 有效 | 正常路径 |
| 3 | timeout=10ms + 复杂图 → nullptr + "async-timeout" | **核心熔断** |
| 4 | timeout=0 + 简单图 → kernel 有效 | 永不超时路径 |
| 5 | 第一次 timeout → 等 background 完成 → 第二次 cache 命中 | 关键场景：超时后仍能受益 |
| 6 | 触发超时 → clearLastCompileError → 再次读为空 | 错误状态管理 |

**关键测试设计**：
- 测试 5a 必须在开头 `sleep 5s` 等前面测试的 background compile 完成，否则会命中 stale `state.pending` 的 future
- 测试 3/4/5 用 buildComplexGraph（MLIR backend 真实卡顿场景），测试 4 用 buildSimpleAddGraph + Handwritten backend 避免 backend 限制干扰

---

## 6. 已知限制与未来工作

### 6.1 已知限制

1. **后台资源浪费**：超时后 compile 线程继续跑，会占用 CPU + 内存（典型 MLIR 图 200-500ms CPU + 1-2GB 内存峰值）。需要 OS-level 资源监控。
2. **不会写入 PGO 错误状态**：超时本身是 `async-timeout` 前缀，但不影响 `PGOCompiledKernel::lastCompileError()`（PGO 自己的 o2/ofast 错误仍走 ADR-010 通道）。
3. **跨平台 timeout 精度**：依赖 `std::condition_variable::wait_for` 精度，macOS 上 cv 唤醒延迟约 1-5ms，对 10ms 级别测试有 ~10% 误差。

### 6.2 未来工作

1. **可取消的 compile**：把 MLIR/clang++ 改为 child process + pipe 信号，watchdog 可 `kill -9` 子进程清理资源
2. **超时分级**：区分"超过 timeout 但仍在跑"和"已 abort"，给运维更细的 metric
3. **per-graph timeout override**：某些图（首次冷启动大模型）允许更长 timeout

---

## 7. 验证（Validation）

```
$ ./test_c3_compile_timeout
=== C3Engine 异步编译 watchdog timeout 测试（ADR-011）===
  PASS [1a]: 默认 timeout = 30000ms
  PASS [1b]: setCompileTimeoutMs / getCompileTimeoutMs 双向同步
  PASS [1c]: timeout=0（永不超时）配置生效
  PASS [2a]: 30s timeout + 简单图 → kernel 有效 (elapsed=120ms)
  PASS [2b]: 编译成功后 getLastCompileError() 仍为空
  PASS [3a]: 10ms timeout + 复杂图 → nullptr (elapsed=0ms)
  PASS [3b]: last_compile_error_ 含 'async-timeout'
  PASS [3c]: 错误信息含 cache_key 便于诊断
  PASS [4]: timeout=0 + 简单图 → kernel 有效（watchdog 退化为直等）
  PASS [4b]: timeout=0 路径无超时错误记录
  PASS [5a]: 第一次 timeout=10ms → nullptr（超时）
  PASS [5b]: 第二次 cache 命中 → kernel 有效 (elapsed=0ms)
  PASS [6a]: 触发超时，错误已记录
  PASS [6b]: clearLastCompileError() 生效

=== 总结 ===
  PASS: 14
  FAIL: 0
```

**全量 C3 回归 73/73 通过**，无回归（test_c3_aot_cache 16, test_c3_compile_and_inject 4, test_c3_compile_error 11, test_c3_compile_merged 10, test_c3_compile_merged_pgo 11, test_c3_compile_timeout 14, test_c3_pgo_deopt 7）。

---

## 8. 变更摘要

| 文件 | 变更 |
|---|---|
| `include/C3/C3Engine.h` | 新增 `setCompileTimeoutMs`, `getCompileTimeoutMs`, `recordCompileError` API |
| `src/C3/C3Engine.cpp` | compileAsync 改为 watchdog 模式；reapCompletedFutures 改为 sink-out 模式；4 个 reaper 调用点改为 caller-scope sink |
| `src/tests/standalone/test_c3_compile_timeout.cpp` | 新增 6 个场景测试 |
| `CMakeLists.txt` | 注册 test_c3_compile_timeout |
| `docs/adr/ADR-011-compile-timeout.md` | 本文档 |
