# ADR-008: AOT (Ahead-Of-Time) 持久化 .so cache

| 字段 | 值 |
|---|---|
| 状态 | Accepted |
| 日期 | 2026-08-03 |
| 作者 | CTorch Agent（苏璃珞） |
| 关联 commit | (pending push) |
| 优先级 | P0 |
| 关联调研 | [compiler-tech-survey-2026.md §5.3](../../skills/reports/2026-08-03/compiler-tech-survey-2026.md) |

---

## 1. 背景（Context）

C3 当前的 JIT 编译流程：

```
graph.toString() → makeCacheKey() → in-memory LRU cache (256 entries) → compileAndLoad()
                                                                          ↓
                                                              clang++ → .so → dlopen
```

**核心问题**：

1. **冷启动开销**：每次进程启动，必须为每个未缓存的图重新调用 clang++ 编译 .so。
   - 单个 kernel：~10-30ms
   - MLP（5 层）：~50-150ms
   - 复杂网络：~100-500ms

2. **跨进程无复用**：当前 cache 是 in-memory（`static EngineState`），进程退出即丢失。
   - 每次冷启动都要重新编译相同图
   - 模型部署场景：必须发布源码而非编译产物

3. **调研报告 5.3 节明确指出**：
   - C3 距离"工业级 JIT"的 6 大核心差距中，**"无 AOT 持久化"** 是 P0
   - 工业级范式（TensorRT-RTX 2025）：AOT 优化（20-30s）+ JIT 特化（< 5s）

---

## 2. 决策（Decision）

为 C3 引入 **AOT 持久化 .so cache**：

- 存储位置：`$HOME/.c3cache/c3_<key>.so`（默认）
- 覆盖优先级：`setAOTCacheDir()` > `C3_AOT_CACHE_DIR` 环境变量 > `$HOME/.c3cache`
- Key 派生：`SHA-256(graph.toString() | device | opt_level | backend_version).substr(0, 32)`
- 失效策略：backend version 不匹配 → 自动重新编译
- 写入模式：`.tmp` → atomic rename（避免读到半截文件）
- 优雅降级：磁盘错误 → 静默回退 in-memory（log warning + 继续）

---

## 3. 替代方案对比（Alternatives Considered）

### 3.1 A1. 传统 .so cache（已选 ✅）

| 维度 | 评估 |
|---|---|
| 实现复杂度 | 中（~300 行） |
| 跨进程 | ✅ |
| 失效策略 | backend version + 显式 evict |
| 调试友好 | 文件可见，ls -la 可查 |
| 跨架构 | 失败时 dlopen 自动 fallback |
| 文件碎片 | 多文件（用 prefix 区分避免污染） |

### 3.2 A2. 单一 .so 多符号

| 维度 | 评估 |
|---|---|
| 优势 | 单文件好管理 |
| 劣势 | 重新链接开销大；版本管理复杂；冷启动需遍历符号表 |
| 决定 | ❌（增加复杂度，收益不明确） |

### 3.3 A3. 共享内存 cache

| 维度 | 评估 |
|---|---|
| 优势 | 极速（无磁盘 I/O） |
| 劣势 | 跨进程复杂；进程崩溃数据丢失；OS 依赖 |
| 决定 | ❌（实现复杂，跨平台难） |

### 3.4 A4. LLVM bitcode cache

| 维度 | 评估 |
|---|---|
| 优势 | 跨后端复用 |
| 劣势 | 需 MLIR 配合；与 Handwritten backend 无关 |
| 决定 | ⏸ 留给 MLIR AOT（Phase D+） |

---

## 4. 详细设计（Detailed Design）

### 4.1 公共 API

```cpp
// C3Engine.h
class C3Engine {
public:
    void setAOTCacheEnabled(bool enabled);
    bool isAOTCacheEnabled() const;

    AOTCacheStats getAOTCacheStats() const;  // hits/misses/writes/load_failures/...
    void evictAOTCache();

    void setAOTCacheDir(const std::string& dir);
    std::string getAOTCacheDir() const;
};

struct AOTCacheStats {
    uint64_t hits;
    uint64_t misses;
    uint64_t writes;
    uint64_t evictions;
    uint64_t load_failures;
    uint64_t invalidations;
    uint64_t disk_errors;
    size_t total_files;
    size_t total_bytes;
};
```

### 4.2 内部流程

```
HandwrittenKernelGen::compileAndLoad(src, func_name, cache_key)
│
├─ 1. AOTCache.lookup(cache_key)
│  ├─ 命中 → dlopen + dlsym → return (避免 clang++)
│  └─ 未命中 ↓
│
├─ 2. mkdtemp + 写源码
│
├─ 3. clang++ 编译 → .so
│
├─ 4. AOTCache.store(cache_key, .so_path)  [静默失败不阻塞]
│  └─ 写入 ~/.c3cache/c3_<key>.so + .meta
│
└─ 5. dlopen + dlsym → return
```

### 4.3 Key 派生

```cpp
std::string AOTCache::makeKey(
    const std::string& graph_str,
    const std::string& device,
    int opt_level,
    const std::string& backend_version)
{
    std::string combined = graph_str + "|" + device + "|" +
                          std::to_string(opt_level) + "|" + backend_version;
    return sha256_hex(combined).substr(0, 32);  // 128-bit 已足够
}
```

**为什么不复用 in-memory cache key**？

In-memory key 包含运行时状态（如 `enable_autotune`），不利于持久化。
AOT key 只包含编译配置因子，更纯粹。

### 4.4 Meta 文件格式

`~/.c3cache/c3_<key>.meta`：
```
backend_version=handwritten-v3
cache_key=<32 hex chars>
```

未来扩展（保留）：
- `created_at=<unix timestamp>`
- `arch=arm64`
- `clang_version=<version>`

### 4.5 失败模式与降级

| 失败 | 处理 |
|---|---|
| 磁盘满 | `mkdir`/`write` 失败 → `disk_errors++` + in-memory 继续 |
| `dlopen` 失败（架构不兼容）| `load_failures++` + 重新编译 + 覆盖 .so |
| `dlsym` 失败（符号名变更）| `load_failures++` + 重新编译 + 覆盖 .so |
| `HOME` 未设置 | fallback 到 `/tmp/.c3cache`（不理想但可用）|
| `~/.c3cache` 权限被拒绝 | `disk_errors++` + 禁用 AOT（log warning）|

### 4.6 线程安全

- 进程级 mutex 保护 disk I/O
- `dlopen` 本身线程安全（POSIX 1003.1）
- 多个线程同时编译同 cache_key：第一个 store，后续 lookup 命中
- 跨进程：fork 子进程 → 父子进程共享同一 .so（OS 引用计数）

---

## 5. 实施（Implementation）

### 5.1 文件改动

| 文件 | 改动 |
|---|---|
| `include/C3/AOTCache.h` | 新增（~200 行）|
| `src/C3/AOTCache.cpp` | 新增（~380 行，含 SHA-256）|
| `include/C3/C3Engine.h` | 新增 6 个公共 API（~70 行）|
| `src/C3/HandwrittenKernelGen.cpp` | 集成 lookup/store（~30 行）|
| `src/tests/standalone/test_c3_aot_cache.cpp` | 新增（~290 行，16 测试）|
| `CMakeLists.txt` | 加入 AOTCache.cpp + test target（~5 行）|

总计：~975 行（+550 行新代码 + ~425 行测试）

### 5.2 编译标志

无新增依赖。SHA-256 是自包含实现（~150 行），不依赖 openssl。

### 5.3 测试覆盖

[test_c3_aot_cache.cpp](../../../CTorch-optimize-AutoDiff/src/tests/standalone/test_c3_aot_cache.cpp) — 16 测试：

1. **SHA-256 正确性**（标准 test vector "abc"）
2. **makeKey 确定性**（相同输入 → 相同 key）
3. **makeKey 唯一性**（不同 graph/device/opt_level/version → 不同 key）
4. **禁用时 lookup** 返回空
5. **从未 store 的 key** → miss 计数++
6. **store + lookup 命中** → writes/hits 计数++
7. **evict 前**：3 个 key 都在
8. **evict 后**：3 个 key 都被清空
9. **当前版本命中**
10. **backend version 不匹配** → invalidations++ + 返回空
11. **setCacheDir** 后 getCacheDir 返回新目录
12. **store 写入**到自定义目录
13. **自定义目录中 .so 文件存在**
14. **C3Engine 集成**：首次 + 二次编译 stats 增加
15. **不同图各自产生独立 cache**（writes >= 2）
16. **dlopen 失败后 fallback**（load_failures++ 或重新写入）

---

## 6. 后果（Consequences）

### 6.1 正面

1. **冷启动加速**：单 kernel 节省 ~10-30ms；MLP 节省 ~50-150ms
2. **模型部署可行**：发布 .so + key 即可，无需发布源码或工具链
3. **跨进程复用**：同图在不同进程间共享编译产物
4. **可观察性**：`getAOTCacheStats()` 提供完整 hit/miss 统计
5. **零依赖**：自包含 SHA-256，不增加 openssl 依赖

### 6.2 负面

1. **磁盘占用**：每个 .so ~10-50KB，1GB 默认上限 = ~20000-100000 个 kernel
2. **失效延迟**：backend version 变更需手写（每次 HandwrittenKernelGen 不兼容变更时手动 ++）
3. **首次启动无收益**：第一次仍需编译所有 kernel
4. **跨架构风险**：在 arm64 编译的 .so 不能在 x86_64 运行（已通过 dlopen failure fallback 缓解）

### 6.3 风险缓解

| 风险 | 缓解 |
|---|---|
| ~/.c3cache 权限被攻击者篡改 | dlopen 失败 → 重新编译 + 覆盖 |
| backend version 漏改导致不兼容 .so 被加载 | SHA-256 校验 graph.toString() — 不同图必产生不同 key |
| 磁盘满导致程序崩溃 | try-catch 所有磁盘操作 → 静默回退 in-memory |
| 跨架构 .so 误加载 | dlopen 失败 → 重新编译并覆盖 |
| 大量失效 .so 占满磁盘 | max_bytes=1GB + max_files=1024（当前未实现 LRU，留待 Phase D+） |

---

## 7. 未来工作（Future Work）

### 7.1 Phase D+ 短期

- **LRU 淘汰**：max_files/max_bytes 超限 → 删最久未用（当前未实现）
- **后端版本自动递增**：用 git hash 作为 backend version（避免漏改）
- **signature 校验**：写 .so 时记录 graph.toString()，load 时校验（防止 key collision）

### 7.2 Phase E 中期

- **MLIR AOT**：MLIR backend 的 AOT 持久化（bitcode 形式）
- **跨架构支持**：x86_64 ↔ arm64 自动选择（用 uname 派生路径）
- **AOT 预热模式**：首次启动时 warmup 关键 kernel

### 7.3 Phase F 长期

- **持久化 .so 签名验证**（防止恶意篡改）
- **加密 cache**（共享计算节点的场景）
- **跨设备 cache**（Lustre/NFS）

---

## 8. 参考（References）

- [TensorRT-RTX 2025 AOT + JIT 混合架构](https://developer.nvidia.com/tensorrt)
- [ONNX Runtime model caching](https://onnxruntime.ai/docs/performance/model-optimizations/ortformat-models.html)
- [TVM Unity Module Caching](https://tvm.apache.org/docs/reference/api/python/relax/index.html)
- [FIPS 180-4 SHA-256 标准](https://csrc.nist.gov/publications/detail/fips/180/4/final)
- [本仓库 compiler-tech-survey-2026.md §5.3](../../skills/reports/2026-08-03/compiler-tech-survey-2026.md)

---

**最后更新**：2026-08-03 18:55 CST
**作者**：CTorch Agent（苏璃珞）
**下次 review**：Phase D 完成时
