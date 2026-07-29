# CTorch ABI 政策

> 更新日期：2026-07-29

## 1. 政策声明

CTorch 当前处于快速演进期（v0.x）。**本项目不保证任何 ABI（Application Binary Interface）稳定性。**

公共头文件（`include/`、`src/kernels/kernels.h`）中的以下内容可能随时变化：

- 类/结构体的内存布局（成员变量顺序、大小、对齐）
- 枚举值及其顺序
- 虚函数表布局
- 内联函数语义
- 函数签名与默认参数

## 2. 为什么不需要向下兼容

经项目维护者确认（HITL 2026-07-29），CTorch 当前阶段**完全不需要向下兼容**。原因：

- 尚未发布稳定版或对外承诺 ABI。
- 下游使用者（如果有）应与 CTorch 同步源码编译。
- 强制 ABI 稳定会显著拖慢新后端、新算子、新数据类型的接入速度。

## 3. 禁止静默 ABI 破坏

虽然不需要兼容，但**每次 ABI 破坏必须显式记录并走审查流程**，禁止"改了但不说"。

### 3.1 必须触发审查的场景

修改以下公共头文件时，必须检查是否影响 ABI 或语义：

- `include/Storage.h`
- `include/Tensor.h`
- `include/Ctools.h`（含 `op`、`DType`、`DeviceType` 等枚举）
- `include/CtorchScheduler.h`
- `include/AutoGrad/Node.h`
- `src/kernels/kernels.h`

### 3.2 新增算子 ABI 检查清单

新增算子（或新增 `DeviceType`、`DType`）时，必须完成以下检查：

1. [ ] 将新枚举值追加到 `kCount` 之前，禁止在已有值之间插入。
2. [ ] 同步更新 `include/CtorchScheduler.h` 中的 `static_assert`（如 `op::kCount == 28`）。
3. [ ] 在 `CtorchScheduler::initKernels()` 中为新算子注册所有目标后端的 kernel，或显式标记为不支持。
4. [ ] 若新增 `DeviceType`，同步更新 `DEVICE_COUNT` static_assert 与所有调度表。
5. [ ] 若新增 `DType`，同步更新 `Storage::checkDType()`、`Tensor` dtype 分发、所有 kernel 的 dtype 支持声明。
6. [ ] 更新本文件第 4 节"已知 ABI 破坏记录"。

### 3.3 编译时自我检查

项目已在以下位置放置 static_assert，作为 ABI 变更的早期预警：

- `include/CtorchScheduler.h`: `op::kCount` 与 `DeviceType::kCount` 维度检查。

新增 enum 或修改 enum 顺序时，这些断言会强制编译失败，提醒维护者同步更新调度表。

## 4. 已知 ABI 破坏记录

| 日期 | 变更 | 影响文件 | 是否已记录 |
|---|---|---|---|
| 2026-07 | Storage 分配器从 Arena 替换为 DeviceAllocator | `include/Storage.h` | 是 |
| 2026-07 | `op` 枚举新增 `GELU`，`op::kCount` 从 27 增至 28 | `include/Ctools.h`, `include/CtorchScheduler.h` | 是 |
| 2026-07 | Tensor 拷贝构造 `_grad` 语义从共享恢复为深拷贝 | `include/Tensor.h` | 是 |

## 5. 未来何时引入 ABI 稳定

当出现以下信号时，应重新评估 ABI 政策：

- 项目发布 v1.0 或对外提供预编译库。
- 出现不可接受的重新编译成本（如下游项目众多、编译时间过长）。
- 需要支持插件或第三方扩展二进制。

届时可引入版本化命名空间（如 `ct::v1::Tensor`）或 SONAME 版本机制。

## 6. 相关资源

- `include/Ctools.h` — `op`、`DType`、`DeviceType` 定义与 ABI 注释。
- `include/Storage.h` — Storage 内存布局 ABI 注释。
- `include/CtorchScheduler.h` — 调度表维度 static_assert。
- `scripts/abi_audit.sh` — 公共头文件变更审计脚本。
