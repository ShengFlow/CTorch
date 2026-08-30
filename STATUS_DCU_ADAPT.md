# 海光 DCU 适配 · 上下文恢复（v0.6 接力棒 · 2026-08-30 ASPLOS 战略）

> **本文件是 `STATUS_CONTEXT.md`（C3 区域融合接力棒）之外的第二个 STATUS 接力棒**
> **专门给"海光 DCU 适配"任务使用**
> **C3 完善优先，DCU 适配在 C3 完善后启动**

**最后更新**：2026-08-30 00:34
**维护者**：苏璃珞（CTorch Agent）
**配套报告**：
- `work/reports/2026-08-08/adapt-dcu-strategy-2257.md`（v0.4 战略报告——**待升级到 v0.6**）
- `work/reports/2026-08-09/framework-dcu-support-1013.md`（AI 框架 DCU 支持矩阵）

---

## 🆕 v0.6 战略调整（2026-08-30 洛锦 HITL）

### 触发事件
洛锦明确：**C3 完善是第一要务**。原"2026-09-10 CGO 2027 截稿"plan 取消，**改投 ASPLOS 2027**（CCF A 类，预计截稿 2027-06~07）。**DCU 适配降级**为"C3 完善后 + ASPLOS 论文需要时再启动"。

### 方向调整

| 维度 | v0.5 假设 | **v0.6 实际** |
|------|----------|---------------|
| **目标会议** | CGO 2027 (CCF B, 2026-09-10 截稿) | **ASPLOS 2027** (CCF A, ~2027-06~07 截稿) |
| **主线** | C3 端到端跑通 + 比 PyTorch-DCU 快 | **C3 完善**（backward 覆盖率 + CrossEntropy + 区域融合性能 + 端到端 ≥ Eager） |
| **时间窗** | 25 天（8.10-9.10） | **~10 个月**（8.30 - 2027-06~07） |
| **DCU 适配** | 优先级 P0 | **降级为 P3**（C3 完善 + ASPLOS 论文需要时再启动） |
| **TT 分解 Pass** | 关键路径 | **暂停**（之前 v0.4 RSVD 封存后未重启） |
| **目标对比基线** | PyTorch-DCU + XLA-DCU + TVM-DCU | **PyTorch-DCU（必须）** + XLA-DCU（如来得及） + TVM-DCU（如来得及） |

### ASPLOS 2027 论文三大 contribution 候选

1. **自研 C3 MLIR Dialect + One-Shot Linalg fusion**（编译时 IR + 融合算法）
2. **异构非阻塞区域融合 + 预走 + MIMO 反向融合**（运行时 fusion 决策 + 多输入多输出）
3. **海光 DCU 适配**（C3 → GCVM → HSACO 全链路）—— **v0.6 阶段才启动**

---

## ✅ DCU 适配已完成（v0.4~v0.5 阶段）

### 1. 战略 + 调研（已落盘）
- ✅ `work/reports/2026-08-08/adapt-dcu-strategy-2257.md` v0.4（25KB，待升级 v0.6）
- ✅ `work/reports/2026-08-09/framework-dcu-support-1013.md` v0.1（17KB，AI 框架 DCU 支持矩阵）
- ✅ `work/reports/2026-08-08/dcu-probe-runbook-2255.md`（探针使用说明）
- ✅ `scripts/probe-dcu.sh`（294 行可执行探针）

### 2. DCU 节点探针（已验证，v0.4 阶段）
- ✅ 节点 `b02r4n13` 验证通过（华东一区昆山）
- ✅ DTK 21.10 / 24.04 / 25.04 / 26.04 全部可 `module load` 切换
- ✅ 设备：**Hygon C86 7285 + ZIFANG C878180**（**gfx906, 16GB VRAM, 64 CU × 4 SIMD**）
- ⚠️ **重要约束**：DTK 21.10（节点默认）**GCVM 路径不可用**，需要切到 **DTK 24.04+** 才能走 GCVM
- ⚠️ **Fast F16 = FALSE**——gfx906 不支持 FP16 加速，只能跑 FP32

### 3. C3 内部 DCU 适配（v0.5 阶段已实装）
- ✅ `GCVMBridge.cpp` 完整实装 5 个 API 链（8.10）
- ✅ `DCUCompiledKernel.h` DCU kernel 容器
- ✅ `MLIRToLLVMIR.cpp` MLIR → LLVM IR 转换
- ✅ `JITCache.cpp` cache key 派生（**注意：只含 graph_str + opt_level，缺平台 / march / 版本**）

### 4. 已知风险（待 C3 完善 + 节点实测）

| ID | 风险 | 缓解 |
|----|------|------|
| DCU-v5-R01 | DTK 21.10 节点默认无 GCVM 路径 | 切到 DTK 24.04+ 重新探针 |
| DCU-v5-R02 | gfx906 不支持 FP16 加速（Fast F16 = FALSE）| FP32 跑或切 INT8 / Mixed Precision workaround |
| DCU-v5-R03 | DCU 显存只有 16G | TT 分解 / INT8 量化 / 1B-3B 模型 |
| DCU-v5-R04 | C3 MLIR → GCVM 路径无现成参考 | 参考 IREE 接入方式（~30 行核心代码）|
| **DCU-v6-R05**（v0.6 新增） | **C3 端到端 ≥ Eager 才能上 DCU 节点** | 先修 P0/P1（P1.3 端到端训练 ≥ Eager）|

---

## 📁 关键文件路径（接手必读）

### CTorch 仓库结构（v0.6 修正）
```
/Users/ghostface/CTorch-optimize-AutoDiff/
├── src/                        # 主仓代码（Tensor / AutoGrad / 调度器）
├── include/                    # 主仓头文件
├── c3/                         # C3 JIT 编译器 SUBMODULE
│   ├── include/C3/             # C3 头文件
│   ├── src/C3/                 # C3 实现
│   └── CMakeLists.txt
├── scripts/probe-dcu.sh        # DCU 节点探针
└── work/reports/2026-08-08/    # 战略报告
```

### c3 submodule DCU 相关文件
- `c3/include/C3/GCVMBridge.h` —— GCVM C API 桥接接口
- `c3/src/C3/GCVMBridge.cpp` —— 5 步 API 链（CreateProgram / SetArch / SetOptLevel / AddModule / Compile）
- `c3/include/C3/DCUCompiledKernel.h` —— DCU kernel 容器
- `c3/include/C3/MLIRToLLVMIR.h` —— MLIR → LLVM IR 转换接口

### 知识库
- `~/skills/docs/dcu-docs/knowledge_base/` —— 海光官方文档 400 文档

---

## 🛠 DCU 节点操作手册（接手必看）

### 登录节点
```
ssh login01  # 然后跳到 b02r4n13
```

### 切 DTK 版本（关键：DTK 21.10 无 GCVM）
```bash
# 默认 DTK 21.10：GCVM 路径不可用
# 切到 DTK 24.04+：
module load compiler/dtk/24.04
# 或 25.04 / 26.04
```

### 探针
```bash
bash scripts/probe-dcu.sh  # 294 行全自动探针
```

### 编译命令
```bash
module load compiler/cmake/3.25.0 compiler/gcc/12.2.0
# 然后用 hipcc 编译（DTK 自带）
```

### DCU 子目标
```
amdgcn-amd-amdhsa--gfx906:sramecc-:xnack-
```

---

## 📋 跨 Session 接力项（接手 agent 必读）

### 🔴 P0（v0.6 阶段：先做 C3 完善）
1. C3 backward 覆盖率统计（P0.1）
2. CrossEntropy/Softmax C3 Graph（P0.2）
3. Add/Mul/MatMul/Sub/Div backward 重新接入（P0.3）
4. C3KernelRegistry tryExecuteBackward stub 完整化（P0.4）

### 🟡 P1（C3 完善阶段并行）
5. MatMul epilogue vector lowering（P1.1）
6. 区域融合性能达标 0.62× → 1.0+×（P1.2）
7. C3 端到端训练 ≥ Eager（P1.3）
8. JITCache key 完整化（P1.4）
9. 统一 C3Cleanup 退出路径（P1.5）
10. MLP benchmark 拆分（P1.6）
11. 多输出 + preAct IR 完善（P1.7）
12. C3KernelRegistry fused lookup 单元测试（P1.8）
13. C3BackwardCapture 生命周期并发安全审查（P1.9）
14. C3HotPathManager deque 并发安全审查（P1.10）

### 🔧 P3（C3 完善 + ASPLOS 论文需要时启动）
15. DTK 24.04+ 探针重做（确认 IR_VERSION_MISMATCH 风险）
16. PyTorch-DCU baseline（选模型跑 eager）
17. C3 → DCU Hello World
18. C3-DCU 端到端跑通
19. ASPLOS 论文 §4 (DCU 适配实现)
20. C3-DCU vs PyTorch-DCU 性能对比

### ⏸ 暂停（不动）
- RSVD 性能起飞
- B 路径修复
- TT 分解 Pass

---

## 📌 重要事实速查

- **DCU 设备**：Hygon C86 7285 + ZIFANG C878180，gfx906，16G VRAM，64 CU × 4 SIMD
- **DTK 现状**：21.10 默认（无 GCVM），24.04+ 可切（待 C3 完善后探针）
- **Fast F16 = FALSE**——gfx906 不支持 FP16 加速
- **C3 → GCVM 路径**：参考 IREE 接入（~30 行核心代码）
- **PyTorch-DCU 是基线**——必须 ≥ PyTorch-DCU 性能才算成功
- **CTorch 差异化**：自研可控 + 轻量级 + fused kernel 自动化
- **ASPLOS 2027 截稿**：~2027-06~07（10 个月窗口）

---

## 📝 修订历史

| 版本 | 日期 | 作者 | 主要变更 |
|------|------|------|---------|
| v0.5 | 2026-08-10 | 苏璃珞 | 方向调整：RSVD 封存，DCU 主线，1 个月窗口 |
| **v0.6** | **2026-08-30** | **苏璃珞** | **战略调整：CGO 2027 → ASPLOS 2027，C3 完善优先，DCU 适配降级为 P3，10 个月窗口** |
