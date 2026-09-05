# RC2 进程级异步 · 跨进程内核交付（P0 难点）定案分析

> 日期：2026-09-04 · 关联：docs/C3_PROCESS_ASYNC_BLUEPRINT.md §3.1
> 目标：c3d 守护进程编译的内核如何交付给主进程执行——C3 进程级异步的架构命门

## 1. 问题

线程级下，\texttt{compileAsync} 返回 \texttt{CompiledKernel}（含指向 LLVM ORC JIT 产物的函数指针），同进程可直接 \texttt{kernel->execute()}。跨进程后，**函数指针是进程内地址，在另一进程无效**。

## 2. 当前内核持有结构（据 C3Engine.h）

\texttt{CompiledKernel} 抽象基类，\texttt{execute(inputs)} 纯虚；派生 \texttt{Concrete/Fused/Multi} 各自持有生成的内核函数入口（ORC JIT 符号地址）并在 execute 时调用。

## 3. 三方案平台分析（macOS 焦点）

### 方案 A：共享内存机器码 + mmap 直接执行
- 思路：c3d 把机器码段写入共享内存，主进程 mmap 到同一虚拟地址后取得可调用函数指针。
- **macOS 可行性：极低。**
  - Linux 可用 \texttt{memfd_create}+\texttt{dlopen("/proc/self/fd/N")} 从匿名内存加载；macOS 无此机制，\texttt{dlopen} 仅接受文件路径。
  - 直接 mmap 可执行段需手工处理代码重定位/符号解析（JIT 产物是位置相关或依赖 reloc），工程上几乎不可行。
- 结论：**放弃**（除非未来仅 Linux 部署时再评估）。

### 方案 B：动态库（.dylib）文件交付 —— 推荐
- 思路：c3d 编译后把内核导出为 \texttt{.dylib}（每个内核或每批一个），写到共享目录（或 \texttt{.c3cache}）；主进程 \texttt{dlopen} 路径 + \texttt{dlsym} 取符号。
- **macOS 可行性：高。**
  - ORC JIT 本就有产物对象模型（\texttt{ObjectLayer}）；可对热内核增量生成 \texttt{.dylib}，或整批写。
  - \texttt{dlopen}/\texttt{dlsym} 是标准 API，跨进程交付只需共享文件路径 + 缓存同步。
- 代价：每次内核交付有一次磁盘写 + dlopen；可用"一次 dlopen、句柄常驻"摊薄（进程内首次后直接符号表缓存）。
- 与现有 \texttt{JITCache}（.c3cache 已落盘）天然衔接——若 cache 存可加载对象，主进程冷启动可直接从 cache dlopen。

### 方案 C：主进程自编译（对等 JIT）
- 思路：不改交付，主进程自己用同源 LLVM 编译（守护进程只做调度/资源隔离）。
- 结论：失去"编译与计算物理隔离/资源零干扰"的设计初衷，退回线程级语义，**违背进程级异步目标**。

## 4. 推荐

**方案 B（动态库 .dylib 交付）** 为 RC2 落地路径：
1. c3d 侧：ORC 产物导出为 .dylib，命名含 cache key，写共享缓存目录；
2. 主进程：\texttt{dlopen} + \texttt{dlsym} 取符号 → 构造 \texttt{CompiledKernel}（保留 execute 接口，调用方无感知）；
3. 复用/扩展现有 JITCache 作为跨进程缓存（一个内核只编译一次，多进程共享）。

## 5. 最小可行性实验（MVE，Phase 1 前做）

验证链路：LLVM ORC 能否把单个融合内核（MatMul+act）稳定导出为 .dylib → 另一进程 dlopen/dlsym 后执行结果与线程级位级一致。
- 通过 → 方案 B 定案，进入 Phase 1-4 实施；
- 若 .dylib 导出有符号/重定位障碍 → 回退到"JITCache 存可加载对象，主进程从 cache 加载"（仍是 B 的变体）。

## 5.5 重要发现：JITCache 存的是 LLVM Bitcode（.bc），非机器码

探查 JITCache（2026-09-04）：`.c3cache` 落盘为 `c3_jit_<key>.bc` + `.meta`（JSON），即 **LLVM IR 层**，非可执行对象/.dylib。
- 含义 1：跨进程缓存**已支持**（多进程可共享 .c3cache 的 .bc，冷启动主进程读 .bc 本地 JIT）；
- 含义 2：方案 B（.dylib 交付）需**新增导出层**——把 JIT 产物（或 .bc）进一步编译并导出为 .dylib；
- 含义 3（架构权衡）：守护进程的增量价值 = **预编译（省主进程 JIT 时间）+ 编译资源隔离 + 崩溃隔离**；若主进程从共享 .bc 本地 JIT 可接受，则机器码交付可弱化——但会牺牲守护进程消除主进程 JIT 卡顿的目标。

**MVE 建议修正**：先验证 `.bc → LLVM 编译 → .dylib 导出 → 另一进程 dlopen/dlsym 执行` 全链路；若导出稳定，方案 B 定案；若 .dylib 导出有签名/重定位障碍，则评估"守护进程预编译 + 主进程从 .bc 重编译"（守护进程价值集中于资源隔离）作为备选。

## 6. 风险
- macOS 代码签名/notarization 对运行时生成的 .dylib 可能要求签名（ad-hoc sign \texttt{ldid -S}）——MVE 需验证；
- dlopen 后 .dylib 更新（Tier-1→Tier-2 升级）需处理旧句柄释放与新句柄热换——参考现有原子指针热替换；
- 多进程共享 .dylib 的写读竞态——写临时文件 + 原子 rename。
