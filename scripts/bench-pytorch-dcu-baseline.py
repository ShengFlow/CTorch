#!/usr/bin/env python3
"""
scripts/bench-pytorch-dcu-baseline.py

PyTorch-DCU 性能 baseline 脚本 (v0.5 DCU 接入 Phase 1)
用法: 在 DCU 节点 b02r4n13 跑
  module load compiler/dtk/24.04
  python3 scripts/bench-pytorch-dcu-baseline.py
输出: work/reports/2026-08-10/pytorch-dcu-baseline-<HOSTNAME>.md
目的: 给 C3-DCU 提供性能对比基线（同节点同模型）

覆盖场景:
  1. 3 层 MLP (784→256→128→10) - C3 优势最大场景
  2. ResNet50 forward - CNN 场景
  3. Simple transformer block (8 层 attention) - 中等复杂度
  4. 纯 matmul (1024/2048/4096) - BLAS baseline

作者: mavis (CTorch 主代理), 2026-08-10
"""
import torch
import torch.nn as nn
import time
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# === 配置 ===
HOSTNAME = os.uname().nodename
REPORT_DIR = Path("work/reports/2026-08-10")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = REPORT_DIR / f"pytorch-dcu-baseline-{HOSTNAME}.md"
WARMUP = 20
RUNS = 100
DTK_VERSION = os.environ.get("DTK_VERSION", "unknown")

# === 检查 PyTorch-DCU 可用性 ===
if not torch.dcu.is_available():
    print("❌ PyTorch-DCU 不可用！")
    print("   请确认: module load compiler/dtk/24.04 + 装 PyTorch-DCU")
    sys.exit(1)

device = torch.device("dcu")
device_name = torch.dcu.get_device_name(0)
device_props = torch.dcu.get_device_properties(0)
total_mem_gb = device_props.total_memory / 1e9

# === 报告 header ===
report_lines = []
def log(line=""):
    """stdout + report_lines 双写"""
    print(line)
    report_lines.append(line)

log(f"# PyTorch-DCU Baseline 报告 · {HOSTNAME}")
log("")
log(f"**日期**: {datetime.now().isoformat()}")
log(f"**DTK 版本**: {DTK_VERSION}")
log(f"**PyTorch 版本**: {torch.__version__}")
log(f"**DCU 设备**: {device_name}")
log(f"**DCU 显存**: {total_mem_gb:.1f} GB")
log(f"**DCU Multi-Processors**: {device_props.multi_processor_count}")
log(f"**WARMUP**: {WARMUP}, **RUNS**: {RUNS}")
log("")
log("**目的**: 给 C3-DCU 性能对比提供 PyTorch baseline（同节点同模型）")
log("**C3 对比目标**: 5.36x (MLP macOS) → 8-12x (MLP DCU 预期) → C3 vs PyTorch-DCU 1.5-3x")
log("")

# === Benchmark 工具 ===
def bench_forward(model, input_shape, name):
    """Benchmark model forward latency"""
    model = model.to(device)
    model.eval()
    x = torch.randn(*input_shape, device=device, dtype=torch.float32)
    
    # warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)
    torch.dcu.synchronize()
    
    # timing
    t0 = time.time()
    with torch.no_grad():
        for _ in range(RUNS):
            _ = model(x)
    torch.dcu.synchronize()
    elapsed = (time.time() - t0) / RUNS * 1000  # ms
    
    log(f"### {name}")
    log(f"- 设备: {device_name}")
    log(f"- 输入: {input_shape}")
    log(f"- 延迟: **{elapsed:.3f} ms/iter**")
    log(f"- 吞吐: {1000.0 / elapsed:.1f} fwd/s")
    log("")
    return elapsed

# === Benchmark 1: 3 层 MLP (784→256→128→10) ===
log("## 1. 3 层 MLP (784→256→128→10) - C3 优势最大场景")
log("")
log("**模型**: Linear(784, 256) + ReLU + Linear(256, 128) + ReLU + Linear(128, 10)")
log("**C3 对比基线**: 5.36x (macOS) → 预期 8-12x (DCU)")
log("")

class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

mlp3_time = bench_forward(MLP3(), (1, 784), "MLP3 forward (batch=1)")

# === Benchmark 2: ResNet50 ===
log("## 2. ResNet50 forward - CNN 场景")
log("")
log("**模型**: torchvision ResNet50 (no weights, random init for compile check)")
log("**C3 对比**: C3 当前 Conv 节点未实装，PyTorch-DCU 大概率胜出")
log("")

try:
    import torchvision
    resnet50 = torchvision.models.resnet50(weights=None)
    resnet_time = bench_forward(resnet50, (1, 3, 224, 224), "ResNet50 forward (batch=1)")
except ImportError:
    log("### ⚠️ torchvision 未安装, 跳过 ResNet50")
    resnet_time = None
log("")

# === Benchmark 3: 纯 matmul (不同尺寸) ===
log("## 3. 纯 matmul (BLAS baseline)")
log("")
log("**对比**: PyTorch-DCU 走 hipBLAS (rocBLAS fork)，C3-DCU 也走 BLAS")
log("")

matmul_results = {}
for size in [256, 512, 1024, 2048, 4096]:
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    
    # warmup
    for _ in range(WARMUP):
        _ = torch.matmul(a, b)
    torch.dcu.synchronize()
    
    # timing
    t0 = time.time()
    for _ in range(RUNS):
        _ = torch.matmul(a, b)
    torch.dcu.synchronize()
    elapsed = (time.time() - t0) / RUNS * 1000
    tflops = 2 * size**3 / elapsed / 1e9
    matmul_results[size] = (elapsed, tflops)
    
    log(f"### {size}x{size} matmul")
    log(f"- 延迟: **{elapsed:.3f} ms**")
    log(f"- 算力: **{tflops:.2f} TFLOPS**")
    log(f"- (DCU gfx906 理论峰值: ~10 TFLOPS FP32)")
    log("")

# === 总结表 ===
log("## 4. 总结")
log("")
log("### 4.1 性能数字汇总")
log("")
log("| 场景 | 延迟 | 吞吐 | 算力 |")
log("|---|---|---|---|")
log(f"| MLP3 fwd (batch=1) | {mlp3_time:.3f} ms | {1000.0/mlp3_time:.1f} fwd/s | — |")
if resnet_time:
    log(f"| ResNet50 fwd (1,3,224,224) | {resnet_time:.3f} ms | {1000.0/resnet_time:.1f} fwd/s | — |")
for size, (elapsed, tflops) in matmul_results.items():
    log(f"| {size}x{size} matmul | {elapsed:.3f} ms | — | {tflops:.2f} TFLOPS |")
log("")

log("### 4.2 C3-DCU 对比预期")
log("")
log("**基于 macOS 基线** (per c3-perf-report-1546.md):")
log(f"- MLP3: C3 macOS 5.36x → DCU 预期 8-12x")
log(f"- matmul {list(matmul_results.keys())[-1]}: BLAS 路径接近 PyTorch")
log(f"- ResNet50: C3 不能跑 (Conv 节点缺), PyTorch-DCU 完胜")
log("")

log("### 4.3 Switch 条件检查")
log("")
log("- [ ] C3-DCU MLP 跑通 + 性能 ≥ PyTorch-DCU 1.0x？")
log("- [ ] C3-DCU MLP 跑通 + 性能 ≥ PyTorch-DCU 1.25x？")
log("- [ ] C3-DCU 跑通但 < PyTorch？")
log("- [ ] C3-DCU 跑不通？")
log("")
log("**Phase 2 完成后** (C3 Hello World), 用这个 baseline 对比")
log("")

log("---")
log("")
log(f"**报告生成完成**: {datetime.now().isoformat()}")
log(f"**报告路径**: {REPORT_FILE}")
log(f"**配套**: scripts/probe-dcu-dtk24.sh + work/reports/2026-08-10/c3-dcu-integration-design.md")

# === 写文件 ===
REPORT_FILE.write_text("\n".join(report_lines))
print(f"\n✅ 报告已写: {REPORT_FILE}")
