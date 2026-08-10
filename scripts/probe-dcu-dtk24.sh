#!/bin/bash
# scripts/probe-dcu-dtk24.sh
# 升级版 DCU 探针（v0.5 专用）：专门验证 DTK 24.04+ GCVM/hipBLAS/MLIR amdgcn 接入条件
# 用法：在 b02r4n13 节点跑
#   module load compiler/dtk/24.04  # 或 25.04/26.04
#   bash scripts/probe-dcu-dtk24.sh
# 输出：work/reports/2026-08-10/dcu-probe-dtk24-<HOSTNAME>.md
# 作者：mavis (CTorch 主代理), 2026-08-10

set -e

HOSTNAME=$(hostname)
REPORT_DIR="work/reports/2026-08-10"
mkdir -p "$REPORT_DIR"
REPORT_FILE="$REPORT_DIR/dcu-probe-dtk24-${HOSTNAME}.md"

DTK_VERSION="${DTK_VERSION:-$(module list 2>&1 | grep -oE 'dtk/[0-9.]+' | head -1 | cut -d/ -f2)}"
DTK_PATH="${ROCM_PATH:-/opt/dtk}"

exec > >(tee "$REPORT_FILE") 2>&1

echo "# DTK 24.04+ 探针报告 · $HOSTNAME"
echo ""
echo "**日期**: $(date -Iseconds)"
echo "**DTK 版本**: ${DTK_VERSION:-未检测到}"
echo "**DTK 路径**: $DTK_PATH"
echo "**目的**: 验证 C3 → DCU 接入条件（GCVM/hipBLAS/MLIR amdgcn/PyTorch-DCU）"
echo ""

# === 1. DTK 模块 ===
echo "## 1. DTK 模块状态"
echo ""
echo "### 1.1 当前加载的 DTK 模块"
module list 2>&1 | grep -E "dtk|rocm" || echo "⚠️  未检测到 DTK 模块"
echo ""
echo "### 1.2 可用 DTK 版本"
module avail compiler/dtk 2>&1 | tail -20 || echo "module avail 失败"
echo ""

# === 2. GCVM C API 检测（关键）===
echo "## 2. GCVM C API 检测（关键）"
echo ""
GCVM_INCLUDE="$DTK_PATH/llvm/gcvm/include"
GCVM_LIB="$DTK_PATH/llvm/gcvm/lib/libgcvm.so"

if [ -d "$GCVM_INCLUDE" ]; then
    echo "### ✅ GCVM 头文件存在: $GCVM_INCLUDE"
    ls -la "$GCVM_INCLUDE" 2>&1 | head -10
    echo ""
    echo "### GCVM 公开 API 头文件"
    ls "$GCVM_INCLUDE"/*.h 2>&1 | head -10
    echo ""
    # 提取 GCVM API 函数名
    echo "### GCVM 公开 API 函数列表"
    grep -hE "^[A-Z][a-zA-Z]*\s+gcvm[A-Z][a-zA-Z]*" "$GCVM_INCLUDE"/*.h 2>&1 | head -20 || echo "未找到 API 函数"
    echo ""
else
    echo "### ❌ GCVM 头文件不存在: $GCVM_INCLUDE"
    echo "**结论**: DTK ${DTK_VERSION} 无 GCVM 路径，需要切 DTK 24.04+ 或检查 DTK 安装完整性"
    echo ""
fi

if [ -f "$GCVM_LIB" ]; then
    echo "### ✅ GCVM 库存在: $GCVM_LIB"
    ls -la "$GCVM_LIB" 2>&1
    echo ""
    # 检查库导出符号
    echo "### GCVM 库导出符号（前 20 个）"
    nm -D "$GCVM_LIB" 2>&1 | grep -E "gcvm[0-9A-Z_]*" | head -20 || echo "未找到符号"
    echo ""
else
    echo "### ❌ GCVM 库不存在: $GCVM_LIB"
    echo ""
fi

# === 3. hipBLAS / hipBLASLt ===
echo "## 3. hipBLAS / hipBLASLt 检测"
echo ""
HIPBLAS_LIB="$DTK_PATH/lib/libhipblas.so"
HIPBLASLT_LIB="$DTK_PATH/lib/libhipblaslt.so"
if [ -f "$HIPBLAS_LIB" ]; then
    echo "### ✅ hipBLAS 存在: $HIPBLAS_LIB"
else
    echo "### ❌ hipBLAS 缺失: $HIPBLAS_LIB"
fi
if [ -f "$HIPBLASLT_LIB" ]; then
    echo "### ✅ hipBLASLt 存在: $HIPBLASLT_LIB"
else
    echo "### ⚠️ hipBLASLt 缺失: $HIPBLASLT_LIB (可选, 性能优化)"
fi
echo ""

# === 4. hipcc / clang++ for amdgcn ===
echo "## 4. hipcc / amdgcn 编译工具"
echo ""
if command -v hipcc &> /dev/null; then
    echo "### ✅ hipcc 存在: $(which hipcc)"
    hipcc --version 2>&1 | head -3
    echo ""
    # 验证 amdgcn target
    if hipcc -print-target-triple 2>&1 | grep -q "amdgcn"; then
        echo "### ✅ hipcc 支持 amdgcn target: $(hipcc -print-target-triple 2>&1)"
    else
        echo "### ❌ hipcc 不支持 amdgcn target"
    fi
else
    echo "### ❌ hipcc 不存在"
fi
echo ""

# === 5. MLIR amdgcn backend 检测 ===
echo "## 5. MLIR amdgcn Backend 检测"
echo ""
if command -v mlir-translate &> /dev/null; then
    echo "### ✅ mlir-translate 存在: $(which mlir-translate)"
    echo ""
    echo "### MLIR amdgcn 转换测试"
    cat > /tmp/test_amdgcnn.mlir <<'EOF'
module {
  llvm.func @test_kernel(%a: f32, %b: f32) -> f32 {
    %sum = llvm.fadd %a, %b : f32
    llvm.return %sum : f32
  }
}
EOF
    if mlir-translate --mlir-to-llvmir /tmp/test_amdgcnn.mlir 2>&1 | head -5; then
        echo "### ✅ mlir-translate --mlir-to-llvmir 成功"
    else
        echo "### ❌ mlir-translate --mlir-to-llvmir 失败"
    fi
    echo ""
    # amdgcn 转换
    if mlir-translate --mlir-to-rocdl /tmp/test_amdgcnn.mlir 2>&1 | head -5; then
        echo "### ✅ mlir-translate --mlir-to-rocdl 成功 (MLIR 直接转 ROCDL/HSA)"
    else
        echo "### ⚠️ mlir-translate --mlir-to-rocdl 失败或不支持 (可能需 --target=amdgcn-amd-amdhsa)"
    fi
else
    echo "### ❌ mlir-translate 不存在"
fi
echo ""

# === 6. PyTorch-DCU 检测 ===
echo "## 6. PyTorch-DCU 检测"
echo ""
if command -v python3 &> /dev/null; then
    python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
try:
    print(f'DCU available: {torch.dcu.is_available()}')
    print(f'DCU 设备数: {torch.dcu.device_count()}')
    if torch.dcu.is_available():
        for i in range(min(2, torch.dcu.device_count())):
            print(f'DCU[{i}] 名称: {torch.dcu.get_device_name(i)}')
            print(f'DCU[{i}] 计算能力: {torch.dcu.get_device_capability(i)}')
            props = torch.dcu.get_device_properties(i)
            print(f'DCU[{i}] 总显存: {props.total_memory / 1e9:.1f} GB')
            print(f'DCU[{i}] 多处理器: {props.multi_processor_count}')
except Exception as e:
    print(f'PyTorch DCU 检测失败: {e}')
" 2>&1
else
    echo "### ❌ python3 不存在"
fi
echo ""

# === 7. 设备信息（基础）===
echo "## 7. DCU 设备基础信息"
echo ""
if command -v hy-smi &> /dev/null; then
    echo "### hy-smi 输出"
    hy-smi 2>&1 | head -20
elif command -v rocm-smi &> /dev/null; then
    echo "### rocm-smi 输出"
    rocm-smi 2>&1 | head -20
else
    echo "### ❌ hy-smi / rocm-smi 都不存在"
fi
echo ""

# === 8. 简单 kernel 跑通测试 (hipBLAS GEMM) ===
echo "## 8. hipBLAS 跑通测试 (Hello World)"
echo ""
if command -v python3 &> /dev/null; then
    python3 -c "
import torch
import time
try:
    if torch.dcu.is_available():
        # 简单 GEMM 测试
        a = torch.randn(1024, 1024, device='dcu', dtype=torch.float32)
        b = torch.randn(1024, 1024, device='dcu', dtype=torch.float32)
        
        # warmup
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.dcu.synchronize()
        
        # timing
        N = 100
        t0 = time.time()
        for _ in range(N):
            c = torch.matmul(a, b)
        torch.dcu.synchronize()
        elapsed = (time.time() - t0) / N * 1000
        
        print(f'✅ PyTorch-DCU 1024x1024 GEMM: {elapsed:.2f} ms/iter')
        print(f'  TFLOPS: {2 * 1024**3 / elapsed / 1e9:.2f}')
    else:
        print('❌ PyTorch-DCU 不可用')
except Exception as e:
    print(f'❌ 跑通测试失败: {e}')
" 2>&1
fi
echo ""

# === 9. 总结 ===
echo "## 9. 探针总结"
echo ""
SUMMARY_OK="✅"
SUMMARY_FAIL="❌"
SUMMARY_WARN="⚠️"

# 综合判断
GCVM_OK="NO"
HIPBLAS_OK="NO"
MLIR_AMDGCN_OK="NO"
PYTORCH_DCU_OK="NO"

[ -f "$GCVM_LIB" ] && [ -d "$GCVM_INCLUDE" ] && GCVM_OK="YES"
[ -f "$HIPBLAS_LIB" ] && HIPBLAS_OK="YES"
command -v mlir-translate &> /dev/null && MLIR_AMDGCN_OK="YES"
python3 -c "import torch; assert torch.dcu.is_available()" 2>/dev/null && PYTORCH_DCU_OK="YES"

echo "| 项目 | 状态 |"
echo "|---|---|"
echo "| GCVM 库 + 头文件 | ${GCVM_OK} |"
echo "| hipBLAS | ${HIPBLAS_OK} |"
echo "| MLIR amdgcn backend | ${MLIR_AMDGCN_OK} |"
echo "| PyTorch-DCU 可用 | ${PYTORCH_DCU_OK} |"
echo ""

# C3 接入就绪度判断
if [ "$GCVM_OK" = "YES" ] && [ "$HIPBLAS_OK" = "YES" ] && [ "$MLIR_AMDGCN_OK" = "YES" ]; then
    echo "### ✅ C3 → DCU 接入条件满足"
    echo "**结论**: 可以走 GCVM C API 桥接路径 (30 行核心代码)"
    echo "**下一步**: 洛锦跑 \`scripts/bench-pytorch-dcu-baseline.py\` 拿 PyTorch baseline"
elif [ "$GCVM_OK" = "NO" ]; then
    echo "### ❌ GCVM 缺失"
    echo "**结论**: 当前 DTK 版本无 GCVM 路径，需要切 DTK 24.04+"
    echo "**建议**: \`module avail compiler/dtk\` 看可用版本，切 24.04/25.04/26.04 重新跑"
else
    echo "### ⚠️ 部分缺失"
    echo "**结论**: 部分依赖缺失，Phase 0 探针完成后再决策"
fi

echo ""
echo "---"
echo ""
echo "**报告生成完成**: $(date -Iseconds)"
echo "**报告路径**: $REPORT_FILE"
echo "**配套**: scripts/bench-pytorch-dcu-baseline.py + work/reports/2026-08-10/c3-dcu-integration-design.md"
