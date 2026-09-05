#!/bin/bash
# ============================================================================
# test_simd_avx512.sh — x86-64 AVX-512 (F+DQ) 机器上验证 C3 的 AVX-512 SIMD
# 用法: ./test_simd_avx512.sh   (可 CC=clang++/g++ 覆盖默认编译器)
# 需 AVX-512 真机; Apple Silicon(Rosetta) 不支持 512 指令会 Illegal instruction
# ============================================================================
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CC="${CC:-clang++}"
echo "=== C3 AVX-512 SIMD 验证 ==="
echo "compiler: $CC"
"$CC" --version | head -1
echo "arch: $(uname -m)"
if ! "$CC" -E -mavx512f -mavx512dq -x c /dev/null >/dev/null 2>&1; then
    echo "错误: 编译器不支持 -mavx512f/-mavx512dq"
    exit 1
fi
echo
echo "--- 编译 test_simd_avx512 ..."
"$CC" -std=c++17 -O2 -mavx512f -mavx512dq -mfma -I"$REPO/include" \
    "$REPO/src/tests/standalone/test_simd_avx512.cpp" \
    "$REPO/src/kernels/CPU-SIMD/SIMDMath.cpp" \
    -o /tmp/test512
echo "--- 运行 ---"
/tmp/test512
echo
rc=$?
echo "exit=$rc"
exit $rc