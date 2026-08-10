/**
 * @file SIMDWrapper.h
 * @brief C-ABI 包装层：将 SIMD 超越函数暴露为 extern "C" 符号
 * @details 供 MLIR JIT 后端通过 LLVM::CallOp 直接调用。
 *          所有函数使用 C ABI 确保 MLIR ExecutionEngine 可正确解析符号。
 *          函数名以 ct_simd_ 前缀避免与 libc 符号冲突。
 *
 *          使用方式（MLIRKernelGen.cpp）：
 *          - 声明 extern @ct_simd_vexp (!llvm.ptr<f32>, !llvm.ptr<f32>, i64) → ()
 *          - 生成 call 指令替代逐元素循环
 *
 * @date 2026/08/03
 * @see ADR-012-simd-mlir-integration
 */

#ifndef CTORCH_KERNELS_SIMD_WRAPPER_H
#define CTORCH_KERNELS_SIMD_WRAPPER_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 向量化 expf（批量 SIMD）
 * @param in  输入数组（连续内存）
 * @param out 输出数组（连续内存）
 * @param n   元素数
 */
void ct_simd_vexp(const float* in, float* out, size_t n);

/**
 * @brief 向量化 logf（批量 SIMD）
 */
void ct_simd_vlog(const float* in, float* out, size_t n);

/**
 * @brief 向量化 tanh（批量 SIMD）
 */
void ct_simd_vtanh(const float* in, float* out, size_t n);

/**
 * @brief 向量化 sigmoid（批量 SIMD）
 */
void ct_simd_vsigmoid(const float* in, float* out, size_t n);

/**
 * @brief 向量化 GELU（批量 SIMD）
 */
void ct_simd_vgelu(const float* in, float* out, size_t n);

// ======================= 批量算术运算 =======================

/**
 * @brief 向量化加法（批量 SIMD）
 * @param a  输入数组 A
 * @param b  输入数组 B
 * @param out 输出数组（可等于 a 实现 in-place）
 * @param n   元素数
 */
void ct_simd_vadd(const float* a, const float* b, float* out, size_t n);

/**
 * @brief 向量化乘法（批量 SIMD）
 */
void ct_simd_vmul(const float* a, const float* b, float* out, size_t n);

/**
 * @brief 向量化减法（批量 SIMD）
 */
void ct_simd_vsub(const float* a, const float* b, float* out, size_t n);

/**
 * @brief 向量化除法（批量 SIMD）
 */
void ct_simd_vdiv(const float* a, const float* b, float* out, size_t n);

// ======================= 批量一元运算 =======================

/**
 * @brief 向量化取负（批量 SIMD）
 */
void ct_simd_vneg(const float* in, float* out, size_t n);

/**
 * @brief 向量化 ReLU（批量 SIMD）
 */
void ct_simd_vrelu(const float* in, float* out, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CTORCH_KERNELS_SIMD_WRAPPER_H