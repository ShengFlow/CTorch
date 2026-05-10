//
// Created by renyz on 2026/3/15.
//

#ifndef CTORCH_FEATURES_H
#define CTORCH_FEATURES_H

#include "CoreDefs.h"

/**
 * Detect compiler
 */
#if defined(_MSC_VER)
#define COMPILER_MSVC 1
#elif defined(__clang__)
#define COMPILER_CLANG 1
#elif defined(__GNUC__)
#define COMPILER_GCC 1
#else
#error "Unsupported compiler"
#endif

/**
 * Detect CPU architecture
 */
// x86_64 / AMD64
#if defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
#define ARCH_X86_64 1
#define ARCH_X86_FAMILY 1
// x86 (32-bit)
#elif defined(_M_IX86) || defined(__i386__) || defined(__i686__)
#define ARCH_X86 1
  #define ARCH_X86_FAMILY 1
// ARM64 / AArch64
#elif defined(_M_ARM64) || defined(__aarch64__) || defined(__arm64__)
  #define ARCH_ARM64 1
  #define ARCH_ARM_FAMILY 1
// ARM (32-bit)
#elif defined(_M_ARM) || defined(__arm__) || defined(__ARM_ARCH)
  #define ARCH_ARM 1
  #define ARCH_ARM_FAMILY 1
#endif

// ============================================================================
//                    x86/x86_64 指令集检测
// ============================================================================

#if defined(ARCH_X86_FAMILY)

// ==================== SSE 系列 ====================

// SSE
#if defined(COMPILER_MSVC)
  #if defined(_M_IX86_FP) && _M_IX86_FP >= 1
    #define HAS_SSE 1
  #elif defined(ARCH_X86_64)  // x64默认有SSE
    #define HAS_SSE 1
  #endif
#elif defined(__SSE__)
  #define HAS_SSE 1
#endif

// SSE2
#if defined(COMPILER_MSVC)
  #if defined(_M_IX86_FP) && _M_IX86_FP >= 2
    #define HAS_SSE2 1
  #elif defined(ARCH_X86_64)  // x64默认有SSE2
    #define HAS_SSE2 1
  #endif
#elif defined(__SSE2__)
  #define HAS_SSE2 1
#endif

// SSE3
#if defined(__SSE3__)
  #define HAS_SSE3 1
#endif

// SSSE3
#if defined(__SSSE3__)
  #define HAS_SSSE3 1
#endif

// SSE4.1
#if defined(__SSE4_1__)
  #define HAS_SSE4_1 1
#endif

// SSE4.2
#if defined(__SSE4_2__)
  #define HAS_SSE4_2 1
#endif

// F16C (Half-precision conversion)
#if defined(__F16C__)
  #define HAS_F16C 1
#endif

// ==================== AVX 系列 ====================

// AVX
#if defined(__AVX__)
  #define HAS_AVX 1
#endif

// AVX2
#if defined(__AVX2__)
  #define HAS_AVX2 1
#endif

// FMA (Fused Multiply-Add)
#if defined(__FMA__)
  #define HAS_FMA 1
#endif

// FMA4 (AMD)
#if defined(__FMA4__)
  #define HAS_FMA4 1
#endif

// XOP (AMD)
#if defined(__XOP__)
  #define HAS_XOP 1
#endif

// ==================== AVX-512 系列 ====================

// AVX512F (基础，必需)
#if defined(__AVX512F__)
  #define HAS_AVX512F 1
#endif

// AVX512BW (Byte/Word操作)
#if defined(__AVX512BW__)
  #define HAS_AVX512BW 1
#endif

// AVX512CD (Conflict Detection)
#if defined(__AVX512CD__)
  #define HAS_AVX512CD 1
#endif

// AVX512DQ (Doubleword/Quadword)
#if defined(__AVX512DQ__)
  #define HAS_AVX512DQ 1
#endif

// AVX512VL (Vector Length)
#if defined(__AVX512VL__)
  #define HAS_AVX512VL 1
#endif

// AVX512IFMA (Integer Fused Multiply Add)
#if defined(__AVX512IFMA__)
  #define HAS_AVX512IFMA 1
#endif

// AVX512VBMI (Vector Byte Manipulation)
#if defined(__AVX512VBMI__)
  #define HAS_AVX512VBMI 1
#endif

// AVX512VBMI2
#if defined(__AVX512VBMI2__)
  #define HAS_AVX512VBMI2 1
#endif

// AVX512VNNI (Vector Neural Network Instructions)
#if defined(__AVX512VNNI__)
  #define HAS_AVX512VNNI 1
#endif

// AVX512BF16 (BFloat16)
#if defined(__AVX512BF16__)
  #define HAS_AVX512_BF16 1
#endif

// AVX512FP16 (FP16)
#if defined(__AVX512FP16__)
  #define HAS_AVX512_FP16 1
#endif

// AVX512VPOPCNTDQ (Population Count)
#if defined(__AVX512VPOPCNTDQ__)
  #define HAS_AVX512VPOPCNTDQ 1
#endif

// AVX512BITALG
#if defined(__AVX512BITALG__)
  #define HAS_AVX512BITALG 1
#endif

// AVX512_4FMAPS
#if defined(__AVX5124FMAPS__)
  #define HAS_AVX512_4FMAPS 1
#endif

// AVX512_4VNNIW
#if defined(__AVX5124VNNIW__)
  #define HAS_AVX512_4VNNIW 1
#endif

// ==================== AMX 系列 ====================

// AMX TILE
#if defined(__AMX_TILE__)
  #define HAS_AMX_TILE 1
#endif

// AMX INT8
#if defined(__AMX_INT8__)
  #define HAS_AMX_INT8 1
#endif

// AMX BF16
#if defined(__AMX_BF16__)
  #define HAS_AMX_BF16 1
#endif

// AMX FP16
#if defined(__AMXFP16__)
  #define HAS_AMX_FP16 1
#endif

// AMX COMPLEX
#if defined(__AMX_COMPLEX__)
  #define HAS_AMX_COMPLEX 1
#endif

// ==================== AVX10 ====================

#if defined(__AVX10_VER__)
  #define HAS_AVX10 1
  #define AVX10_VERSION __AVX10_VER__
#endif

// ==================== 其他指令集 ====================

// POPCNT
#if defined(__POPCNT__)
  #define HAS_POPCNT 1
#endif

// BMI (Bit Manipulation Instructions)
#if defined(__BMI__)
  #define HAS_BMI 1
#endif

// BMI2
#if defined(__BMI2__)
  #define HAS_BMI2 1
#endif

// LZCNT
#if defined(__LZCNT__)
  #define HAS_LZCNT 1
#endif

// TZCNT
#if defined(__TZCNT__)
  #define HAS_TZCNT 1
#endif

// AES-NI
#if defined(__AES__)
  #define HAS_AES_NI 1
#endif

// SHA
#if defined(__SHA__)
  #define HAS_SHA 1
#endif

// PCLMULQDQ
#if defined(__PCLMUL__)
  #define HAS_PCLMULQDQ 1
#endif

// RDRAND
#if defined(__RDRND__)
  #define HAS_RDRAND 1
#endif

// RDSEED
#if defined(__RDSEED__)
  #define HAS_RDSEED 1
#endif

// ADX (Multi-Precision Add-Carry)
#if defined(__ADX__)
  #define HAS_ADX 1
#endif

#endif // ARCH_X86_FAMILY


// ============================================================================
//                    ARM 指令集检测
// ============================================================================

#if defined(ARCH_ARM_FAMILY)

// ==================== 架构版本检测 ====================

// ARM架构版本 (整数: 7, 8, 9 等)
#if defined(__ARM_ARCH)
  #define ARM_ARCH_VERSION __ARM_ARCH
#elif defined(__ARM_ARCH_9__)
  #define ARM_ARCH_VERSION 9
#elif defined(__ARM_ARCH_8__)
  #define ARM_ARCH_VERSION 8
#elif defined(__ARM_ARCH_7__)
  #define ARM_ARCH_VERSION 7
#elif defined(__ARM_ARCH_6__)
  #define ARM_ARCH_VERSION 6
#elif defined(__ARM_ARCH_5__)
  #define ARM_ARCH_VERSION 5
#elif defined(__ARM_ARCH_4__)
  #define ARM_ARCH_VERSION 4
#endif

// 架构配置文件 ('A' = Application, 'R' = Real-time, 'M' = Microcontroller)
#if defined(__ARM_ARCH_PROFILE)
  #define ARM_ARCH_PROFILE __ARM_ARCH_PROFILE
#elif defined(__ARM_ARCH_9A__) || defined(__ARM_ARCH_8A__)
  #define ARM_ARCH_PROFILE 'A'
#elif defined(__ARM_ARCH_8R__)
  #define ARM_ARCH_PROFILE 'R'
#elif defined(__ARM_ARCH_8M__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_6M__)
  #define ARM_ARCH_PROFILE 'M'
#endif

// ARMv8 具体版本
#if defined(__ARM_ARCH_8_1A__)
  #define ARM_ARCH_8_1A 1
#elif defined(__ARM_ARCH_8_2A__)
  #define ARM_ARCH_8_2A 1
#elif defined(__ARM_ARCH_8_3A__)
  #define ARM_ARCH_8_3A 1
#elif defined(__ARM_ARCH_8_4A__)
  #define ARM_ARCH_8_4A 1
#elif defined(__ARM_ARCH_8_5A__)
  #define ARM_ARCH_8_5A 1
#elif defined(__ARM_ARCH_8_6A__)
  #define ARM_ARCH_8_6A 1
#elif defined(__ARM_ARCH_8_7A__)
  #define ARM_ARCH_8_7A 1
#elif defined(__ARM_ARCH_8_8A__)
  #define ARM_ARCH_8_8A 1
#elif defined(__ARM_ARCH_8_9A__)
  #define ARM_ARCH_8_9A 1
#endif

// ARMv9 检测
#if defined(__ARM_ARCH_9A__)
  #define ARM_ARCH_V9 1
#elif defined(__ARM_ARCH_9_1A__)
  #define ARM_ARCH_9_1A 1
#elif defined(__ARM_ARCH_9_2A__)
  #define ARM_ARCH_9_2A 1
#elif defined(__ARM_ARCH_9_3A__)
  #define ARM_ARCH_9_3A 1
#elif defined(__ARM_ARCH_9_4A__)
  #define ARM_ARCH_9_4A 1
#endif

// ARMv9 总体检测
#if defined(ARM_ARCH_V9) || defined(ARM_ARCH_9_1A) || defined(ARM_ARCH_9_2A) || \
    defined(ARM_ARCH_9_3A) || defined(ARM_ARCH_9_4A)
  #define IS_ARMV9_OR_LATER 1
#endif

// ==================== NEON (Advanced SIMD) ====================

// NEON 支持 (ACLE 标准宏)
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #define HAS_NEON 1
#endif

// NEON 浮点支持
#if defined(__ARM_NEON_FP)
  #define HAS_NEON_FP __ARM_NEON_FP
#endif

// NEON FP16 算术支持
#if defined(__ARM_FEATURE_FP16_ARITHMETIC)
  #define HAS_NEON_FP16_ARITH 1
#endif

// NEON FML 扩展 (FP16 Fused Multiply-Add)
#if defined(__ARM_FEATURE_FP16_FML)
  #define HAS_NEON_FP16_FML 1
#endif

// 32-bit SIMD (ARMv6 SIMD)
#if defined(__ARM_FEATURE_SIMD32)
  #define HAS_ARM_SIMD32 1
#endif

// ==================== SVE (Scalable Vector Extension) ====================

// SVE 基础支持
#if defined(__ARM_FEATURE_SVE)
  #define HAS_SVE 1
  #define SVE_VECTOR_BITS __ARM_FEATURE_SVE
#endif

// SVE 向量长度 (如果编译时指定)
#if defined(__ARM_FEATURE_SVE_BITS)
  #define HAS_FIXED_SVE_BITS 1
  #define FIXED_SVE_BITS __ARM_FEATURE_SVE_BITS
#endif

// SVE 向量运算符
#if defined(__ARM_FEATURE_SVE_VECTOR_OPERATORS)
  #define HAS_SVE_VECTOR_OPERATORS 1
#endif

// ==================== SVE2 ====================

// SVE2 基础支持
#if defined(__ARM_FEATURE_SVE2)
  #define HAS_SVE2 1
#endif

// SVE2 AES 扩展
#if defined(__ARM_FEATURE_SVE2_AES)
  #define HAS_SVE2_AES 1
#endif

// SVE2 Bitperm 扩展
#if defined(__ARM_FEATURE_SVE2_BITPERM)
  #define HAS_SVE2_BITPERM 1
#endif

// SVE2 SHA3 扩展
#if defined(__ARM_FEATURE_SVE2_SHA3)
  #define HAS_SVE2_SHA3 1
#endif

// SVE2 SM4 扩展
#if defined(__ARM_FEATURE_SVE2_SM4)
  #define HAS_SVE2_SM4 1
#endif

// SVE2.1
#if defined(__ARM_FEATURE_SVE2p1)
  #define HAS_SVE2p1 1
#endif

// ==================== SME (Scalable Matrix Extension) ====================

// SME 基础支持
#if defined(__ARM_FEATURE_SME)
  #define HAS_SME 1
#endif

// SME F64F64 (64-bit 浮点矩阵)
#if defined(__ARM_FEATURE_SME_F64F64)
  #define HAS_SME_F64F64 1
#endif

// SME F16F16 (FP16 矩阵)
#if defined(__ARM_FEATURE_SME_F16F16)
  #define HAS_SME_F16F16 1
#endif

// SME F16F32 (FP16->FP32 矩阵)
#if defined(__ARM_FEATURE_SME_F16F32)
  #define HAS_SME_F16F32 1
#endif

// SME F8F16 (FP8->FP16 矩阵)
#if defined(__ARM_FEATURE_SME_F8F16)
  #define HAS_SME_F8F16 1
#endif

// SME F8F32 (FP8->FP32 矩阵)
#if defined(__ARM_FEATURE_SME_F8F32)
  #define HAS_SME_F8F32 1
#endif

// SME I16I32 (INT16->INT32 矩阵)
#if defined(__ARM_FEATURE_SME_I16I32)
  #define HAS_SME_I16I32 1
#endif

// SME I16I64 (INT16->INT64 矩阵)
#if defined(__ARM_FEATURE_SME_I16I64)
  #define HAS_SME_I16I64 1
#endif

// SME BI16I32
#if defined(__ARM_FEATURE_SME_BI16I32)
  #define HAS_SME_BI16I32 1
#endif

// SME2 支持
#if defined(__ARM_FEATURE_SME2)
  #define HAS_SME2 1
#endif

// SME2.1 支持
#if defined(__ARM_FEATURE_SME2p1)
  #define HAS_SME2p1 1
#endif

// ==================== MVE (M-profile Vector Extension) ====================

// MVE (Cortex-M 系列 SIMD)
#if defined(__ARM_FEATURE_MVE)
  #define HAS_MVE 1
#endif

// MVE 整数支持
#if defined(__ARM_FEATURE_MVE_I8)
  #define HAS_MVE_I8 1
#endif

#if defined(__ARM_FEATURE_MVE_I16)
  #define HAS_MVE_I16 1
#endif

#if defined(__ARM_FEATURE_MVE_I32)
  #define HAS_MVE_I32 1
#endif

// MVE 浮点支持
#if defined(__ARM_FEATURE_MVE_FP)
  #define HAS_MVE_FP 1
#endif

// ==================== 加密扩展 ====================

// CRC32
#if defined(__ARM_FEATURE_CRC32)
  #define HAS_CRC32 1
#endif

// AES
#if defined(__ARM_FEATURE_AES)
  #define HAS_AES 1
#endif

// SHA2 (SHA-256)
#if defined(__ARM_FEATURE_SHA2)
  #define HAS_SHA2 1
#endif

// SHA3
#if defined(__ARM_FEATURE_SHA3)
  #define HAS_SHA3 1
#endif

// SHA512
#if defined(__ARM_FEATURE_SHA512)
  #define HAS_SHA512 1
#endif

// SM3 (国密)
#if defined(__ARM_FEATURE_SM3)
  #define HAS_SM3 1
#endif

// SM4 (国密)
#if defined(__ARM_FEATURE_SM4)
  #define HAS_SM4 1
#endif

// PMULL (64-bit多项式乘法)
#if defined(__ARM_FEATURE_PMULL)
  #define HAS_PMULL 1
#endif

// ==================== 计算扩展 ====================

// Dot Product (点积)
#if defined(__ARM_FEATURE_DOTPROD)
  #define HAS_DOTPROD 1
#endif

// BFloat16
#if defined(__ARM_FEATURE_BF16)
  #define HAS_BF16 1
#endif

// BFloat16 算术
#if defined(__ARM_FEATURE_BF16_ARITHMETIC)
  #define HAS_BF16_ARITHMETIC 1
#endif

// INT8 矩阵乘法
#if defined(__ARM_FEATURE_MATMUL_INT8)
  #define HAS_MATMUL_INT8 1
#endif

// FP32 矩阵乘法
#if defined(__ARM_FEATURE_MATMUL_FP32)
  #define HAS_MATMUL_FP32 1
#endif

// FP64 矩阵乘法
#if defined(__ARM_FEATURE_MATMUL_FP64)
  #define HAS_MATMUL_FP64 1
#endif

// BFloat16 矩阵乘法
#if defined(__ARM_FEATURE_MATMUL_BF16)
  #define HAS_MATMUL_BF16 1
#endif

// I8MM (Int8 Matrix Multiply)
#if defined(__ARM_FEATURE_I8MM)
  #define HAS_I8MM 1
#endif

// ==================== 浮点扩展 ====================

// 硬件浮点
#if defined(__ARM_FP)
  #define HAS_VFP __ARM_FP
#endif

// FMA (Fused Multiply-Add)
#if defined(__ARM_FEATURE_FMA)
  #define HAS_ARM_FMA 1
#endif

// FP16 (Half-precision)
#if defined(__ARM_FEATURE_FP16)
  #define HAS_FP16 1
#endif

// FP16 算术
#if defined(__ARM_FEATURE_FP16_ARITHMETIC)
  #define HAS_FP16_ARITHMETIC 1
#endif

// ==================== 其他特性 ====================

// 硬件整数除法
#if defined(__ARM_FEATURE_IDIV)
  #define HAS_IDIV 1
#endif

// Atomics (LSE - Large System Extensions)
#if defined(__ARM_FEATURE_ATOMICS)
  #define HAS_LSE 1
#endif

// RCPC (Release Consistent Processor Consistent)
#if defined(__ARM_FEATURE_RCPC)
  #define HAS_RCPC 1
#endif

// RCPC3
#if defined(__ARM_FEATURE_RCPC3)
  #define HAS_RCPC3 1
#endif

// MOPS (Memory Operations)
#if defined(__ARM_FEATURE_MOPS)
  #define HAS_MOPS 1
#endif

// Branch Target Identification
#if defined(__ARM_FEATURE_BTI)
  #define HAS_BTI 1
#endif

// Pointer Authentication
#if defined(__ARM_FEATURE_PAUTH)
  #define HAS_PAUTH 1
#endif

// Memory Tagging Extension
#if defined(__ARM_FEATURE_MEMORY_TAGGING)
  #define HAS_MTE 1
#endif

// Guarded Control Stack
#if defined(__ARM_FEATURE_GCS)
  #define HAS_GCS 1
#endif

// JSCONV (JavaScript Conversion)
#if defined(__ARM_FEATURE_JCVT)
  #define HAS_JSCONV 1
#endif

// FRINT (Floating-point Round to Integer)
#if defined(__ARM_FEATURE_FRINT)
  #define HAS_FRINT 1
#endif

// RDM (Round Doubling Multiply)
#if defined(__ARM_FEATURE_RDM)
  #define HAS_RDM 1
#endif

// Complex Number
#if defined(__ARM_FEATURE_COMPLEX)
  #define HAS_COMPLEX 1
#endif

// Q (Saturation) Flag
#if defined(__ARM_FEATURE_QBIT)
  #define HAS_QBIT 1
#endif

// DSP Instructions
#if defined(__ARM_FEATURE_DSP)
  #define HAS_DSP 1
#endif

// Unaligned Access
#if defined(__ARM_FEATURE_UNALIGNED)
  #define HAS_UNALIGNED 1
#endif

// CLZ (Count Leading Zeros)
#if defined(__ARM_FEATURE_CLZ)
  #define HAS_CLZ 1
#endif

// CB/CZB (Compare and Branch)
#if defined(__ARM_FEATURE_CB)
  #define HAS_CB 1
#endif

// SEL (Select)
#if defined(__ARM_FEATURE_SEL)
  #define HAS_SEL 1
#endif

// LDREX/STREX (Exclusive Load/Store)
#if defined(__ARM_FEATURE_LDREX)
  #define HAS_LDREX 1
#endif

// ==================== MSVC 特殊处理 ====================

#if defined(COMPILER_MSVC) && defined(ARCH_ARM64)
  // MSVC for ARM64
  #if defined(_M_ARM64)
    #define HAS_NEON 1
    #define HAS_VFP 1
  #endif

  // MSVC ARM64 Crypto
  #if defined(_M_ARM64CRYPTO)
    #define HAS_AES 1
    #define HAS_SHA2 1
    #define HAS_PMULL 1
  #endif
#endif

#endif // ARCH_ARM_FAMILY


// ============================================================================
//                           便捷宏定义
// ============================================================================

// SIMD 宽度级别 (适用于两种架构)
#if defined(HAS_AVX512F)
  #define SIMD_WIDTH 512
#elif defined(HAS_AVX)
  #define SIMD_WIDTH 256
#elif defined(HAS_SSE) || defined(HAS_NEON)
  #define SIMD_WIDTH 128
#elif defined(HAS_SVE)
  #define SIMD_WIDTH (-1) // Special width denoting scalable size
#else
  #define SIMD_WIDTH 0  // Scalar
#endif

// 检测是否有任何 SIMD 支持
#if defined(HAS_SSE) || defined(HAS_NEON) || defined(HAS_SVE) || defined(HAS_MVE)
  #define HAS_SIMD 1
#endif

// 检测是否有矩阵扩展
#if defined(HAS_AMX_TILE) || defined(HAS_SME)
#define HAS_MATRIX_EXTENSION 1
#endif

// 检测是否支持 BF16
#if defined(HAS_AVX512_BF16) || defined(HAS_BF16) || defined(HAS_SME_BI16I32)
  #define HAS_BFLOAT16 1
#endif

// 检测是否支持 FP16
#if defined(HAS_AVX512_FP16) || defined(HAS_FP16) || defined(HAS_NEON_FP16_ARITH)
  #define HAS_HALF_PRECISION 1
#endif


// ============================================================================
//                           辅助函数宏
// ============================================================================

// 架构名称字符串
#if defined(ARCH_X86_64)
  #define ARCH_NAME "x86_64"
#elif defined(ARCH_X86)
  #define ARCH_NAME "x86"
#elif defined(ARCH_ARM64)
  #define ARCH_NAME "ARM64"
#elif defined(ARCH_ARM)
  #define ARCH_NAME "ARM"
#else
  #define ARCH_NAME "Unknown"
#endif

// SIMD 名称
#if defined(HAS_AVX512F)
  #define SIMD_NAME "AVX-512"
#elif defined(HAS_AVX2)
  #define SIMD_NAME "AVX2"
#elif defined(HAS_AVX)
  #define SIMD_NAME "AVX"
#elif defined(HAS_SVE2)
  #define SIMD_NAME "SVE2"
#elif defined(HAS_SVE)
  #define SIMD_NAME "SVE"
#elif defined(HAS_SSE4_2)
  #define SIMD_NAME "SSE4.2"
#elif defined(HAS_SSE4_1)
  #define SIMD_NAME "SSE4.1"
#elif defined(HAS_SSSE3)
  #define SIMD_NAME "SSSE3"
#elif defined(HAS_SSE3)
  #define SIMD_NAME "SSE3"
#elif defined(HAS_SSE2)
  #define SIMD_NAME "SSE2"
#elif defined(HAS_SSE)
  #define SIMD_NAME "SSE"
#elif defined(HAS_NEON)
  #define SIMD_NAME "NEON"
#elif defined(HAS_MVE)
  #define SIMD_NAME "MVE"
#else
  #define SIMD_NAME "Scalar"
#endif

#endif //CTORCH_FEATURES_H
