/**
 * @file kernels.h
 * @brief Ctorch 算子统一声明头
 * @author GhostFace
 * @date 2025/12/20
 */

#ifndef KERNELS_H
#define KERNELS_H
#include "./../Tensor.h"

/**
 * @brief 统一函数指针类型定义
 */

/**
 * @brief 双输入算子函数指针类型
 * @details 用于表示需要两个输入张量的算子，如+、-、*、/等
 * @param a 第一个输入张量
 * @param b 第二个输入张量
 * @return 计算结果张量
 */
typedef Tensor (*BinaryKernelFunc)(const Tensor& a, const Tensor& b);

/**
 * @brief 单输入算子函数指针类型
 * @details 用于表示只需要一个输入张量的算子，如负号、ReLU等
 * @param a 输入张量
 * @return 计算结果张量
 */
typedef Tensor (*UnaryKernelFunc)(const Tensor& a);

/**
 * @brief 基本加法算子实现
 * @details 执行两个张量的元素级加法操作
 * @param a 第一个输入张量
 * @param b 第二个输入张量
 * @return 加法结果张量
 */
Tensor Add_BASIC_kernel(const Tensor& a, const Tensor& b);
Tensor Add_SIMD_kernel(const Tensor& a, const Tensor& b);
Tensor Add_CUDA_kernel(const Tensor& a, const Tensor& b);
Tensor Add_AMX_kernel(const Tensor& a, const Tensor& b);
Tensor Add_MPS_kernel(const Tensor& a, const Tensor& b);

/**
 * @brief 基本减法算子实现
 * @details 执行两个张量的元素级减法操作
 * @param a 第一个输入张量
 * @param b 第二个输入张量
 * @return 减法结果张量
 */
Tensor Sub_BASIC_kernel(const Tensor& a, const Tensor& b);
Tensor Sub_SIMD_kernel(const Tensor& a, const Tensor& b);
Tensor Sub_CUDA_kernel(const Tensor& a, const Tensor& b);
Tensor Sub_AMX_kernel(const Tensor& a, const Tensor& b);
Tensor Sub_MPS_kernel(const Tensor& a, const Tensor& b);

/**
 * @brief 基本乘法算子实现
 * @details 执行两个张量的元素级乘法操作
 * @param a 第一个输入张量
 * @param b 第二个输入张量
 * @return 乘法结果张量
 */
Tensor Mul_BASIC_kernel(const Tensor& a, const Tensor& b);
Tensor Mul_SIMD_kernel(const Tensor& a, const Tensor& b);
Tensor Mul_CUDA_kernel(const Tensor& a, const Tensor& b);
Tensor Mul_AMX_kernel(const Tensor& a, const Tensor& b);
Tensor Mul_MPS_kernel(const Tensor& a, const Tensor& b);

/**
 * @brief 基本除法算子实现
 * @details 执行两个张量的元素级除法操作
 * @param a 第一个输入张量
 * @param b 第二个输入张量
 * @return 除法结果张量
 */
Tensor Div_BASIC_kernel(const Tensor& a, const Tensor& b);
Tensor Div_SIMD_kernel(const Tensor& a, const Tensor& b);
Tensor Div_CUDA_kernel(const Tensor& a, const Tensor& b);
Tensor Div_AMX_kernel(const Tensor& a, const Tensor& b);
Tensor Div_MPS_kernel(const Tensor& a, const Tensor& b);

/**
 * @brief 基本矩阵乘法算子实现
 * @details 执行两个张量的矩阵乘法操作
 * @param a 第一个输入张量
 * @param b 第二个输入张量
 * @return 矩阵乘法结果张量
 */
Tensor MatMul_BASIC_kernel(const Tensor& a, const Tensor& b);
Tensor MatMul_SIMD_kernel(const Tensor& a, const Tensor& b);
Tensor MatMul_CUDA_kernel(const Tensor& a, const Tensor& b);
Tensor MatMul_AMX_kernel(const Tensor& a, const Tensor& b);
Tensor MatMul_MPS_kernel(const Tensor& a, const Tensor& b);

/**
 * @brief 基本点乘算子实现
 * @details 执行两个张量的点乘操作
 * @param a 第一个输入张量
 * @param b 第二个输入张量
 * @return 点乘结果张量
 */
Tensor Dot_BASIC_kernel(const Tensor& a, const Tensor& b);
Tensor Dot_SIMD_kernel(const Tensor& a, const Tensor& b);
Tensor Dot_CUDA_kernel(const Tensor& a, const Tensor& b);
Tensor Dot_AMX_kernel(const Tensor& a, const Tensor& b);
Tensor Dot_MPS_kernel(const Tensor& a, const Tensor& b);

/**
 * @brief 基本负号算子实现
 * @details 执行张量的负号操作
 * @param a 输入张量
 * @return 负号结果张量
 */
Tensor Neg_BASIC_kernel(const Tensor& a);
Tensor Neg_SIMD_kernel(const Tensor& a);
Tensor Neg_CUDA_kernel(const Tensor& a);
Tensor Neg_AMX_kernel(const Tensor& a);
Tensor Neg_MPS_kernel(const Tensor& a);

/**
 * @brief 基本余弦算子实现
 * @details 执行张量的余弦操作
 * @param a 输入张量
 * @return 余弦结果张量
 */
Tensor Cos_BASIC_kernel(const Tensor& a);
Tensor Cos_SIMD_kernel(const Tensor& a);
Tensor Cos_CUDA_kernel(const Tensor& a);
Tensor Cos_AMX_kernel(const Tensor& a);
Tensor Cos_MPS_kernel(const Tensor& a);

/**
 * @brief 基本正弦算子实现
 * @details 执行张量的正弦操作
 * @param a 输入张量
 * @return 正弦结果张量
 */
Tensor Sin_BASIC_kernel(const Tensor& a);
Tensor Sin_SIMD_kernel(const Tensor& a);
Tensor Sin_CUDA_kernel(const Tensor& a);
Tensor Sin_AMX_kernel(const Tensor& a);
Tensor Sin_MPS_kernel(const Tensor& a);

/**
 * @brief 基本ReLU算子实现
 * @details 执行张量的ReLU操作
 * @param a 输入张量
 * @return ReLU结果张量
 */
Tensor ReLU_BASIC_kernel(const Tensor& a);
Tensor ReLU_SIMD_kernel(const Tensor& a);
Tensor ReLU_CUDA_kernel(const Tensor& a);
Tensor ReLU_AMX_kernel(const Tensor& a);
Tensor ReLU_MPS_kernel(const Tensor& a);

/**
 * @brief 基本Tanh算子实现
 * @details 执行张量的Tanh操作
 * @param a 输入张量
 * @return Tanh结果张量
 */
Tensor Tanh_BASIC_kernel(const Tensor& a);
Tensor Tanh_SIMD_kernel(const Tensor& a);
Tensor Tanh_CUDA_kernel(const Tensor& a);
Tensor Tanh_AMX_kernel(const Tensor& a);
Tensor Tanh_MPS_kernel(const Tensor& a);

/**
 * @brief 基本Sigmoid算子实现
 * @details 执行张量的Sigmoid操作
 * @param a 输入张量
 * @return Sigmoid结果张量
 */
Tensor Sigmoid_BASIC_kernel(const Tensor& a);
Tensor Sigmoid_SIMD_kernel(const Tensor& a);
Tensor Sigmoid_CUDA_kernel(const Tensor& a);
Tensor Sigmoid_AMX_kernel(const Tensor& a);
Tensor Sigmoid_MPS_kernel(const Tensor& a);

/**
 * @brief 基本Softmax算子实现
 * @details 执行张量的Softmax操作
 * @param a 输入张量
 * @return Softmax结果张量
 */
Tensor Softmax_BASIC_kernel(const Tensor& a);
Tensor Softmax_SIMD_kernel(const Tensor& a);
Tensor Softmax_CUDA_kernel(const Tensor& a);
Tensor Softmax_AMX_kernel(const Tensor& a);
Tensor Softmax_MPS_kernel(const Tensor& a);

/**
 * @brief 基本LReLU算子实现
 * @details 执行张量的LReLU操作
 * @param a 输入张量
 * @return LReLU结果张量
 */
Tensor LReLU_BASIC_kernel(const Tensor& a);
Tensor LReLU_SIMD_kernel(const Tensor& a);
Tensor LReLU_CUDA_kernel(const Tensor& a);
Tensor LReLU_AMX_kernel(const Tensor& a);
Tensor LReLU_MPS_kernel(const Tensor& a);

/**
 * @brief 基本MSE算子实现
 * @details 执行张量的MSE操作
 * @param a 第一个输入张量
 * @param b 第二个输入张量
 * @return MSE结果张量
 */
Tensor MSE_BASIC_kernel(const Tensor& a, const Tensor& b);
Tensor MSE_SIMD_kernel(const Tensor& a, const Tensor& b);
Tensor MSE_CUDA_kernel(const Tensor& a, const Tensor& b);
Tensor MSE_AMX_kernel(const Tensor& a, const Tensor& b);
Tensor MSE_MPS_kernel(const Tensor& a, const Tensor& b);

/**
 * @brief 基本CrossEntropy算子实现
 * @details 执行张量的CrossEntropy操作
 * @param a 第一个输入张量
 * @param b 第二个输入张量
 * @return CrossEntropy结果张量
 */
Tensor CrossEntropy_BASIC_kernel(const Tensor& a, const Tensor& b);
Tensor CrossEntropy_SIMD_kernel(const Tensor& a, const Tensor& b);
Tensor CrossEntropy_CUDA_kernel(const Tensor& a, const Tensor& b);
Tensor CrossEntropy_AMX_kernel(const Tensor& a, const Tensor& b);
Tensor CrossEntropy_MPS_kernel(const Tensor& a, const Tensor& b);

/**
 * @brief 基本MAE算子实现
 * @details 执行张量的MAE操作
 * @param a 第一个输入张量
 * @param b 第二个输入张量
 * @return MAE结果张量
 */
Tensor MAE_BASIC_kernel(const Tensor& a, const Tensor& b);
Tensor MAE_SIMD_kernel(const Tensor& a, const Tensor& b);
Tensor MAE_CUDA_kernel(const Tensor& a, const Tensor& b);
Tensor MAE_AMX_kernel(const Tensor& a, const Tensor& b);
Tensor MAE_MPS_kernel(const Tensor& a, const Tensor& b);

#endif //KERNELS_H
