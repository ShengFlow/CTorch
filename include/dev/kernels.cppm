module;                                                                   
import Ctools;                                                          
import Tensor;                                                          
import Ctorch_Error;                                                    

export module kernels;                                                   

export typedef Tensor (*BinaryKernelFunc)(const Tensor& a, const Tensor& b);
export typedef Tensor (*UnaryKernelFunc)(const Tensor& a);

export Tensor Add_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor Add_SIMD_kernel(const Tensor& a, const Tensor& b);
export Tensor Add_CUDA_kernel(const Tensor& a, const Tensor& b);
export Tensor Add_AMX_kernel(const Tensor& a, const Tensor& b);
export Tensor Add_MPS_kernel(const Tensor& a, const Tensor& b);

export Tensor Sub_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor Sub_SIMD_kernel(const Tensor& a, const Tensor& b);
export Tensor Sub_CUDA_kernel(const Tensor& a, const Tensor& b);
export Tensor Sub_AMX_kernel(const Tensor& a, const Tensor& b);
export Tensor Sub_MPS_kernel(const Tensor& a, const Tensor& b);

export Tensor Mul_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor Mul_SIMD_kernel(const Tensor& a, const Tensor& b);
export Tensor Mul_CUDA_kernel(const Tensor& a, const Tensor& b);
export Tensor Mul_AMX_kernel(const Tensor& a, const Tensor& b);
export Tensor Mul_MPS_kernel(const Tensor& a, const Tensor& b);

export Tensor Div_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor Div_SIMD_kernel(const Tensor& a, const Tensor& b);
export Tensor Div_CUDA_kernel(const Tensor& a, const Tensor& b);
export Tensor Div_AMX_kernel(const Tensor& a, const Tensor& b);
export Tensor Div_MPS_kernel(const Tensor& a, const Tensor& b);

export Tensor MatMul_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor MatMul_SIMD_kernel(const Tensor& a, const Tensor& b);
export Tensor MatMul_CUDA_kernel(const Tensor& a, const Tensor& b);
export Tensor MatMul_AMX_kernel(const Tensor& a, const Tensor& b);
export Tensor MatMul_MPS_kernel(const Tensor& a, const Tensor& b);

export Tensor Dot_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor Dot_SIMD_kernel(const Tensor& a, const Tensor& b);
export Tensor Dot_CUDA_kernel(const Tensor& a, const Tensor& b);
export Tensor Dot_AMX_kernel(const Tensor& a, const Tensor& b);
export Tensor Dot_MPS_kernel(const Tensor& a, const Tensor& b);

export Tensor Neg_BASIC_kernel(const Tensor& a);
export Tensor Neg_SIMD_kernel(const Tensor& a);
export Tensor Neg_CUDA_kernel(const Tensor& a);
export Tensor Neg_AMX_kernel(const Tensor& a);
export Tensor Neg_MPS_kernel(const Tensor& a);

export Tensor Cos_BASIC_kernel(const Tensor& a);
export Tensor Cos_SIMD_kernel(const Tensor& a);
export Tensor Cos_CUDA_kernel(const Tensor& a);
export Tensor Cos_AMX_kernel(const Tensor& a);
export Tensor Cos_MPS_kernel(const Tensor& a);

export Tensor Sin_BASIC_kernel(const Tensor& a);
export Tensor Sin_SIMD_kernel(const Tensor& a);
export Tensor Sin_CUDA_kernel(const Tensor& a);
export Tensor Sin_AMX_kernel(const Tensor& a);
export Tensor Sin_MPS_kernel(const Tensor& a);

export Tensor ReLU_BASIC_kernel(const Tensor& a);
export Tensor ReLU_SIMD_kernel(const Tensor& a);
export Tensor ReLU_CUDA_kernel(const Tensor& a);
export Tensor ReLU_AMX_kernel(const Tensor& a);
export Tensor ReLU_MPS_kernel(const Tensor& a);

export Tensor Tanh_BASIC_kernel(const Tensor& a);
export Tensor Tanh_SIMD_kernel(const Tensor& a);
export Tensor Tanh_CUDA_kernel(const Tensor& a);
export Tensor Tanh_AMX_kernel(const Tensor& a);
export Tensor Tanh_MPS_kernel(const Tensor& a);

export Tensor Sigmoid_BASIC_kernel(const Tensor& a);
export Tensor Sigmoid_SIMD_kernel(const Tensor& a);
export Tensor Sigmoid_CUDA_kernel(const Tensor& a);
export Tensor Sigmoid_AMX_kernel(const Tensor& a);
export Tensor Sigmoid_MPS_kernel(const Tensor& a);

export Tensor Softmax_BASIC_kernel(const Tensor& a);
export Tensor Softmax_SIMD_kernel(const Tensor& a);
export Tensor Softmax_CUDA_kernel(const Tensor& a);
export Tensor Softmax_AMX_kernel(const Tensor& a);
export Tensor Softmax_MPS_kernel(const Tensor& a);

export Tensor LReLU_BASIC_kernel(const Tensor& a);
export Tensor LReLU_SIMD_kernel(const Tensor& a);
export Tensor LReLU_CUDA_kernel(const Tensor& a);
export Tensor LReLU_AMX_kernel(const Tensor& a);
export Tensor LReLU_MPS_kernel(const Tensor& a);

export Tensor MSE_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor MSE_SIMD_kernel(const Tensor& a, const Tensor& b);
export Tensor MSE_CUDA_kernel(const Tensor& a, const Tensor& b);
export Tensor MSE_AMX_kernel(const Tensor& a, const Tensor& b);
export Tensor MSE_MPS_kernel(const Tensor& a, const Tensor& b);

export Tensor CrossEntropy_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor CrossEntropy_SIMD_kernel(const Tensor& a, const Tensor& b);
export Tensor CrossEntropy_CUDA_kernel(const Tensor& a, const Tensor& b);
export Tensor CrossEntropy_AMX_kernel(const Tensor& a, const Tensor& b);
export Tensor CrossEntropy_MPS_kernel(const Tensor& a, const Tensor& b);

export Tensor MAE_BASIC_kernel(const Tensor& a, const Tensor& b);
export Tensor MAE_SIMD_kernel(const Tensor& a, const Tensor& b);
export Tensor MAE_CUDA_kernel(const Tensor& a, const Tensor& b);
export Tensor MAE_AMX_kernel(const Tensor& a, const Tensor& b);
export Tensor MAE_MPS_kernel(const Tensor& a, const Tensor& b);