// Linux stubs for macOS MPS (Metal Performance Shaders) kernel symbols
#include "Tensor.h"
using namespace ct;

// MPS flush/mark - no-ops on Linux
extern "C" void MPS_flush_wait() {}
extern "C" void MPS_markBufferModified(void*) {}

// Unary and Binary MPS kernels - throw or return dummy on Linux
#define STUB_UNARY(name) Tensor name##_MPS_kernel(const Tensor& a) { return Tensor(); }
#define STUB_BINARY(name) Tensor name##_MPS_kernel(const Tensor& a, const Tensor& b) { return Tensor(); }

STUB_UNARY(Neg) STUB_UNARY(ReLU) STUB_UNARY(Sigmoid) STUB_UNARY(Tanh)
STUB_UNARY(Sin) STUB_UNARY(Cos) STUB_UNARY(GELU) STUB_UNARY(LReLU)
STUB_UNARY(Log) STUB_UNARY(Exp) STUB_UNARY(Abs)
STUB_BINARY(Add) STUB_BINARY(Sub) STUB_BINARY(Mul) STUB_BINARY(Div)
STUB_BINARY(MatMul) STUB_BINARY(Dot) STUB_BINARY(MSE) STUB_BINARY(MAE)
STUB_BINARY(CrossEntropy) STUB_BINARY(Max) STUB_BINARY(Min)
STUB_BINARY(LReLU_Grad)

// Inplace variants
#define STUB_INPLACE(name) void name##_MPS_inplace(Tensor& a) {}
STUB_INPLACE(Neg) STUB_INPLACE(ReLU) STUB_INPLACE(Sigmoid) STUB_INPLACE(Tanh)
STUB_INPLACE(Sin) STUB_INPLACE(Cos) STUB_INPLACE(GELU) STUB_INPLACE(LReLU)
STUB_INPLACE(Log) STUB_INPLACE(Exp) STUB_INPLACE(Abs)

// Special signatures
extern "C" void Softmax_MPS_kernel(const Tensor& a, int dim) {}
extern "C" void Zero_MPS_kernel(const Tensor& a) {}
