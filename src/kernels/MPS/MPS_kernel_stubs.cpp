// Linux stubs for macOS MPS (Metal Performance Shaders) kernel symbols
#include "Tensor.h"
using namespace ct;

// MPS flush/mark - no-ops on Linux
//
// [Linux 首次构建修复] 这两处原先的参数个数与 Tensor.h 的声明不一致：
//   Tensor.h:21  extern "C" void MPS_flush_wait(bool wait);
//   Tensor.h:22  extern "C" void MPS_markBufferModified(void* ptr, size_t bytes);
// 而旧 stub 写成 MPS_flush_wait() 与 MPS_markBufferModified(void*)。
// 由于是 extern "C"，参数不参与 name mangling，链接器**不会报错**——
// 于是这个不匹配静默存在，只在调用方传实参（MPS_flush_wait(true) /
// markBufferModified(ptr, n*sizeof(T))）时语义才错。macOS 走真 MPS 实现故从未暴露。
extern "C" void MPS_flush_wait(bool wait) {
    (void)wait;
}
extern "C" void MPS_markBufferModified(void* ptr, size_t bytes) {
    (void)ptr;
    (void)bytes;
}

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
//
// [Linux 首次构建修复] Softmax 原写成 `extern "C" void ...(const Tensor&, int)`，
// 与 src/kernels/kernels.h:240 的声明 `Tensor Softmax_MPS_kernel(const Tensor&, int = -1)`
// 在返回类型上冲突，且 CtorchScheduler.cpp:205 的 set_softmax 需要取返回值 ——
// void 版本会让该调度项拿到无效结果。改为与宏生成的其余 kernel 一致：
// C++ 链接（非 extern "C"，因为参数含 C++ 类型 Tensor）+ 返回 Tensor。
Tensor Softmax_MPS_kernel(const Tensor& a, int dim) {
    (void)a;
    (void)dim;
    return Tensor();
}

// Zero_MPS_kernel 在仓库内无对应声明（mnist 用的是 SGD_Step_Zero_MPS_kernel，
// 为另一个符号）；保留此定义仅为兼容潜在外部引用，同样去掉 extern "C"。
void Zero_MPS_kernel(const Tensor& a) { (void)a; }
