/**
 * @file MatMul_SIMD_kernel.cpp
 * @brief CPU-SIMD 矩阵乘法算子（朴素向量化，作为 AMX 不可用时的 fallback）
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"

// [Eager 优化 2026-08-27] SIMD MatMul 内部委托 cblas_sgemm（Accelerate）+ 转置折叠。
//   eager 的 CPU MatMul 全部落到本 kernel，朴素 i-k-j 三重循环比 cblas 慢 ~50×（采样证实）。
//   改造后：行列连续 → cblas NoTrans；转置视图(1,R) → cblas Trans 直读（零拷贝，同 MIMO 折叠）；
//   仅真正任意 stride 才回退朴素循环（保正确性）。
//   ⚠️ Accelerate 是 macOS 专属，Linux/DCU 构建需回退原 SIMD 循环（同 AMX 守卫策略）。
#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

#ifdef __APPLE__
/**
 * @brief 把 Tensor 映射为 cblas (trans, lda) 描述。无法映射（任意 stride）返回 nullptr。
 * @param t    目标矩阵（逻辑 rows×cols）
 * @param rows 逻辑行数
 * @param cols 逻辑列数
 * @param trans 输出：是否需要 Transpose
 * @param lda  输出：物理 leading dimension
 * @return 物理数据指针，或 nullptr（无法映射，调用方需回退）
 */
static const float* mapToCblasView(const Tensor& t, size_t rows, size_t cols,
                                   bool& trans, int& lda) {
    const auto& st = t.strides();
    // 行列连续：dense row-major，NoTrans
    if (st[0] == cols && st[1] == 1) {
        trans = false;
        lda = (int)cols;
        return t.data_read<float>();
    }
    // 转置视图：stride==(1, R) 等价于某行列连续的转置，Trans 直读原存储零拷贝
    if (st[0] == 1 && st[1] == rows) {
        trans = true;
        lda = (int)rows;
        return t.data_read<float>();
    }
    return nullptr;
}
#endif

Tensor MatMul_SIMD_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                         ErrorType::DEVICE_COMPAT,
                         "CPU-SIMD MatMul_Kernel: 仅在CPU支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                         "CPU-SIMD MatMul_Kernel: 张量数据类型不一致");
    }
    if (a.sizes().size() != 2 || b.sizes().size() != 2) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                         "CPU-SIMD MatMul_Kernel: 仅支持 2D 矩阵");
        return Tensor();
    }

    size_t m = a.sizes()[0];
    size_t k = a.sizes()[1];
    size_t n = b.sizes()[1];
    if (k != b.sizes()[0]) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                         "CPU-SIMD MatMul_Kernel: 矩阵维度不匹配");
        return Tensor();
    }

    Tensor result(ShapeTag{}, {m, n}, a.dtype(), a.device());

#ifdef __APPLE__
    // [Eager 2026-08-27] 优先 cblas_sgemm + 转置折叠路径（macOS 专属）
    bool transA, transB;
    int lda, ldb;
    const float* pa = mapToCblasView(a, m, k, transA, lda);
    const float* pb = mapToCblasView(b, k, n, transB, ldb);
    if (pa && pb) {
        // 两输入均能映射为连续/转置视图 → cblas_sgemm 直读，零拷贝
        CBLAS_TRANSPOSE cta = transA ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE ctb = transB ? CblasTrans : CblasNoTrans;
        cblas_sgemm(CblasRowMajor, cta, ctb,
                    (int)m, (int)n, (int)k,
                    1.0f, pa, lda, pb, ldb,
                    0.0f, result.data_write<float>(), (int)n);
        return result;
    }
#endif

    // [Eager 2026-08-27] 任意 stride / 非 macOS 回退：先 zero 输出，再用原 SIMD 循环
    result.zero();

    const float* CT_RESTRICT a_data = a.data_read<float>();
    const float* CT_RESTRICT b_data = b.data_read<float>();
    float* CT_RESTRICT r_data = result.data_write<float>();

    const auto& a_strides = a.strides();
    const auto& b_strides = b.strides();
    size_t a_stride0 = a_strides[0];
    size_t a_stride1 = a_strides[1];
    size_t b_stride0 = b_strides[0];
    size_t b_stride1 = b_strides[1];

    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        for (size_t l = 0; l < k; ++l) {
            float a_val = a_data[i * a_stride0 + l * a_stride1];
            #pragma omp simd
            for (size_t j = 0; j < n; ++j) {
                r_data[i * n + j] += a_val * b_data[l * b_stride0 + j * b_stride1];
            }
        }
    }

    return result;
}
