/**
 * @file Rot_AMX_kernel.cpp
 * @brief Apple AMX Acceleration Givens 旋转算子（cblas_srot from Accelerate）
 * @details 原地修改两向量：x[i] ← c*x[i] + s*y[i], y[i] ← c*y[i] - s*x[i]
 *          用于 JacobiSVD 的列对正交化（一侧 Jacobi 算法核心）
 * @author GhostFace
 * @date 2026/08/10
 */

#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../kernels.h"

// 包含 Accelerate 时禁用可能冲突的宏
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

/**
 * @brief Givens 旋转算子（AMX 加速，使用 Accelerate cblas_srot）
 *
 * @param x 第一个向量（m 维，1D 或任意连续内存），会被原地修改
 * @param y 第二个向量（m 维，1D 或任意连续内存），会被原地修改
 * @param c cos 系数（Givens 旋转矩阵 [c s; -s c]）
 * @param s sin 系数
 *
 * @details 数学定义：
 *          [ x' ]   [  c  s ] [ x ]
 *          [ y' ] = [ -s  c ] [ y ]
 *          = c*x + s*y
 *          = c*y - s*x（按 cblas_srot 的 LAPACK 约定）
 *
 *          等价于：
 *            x[i] ← c * x[i] + s * y[i]
 *            y[i] ← c * y[i] - s * x[i]  (在 cblas_srot 内 x 已更新)
 *
 * @note cblas_srot 要求向量按 stride 访问，stride=1 表示 row-major 连续
 *       如果 x/y 是非连续（带 stride > 1），会回退到朴素循环
 *
 * @warning x 和 y 内存不能重叠
 */
void Rot_AMX_kernel(Tensor& x, Tensor& y, float c, float s) {
    // 校验设备：仅支持 CPU/MPS（MPS 内存是 shared）
    if (x.device() != DeviceType::kCPU && x.device() != DeviceType::kMPS) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(x.device()),
                         ErrorType::DEVICE_COMPAT,
                         "AMX Rot_Kernel: 仅在 CPU/MPS 支持");
    }
    if (x.device() != y.device()) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(x.device()),
                         ErrorType::DEVICE_COMPAT,
                         "AMX Rot_Kernel: x 和 y 设备不一致");
    }
    if (x.dtype() != y.dtype() || x.dtype() != DType::kFloat) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                         "AMX Rot_Kernel: 仅支持 float32");
    }

    // 1D 或 N 维都支持：取 numel 当长度
    const std::size_t n = x.numel();
    if (n != y.numel()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                     "AMX Rot_Kernel: x 和 y 元素数不一致");
    }
    if (n == 0) return;

    float* x_data = x.data_write<float>();
    float* y_data = y.data_write<float>();

#ifdef __APPLE__
    // MPS 路径：host 写入 shared buffer 后需 mark
    if (x.device() == DeviceType::kMPS) {
        MPS_markBufferModified(x_data, n * sizeof(float));
        MPS_markBufferModified(y_data, n * sizeof(float));
    }
#endif

    // cblas_srot 要求 stride=1（连续 row-major 访问）
    // 对于行主序 1D Tensor：data 连续，stride=1
    // 对于行主序 2D Tensor 的列访问：stride = leading_dim > 1，需要 pack/unpack
    // 简化处理：要求 x/y 必须是 1D 或 stride=1 的连续内存
    const std::size_t x_stride = x.numel() > 0 ? (x.strides().empty() ? 1 : x.strides().back()) : 1;
    const std::size_t y_stride = y.numel() > 0 ? (y.strides().empty() ? 1 : y.strides().back()) : 1;

    if (x.is_contiguous() && y.is_contiguous()) {
        // 快速路径：cblas_srot（BLAS 内部用 NEON SIMD）
        cblas_srot((int)n, x_data, 1, y_data, 1, c, s);
    } else if (x.dim() == 1 && y.dim() == 1) {
        // 1D 跨步路径：BLAS 原生支持任意正跨步
        int sx = static_cast<int>(x.strides().empty() ? 1 : x.strides()[0]);
        int sy = static_cast<int>(y.strides().empty() ? 1 : y.strides()[0]);
        cblas_srot((int)n, x_data, sx, y_data, sy, c, s);
    } else {
        // 回退路径：高维非连续视图按各轴 strides 安全寻址，防止跨步累乘越界
        const auto& shape = x.shape();
        const auto& x_strides = x.strides();
        const auto& y_strides = y.strides();
        std::vector<size_t> coord(shape.size(), 0);
        for (std::size_t lin = 0; lin < n; ++lin) {
            size_t off_x = 0;
            size_t off_y = 0;
            for (size_t d = 0; d < coord.size(); ++d) {
                off_x += coord[d] * x_strides[d];
                off_y += coord[d] * y_strides[d];
            }
            const float xi = x_data[off_x];
            const float yi = y_data[off_y];
            x_data[off_x] = c * xi + s * yi;
            y_data[off_y] = c * yi - s * xi;
            for (int d = static_cast<int>(coord.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) break;
                coord[d] = 0;
            }
        }
    }
}
