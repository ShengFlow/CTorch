//
// RandomMatrix.h
// 标准正态随机矩阵生成器（RSVD 用）
// 2026-08-10
//
// 设计要点：
//   * 复用 ctQALS::rng::ZigguratNormal，不重复造正态采样
//   * 行主序（row-major），与 CTorch Tensor 内存布局一致
//   * header-only，仅依赖 <cstddef> 和 Random.h
//   * 提供 fill_gaussian 自由函数 + RAII 包装 MatrixXf
//
// 这是 RSVD 的 Layer 0-a 子模块，无前置依赖。
//

#ifndef CTORCH_RANDOM_MATRIX_H
#define CTORCH_RANDOM_MATRIX_H
#pragma once

#include "Random.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ctQALS {
namespace linalg {

// ============================================================
// RAII 轻量矩阵：拥有 row-major float 缓冲
// ============================================================
class MatrixXf {
public:
    MatrixXf() : rows_(0), cols_(0) {}

    MatrixXf(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols),
          data_(rows * cols, 0.0f) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("MatrixXf: rows and cols must be > 0");
        }
    }

    // 从现有缓冲构造（拷贝）
    MatrixXf(const float* src, std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(src, src + rows * cols) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("MatrixXf: rows and cols must be > 0");
        }
    }

    // 移动构造/赋值
    MatrixXf(MatrixXf&&) noexcept = default;
    MatrixXf& operator=(MatrixXf&&) noexcept = default;
    MatrixXf(const MatrixXf&) = default;
    MatrixXf& operator=(const MatrixXf&) = default;

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size()  const noexcept { return rows_ * cols_; }

    float*       data()       noexcept { return data_.data(); }
    const float* data() const noexcept { return data_.data(); }

    // 元素访问（行主序）
    float&       operator()(std::size_t i, std::size_t j)       { return data_[i * cols_ + j]; }
    float        operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

    // 填充零
    void zero() noexcept { std::fill(data_.begin(), data_.end(), 0.0f); }

    // 缩放（in-place）
    void scale(float s) noexcept {
        for (auto& v : data_) v *= s;
    }

private:
    std::size_t           rows_;
    std::size_t           cols_;
    std::vector<float>    data_;
};

// ============================================================
// 自由函数：把标准正态 N(0,1) 填进 m×n 矩阵
// ============================================================

// 用全局线程本地引擎（适合普通调用方）
inline void fill_gaussian(float* dst, std::size_t m, std::size_t n) {
    if (m == 0 || n == 0) return;
    auto& normal = ctQALS::rng::local_normal();
    normal.fill(dst, m * n);
}

// 用指定引擎（适合需要可复现实验的调用方）
inline void fill_gaussian(float* dst, std::size_t m, std::size_t n,
                          ctQALS::rng::ZigguratNormal& rng) {
    if (m == 0 || n == 0) return;
    rng.fill(dst, m * n);
}

// 构造一个 m×n 标准正态随机矩阵
inline MatrixXf randn(std::size_t m, std::size_t n) {
    MatrixXf M(m, n);
    fill_gaussian(M.data(), m, n);
    return M;
}

// 指定引擎版本
inline MatrixXf randn(std::size_t m, std::size_t n,
                      ctQALS::rng::ZigguratNormal& rng) {
    MatrixXf M(m, n);
    fill_gaussian(M.data(), m, n, rng);
    return M;
}

// ============================================================
// 矩阵-矩阵乘基础（C = A * B，全部 row-major float）
// 这些是 RSVD 和 Householder QR 都要用到的原语。
// 不调用任何 BLAS/LAPACK。
//
// 复杂度：O(m * n * k)，i-k-j 内层循环（连续写 C）利于缓存。
// ============================================================
inline void matmul(const float* A, std::size_t A_rows, std::size_t A_cols,
                   const float* B, std::size_t B_rows, std::size_t B_cols,
                   float* C) {
    if (A_cols != B_rows) {
        throw std::invalid_argument("matmul: A_cols (" + std::to_string(A_cols)
                                    + ") != B_rows (" + std::to_string(B_rows) + ")");
    }
    const std::size_t M = A_rows;
    const std::size_t N = B_cols;
    const std::size_t K = A_cols;
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const float a = A[i * A_cols + k];
            const float* Brow = B + k * B_cols;
            float* Crow = C + i * N;
            for (std::size_t j = 0; j < N; ++j) {
                Crow[j] += a * Brow[j];
            }
        }
    }
}

// C = A^T * B  （A 是 A_rows×A_cols，结果 C 是 A_cols×B_cols）
inline void matmul_AtB(const float* A, std::size_t A_rows, std::size_t A_cols,
                       const float* B, std::size_t B_rows, std::size_t B_cols,
                       float* C) {
    if (A_rows != B_rows) {
        throw std::invalid_argument("matmul_AtB: A_rows (" + std::to_string(A_rows)
                                    + ") != B_rows (" + std::to_string(B_rows) + ")");
    }
    const std::size_t M = A_cols;
    const std::size_t N = B_cols;
    const std::size_t K = A_rows;
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const float a = A[k * A_cols + i];   // A^T[i,k] = A[k,i]
            const float* Brow = B + k * B_cols;
            float* Crow = C + i * N;
            for (std::size_t j = 0; j < N; ++j) {
                Crow[j] += a * Brow[j];
            }
        }
    }
}

// C = A * B^T  （B 是 B_rows×B_cols，B^T 是 B_cols×B_rows，结果 A_rows×B_rows）
inline void matmul_ABt(const float* A, std::size_t A_rows, std::size_t A_cols,
                       const float* B, std::size_t B_rows, std::size_t B_cols,
                       float* C) {
    if (A_cols != B_cols) {
        throw std::invalid_argument("matmul_ABt: A_cols (" + std::to_string(A_cols)
                                    + ") != B_cols (" + std::to_string(B_cols) + ")");
    }
    const std::size_t M = A_rows;
    const std::size_t N = B_rows;
    const std::size_t K = A_cols;
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const float a = A[i * A_cols + k];
            for (std::size_t j = 0; j < N; ++j) {
                C[i * N + j] += a * B[j * B_cols + k];  // B^T[k,j] = B[j,k]
            }
        }
    }
}

} // namespace linalg
} // namespace ctQALS
#endif // CTORCH_RANDOM_MATRIX_H
