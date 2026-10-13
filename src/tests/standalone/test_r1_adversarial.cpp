/**
 * @file test_r1_adversarial.cpp
 * @brief Adversarial Stress Test Suite for Requirement R1 (Tensor View & Reduction Memory Safety)
 * @details Formulates adversarial tests for:
 *   1. 3D Transposed Views (transpose(0, 2), transpose(1, 2), chained transpositions)
 *   2. Sliced and Strided Sub-Views (slice + transpose, transpose + slice, multi-dim slicing)
 *   3. 1x1 and Unit Dimensions Edge Cases ({1, 1}, {1, 8}, {8, 1}, reduction along dim_size=1)
 *   4. High-Dimensional Reductions (4D shape {2, 3, 4, 5}, 5D shape {2, 2, 3, 2, 4})
 *   5. Double Precision (kDouble) exactness
 *   6. Autograd Backward Pass under Non-Uniform Gradients and Complex Views
 *   7. Dimension bounds validation (negative dims, out-of-range dims)
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "Tensor.h"
#include "AutoGrad.h"

using namespace ct;

static int g_adversarial_fails = 0;
static int g_adversarial_passes = 0;

#define ADV_CHECK(cond, msg)                                             \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::cerr << "FAIL [Adversarial]: " << msg << "\n";          \
            ++g_adversarial_fails;                                       \
        } else {                                                         \
            std::cout << "PASS [Adversarial]: " << msg << "\n";          \
            ++g_adversarial_passes;                                      \
        }                                                                \
    } while (0)

// Reference Dense Tensor Helper
struct DenseTensorF {
    std::vector<size_t> shape;
    std::vector<float> data;

    DenseTensorF(std::vector<size_t> s) : shape(s) {
        size_t n = 1;
        for (auto d : s) n *= d;
        data.resize(n, 0.0f);
    }

    size_t offset(const std::vector<size_t>& idx) const {
        size_t off = 0;
        size_t stride = 1;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
            off += idx[d] * stride;
            stride *= shape[d];
        }
        return off;
    }

    float get(const std::vector<size_t>& idx) const {
        return data[offset(idx)];
    }

    void set(const std::vector<size_t>& idx, float v) {
        data[offset(idx)] = v;
    }

    DenseTensorF reduce_sum(int dim, bool keepdim) const {
        std::vector<size_t> out_shape;
        for (size_t d = 0; d < shape.size(); ++d) {
            if (static_cast<int>(d) == dim) {
                if (keepdim) out_shape.push_back(1);
            } else {
                out_shape.push_back(shape[d]);
            }
        }
        if (out_shape.empty()) out_shape = {1};
        DenseTensorF res(out_shape);

        size_t total_in = data.size();
        std::vector<size_t> in_coord(shape.size(), 0);
        for (size_t lin = 0; lin < total_in; ++lin) {
            std::vector<size_t> out_coord;
            for (size_t d = 0; d < shape.size(); ++d) {
                if (static_cast<int>(d) == dim) {
                    if (keepdim) out_coord.push_back(0);
                } else {
                    out_coord.push_back(in_coord[d]);
                }
            }
            if (out_coord.empty()) out_coord = {0};
            res.data[res.offset(out_coord)] += get(in_coord);

            for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
                if (++in_coord[d] < shape[d]) break;
                in_coord[d] = 0;
            }
        }
        return res;
    }

    DenseTensorF reduce_mean(int dim, bool keepdim) const {
        DenseTensorF s = reduce_sum(dim, keepdim);
        float scale = 1.0f / static_cast<float>(shape[dim]);
        for (auto& v : s.data) v *= scale;
        return s;
    }
};

struct DenseTensorD {
    std::vector<size_t> shape;
    std::vector<double> data;

    DenseTensorD(std::vector<size_t> s) : shape(s) {
        size_t n = 1;
        for (auto d : s) n *= d;
        data.resize(n, 0.0);
    }

    size_t offset(const std::vector<size_t>& idx) const {
        size_t off = 0;
        size_t stride = 1;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
            off += idx[d] * stride;
            stride *= shape[d];
        }
        return off;
    }

    double get(const std::vector<size_t>& idx) const {
        return data[offset(idx)];
    }

    void set(const std::vector<size_t>& idx, double v) {
        data[offset(idx)] = v;
    }

    DenseTensorD reduce_sum(int dim, bool keepdim) const {
        std::vector<size_t> out_shape;
        for (size_t d = 0; d < shape.size(); ++d) {
            if (static_cast<int>(d) == dim) {
                if (keepdim) out_shape.push_back(1);
            } else {
                out_shape.push_back(shape[d]);
            }
        }
        if (out_shape.empty()) out_shape = {1};
        DenseTensorD res(out_shape);

        size_t total_in = data.size();
        std::vector<size_t> in_coord(shape.size(), 0);
        for (size_t lin = 0; lin < total_in; ++lin) {
            std::vector<size_t> out_coord;
            for (size_t d = 0; d < shape.size(); ++d) {
                if (static_cast<int>(d) == dim) {
                    if (keepdim) out_coord.push_back(0);
                } else {
                    out_coord.push_back(in_coord[d]);
                }
            }
            if (out_coord.empty()) out_coord = {0};
            res.data[res.offset(out_coord)] += get(in_coord);

            for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
                if (++in_coord[d] < shape[d]) break;
                in_coord[d] = 0;
            }
        }
        return res;
    }

    DenseTensorD reduce_mean(int dim, bool keepdim) const {
        DenseTensorD s = reduce_sum(dim, keepdim);
        double scale = 1.0 / static_cast<double>(shape[dim]);
        for (auto& v : s.data) v *= scale;
        return s;
    }
};


static bool compare_two_tensors(const Tensor& a, const Tensor& b, float tol = 1e-4f) {
    if (a.shape() != b.shape()) return false;
    Tensor ca = a.is_contiguous() ? a : a.contiguous();
    Tensor cb = b.is_contiguous() ? b : b.contiguous();
    const float* pa = ca.data_read<float>();
    const float* pb = cb.data_read<float>();
    for (size_t i = 0; i < ca.numel(); ++i) {
        if (std::abs(pa[i] - pb[i]) > tol) return false;
    }
    return true;
}

static bool compare_tensor_dense(const Tensor& t, const DenseTensorF& ref, float tol = 1e-4f) {
    if (t.shape() != ref.shape) {
        std::cerr << "Shape mismatch: tensor [";
        for (auto d : t.shape()) std::cerr << d << " ";
        std::cerr << "] vs ref [";
        for (auto d : ref.shape) std::cerr << d << " ";
        std::cerr << "]\n";
        return false;
    }
    Tensor c = t.is_contiguous() ? t : t.contiguous();
    const float* ptr = c.data_read<float>();
    for (size_t i = 0; i < ref.data.size(); ++i) {
        if (std::abs(ptr[i] - ref.data[i]) > tol) {
            std::cerr << "Mismatch at index " << i << ": got " << ptr[i] << " vs expected " << ref.data[i] << "\n";
            return false;
        }
    }
    return true;
}

static bool compare_tensor_dense_d(const Tensor& t, const DenseTensorD& ref, double tol = 1e-9) {
    if (t.shape() != ref.shape) return false;
    Tensor c = t.is_contiguous() ? t : t.contiguous();
    const double* ptr = c.data_read<double>();
    for (size_t i = 0; i < ref.data.size(); ++i) {
        if (std::abs(ptr[i] - ref.data[i]) > tol) {
            std::cerr << "Double mismatch at index " << i << ": got " << ptr[i] << " vs ref " << ref.data[i] << "\n";
            return false;
        }
    }
    return true;
}

void test_3d_transposed_views() {
    std::cout << "\n=== [Test 1] 3D Transposed Views Reductions ===\n";
    Tensor t(ShapeTag{}, {3, 4, 5}, DType::kFloat, DeviceType::kCPU);
    DenseTensorF ref_orig({3, 4, 5});
    float* p = t.data_write<float>();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 5; ++k) {
                float val = static_cast<float>(i * 20 + j * 5 + k + 1);
                p[i * 20 + j * 5 + k] = val;
                ref_orig.set({i, j, k}, val);
            }
        }
    }

    // 1.1 transpose(0, 2): shape (5, 4, 3)
    {
        Tensor t_02 = t.transpose(0, 2);
        ADV_CHECK(!t_02.is_contiguous(), "3D transpose(0, 2) is non-contiguous");

        DenseTensorF ref_02({5, 4, 3});
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    ref_02.set({i, j, k}, ref_orig.get({k, j, i}));
                }
            }
        }

        ADV_CHECK(compare_tensor_dense(t_02.sum(0), ref_02.reduce_sum(0, false)), "t_02.sum(0) exact match");
        ADV_CHECK(compare_tensor_dense(t_02.sum(1), ref_02.reduce_sum(1, false)), "t_02.sum(1) exact match");
        ADV_CHECK(compare_tensor_dense(t_02.sum(2), ref_02.reduce_sum(2, false)), "t_02.sum(2) exact match");

        ADV_CHECK(compare_tensor_dense(t_02.sum(0, true), ref_02.reduce_sum(0, true)), "t_02.sum(0, keepdim=true) exact match");
        ADV_CHECK(compare_tensor_dense(t_02.sum(1, true), ref_02.reduce_sum(1, true)), "t_02.sum(1, keepdim=true) exact match");
        ADV_CHECK(compare_tensor_dense(t_02.sum(2, true), ref_02.reduce_sum(2, true)), "t_02.sum(2, keepdim=true) exact match");

        ADV_CHECK(compare_tensor_dense(t_02.mean(0), ref_02.reduce_mean(0, false)), "t_02.mean(0) exact match");
        ADV_CHECK(compare_tensor_dense(t_02.mean(1), ref_02.reduce_mean(1, false)), "t_02.mean(1) exact match");
        ADV_CHECK(compare_tensor_dense(t_02.mean(2), ref_02.reduce_mean(2, false)), "t_02.mean(2) exact match");

        ADV_CHECK(compare_tensor_dense(t_02.mean(0, true), ref_02.reduce_mean(0, true)), "t_02.mean(0, keepdim=true) exact match");
        ADV_CHECK(compare_tensor_dense(t_02.mean(1, true), ref_02.reduce_mean(1, true)), "t_02.mean(1, keepdim=true) exact match");
        ADV_CHECK(compare_tensor_dense(t_02.mean(2, true), ref_02.reduce_mean(2, true)), "t_02.mean(2, keepdim=true) exact match");
    }

    // 1.2 transpose(1, 2): shape (3, 5, 4)
    {
        Tensor t_12 = t.transpose(1, 2);
        ADV_CHECK(!t_12.is_contiguous(), "3D transpose(1, 2) is non-contiguous");

        DenseTensorF ref_12({3, 5, 4});
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    ref_12.set({i, j, k}, ref_orig.get({i, k, j}));
                }
            }
        }

        ADV_CHECK(compare_tensor_dense(t_12.sum(0), ref_12.reduce_sum(0, false)), "t_12.sum(0) exact match");
        ADV_CHECK(compare_tensor_dense(t_12.sum(1), ref_12.reduce_sum(1, false)), "t_12.sum(1) exact match");
        ADV_CHECK(compare_tensor_dense(t_12.sum(2), ref_12.reduce_sum(2, false)), "t_12.sum(2) exact match");

        ADV_CHECK(compare_tensor_dense(t_12.mean(0), ref_12.reduce_mean(0, false)), "t_12.mean(0) exact match");
        ADV_CHECK(compare_tensor_dense(t_12.mean(1), ref_12.reduce_mean(1, false)), "t_12.mean(1) exact match");
        ADV_CHECK(compare_tensor_dense(t_12.mean(2), ref_12.reduce_mean(2, false)), "t_12.mean(2) exact match");
    }

    // 1.3 Chained transpositions: transpose(0, 2).transpose(1, 2): shape (5, 3, 4)
    {
        Tensor t_chain = t.transpose(0, 2).transpose(1, 2);
        ADV_CHECK(!t_chain.is_contiguous(), "Chained transpose is non-contiguous");

        DenseTensorF ref_chain({5, 3, 4});
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    ref_chain.set({i, j, k}, ref_orig.get({j, k, i}));
                }
            }
        }

        ADV_CHECK(compare_tensor_dense(t_chain.sum(0), ref_chain.reduce_sum(0, false)), "t_chain.sum(0) exact match");
        ADV_CHECK(compare_tensor_dense(t_chain.sum(1), ref_chain.reduce_sum(1, false)), "t_chain.sum(1) exact match");
        ADV_CHECK(compare_tensor_dense(t_chain.sum(2), ref_chain.reduce_sum(2, false)), "t_chain.sum(2) exact match");

        ADV_CHECK(compare_tensor_dense(t_chain.mean(0), ref_chain.reduce_mean(0, false)), "t_chain.mean(0) exact match");
        ADV_CHECK(compare_tensor_dense(t_chain.mean(1), ref_chain.reduce_mean(1, false)), "t_chain.mean(1) exact match");
        ADV_CHECK(compare_tensor_dense(t_chain.mean(2), ref_chain.reduce_mean(2, false)), "t_chain.mean(2) exact match");
    }
}

void test_sliced_strided_views() {
    std::cout << "\n=== [Test 2] Sliced & Strided Sub-Views ===\n";
    {
        Tensor t(ShapeTag{}, {4, 6}, DType::kFloat, DeviceType::kCPU);
        DenseTensorF ref_orig({4, 6});
        float* p = t.data_write<float>();
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                float v = static_cast<float>(i * 10 + j + 1);
                p[i * 6 + j] = v;
                ref_orig.set({i, j}, v);
            }
        }

        Tensor t_slice1 = t.slice(1, 1, 4);
        ADV_CHECK(!t_slice1.is_contiguous(), "Slice on innermost dim is non-contiguous");

        DenseTensorF ref_s1({4, 4});
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                ref_s1.set({i, j}, ref_orig.get({i, j + 1}));
            }
        }

        ADV_CHECK(compare_tensor_dense(t_slice1.sum(0), ref_s1.reduce_sum(0, false)), "t_slice1.sum(0) exact match");
        ADV_CHECK(compare_tensor_dense(t_slice1.sum(1), ref_s1.reduce_sum(1, false)), "t_slice1.sum(1) exact match");
        ADV_CHECK(compare_tensor_dense(t_slice1.mean(0), ref_s1.reduce_mean(0, false)), "t_slice1.mean(0) exact match");
        ADV_CHECK(compare_tensor_dense(t_slice1.mean(1), ref_s1.reduce_mean(1, false)), "t_slice1.mean(1) exact match");

        Tensor t_ts = t.transpose(0, 1).slice(0, 2, 3);
        ADV_CHECK(!t_ts.is_contiguous(), "Transpose then slice is non-contiguous");
        DenseTensorF ref_ts({3, 4});
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                ref_ts.set({i, j}, ref_orig.get({j, i + 2}));
            }
        }

        ADV_CHECK(compare_tensor_dense(t_ts.sum(0), ref_ts.reduce_sum(0, false)), "t_ts.sum(0) exact match");
        ADV_CHECK(compare_tensor_dense(t_ts.sum(1), ref_ts.reduce_sum(1, false)), "t_ts.sum(1) exact match");
        ADV_CHECK(compare_tensor_dense(t_ts.mean(0), ref_ts.reduce_mean(0, false)), "t_ts.mean(0) exact match");
        ADV_CHECK(compare_tensor_dense(t_ts.mean(1), ref_ts.reduce_mean(1, false)), "t_ts.mean(1) exact match");

        Tensor t_st = t.slice(1, 2, 3).transpose(0, 1);
        ADV_CHECK(!t_st.is_contiguous(), "Slice then transpose is non-contiguous");
        ADV_CHECK(compare_tensor_dense(t_st.sum(0), ref_ts.reduce_sum(0, false)), "t_st.sum(0) exact match");
        ADV_CHECK(compare_tensor_dense(t_st.sum(1), ref_ts.reduce_sum(1, false)), "t_st.sum(1) exact match");
    }

    {
        Tensor t3(ShapeTag{}, {3, 4, 5}, DType::kFloat, DeviceType::kCPU);
        DenseTensorF ref3({3, 4, 5});
        float* p = t3.data_write<float>();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                for (size_t k = 0; k < 5; ++k) {
                    float v = static_cast<float>(i * 50 + j * 10 + k + 1);
                    p[i * 20 + j * 5 + k] = v;
                    ref3.set({i, j, k}, v);
                }
            }
        }

        Tensor t3_s = t3.slice(1, 1, 2);
        Tensor t3_st = t3_s.transpose(0, 2);
        ADV_CHECK(!t3_st.is_contiguous(), "3D slice then transpose is non-contiguous");

        DenseTensorF ref3_st({5, 2, 3});
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    ref3_st.set({i, j, k}, ref3.get({k, j + 1, i}));
                }
            }
        }

        ADV_CHECK(compare_tensor_dense(t3_st.sum(0), ref3_st.reduce_sum(0, false)), "t3_st.sum(0) exact match");
        ADV_CHECK(compare_tensor_dense(t3_st.sum(1), ref3_st.reduce_sum(1, false)), "t3_st.sum(1) exact match");
        ADV_CHECK(compare_tensor_dense(t3_st.sum(2), ref3_st.reduce_sum(2, false)), "t3_st.sum(2) exact match");
        ADV_CHECK(compare_tensor_dense(t3_st.mean(0), ref3_st.reduce_mean(0, false)), "t3_st.mean(0) exact match");
        ADV_CHECK(compare_tensor_dense(t3_st.mean(1), ref3_st.reduce_mean(1, false)), "t3_st.mean(1) exact match");
        ADV_CHECK(compare_tensor_dense(t3_st.mean(2), ref3_st.reduce_mean(2, false)), "t3_st.mean(2) exact match");
    }
}

void test_unit_and_edge_dimensions() {
    std::cout << "\n=== [Test 3] 1x1 and Unit Dimensions Edge Cases ===\n";

    {
        Tensor t(ShapeTag{}, {1, 1}, DType::kFloat, DeviceType::kCPU);
        t.data_write<float>()[0] = 42.0f;
        DenseTensorF ref({1, 1});
        ref.set({0, 0}, 42.0f);

        Tensor tt = t.transpose(0, 1);
        ADV_CHECK(compare_tensor_dense(tt.sum(0), ref.reduce_sum(0, false)), "1x1 transposed sum(0) = 42");
        ADV_CHECK(compare_tensor_dense(tt.sum(1), ref.reduce_sum(1, false)), "1x1 transposed sum(1) = 42");
        ADV_CHECK(compare_tensor_dense(tt.mean(0), ref.reduce_mean(0, false)), "1x1 transposed mean(0) = 42");
        ADV_CHECK(compare_tensor_dense(tt.mean(1), ref.reduce_mean(1, false)), "1x1 transposed mean(1) = 42");
    }

    {
        Tensor t(ShapeTag{}, {1, 8}, DType::kFloat, DeviceType::kCPU);
        DenseTensorF ref({1, 8});
        for (size_t i = 0; i < 8; ++i) {
            float v = static_cast<float>(i + 1);
            t.data_write<float>()[i] = v;
            ref.set({0, i}, v);
        }

        Tensor tt = t.transpose(0, 1);
        ADV_CHECK(!tt.is_contiguous(), "{1, 8}.transpose(0, 1) is non-contiguous");

        DenseTensorF ref_t({8, 1});
        for (size_t i = 0; i < 8; ++i) ref_t.set({i, 0}, ref.get({0, i}));

        ADV_CHECK(compare_tensor_dense(tt.sum(0), ref_t.reduce_sum(0, false)), "{8, 1} sum(0) exact match");
        ADV_CHECK(compare_tensor_dense(tt.sum(1), ref_t.reduce_sum(1, false)), "{8, 1} sum(1) exact match");
        ADV_CHECK(compare_tensor_dense(tt.mean(0), ref_t.reduce_mean(0, false)), "{8, 1} mean(0) exact match");
        ADV_CHECK(compare_tensor_dense(tt.mean(1), ref_t.reduce_mean(1, false)), "{8, 1} mean(1) exact match");
    }

    {
        Tensor t(ShapeTag{}, {3, 1, 4}, DType::kFloat, DeviceType::kCPU);
        DenseTensorF ref({3, 1, 4});
        for (size_t i = 0; i < 3; ++i) {
            for (size_t k = 0; k < 4; ++k) {
                float v = static_cast<float>(i * 10 + k + 1);
                t.data_write<float>()[i * 4 + k] = v;
                ref.set({i, 0, k}, v);
            }
        }

        ADV_CHECK(compare_tensor_dense(t.sum(1), ref.reduce_sum(1, false)), "{3, 1, 4} sum(1) identity");
        ADV_CHECK(compare_tensor_dense(t.mean(1), ref.reduce_mean(1, false)), "{3, 1, 4} mean(1) identity");

        Tensor tt = t.transpose(0, 2);
        ADV_CHECK(!tt.is_contiguous(), "{3, 1, 4} transpose(0, 2) is non-contiguous");
        DenseTensorF ref_tt({4, 1, 3});
        for (size_t i = 0; i < 4; ++i) {
            for (size_t k = 0; k < 3; ++k) {
                ref_tt.set({i, 0, k}, ref.get({k, 0, i}));
            }
        }
        ADV_CHECK(compare_tensor_dense(tt.sum(1), ref_tt.reduce_sum(1, false)), "{4, 1, 3} sum(1) identity on non-contiguous");
        ADV_CHECK(compare_tensor_dense(tt.mean(1), ref_tt.reduce_mean(1, false)), "{4, 1, 3} mean(1) identity on non-contiguous");
    }

    {
        Tensor t(ShapeTag{}, {1, 1, 1}, DType::kFloat, DeviceType::kCPU);
        t.data_write<float>()[0] = 7.0f;
        DenseTensorF ref({1, 1, 1});
        ref.set({0, 0, 0}, 7.0f);

        ADV_CHECK(compare_tensor_dense(t.sum(0, true), ref.reduce_sum(0, true)), "{1, 1, 1} sum(0, keepdim=true)");
        ADV_CHECK(compare_tensor_dense(t.sum(1, true), ref.reduce_sum(1, true)), "{1, 1, 1} sum(1, keepdim=true)");
        ADV_CHECK(compare_tensor_dense(t.sum(2, true), ref.reduce_sum(2, true)), "{1, 1, 1} sum(2, keepdim=true)");
        ADV_CHECK(compare_tensor_dense(t.mean(0, true), ref.reduce_mean(0, true)), "{1, 1, 1} mean(0, keepdim=true)");
    }
}

void test_high_dimensional_reductions() {
    std::cout << "\n=== [Test 4] High-Dimensional Reductions (4D and 5D) ===\n";

    {
        Tensor t4(ShapeTag{}, {2, 3, 4, 5}, DType::kFloat, DeviceType::kCPU);
        DenseTensorF ref4({2, 3, 4, 5});
        float* p = t4.data_write<float>();
        size_t idx = 0;
        for (size_t a = 0; a < 2; ++a) {
            for (size_t b = 0; b < 3; ++b) {
                for (size_t c = 0; c < 4; ++c) {
                    for (size_t d = 0; d < 5; ++d) {
                        float v = static_cast<float>(idx + 1);
                        p[idx++] = v;
                        ref4.set({a, b, c, d}, v);
                    }
                }
            }
        }

        Tensor t4_03 = t4.transpose(0, 3);
        ADV_CHECK(!t4_03.is_contiguous(), "4D transpose(0, 3) is non-contiguous");

        DenseTensorF ref4_03({5, 3, 4, 2});
        for (size_t a = 0; a < 5; ++a) {
            for (size_t b = 0; b < 3; ++b) {
                for (size_t c = 0; c < 4; ++c) {
                    for (size_t d = 0; d < 2; ++d) {
                        ref4_03.set({a, b, c, d}, ref4.get({d, b, c, a}));
                    }
                }
            }
        }

        ADV_CHECK(compare_tensor_dense(t4_03.sum(0), ref4_03.reduce_sum(0, false)), "4D t4_03.sum(0) exact match");
        ADV_CHECK(compare_tensor_dense(t4_03.sum(1), ref4_03.reduce_sum(1, false)), "4D t4_03.sum(1) exact match");
        ADV_CHECK(compare_tensor_dense(t4_03.sum(2), ref4_03.reduce_sum(2, false)), "4D t4_03.sum(2) exact match");
        ADV_CHECK(compare_tensor_dense(t4_03.sum(3), ref4_03.reduce_sum(3, false)), "4D t4_03.sum(3) exact match");

        ADV_CHECK(compare_tensor_dense(t4_03.mean(0), ref4_03.reduce_mean(0, false)), "4D t4_03.mean(0) exact match");
        ADV_CHECK(compare_tensor_dense(t4_03.mean(1), ref4_03.reduce_mean(1, false)), "4D t4_03.mean(1) exact match");
        ADV_CHECK(compare_tensor_dense(t4_03.mean(2), ref4_03.reduce_mean(2, false)), "4D t4_03.mean(2) exact match");
        ADV_CHECK(compare_tensor_dense(t4_03.mean(3), ref4_03.reduce_mean(3, false)), "4D t4_03.mean(3) exact match");
    }

    {
        Tensor t5(ShapeTag{}, {2, 2, 3, 2, 4}, DType::kFloat, DeviceType::kCPU);
        DenseTensorF ref5({2, 2, 3, 2, 4});
        float* p = t5.data_write<float>();
        for (size_t i = 0; i < 96; ++i) {
            p[i] = static_cast<float>(i + 1);
        }
        size_t idx = 0;
        for (size_t a = 0; a < 2; ++a) {
            for (size_t b = 0; b < 2; ++b) {
                for (size_t c = 0; c < 3; ++c) {
                    for (size_t d = 0; d < 2; ++d) {
                        for (size_t e = 0; e < 4; ++e) {
                            ref5.set({a, b, c, d, e}, p[idx++]);
                        }
                    }
                }
            }
        }

        Tensor t5_14 = t5.transpose(1, 4);
        ADV_CHECK(!t5_14.is_contiguous(), "5D transpose(1, 4) is non-contiguous");

        DenseTensorF ref5_14({2, 4, 3, 2, 2});
        for (size_t a = 0; a < 2; ++a) {
            for (size_t b = 0; b < 4; ++b) {
                for (size_t c = 0; c < 3; ++c) {
                    for (size_t d = 0; d < 2; ++d) {
                        for (size_t e = 0; e < 2; ++e) {
                            ref5_14.set({a, b, c, d, e}, ref5.get({a, e, c, d, b}));
                        }
                    }
                }
            }
        }

        ADV_CHECK(compare_tensor_dense(t5_14.sum(0), ref5_14.reduce_sum(0, false)), "5D t5_14.sum(0) exact match");
        ADV_CHECK(compare_tensor_dense(t5_14.sum(1), ref5_14.reduce_sum(1, false)), "5D t5_14.sum(1) exact match");
        ADV_CHECK(compare_tensor_dense(t5_14.sum(2), ref5_14.reduce_sum(2, false)), "5D t5_14.sum(2) exact match");
        ADV_CHECK(compare_tensor_dense(t5_14.sum(3), ref5_14.reduce_sum(3, false)), "5D t5_14.sum(3) exact match");
        ADV_CHECK(compare_tensor_dense(t5_14.sum(4), ref5_14.reduce_sum(4, false)), "5D t5_14.sum(4) exact match");

        ADV_CHECK(compare_tensor_dense(t5_14.mean(2), ref5_14.reduce_mean(2, false)), "5D t5_14.mean(2) exact match");
        ADV_CHECK(compare_tensor_dense(t5_14.mean(4), ref5_14.reduce_mean(4, false)), "5D t5_14.mean(4) exact match");
    }
}

void test_double_precision_adversarial() {
    std::cout << "\n=== [Test 5] Double Precision (kDouble) Adversarial ===\n";
    Tensor td(ShapeTag{}, {3, 4, 5}, DType::kDouble, DeviceType::kCPU);
    DenseTensorD refd({3, 4, 5});
    double* p = td.data_write<double>();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 5; ++k) {
                double val = static_cast<double>(i * 100.0 + j * 10.0 + k + 0.125);
                p[i * 20 + j * 5 + k] = val;
                refd.set({i, j, k}, val);
            }
        }
    }

    Tensor td_02 = td.transpose(0, 2);
    DenseTensorD refd_02({5, 4, 3});
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                refd_02.set({i, j, k}, refd.get({k, j, i}));
            }
        }
    }

    ADV_CHECK(compare_tensor_dense_d(td_02.sum(0), refd_02.reduce_sum(0, false)), "Double td_02.sum(0) exact match");
    ADV_CHECK(compare_tensor_dense_d(td_02.sum(1), refd_02.reduce_sum(1, false)), "Double td_02.sum(1) exact match");
    ADV_CHECK(compare_tensor_dense_d(td_02.sum(2), refd_02.reduce_sum(2, false)), "Double td_02.sum(2) exact match");

    ADV_CHECK(compare_tensor_dense_d(td_02.mean(0), refd_02.reduce_mean(0, false)), "Double td_02.mean(0) exact match");
    ADV_CHECK(compare_tensor_dense_d(td_02.mean(1), refd_02.reduce_mean(1, false)), "Double td_02.mean(1) exact match");
    ADV_CHECK(compare_tensor_dense_d(td_02.mean(2), refd_02.reduce_mean(2, false)), "Double td_02.mean(2) exact match");
}

void test_autograd_complex_views() {
    std::cout << "\n=== [Test 6] Autograd Backward with Non-Uniform Gradients ===\n";
    AutoGrad::EnableGrad = true;

    {
        Tensor t(ShapeTag{}, {2, 3, 4}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < 24; ++i) t.data_write<float>()[i] = static_cast<float>(i + 1);
        t.requires_grad(true);

        Tensor v = t.transpose(0, 2);
        Tensor s = v.sum(1);

        Tensor w(ShapeTag{}, {4, 2}, DType::kFloat, DeviceType::kCPU);
        float* wp = w.data_write<float>();
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 2; ++k) {
                wp[i * 2 + k] = static_cast<float>((i + 1) * 10 + (k + 1));
            }
        }

        Tensor loss = (s * w).sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        Tensor grad_t = t.grad();
        ADV_CHECK(grad_t.data_read<float>() != nullptr, "3D transpose sum autograd: leaf grad exists");

        bool ok = true;
        Tensor c_grad = grad_t.is_contiguous() ? grad_t : grad_t.contiguous();
        const float* gp = c_grad.data_read<float>();
        for (size_t k = 0; k < 2; ++k) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t i = 0; i < 4; ++i) {
                    float expected = wp[i * 2 + k];
                    size_t flat_idx = k * 12 + j * 4 + i;
                    if (std::abs(gp[flat_idx] - expected) > 1e-4f) {
                        std::cerr << "Grad mismatch at T(" << k << "," << j << "," << i << "): got "
                                  << gp[flat_idx] << " expected " << expected << "\n";
                        ok = false;
                    }
                }
            }
        }
        ADV_CHECK(ok, "3D transpose sum autograd: leaf gradient exactly matches analytical W(i, k)");
    }

    {
        Tensor t(ShapeTag{}, {2, 3, 4}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < 24; ++i) t.data_write<float>()[i] = static_cast<float>(i + 1);
        t.requires_grad(true);

        Tensor v = t.transpose(0, 2);
        Tensor m = v.mean(1);

        Tensor w(ShapeTag{}, {4, 2}, DType::kFloat, DeviceType::kCPU);
        float* wp = w.data_write<float>();
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 2; ++k) {
                wp[i * 2 + k] = static_cast<float>((i + 1) * 6 + (k + 1));
            }
        }

        Tensor loss = (m * w).sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        Tensor grad_t = t.grad();
        ADV_CHECK(grad_t.data_read<float>() != nullptr, "3D transpose mean autograd: leaf grad exists");

        bool ok = true;
        Tensor c_grad = grad_t.is_contiguous() ? grad_t : grad_t.contiguous();
        const float* gp = c_grad.data_read<float>();
        for (size_t k = 0; k < 2; ++k) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t i = 0; i < 4; ++i) {
                    float expected = wp[i * 2 + k] / 3.0f;
                    size_t flat_idx = k * 12 + j * 4 + i;
                    if (std::abs(gp[flat_idx] - expected) > 1e-4f) {
                        std::cerr << "Mean grad mismatch at T(" << k << "," << j << "," << i << "): got "
                                  << gp[flat_idx] << " expected " << expected << "\n";
                        ok = false;
                    }
                }
            }
        }
        ADV_CHECK(ok, "3D transpose mean autograd: leaf gradient exactly matches analytical W(i, k)/3");
    }

    {
        Tensor t(ShapeTag{}, {4, 6}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < 24; ++i) t.data_write<float>()[i] = static_cast<float>(i + 1);
        t.requires_grad(true);

        Tensor s = t.slice(1, 1, 4);
        Tensor loss = s.sum(1).sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        Tensor grad_t = t.grad();
        ADV_CHECK(grad_t.data_read<float>() != nullptr, "Sliced view sum autograd: leaf grad exists");

        bool ok = true;
        Tensor c_grad = grad_t.is_contiguous() ? grad_t : grad_t.contiguous();
        const float* gp = c_grad.data_read<float>();
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                float expected = (j >= 1 && j <= 4) ? 1.0f : 0.0f;
                float actual = gp[i * 6 + j];
                if (std::abs(actual - expected) > 1e-5f) {
                    std::cerr << "Slice grad mismatch at (" << i << "," << j << "): got " << actual << " vs " << expected << "\n";
                    ok = false;
                }
            }
        }
        ADV_CHECK(ok, "Sliced view sum autograd: inside cols receive 1.0, outside cols receive 0.0");
    }
}

void test_dimension_bounds() {
    std::cout << "\n=== [Test 7] Negative Dimension Indexing & Bounds Validation ===\n";
    Tensor t(ShapeTag{}, {3, 4, 5}, DType::kFloat, DeviceType::kCPU);
    for (size_t i = 0; i < 60; ++i) {
        float v = static_cast<float>(i + 1);
        t.data_write<float>()[i] = v;
    }

    Tensor tt = t.transpose(0, 2);

    ADV_CHECK(compare_two_tensors(tt.sum(-1), tt.sum(2)), "tt.sum(-1) == tt.sum(2)");
    ADV_CHECK(compare_two_tensors(tt.sum(-2), tt.sum(1)), "tt.sum(-2) == tt.sum(1)");
    ADV_CHECK(compare_two_tensors(tt.sum(-3), tt.sum(0)), "tt.sum(-3) == tt.sum(0)");

    ADV_CHECK(compare_two_tensors(tt.mean(-1), tt.mean(2)), "tt.mean(-1) == tt.mean(2)");
    ADV_CHECK(compare_two_tensors(tt.mean(-2), tt.mean(1)), "tt.mean(-2) == tt.mean(1)");
    ADV_CHECK(compare_two_tensors(tt.mean(-3), tt.mean(0)), "tt.mean(-3) == tt.mean(0)");

    bool caught_sum_pos = false;
    try {
        tt.sum(3);
    } catch (...) {
        caught_sum_pos = true;
    }
    ADV_CHECK(caught_sum_pos, "tt.sum(3) throws DIMENSION error for out of bounds");

    bool caught_sum_neg = false;
    try {
        tt.sum(-4);
    } catch (...) {
        caught_sum_neg = true;
    }
    ADV_CHECK(caught_sum_neg, "tt.sum(-4) throws DIMENSION error for out of bounds");

    bool caught_mean_pos = false;
    try {
        tt.mean(3);
    } catch (...) {
        caught_mean_pos = true;
    }
    ADV_CHECK(caught_mean_pos, "tt.mean(3) throws DIMENSION error for out of bounds");

    bool caught_mean_neg = false;
    try {
        tt.mean(-4);
    } catch (...) {
        caught_mean_neg = true;
    }
    ADV_CHECK(caught_mean_neg, "tt.mean(-4) throws DIMENSION error for out of bounds");
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "STARTING ADVERSARIAL STRESS TEST SUITE FOR R1\n";
    std::cout << "========================================================\n";

    test_3d_transposed_views();
    test_sliced_strided_views();
    test_unit_and_edge_dimensions();
    test_high_dimensional_reductions();
    test_double_precision_adversarial();
    test_autograd_complex_views();
    test_dimension_bounds();

    std::cout << "\n========================================================\n";
    std::cout << "ADVERSARIAL SUMMARY:\n";
    std::cout << "  Passed: " << g_adversarial_passes << "\n";
    std::cout << "  Failed: " << g_adversarial_fails << "\n";
    std::cout << "========================================================\n";

    return (g_adversarial_fails == 0 ? 0 : 1);
}
