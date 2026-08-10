/**
 * @file EntropyAwareCompressor.h
 * @brief 率失真自适应压缩器 — 继承 Gen 1 RD-LocalSGD 的信息论驱动压缩
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details 本文件实现 EntropyAwareCompressor，是 Gen 1 RD-LocalSGD 的
 *          率失真（Rate-Distortion）自适应压缩在 Gen 2 中的独立实现。
 *
 *          设计原则（MacKay ITILA #4-7 — 信源编码定理）：
 *          1. 梯度值视为信源，压缩器在"率"（压缩比）和"失真"（精度损失）
 *             之间做 Pareto 最优权衡
 *          2. 自适应量化：根据梯度熵选择量化位数（8/16/32-bit）
 *          3. 数据处理不等式（ITILA #9）：压缩应单步联合设计，
 *             避免"先量化再编码"的信息级联损失
 *
 *          本模块与 CommEngine 完全解耦：
 *          EntropyAwareCompressor 只处理字节流（float* → uint8_t*），
 *          不关心张量语义或后端类型。
 *          CommEngine 在使用 CDTF 序列化前，可选调用本压缩器进行
 *          率失真优化。
 */

#ifndef CTORCH_DISTRIBUTED_ENTROPY_AWARE_COMPRESSOR_H
#define CTORCH_DISTRIBUTED_ENTROPY_AWARE_COMPRESSOR_H

#include "Ctools.h"
#include "CtorchError.h"
#include "Tensor.h"

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <limits>

namespace ct {
namespace distributed {

/**
 * @brief 量化精度枚举
 */
enum class QuantizePrecision : uint8_t {
    Float32 = 32,   ///< 无量化，保持 float32
    Float16 = 16,   ///< 量化到 16-bit float
    Int8    = 8,    ///< 量化到 8-bit integer
};

/**
 * @struct CompressionResult
 * @brief 压缩结果
 */
struct CompressionResult {
    std::vector<uint8_t> compressed_data;  ///< 压缩后的字节流
    size_t original_size;                  ///< 原始数据大小（字节）
    size_t compressed_size;                ///< 压缩后数据大小（字节）
    float compression_ratio;               ///< 压缩比 (compressed/original)
    QuantizePrecision precision;           ///< 使用的量化精度
    float estimated_entropy;               ///< 估计的梯度熵 (bits/sample)
    float max_distortion;                  ///< 最大失真（逐元素最大差异）
    float r2_score;                        ///< 压缩保真度 (R²)
    bool lossless;                         ///< 是否无损压缩
};

/**
 * @struct DecompressionResult
 * @brief 解压结果
 */
struct DecompressionResult {
    std::vector<float> data;    ///< 解压后的数据
    size_t num_elements;        ///< 元素数量
    float max_distortion;       ///< 解压误差（仅当有原始数据可比时有效）
};

/**
 * @struct QuantizationConfig
 * @brief 量化配置
 */
struct QuantizationConfig {
    float entropy_threshold_8bit;   ///< 启用 8-bit 量化的熵阈值
    float entropy_threshold_16bit;  ///< 启用 16-bit 量化的熵阈值
    size_t histogram_bins;          ///< 熵估计的直方图桶数
    bool enable_entropy_coding;     ///< 是否启用熵编码
    bool enable_outlier_protection; ///< 是否保护离群值（使用更高精度）
    float outlier_std_threshold;    ///< 离群值的标准差阈值

    static QuantizationConfig defaultConfig() {
        return QuantizationConfig{
            1.5f,       // entropy_threshold_8bit
            3.0f,       // entropy_threshold_16bit
            256,        // histogram_bins
            true,       // enable_entropy_coding
            true,       // enable_outlier_protection
            3.0f        // outlier_std_threshold
        };
    }
};

/**
 * @struct CompressionStats
 * @brief 压缩统计信息
 */
struct CompressionStats {
    size_t total_compressions;       ///< 总压缩次数
    size_t total_decompressions;     ///< 总解压次数
    size_t total_original_bytes;     ///< 总原始字节数
    size_t total_compressed_bytes;   ///< 总压缩后字节数
    float avg_compression_ratio;     ///< 平均压缩比
    float avg_compression_time_ms;   ///< 平均压缩时间 (ms)
    float avg_max_distortion;        ///< 平均最大失真
    size_t lossless_count;           ///< 无损压缩次数
    size_t lossy_count;              ///< 有损压缩次数

    void reset() {
        total_compressions = 0;
        total_decompressions = 0;
        total_original_bytes = 0;
        total_compressed_bytes = 0;
        avg_compression_ratio = 1.0f;
        avg_compression_time_ms = 0.0f;
        avg_max_distortion = 0.0f;
        lossless_count = 0;
        lossy_count = 0;
    }
};

/**
 * @class EntropyAwareCompressor
 * @brief 率失真自适应压缩器
 *
 * 基于信息论的梯度压缩器，核心算法：
 * 1. 熵估计：使用直方图估计梯度值的经验熵
 * 2. 量化选择：根据熵阈值自动选择量化精度
 * 3. 量化编码：支持 8-bit int、16-bit float、32-bit float
 * 4. 离群值保护：检测并保护离群值（使用更高精度存储）
 * 5. 熵编码：对量化后的数据进行简单的熵编码
 *
 * 本压缩器是无状态的函数式接口，所有方法都是线程安全的。
 * 压缩/解压不依赖任何外部状态。
 */
class EntropyAwareCompressor {
public:
    /**
     * @brief 构造压缩器
     * @param config 量化配置
     */
    explicit EntropyAwareCompressor(
        QuantizationConfig config = QuantizationConfig::defaultConfig());

    ~EntropyAwareCompressor() = default;

    // ======================= 核心压缩/解压 =======================

    /**
     * @brief 压缩张量数据
     * @param tensor 输入张量
     * @return 压缩结果
     *
     * 自动选择量化精度，执行率失真最优压缩。
     * 如果张量在非 CPU 设备上，先移动到 CPU 再压缩。
     */
    CompressionResult compress(const Tensor& tensor);

    /**
     * @brief 压缩原始 float 数据
     * @param data 输入 float 数据
     * @param num_elements 元素数量
     * @param forced_precision 强制使用指定精度（可选）
     * @return 压缩结果
     */
    CompressionResult compress(const float* data, size_t num_elements,
                                QuantizePrecision forced_precision = QuantizePrecision::Float32);

    /**
     * @brief 解压为 float 数组
     * @param compressed 压缩后的字节流
     * @return 解压结果
     * @throws CtorchError 如果格式无效
     */
    DecompressionResult decompress(const std::vector<uint8_t>& compressed);

    /**
     * @brief 解压为 Tensor
     * @param compressed 压缩后的字节流
     * @param shape 目标形状
     * @return 解压后的 Tensor（在 CPU 上）
     */
    Tensor decompressToTensor(const std::vector<uint8_t>& compressed,
                               const std::vector<size_t>& shape);

    // ======================= 熵估计 =======================

    /**
     * @brief 估计梯度数据的经验熵
     * @param data 输入数据
     * @param num_elements 元素数量
     * @return 经验熵 (bits/sample)
     *
     * 使用直方图方法估计，直方图桶数由 config.histogram_bins 控制。
     * 熵值越低 → 压缩潜力越大。
     */
    float estimateEntropy(const float* data, size_t num_elements) const;

    /**
     * @brief 根据熵自动选择量化精度
     * @param entropy 估计熵值
     * @return 选择的量化精度
     */
    QuantizePrecision selectPrecision(float entropy) const;

    // ======================= 配置 =======================

    /**
     * @brief 设置量化配置
     * @param config 量化配置
     */
    void setConfig(const QuantizationConfig& config) { _config = config; }

    /**
     * @brief 获取当前量化配置
     * @return 量化配置
     */
    const QuantizationConfig& config() const { return _config; }

    // ======================= 统计信息 =======================

    /**
     * @brief 获取压缩统计信息
     * @return 统计信息
     */
    CompressionStats stats() const { return _stats; }

    /**
     * @brief 重置统计信息
     */
    void resetStats() { _stats.reset(); }

    /**
     * @brief 预测压缩比
     * @param entropy 估计熵值
     * @return 预期的压缩比
     *
     * 基于率失真理论的上界估计：
     * compress_ratio ≈ entropy / (bits_per_sample * 8)
     * 其中 bits_per_sample 是量化位数
     */
    float predictCompressionRatio(float entropy) const;

private:
    QuantizationConfig _config;
    mutable std::mutex _mtx;
    CompressionStats _stats;

    // ======================= 内部压缩实现 =======================

    /**
     * @brief 执行 8-bit 量化
     * @param data 输入 float 数据
     * @param num_elements 元素数量
     * @param min_val 最小值（用于缩放）
     * @param max_val 最大值（用于缩放）
     * @param outlier_mask 离群值掩码（可选）
     * @return 量化后的字节流
     */
    std::vector<uint8_t> quantize8bit(const float* data, size_t num_elements,
                                       float min_val, float max_val,
                                       const std::vector<bool>& outlier_mask = {});

    /**
     * @brief 执行 16-bit float 量化
     * @param data 输入 float 数据
     * @param num_elements 元素数量
     * @return 量化后的字节流
     */
    std::vector<uint8_t> quantize16bit(const float* data, size_t num_elements);

    /**
     * @brief 解压 8-bit 量化数据
     * @param data 量化后的字节流
     * @param offset 起始偏移
     * @param num_elements 元素数量
     * @param min_val 缩放最小值
     * @param max_val 缩放最大值
     * @return 解压后的 float 数据
     */
    std::vector<float> dequantize8bit(const uint8_t* data, size_t offset,
                                       size_t num_elements, float min_val, float max_val);

    /**
     * @brief 解压 16-bit float 量化数据
     * @param data 压缩字节流
     * @param offset 起始偏移
     * @param num_elements 元素数量
     * @return 解压后的 float 数据
     */
    std::vector<float> dequantize16bit(const uint8_t* data, size_t offset,
                                        size_t num_elements);

    /**
     * @brief 检测离群值
     * @param data 输入数据
     * @param num_elements 元素数量
     * @return 离群值掩码
     */
    std::vector<bool> detectOutliers(const float* data, size_t num_elements) const;

    /**
     * @brief 简单的运行长度编码（RLE）
     * @param data 输入字节流
     * @return RLE 编码后的字节流
     */
    std::vector<uint8_t> runLengthEncode(const std::vector<uint8_t>& data) const;

    /**
     * @brief 解码 RLE
     * @param data RLE 编码的字节流
     * @return 解码后的字节流
     */
    std::vector<uint8_t> runLengthDecode(const std::vector<uint8_t>& data) const;

    /**
     * @brief 计算最大失真
     * @param original 原始数据
     * @param decompressed 解压后数据
     * @param num_elements 元素数量
     * @return 逐元素最大绝对差异
     */
    float computeMaxDistortion(const float* original, const float* decompressed,
                                size_t num_elements) const;

    /**
     * @brief 计算 R² 保真度
     * @param original 原始数据
     * @param decompressed 解压后数据
     * @param num_elements 元素数量
     * @return R² 值（1.0 = 完美保真）
     */
    float computeR2Score(const float* original, const float* decompressed,
                          size_t num_elements) const;
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_ENTROPY_AWARE_COMPRESSOR_H