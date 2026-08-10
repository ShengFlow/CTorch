/**
 * @file CDTF.h
 * @brief 跨后端梯度序列化协议 — CTorch Distributed Tensor Format v1.0
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details 本文件实现 CTorch Distributed Tensor Format (CDTF) v1.0，
 *          一个后端无关的梯度序列化协议。
 *
 *          设计原则（MacKay ITILA #4-7 — 信源编码定理）：
 *          1. 编码 dtype + shape + backend type + 数据
 *          2. 保证跨后端字节序和内存对齐一致性
 *          3. 支持自适应压缩（根据梯度熵选择量化精度）
 *          4. 包含校验和用于检测传输错误
 *
 *          协议格式：
 *          ┌──────────────────────────────────────────────────┐
 *          │ Header (32 bytes)                                 │
 *          │  - magic: 0x43445446 ("CDTF")        [4 bytes]   │
 *          │  - version: 1                         [2 bytes]   │
 *          │  - flags: compression/quantization     [2 bytes]   │
 *          │  - dtype: float32/float16/bfloat16     [2 bytes]   │
 *          │  - ndim: 1-8                          [2 bytes]   │
 *          │  - reserved                            [4 bytes]   │
 *          │  - checksum: CRC32                     [4 bytes]   │
 *          │  - data_size: total bytes              [8 bytes]   │
 *          │  - backend: source device type         [2 bytes]   │
 *          │  - endianness: little/big              [2 bytes]   │
 *          ├──────────────────────────────────────────────────┤
 *          │ Shape (ndim * 8 bytes)                            │
 *          │  - dim[0], dim[1], ..., dim[ndim-1]  [each 8 bytes]│
 *          ├──────────────────────────────────────────────────┤
 *          │ Data (data_size bytes)                            │
 *          │  - raw or compressed gradient data                │
 *          │  - 按行优先（row-major）排列，IEEE 754 格式          │
 *          └──────────────────────────────────────────────────┘
 *
 * @note 序列化协议设计为与后端无关：CommEngine 使用 CDTF 进行跨后端通信，
 *       DeviceBackend 子类使用 CDTF 的 serialize/deserialize 便捷接口。
 */

#ifndef CTORCH_DISTRIBUTED_CDTF_H
#define CTORCH_DISTRIBUTED_CDTF_H

#include "Ctools.h"
#include "CtorchError.h"
#include "Tensor.h"

#include <cstdint>
#include <vector>
#include <array>
#include <cstring>
#include <string>
#include <algorithm>

namespace ct {
namespace distributed {

/**
 * @brief CDTF 魔数 — "CDTF" 的 ASCII 表示
 */
static constexpr uint32_t kCDTFMagic = 0x43445446;

/**
 * @brief CDTF 当前版本
 */
static constexpr uint16_t kCDTFVersion = 1;

/**
 * @brief CDTF 最大支持维度数
 */
static constexpr uint8_t kCDTFMaxNDim = 8;

/**
 * @brief CDTF Header 大小（固定 32 字节）
 */
static constexpr size_t kCDTFHeaderSize = 32;

/**
 * @struct CDTFHeader
 * @brief CDTF 协议头部结构（32 字节）
 */
struct __attribute__((packed)) CDTFHeader {
    uint32_t magic;          ///< 魔数: 0x43445446 ("CDTF")
    uint16_t version;        ///< 协议版本
    uint16_t flags;          ///< 标志位: 压缩/量化标志
    uint16_t dtype;          ///< 数据类型: 映射到 DType 枚举
    uint16_t ndim;           ///< 维度数量 (1-8)
    uint32_t reserved;       ///< 保留字段
    uint32_t checksum;       ///< CRC32 校验和
    uint64_t data_size;      ///< 数据体大小（字节）
    uint16_t backend_type;   ///< 源后端类型
    uint16_t endianness;     ///< 字节序: 0=little, 1=big
};

/**
 * @brief CDTF 标志位定义
 */
enum CDTF_Flags : uint16_t {
    CDTF_FLAG_NONE        = 0x0000,  ///< 无压缩
    CDTF_FLAG_QUANTIZE_8  = 0x0001,  ///< 8-bit 量化
    CDTF_FLAG_QUANTIZE_16 = 0x0002,  ///< 16-bit 量化
    CDTF_FLAG_COMPRESSED  = 0x0004,  ///< 熵编码压缩
};

/**
 * @class CDTF
 * @brief CDTF 序列化/反序列化引擎
 *
 * 提供一个静态方法集合，用于将 Tensor 序列化为 CDTF 字节流，
 * 以及从 CDTF 字节流反序列化为 Tensor。
 *
 * 该类不持有状态，所有方法都是线程安全的。
 */
class CDTF {
public:
    /**
     * @brief 将 Tensor 序列化为 CDTF 字节流
     * @param tensor 输入张量
     * @param flags 序列化标志（压缩/量化选项）
     * @return CDTF 编码的字节流
     * @throws CtorchError 如果序列化失败（维度超限、dtype 不支持等）
     */
    static std::vector<uint8_t> serialize(const Tensor& tensor,
                                           uint16_t flags = CDTF_FLAG_NONE);

    /**
     * @brief 从 CDTF 字节流反序列化为 Tensor
     * @param data CDTF 编码的字节流
     * @return 反序列化后的 Tensor（在 CPU 设备上）
     * @throws CtorchError 如果反序列化失败（魔数不匹配、校验和错误等）
     */
    static Tensor deserialize(const std::vector<uint8_t>& data);

    /**
     * @brief 验证 CDTF 字节流的完整性
     * @param data CDTF 编码的字节流
     * @return true 如果格式正确且校验和匹配
     */
    static bool validate(const std::vector<uint8_t>& data);

    /**
     * @brief 获取 CDTF 字节流中 Tensor 的尺寸（不反序列化数据）
     * @param data CDTF 编码的字节流
     * @return 形状向量
     * @throws CtorchError 如果格式无效
     */
    static std::vector<size_t> peekShape(const std::vector<uint8_t>& data);

    /**
     * @brief 获取 CDTF 字节流中 Tensor 的数据类型
     * @param data CDTF 编码的字节流
     * @return DType 枚举值
     */
    static DType peekDType(const std::vector<uint8_t>& data);

    /**
     * @brief 获取 CDTF 字节流中 Tensor 的元素总数
     * @param data CDTF 编码的字节流
     * @return 元素数量
     */
    static size_t peekNumel(const std::vector<uint8_t>& data);

    /**
     * @brief 计算 CDTF 往返精度损失
     * @param tensor 输入张量
     * @return 序列化→反序列化后逐元素最大差异
     *
     * 用于 MVE-G2-1 实验验证，评估跨后端序列化的精度损失。
     */
    static float roundtripError(const Tensor& tensor);

private:
    /**
     * @brief 计算 CRC32 校验和
     * @param data 数据指针
     * @param len 数据长度
     * @return CRC32 值
     */
    static uint32_t crc32(const uint8_t* data, size_t len);

    /**
     * @brief 将 DType 编码为 uint16_t
     */
    static uint16_t dtypeToU16(DType dtype);

    /**
     * @brief 将 uint16_t 解码为 DType
     */
    static DType u16ToDType(uint16_t val);

    /**
     * @brief 检测当前平台的字节序
     * @return 0=little endian, 1=big endian
     */
    static uint16_t detectEndianness() {
        const uint16_t test = 0x0001;
        return (*reinterpret_cast<const uint8_t*>(&test) == 0x01) ? 0 : 1;
    }
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_CDTF_H