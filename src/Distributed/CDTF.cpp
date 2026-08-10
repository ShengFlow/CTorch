#include "Distributed/CDTF.h"

#include <cstring>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace ct {
namespace distributed {

// ======================= CRC32 查找表 =======================
static const uint32_t kCRC32Table[256] = {
    0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F,
    0xE963A535, 0x9E6495A3, 0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988,
    0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91, 0x1DB71064, 0x6AB020F2,
    0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
    0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9,
    0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
    0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C,
    0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
    0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423,
    0xCFBA9599, 0xB8BDA50F, 0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
    0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D, 0x76DC4190, 0x01DB7106,
    0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
    0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D,
    0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
    0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950,
    0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
    0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7,
    0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
    0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9, 0x5005713C, 0x270241AA,
    0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
    0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81,
    0xB7BD5C3B, 0xC0BA6CAD, 0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A,
    0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683, 0xE3630B12, 0x94643B84,
    0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
    0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB,
    0x196C3671, 0x6E6B06E7, 0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC,
    0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5, 0xD6D6A3E8, 0xA1D1937E,
    0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
    0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55,
    0x316E8EEF, 0x4669BE79, 0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236,
    0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F, 0xC5BA3BBE, 0xB2BD0B28,
    0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
    0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F,
    0x72076785, 0x05005713, 0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38,
    0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21, 0x86D3D2D4, 0xF1D4E242,
    0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
    0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69,
    0x616BFFD3, 0x166CCF45, 0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2,
    0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB, 0xAED16A4A, 0xD9D65ADC,
    0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
    0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD70693,
    0x54DE5729, 0x23D967BF, 0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
    0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
};

uint32_t CDTF::crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc = kCRC32Table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

// ======================= Float16 转换 =======================

/**
 * @brief IEEE 754 float32 → float16 转换
 *
 * float16 格式：
 *   sign: 1 bit, exponent: 5 bits (bias 15), mantissa: 10 bits
 */
static uint16_t float32ToFloat16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));

    uint16_t sign = (u >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = u & 0x007FFFFF;

    if (exp <= 0) {
        // 非规格化数或零：float32 中 exp=0 不代表 float16 的 exp=0
        // 直接截断为 0
        return sign;
    }
    if (exp >= 31) {
        // 无穷大或 NaN：饱和到 float16 的无穷大
        return sign | 0x7C00 | (mantissa != 0 ? 0x0200 : 0);
    }

    // 正常情况：截断 mantissa 到 10 bits
    return sign | (static_cast<uint16_t>(exp) << 10) | (mantissa >> 13);
}

/**
 * @brief IEEE 754 float16 → float32 转换
 */
static float float16ToFloat32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;

    uint32_t u;
    if (exp == 0) {
        // 非规格化数或零
        if (mantissa == 0) {
            u = sign;
        } else {
            // 非规格化 float16 → 规格化 float32
            int32_t e = -1;
            uint32_t m = mantissa;
            while ((m & 0x0400) == 0) { m <<= 1; e--; }
            exp = 15 + e + 127;  // float16 bias 15 → float32 bias 127
            mantissa = (m & 0x03FF) << 13;
            u = sign | (static_cast<uint32_t>(exp) << 23) | mantissa;
        }
    } else if (exp == 31) {
        // 无穷大或 NaN
        u = sign | 0x7F800000 | (mantissa << 13);
    } else {
        // 正常情况
        exp = exp - 15 + 127;
        u = sign | (static_cast<uint32_t>(exp) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &u, sizeof(result));
    return result;
}

uint16_t CDTF::dtypeToU16(DType dtype) {
    switch (dtype) {
        case DType::kFloat:  return 0;
        case DType::kDouble: return 1;
        case DType::kInt:    return 2;
        case DType::kLong:   return 3;
        case DType::kBool:   return 4;
        default:
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                "CDTF: unsupported dtype for serialization");
            return 0xFFFF;
    }
}

DType CDTF::u16ToDType(uint16_t val) {
    switch (val) {
        case 0:  return DType::kFloat;
        case 1:  return DType::kDouble;
        case 2:  return DType::kInt;
        case 3:  return DType::kLong;
        case 4:  return DType::kBool;
        default:
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                "CDTF: unknown dtype code in header");
            return DType::kFloat;
    }
}

std::vector<uint8_t> CDTF::serialize(const Tensor& tensor, uint16_t flags) {
    const auto& shape = tensor.shape();
    size_t ndim = shape.size();
    if (ndim > kCDTFMaxNDim) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "CDTF: tensor dimension exceeds max (8)");
    }
    if (ndim == 0) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "CDTF: scalar tensor serialization not supported");
    }

    // 计算数据大小
    size_t elem_size = dtypeSize(tensor.dtype());
    size_t numel = tensor.numel();

    // 读取数据到 CPU buffer
    Tensor cpu_tensor = (tensor.device() == DeviceType::kCPU)
        ? tensor : tensor.to(DeviceType::kCPU);
    const float* data_ptr = cpu_tensor.data_read<float>();
    if (numel > 0 && !data_ptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
            "CDTF: failed to read tensor data for serialization");
    }

    // 根据 flags 选择实际存储格式
    bool use_float16 = (flags & CDTF_FLAG_QUANTIZE_16) != 0;
    bool use_int8 = (flags & CDTF_FLAG_QUANTIZE_8) != 0;
    size_t data_size;
    if (use_int8) {
        // 8-bit 量化：min/max 前缀 (8 bytes) + 每元素 1 byte
        data_size = 2 * sizeof(float) + numel;
    } else if (use_float16) {
        data_size = numel * sizeof(uint16_t);
    } else {
        data_size = numel * elem_size;
    }

    // 计算总大小：header + shape + data
    size_t shape_bytes = ndim * sizeof(uint64_t);
    size_t total_size = kCDTFHeaderSize + shape_bytes + data_size;

    std::vector<uint8_t> buffer(total_size);

    // 填充 header
    CDTFHeader header;
    header.magic = kCDTFMagic;
    header.version = kCDTFVersion;
    header.flags = flags;
    header.dtype = dtypeToU16(tensor.dtype());
    header.ndim = static_cast<uint16_t>(ndim);
    header.reserved = 0;
    header.data_size = data_size;
    header.backend_type = static_cast<uint16_t>(tensor.device());
    header.endianness = detectEndianness();
    // checksum 先置 0，序列化完再计算
    header.checksum = 0;

    std::memcpy(buffer.data(), &header, sizeof(CDTFHeader));

    // 填充 shape
    uint64_t* shape_buf = reinterpret_cast<uint64_t*>(buffer.data() + kCDTFHeaderSize);
    for (size_t i = 0; i < ndim; ++i) {
        shape_buf[i] = static_cast<uint64_t>(shape[i]);
    }

    // 填充数据
    uint8_t* data_buf = buffer.data() + kCDTFHeaderSize + shape_bytes;
    if (data_size > 0 && data_ptr) {
        if (use_int8) {
            // 8-bit 量化：min-max 缩放 + 量化到 [0, 255]
            float min_val = data_ptr[0], max_val = data_ptr[0];
            for (size_t i = 1; i < numel; ++i) {
                if (data_ptr[i] < min_val) min_val = data_ptr[i];
                if (data_ptr[i] > max_val) max_val = data_ptr[i];
            }
            if (max_val - min_val < 1e-12f) {
                max_val = min_val + 1e-12f;
            }
            // 写入 min/max
            std::memcpy(data_buf, &min_val, sizeof(float));
            std::memcpy(data_buf + sizeof(float), &max_val, sizeof(float));
            // 量化
            float scale = 255.0f / (max_val - min_val);
            uint8_t* q_buf = data_buf + 2 * sizeof(float);
            for (size_t i = 0; i < numel; ++i) {
                float normalized = (data_ptr[i] - min_val) * scale;
                if (normalized < 0.0f) normalized = 0.0f;
                if (normalized > 255.0f) normalized = 255.0f;
                q_buf[i] = static_cast<uint8_t>(std::round(normalized));
            }
        } else if (use_float16) {
            // Float16 量化：逐元素转换
            uint16_t* f16_buf = reinterpret_cast<uint16_t*>(data_buf);
            for (size_t i = 0; i < numel; ++i) {
                f16_buf[i] = float32ToFloat16(data_ptr[i]);
            }
        } else {
            // 原始 float32 数据
            std::memcpy(data_buf, data_ptr, data_size);
        }
    }

    // 计算并写入校验和
    CDTFHeader* hdr = reinterpret_cast<CDTFHeader*>(buffer.data());
    hdr->checksum = crc32(buffer.data() + sizeof(CDTFHeader),
                           buffer.size() - sizeof(CDTFHeader));

    return buffer;
}

Tensor CDTF::deserialize(const std::vector<uint8_t>& data) {
    if (data.size() < kCDTFHeaderSize) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
            "CDTF: data too small for header");
    }

    // 解析 header
    const CDTFHeader& header = *reinterpret_cast<const CDTFHeader*>(data.data());

    // 验证魔数
    if (header.magic != kCDTFMagic) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
            "CDTF: invalid magic number");
    }

    // 验证版本
    if (header.version > kCDTFVersion) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
            "CDTF: unsupported version");
    }

    // 验证校验和
    uint32_t stored_checksum = header.checksum;
    // 计算校验和时使用 checksum=0 的 header
    std::vector<uint8_t> verify_data(data);
    CDTFHeader* verify_header = reinterpret_cast<CDTFHeader*>(verify_data.data());
    verify_header->checksum = 0;
    uint32_t computed = crc32(verify_data.data() + sizeof(CDTFHeader),
                               verify_data.size() - sizeof(CDTFHeader));
    if (stored_checksum != computed) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
            "CDTF: checksum mismatch — data may be corrupted");
    }

    // 解析 dtype
    DType dtype = u16ToDType(header.dtype);

    // 验证 ndim
    if (header.ndim > kCDTFMaxNDim || header.ndim == 0) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "CDTF: invalid ndim in header");
    }

    // 验证 data_size 不超出缓冲区
    size_t ndim = header.ndim;
    size_t shape_bytes = ndim * sizeof(uint64_t);
    size_t min_expected_size = kCDTFHeaderSize + shape_bytes + header.data_size;
    if (data.size() < min_expected_size) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
            "CDTF: data size mismatch — buffer too small for claimed data_size");
    }

    // 解析 shape
    const uint64_t* shape_buf = reinterpret_cast<const uint64_t*>(
        data.data() + kCDTFHeaderSize);
    std::vector<size_t> shape(ndim);
    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        shape[i] = static_cast<size_t>(shape_buf[i]);
        numel *= shape[i];
    }

    // 验证 numel 与 data_size 一致
    bool is_float16 = (header.flags & CDTF_FLAG_QUANTIZE_16) != 0;
    bool is_int8 = (header.flags & CDTF_FLAG_QUANTIZE_8) != 0;
    size_t expected_data_size;
    if (is_int8) {
        expected_data_size = 2 * sizeof(float) + numel;  // min/max 前缀 + 量化值
    } else if (is_float16) {
        expected_data_size = numel * sizeof(uint16_t);
    } else {
        expected_data_size = numel * sizeof(float);
    }
    if (header.data_size != expected_data_size) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "CDTF: data_size (" + std::to_string(header.data_size)
            + ") does not match tensor dimensions (" + std::to_string(expected_data_size) + ")");
    }

    // 构造 Tensor
    Tensor result(ShapeTag{}, shape, dtype, DeviceType::kCPU, false);

    // 复制数据（处理 Float16/Int8 反量化）
    const uint8_t* src_data = data.data() + kCDTFHeaderSize + shape_bytes;
    float* dst_data = result.data_write<float>();

    if (numel > 0 && src_data && dst_data) {
        if (is_int8) {
            // Int8 → Float32 反量化：读取 min/max，还原
            float min_val, max_val;
            std::memcpy(&min_val, src_data, sizeof(float));
            std::memcpy(&max_val, src_data + sizeof(float), sizeof(float));
            float inv_scale = (max_val - min_val) / 255.0f;
            const uint8_t* q_data = src_data + 2 * sizeof(float);
            for (size_t i = 0; i < numel; ++i) {
                dst_data[i] = min_val + static_cast<float>(q_data[i]) * inv_scale;
            }
        } else if (is_float16) {
            // Float16 → Float32 反量化
            const uint16_t* f16_data = reinterpret_cast<const uint16_t*>(src_data);
            for (size_t i = 0; i < numel; ++i) {
                dst_data[i] = float16ToFloat32(f16_data[i]);
            }
        } else {
            std::memcpy(dst_data, src_data, numel * sizeof(float));
        }
    }

    return result;
}

bool CDTF::validate(const std::vector<uint8_t>& data) {
    if (data.size() < kCDTFHeaderSize) return false;
    const CDTFHeader& header = *reinterpret_cast<const CDTFHeader*>(data.data());
    if (header.magic != kCDTFMagic) return false;
    if (header.version > kCDTFVersion) return false;
    if (header.ndim > kCDTFMaxNDim) return false;

    // 验证校验和
    uint32_t stored_checksum = header.checksum;
    std::vector<uint8_t> verify_data(data);
    CDTFHeader* verify_header = reinterpret_cast<CDTFHeader*>(verify_data.data());
    verify_header->checksum = 0;
    uint32_t computed = crc32(verify_data.data() + sizeof(CDTFHeader),
                               verify_data.size() - sizeof(CDTFHeader));
    return stored_checksum == computed;
}

std::vector<size_t> CDTF::peekShape(const std::vector<uint8_t>& data) {
    if (data.size() < kCDTFHeaderSize) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
            "CDTF: data too small for header");
    }
    const CDTFHeader& header = *reinterpret_cast<const CDTFHeader*>(data.data());
    if (header.magic != kCDTFMagic) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
            "CDTF: invalid magic number");
    }
    size_t ndim = header.ndim;
    if (ndim > kCDTFMaxNDim || ndim == 0) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "CDTF: invalid ndim in peekShape");
    }
    size_t shape_bytes = ndim * sizeof(uint64_t);
    if (data.size() < kCDTFHeaderSize + shape_bytes) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
            "CDTF: data too small for shape data");
    }
    const uint64_t* shape_buf = reinterpret_cast<const uint64_t*>(
        data.data() + kCDTFHeaderSize);
    std::vector<size_t> shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        shape[i] = static_cast<size_t>(shape_buf[i]);
    }
    return shape;
}

DType CDTF::peekDType(const std::vector<uint8_t>& data) {
    if (data.size() < kCDTFHeaderSize) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
            "CDTF: data too small for header");
    }
    const CDTFHeader& header = *reinterpret_cast<const CDTFHeader*>(data.data());
    return u16ToDType(header.dtype);
}

size_t CDTF::peekNumel(const std::vector<uint8_t>& data) {
    auto shape = peekShape(data);
    if (shape.empty()) return 0;
    size_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

float CDTF::roundtripError(const Tensor& tensor) {
    auto serialized = serialize(tensor, CDTF_FLAG_NONE);
    auto deserialized = deserialize(serialized);

    // 确保两者都在 CPU 上
    Tensor orig = (tensor.device() == DeviceType::kCPU)
        ? tensor : tensor.to(DeviceType::kCPU);

    if (orig.numel() != deserialized.numel()) {
        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "CDTF: roundtrip numel mismatch — orig=" + std::to_string(orig.numel())
            + ", deserialized=" + std::to_string(deserialized.numel()));
    }

    if (orig.numel() == 0 || deserialized.numel() == 0) {
        return 0.0f;
    }

    float max_diff = 0.0f;
    size_t n = std::min(orig.numel(), deserialized.numel());
    const float* orig_data = orig.data_read<float>();
    const float* des_data = deserialized.data_read<float>();

    if (!orig_data || !des_data) return -1.0f;

    for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(orig_data[i] - des_data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

} // namespace distributed
} // namespace ct