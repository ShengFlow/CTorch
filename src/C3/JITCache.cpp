/**
 * @file JITCache.cpp
 * @brief MLIR JIT bitcode 磁盘缓存实现
 * @see JITCache.h
 *
 * 关键设计决策：
 *  - 文件命名：c3_jit_<key>.bc + c3_jit_<key>.meta (JSON)
 *  - 写入模式：先写 .tmp，rename 原子替换
 *  - 失效：JIT backend version 不匹配 → 重新编译
 *  - 降级：磁盘错误 → 静默回退（log warning）
 *  - 自带轻量级 SHA-256（FIPS 180-4，无外部依赖，派生 cache key）
 *
 * 命名澄清（2026-08-15）：本类是把【JIT 编译产物（LLVM bitcode）】持久化到磁盘的
 * 缓存，本质是"JIT 缓存的磁盘版"，并非 Ahead-Of-Time（运行期仍需 LLVM JIT 编译成
 * 机器码）。目录仍复用 $C3_AOT_CACHE_DIR 环境变量（sandbox 硬约束，历史命名保留）。
 *
 * @date 2026/08/06
 */

#include "../../include/C3/JITCache.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

namespace ct {
namespace c3 {

// ======================= JIT backend version =======================

const char* JITCache::currentJITVersion() {
    // v1 -> v2: 修复 buildTiledMatMulWithEpilogue 中 2D bias 索引（改用 modulo 广播），
    // 使旧的错误缓存失效
    return "jit_v2";
}

// ======================= Cache directory =======================

std::string JITCache::resolveCacheDir() {
    // 优先级：$C3_AOT_CACHE_DIR > $HOME/.c3cache > /tmp/.c3cache
    // 新增：HOME/.c3cache 路径若不可写（sandbox 拦截等）→ 自动 fallback 到 /tmp/.c3cache，避免挂死
    const char* env = getenv("C3_AOT_CACHE_DIR");
    if (env && strlen(env) > 0) {
        return env;
    }

    // 检查 HOME 目录下的 .c3cache 是否真的可写
    const char* home = getenv("HOME");
    if (home && strlen(home) > 0) {
        std::string candidate = std::string(home) + "/.c3cache";
        // 先尝试 mkdir（如果不存在）
        int r = mkdir(candidate.c_str(), 0755);
        if (r == 0 || errno == EEXIST) {
            // 目录已存在或创建成功 → 再检查写权限（用候选目录下的临时文件来测，或直接 access W_OK）
            if (access(candidate.c_str(), W_OK) == 0) {
                return candidate;
            }
        }
        // 走到这里：HOME/.c3cache 创建失败或写权限不足 → sandbox 拦截等原因，fallback
#ifdef CT_DEBUG
        fprintf(stderr, "[JITCache] WARN: HOME cache dir '%s' not writable (errno=%d), "
                        "fallback to /tmp/.c3cache\n", candidate.c_str(), errno);
#endif
    }
    return "/tmp/.c3cache";
}

std::string JITCache::cacheDir() const {
    return resolveCacheDir();
}

// ======================= SHA-256（轻量级、自包含、无外部依赖） =======================
//
// 标准 FIPS 180-4 SHA-256 实现，用于派生 cache key（key 派生不是热路径，
// 追求正确性 + 零依赖）。自 AOTCache 移植（2026-08-15，AOTCache 已删除）。

namespace {

constexpr uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

struct SHA256Ctx {
    uint32_t state[8];
    uint64_t bit_count;        ///< 仅在 sha256_final 中设置（原始消息 bit 数）
    uint64_t processed_bytes;  ///< 已处理的原始消息字节数（在 update 中累加）
    uint8_t  buffer[64];
    size_t   buffer_len;
};

inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

void sha256_transform(SHA256Ctx& ctx, const uint8_t* data) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(data[i*4]) << 24) | (uint32_t(data[i*4+1]) << 16) |
               (uint32_t(data[i*4+2]) << 8) | uint32_t(data[i*4+3]);
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = ctx.state[0], b = ctx.state[1], c = ctx.state[2], d = ctx.state[3];
    uint32_t e = ctx.state[4], f = ctx.state[5], g = ctx.state[6], h = ctx.state[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx.state[0] += a; ctx.state[1] += b; ctx.state[2] += c; ctx.state[3] += d;
    ctx.state[4] += e; ctx.state[5] += f; ctx.state[6] += g; ctx.state[7] += h;
}

void sha256_init(SHA256Ctx& ctx) {
    ctx.state[0] = 0x6a09e667; ctx.state[1] = 0xbb67ae85;
    ctx.state[2] = 0x3c6ef372; ctx.state[3] = 0xa54ff53a;
    ctx.state[4] = 0x510e527f; ctx.state[5] = 0x9b05688c;
    ctx.state[6] = 0x1f83d9ab; ctx.state[7] = 0x5be0cd19;
    ctx.bit_count = 0;
    ctx.processed_bytes = 0;
    ctx.buffer_len = 0;
}

void sha256_update(SHA256Ctx& ctx, const uint8_t* data, size_t len) {
    // 注意：bit_count 在 sha256_final 中累加，**不**在 update 中。
    // 因为 final 内部会再次调 update 来写 padding 和 length，
    // 若在 update 中累加 bit_count，会把 padding 的字节也算进 message length。
    // 标准 FIPS 180-4 要求 length 是**原始消息**的 bit 数（不含 padding）。
    // 这里只累加 processed_bytes（原始消息字节数），不影响 padding。
    ctx.processed_bytes += len;
    while (len > 0) {
        size_t take = std::min(len, size_t(64) - ctx.buffer_len);
        std::memcpy(ctx.buffer + ctx.buffer_len, data, take);
        ctx.buffer_len += take;
        data += take;
        len -= take;
        if (ctx.buffer_len == 64) {
            sha256_transform(ctx, ctx.buffer);
            ctx.buffer_len = 0;
        }
    }
}

void sha256_final(SHA256Ctx& ctx, uint8_t out[32]) {
    // 在 padding 之前累加 bit_count：只算原始消息
    ctx.bit_count = ctx.processed_bytes * 8;

    uint8_t pad[64] = {0x80};
    size_t pad_len = (ctx.buffer_len < 56) ? (56 - ctx.buffer_len) : (120 - ctx.buffer_len);
    sha256_update(ctx, pad, pad_len);
    uint8_t len_bytes[8];
    for (int i = 0; i < 8; ++i) {
        len_bytes[i] = uint8_t(ctx.bit_count >> ((7 - i) * 8));
    }
    sha256_update(ctx, len_bytes, 8);
    for (int i = 0; i < 8; ++i) {
        out[i*4]   = uint8_t(ctx.state[i] >> 24);
        out[i*4+1] = uint8_t(ctx.state[i] >> 16);
        out[i*4+2] = uint8_t(ctx.state[i] >> 8);
        out[i*4+3] = uint8_t(ctx.state[i]);
    }
}

std::string sha256_hex(const std::string& data) {
    SHA256Ctx ctx;
    sha256_init(ctx);
    sha256_update(ctx, reinterpret_cast<const uint8_t*>(data.data()), data.size());
    uint8_t out[32];
    sha256_final(ctx, out);
    static const char* hex = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (int i = 0; i < 32; ++i) {
        result.push_back(hex[(out[i] >> 4) & 0xF]);
        result.push_back(hex[out[i] & 0xF]);
    }
    return result;
}

} // anonymous namespace

// ======================= Cache key =======================

std::string JITCache::makeKey(const std::string& graph_str, int opt_level) {
    std::ostringstream ss;
    ss << "c3_jit_" << currentJITVersion() << "_"
       << opt_level << "_"
       << graph_str;
    return sha256_hex(ss.str());
}

// ======================= Lookup =======================

std::string JITCache::lookup(const std::string& cache_key) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string dir = resolveCacheDir();
    std::string bc_path = dir + "/c3_jit_" + cache_key + ".bc";
    std::string meta_path = dir + "/c3_jit_" + cache_key + ".meta";

    // 检查 .bc 和 .meta 文件是否存在
    if (access(bc_path.c_str(), R_OK) != 0) {
        misses_++;
        return "";
    }

    // 检查 meta 中的版本号
    std::ifstream meta(meta_path);
    if (meta.is_open()) {
        std::string version;
        std::getline(meta, version);
        if (version != currentJITVersion()) {
            // 版本不匹配，删除旧文件
            unlink(bc_path.c_str());
            unlink(meta_path.c_str());
            misses_++;
            return "";
        }
    }

    hits_++;
    return bc_path;
}

// ======================= Store =======================

std::string JITCache::store(const std::string& cache_key, llvm::Module& module) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string dir = resolveCacheDir();
    std::string bc_path = dir + "/c3_jit_" + cache_key + ".bc";
    std::string meta_path = dir + "/c3_jit_" + cache_key + ".meta";

    #ifdef CT_DEBUG
    fprintf(stderr, "[JITCache] store: key=%s, bc_path=%s\n", cache_key.c_str(), bc_path.c_str());
#endif

    // 确保目录存在
    mkdir(dir.c_str(), 0755);

    // 写入 .bc 文件（先写 .tmp，再 rename）
    std::string tmp_bc = bc_path + ".tmp";
    std::string tmp_meta = meta_path + ".tmp";

    {
        std::error_code ec;
        llvm::raw_fd_ostream os(tmp_bc, ec);
        if (ec) {
            return "";
        }
        llvm::WriteBitcodeToFile(module, os);
        os.flush();
    }

    // 写入 .meta 文件
    {
        std::ofstream meta(tmp_meta);
        if (!meta.is_open()) {
            unlink(tmp_bc.c_str());
            return "";
        }
        meta << currentJITVersion() << "\n";
        meta.close();
    }

    // 原子 rename
    if (rename(tmp_bc.c_str(), bc_path.c_str()) != 0) {
        unlink(tmp_bc.c_str());
        unlink(tmp_meta.c_str());
        return "";
    }
    if (rename(tmp_meta.c_str(), meta_path.c_str()) != 0) {
        unlink(bc_path.c_str());
        unlink(tmp_meta.c_str());
        return "";
    }

    // [Dev] v0.5.2 (4) 1.0 store-only 计数器 (2026-08-09)
    stores_.fetch_add(1, std::memory_order_relaxed);
    return bc_path;
}

// ======================= Load =======================

std::unique_ptr<llvm::Module> JITCache::loadBitcode(
    const std::string& bc_path, llvm::LLVMContext& ctx) {
    auto buf = llvm::MemoryBuffer::getFile(bc_path);
    if (!buf) {
        return nullptr;
    }

    auto mod_or_err = llvm::parseBitcodeFile(buf->get()->getMemBufferRef(), ctx);
    if (!mod_or_err) {
        return nullptr;
    }

    return std::move(mod_or_err.get());
}

// ======================= Evict =======================

void JITCache::evict() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string dir = resolveCacheDir();
    DIR* d = opendir(dir.c_str());
    if (!d) return;

    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name(entry->d_name);
        // 只删除 JIT cache 文件
        if (name.find("c3_jit_") == 0) {
            std::string full = dir + "/" + name;
            unlink(full.c_str());
        }
    }
    closedir(d);
}

// ======================= Singleton =======================

JITCache& JITCache::getInstance() {
    static JITCache instance;
    return instance;
}

} // namespace c3
} // namespace ct