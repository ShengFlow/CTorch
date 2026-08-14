/**
 * @file AOTCache.cpp
 * @brief AOT 持久化 .so cache 实现
 * @see AOTCache.h
 * @see ADR-008-aot-persistent-cache
 *
 * 关键设计决策：
 *  - 文件命名：c3_<key>.so + c3_<key>.meta (JSON)
 *  - 写入模式：先写 .tmp，fsync，rename 原子替换
 *  - 失效：backend version 不匹配 → 重新编译
 *  - 降级：磁盘错误 → 静默回退 in-memory（log warning）
 *  - 跨进程：fork 子进程测试验证 dlopen 共享
 *
 * @date 2026/08/03
 */

#include "../../include/C3/AOTCache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace ct {
namespace c3 {

// ======================= SHA-256 实现（轻量级、自包含、无外部依赖） =======================
//
// 标准 FIPS 180-4 SHA-256 实现，用于派生 cache key。
// 不追求极致性能（key 派生不是热路径），追求正确性 + 零依赖。
// ~150 行，可独立测试。

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

// ======================= AOTCache 实现 =======================

AOTCache& AOTCache::getInstance() {
    static AOTCache instance;
    return instance;
}

const char* AOTCache::currentBackendVersion() {
    // Handwritten kernel 生成器后端版本号
    // 不兼容变更（如：生成代码签名变化、buffer layout 变化）必须递增此版本
    // 兼容变更（如：内部优化）无需递增
    //
    // 版本历史：
    //   v1: 初始版本（支持单节点 + 多节点）
    //   v2: 引入 multi-node kernel，签名变化
    //   v3: 引入 fused kernel（cache key 前缀区分）
    //   v4: 2026-08-11 反向图 bug 修复：TransposeNode 改用 external_input_map 解析
    //       输入指针 + 新增 GtNode/ExpNode/LogNode 处理 + 多输出平面 buffer 布局。
    //       生成代码签名变化，必须 bump 以便失效旧 AOT 缓存（否则旧 kernel 持续被加载）。
    //   v5: 2026-08-15 并行切片越界修复：所有逐元素循环上界从编译期 node_n
    //       改为 std::min(node_n, n)（n=运行时切片大小），生成代码签名变化，
    //       必须 bump 失效旧 AOT 缓存（否则旧 kernel 越界写继续被加载）。
    return "handwritten-v5";
}

std::string AOTCache::makeKey(
    const std::string& graph_str,
    const std::string& device,
    int opt_level,
    const std::string& backend_version)
{
    // 拼接所有影响编译产物的因素，SHA-256 后取前 32 hex chars (128 bit)
    // 32 chars 足够避免冲突（生日悖论：~2^64 个 key 才 50% 概率冲突）
    std::string combined = graph_str + "|" + device + "|" +
                          std::to_string(opt_level) + "|" + backend_version;
    std::string full_hash = sha256_hex(combined);
    return full_hash.substr(0, 32);
}

std::string AOTCache::sha256Hex(const std::string& data) {
    return sha256_hex(data);
}

std::string AOTCache::effectiveCacheDir(const AOTCacheConfig& cfg) {
    if (!cfg.custom_dir.empty()) {
        return cfg.custom_dir;
    }
    const char* env = std::getenv("C3_AOT_CACHE_DIR");
    if (env != nullptr && env[0] != '\0') {
        return std::string(env);
    }
    const char* home = std::getenv("HOME");
    if (home != nullptr && home[0] != '\0') {
        return std::string(home) + "/.c3cache";
    }
    // Fallback: /tmp（不理想但总比没有好）
    return "/tmp/.c3cache";
}

/// 确保目录存在（不存在则创建）。返回是否成功。
static bool ensureDir(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    // 0777 & ~umask = 0755（用户读写，组/其他只读）
    return mkdir(dir.c_str(), 0755) == 0;
}

std::string AOTCache::getCacheDir() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!dir_initialized_) {
        cached_dir_ = effectiveCacheDir(config_);
        dir_initialized_ = true;
    }
    return cached_dir_;
}

std::string AOTCache::lookup(const std::string& cache_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.enabled) {
        return "";
    }

    std::string dir = effectiveCacheDir(config_);
    if (dir.empty() || !ensureDir(dir)) {
        stats_.disk_errors++;
        return ""; // 磁盘不可用 → 静默降级
    }
    cached_dir_ = dir;
    dir_initialized_ = true;

    std::string so_path = dir + "/c3_" + cache_key + ".so";
    std::string meta_path = dir + "/c3_" + cache_key + ".meta";

    // 检查 .so 文件存在
    struct stat so_st;
    if (stat(so_path.c_str(), &so_st) != 0) {
        stats_.misses++;
        return ""; // 文件不存在 → miss
    }

    // 检查 .meta 文件存在
    struct stat meta_st;
    if (stat(meta_path.c_str(), &meta_st) != 0) {
        // .so 存在但 .meta 缺失 → 视为 miss（让 store 重新写）
        stats_.misses++;
        return "";
    }

    // 读取 .meta 并校验 backend version
    std::ifstream meta_file(meta_path);
    if (!meta_file) {
        stats_.misses++;
        return "";
    }
    std::string line, stored_version;
    while (std::getline(meta_file, line)) {
        if (line.rfind("backend_version=", 0) == 0) {
            stored_version = line.substr(strlen("backend_version="));
        }
    }
    if (stored_version != currentBackendVersion()) {
        // 版本不匹配 → 视为 miss + invalidation
        stats_.misses++;
        stats_.invalidations++;
        return "";
    }

    // 命中
    stats_.hits++;
    return so_path;
}

std::string AOTCache::store(const std::string& cache_key, const std::string& so_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.enabled) {
        return "";
    }

    std::string dir = effectiveCacheDir(config_);
    if (dir.empty() || !ensureDir(dir)) {
        stats_.disk_errors++;
        return "";
    }
    cached_dir_ = dir;
    dir_initialized_ = true;

    std::string final_path = dir + "/c3_" + cache_key + ".so";
    std::string meta_path = dir + "/c3_" + cache_key + ".meta";
    std::string tmp_so = dir + "/.c3_" + cache_key + ".so.tmp";
    std::string tmp_meta = dir + "/.c3_" + cache_key + ".meta.tmp";

    // 读取 .so 内容并写到 .tmp
    std::ifstream src(so_path, std::ios::binary);
    if (!src) {
        stats_.disk_errors++;
        return "";
    }
    {
        std::ofstream dst(tmp_so, std::ios::binary);
        if (!dst) {
            stats_.disk_errors++;
            return "";
        }
        dst << src.rdbuf();
        if (!dst) {
            stats_.disk_errors++;
            std::remove(tmp_so.c_str());
            return "";
        }
    }
    // [Fix 2026-08-10 code-review-001910 P0-1]: fsync tmp_so
    //   之前 rename 前没 fsync tmp_so, 系统崩溃时 rename 后的 .so 实际内容可能没落盘,
    //   下次 dlopen 失败但 cache 表面"已写" → 静默损坏
    //   修法: reopen tmp_so 拿 fd, fsync, close. rename 前必须 fsync 数据文件
    {
        int fd = ::open(tmp_so.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::fsync(fd);
            ::close(fd);
        }
    }

    // 原子 rename 到最终路径
    if (std::rename(tmp_so.c_str(), final_path.c_str()) != 0) {
        stats_.disk_errors++;
        std::remove(tmp_so.c_str());
        return "";
    }
    // [Fix 2026-08-10 code-review-001910 P0-1]: fsync 父目录
    //   rename 后必须 fsync 父目录, 确保目录项变更 (新文件名) 落盘.
    //   否则系统崩溃时 .so 文件存在但目录项没更新 → 文件名找不到 → 静默损坏
    {
        int dirfd = ::open(dir.c_str(), O_RDONLY);
        if (dirfd >= 0) {
            ::fsync(dirfd);
            ::close(dirfd);
        }
    }

    // 写 .meta 文件 (per P2-1 合并修法: 同样 atomic rename + fsync 父目录)
    {
        std::ofstream meta(tmp_meta);
        if (!meta) {
            // .so 已写入, .meta 失败不算致命 (下次启动会重新生成)
            stats_.disk_errors++;
        } else {
            meta << "backend_version=" << currentBackendVersion() << "\n";
            meta << "cache_key=" << cache_key << "\n";
        }
    }
    // [Fix 2026-08-10 code-review-001910 P2-1]: .meta 也走 atomic rename
    //   之前直接 ofstream 写 final .meta, 中途崩溃会留半截文件
    //   修法: 写 .tmp_meta + atomic rename, 跟 .so 一致
    if (std::rename(tmp_meta.c_str(), meta_path.c_str()) != 0) {
        // .meta 失败不算致命 (.so 已落盘), 但记录 disk_errors
        std::remove(tmp_meta.c_str());
        stats_.disk_errors++;
    } else {
        // .meta rename 成功, fsync 父目录
        int dirfd = ::open(dir.c_str(), O_RDONLY);
        if (dirfd >= 0) {
            ::fsync(dirfd);
            ::close(dirfd);
        }
    }

    stats_.writes++;
    return final_path;
}

void AOTCache::evict() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string dir = effectiveCacheDir(config_);
    if (dir.empty()) return;

    DIR* d = opendir(dir.c_str());
    if (!d) {
        // 目录不存在 = 已经清空
        return;
    }

    struct dirent* entry;
    int removed = 0;
    while ((entry = readdir(d)) != nullptr) {
        std::string name = entry->d_name;
        // 只删除 c3_ 前缀的文件（不破坏用户其他文件）
        if (name.find("c3_") == 0) {
            std::string full = dir + "/" + name;
            if (std::remove(full.c_str()) == 0) {
                removed++;
            }
        }
    }
    closedir(d);

    stats_.evictions++;
    (void)removed;
}

void AOTCache::scanDiskUsage() {
    // 内部辅助：扫描 c3_* 文件统计占用
    // 当前未在 hot path 使用（仅 evict 时考虑），保留供未来 LRU 扩展
    std::string dir = effectiveCacheDir(config_);
    if (dir.empty()) return;

    DIR* d = opendir(dir.c_str());
    if (!d) return;

    size_t total_files = 0;
    size_t total_bytes = 0;
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name = entry->d_name;
        if (name.find("c3_") != 0) continue;
        std::string full = dir + "/" + name;
        if (name.size() < 4 || name.substr(name.size() - 3) != ".so") continue;
        struct stat st;
        if (stat(full.c_str(), &st) == 0) {
            total_files++;
            total_bytes += size_t(st.st_size);
        }
    }
    closedir(d);

    stats_.total_files = total_files;
    stats_.total_bytes = total_bytes;
}

} // namespace c3
} // namespace ct
