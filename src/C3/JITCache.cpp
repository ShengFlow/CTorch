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
 *  - 复用 AOTCache 的 SHA-256 实现
 *
 * @date 2026/08/06
 */

#include "../../include/C3/JITCache.h"
#include "../../include/C3/AOTCache.h"

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

// ======================= Cache key =======================

std::string JITCache::makeKey(const std::string& graph_str, int opt_level) {
    std::ostringstream ss;
    ss << "c3_jit_" << currentJITVersion() << "_"
       << opt_level << "_"
       << graph_str;
    std::string key = AOTCache::sha256Hex(ss.str());
#ifdef CT_DEBUG
    fprintf(stderr, "[JITCache] makeKey: input_len=%zu, key=%s\n", ss.str().size(), key.c_str());
#endif
    return key;
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