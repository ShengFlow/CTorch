/**
 * @file CheckpointManager.cpp
 * @brief 分布式检查点管理器实现 — 模型参数和优化器状态持久化
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 */

#include "Distributed/CheckpointManager.h"
#include "Distributed/CDTF.h"

#include <sstream>
#include <iomanip>
#include <cstdio>
#include <cinttypes>

namespace ct {
namespace distributed {

// ======================= 内部常量 =======================

/// 检查点文件魔数 "CTCK"
static constexpr uint32_t kCheckpointMagic = 0x4354434B;
/// 检查点文件格式版本
static constexpr uint32_t kCheckpointVersion = 1;
/// 检查点文件头大小（固定 64 字节）
static constexpr size_t kCheckpointHeaderSize = 64;
/// 标志位：包含优化器状态
static constexpr uint32_t kCheckpointFlagHasOptimizerState = 0x00000001;

/// 临时文件后缀
static constexpr const char* kTempSuffix = ".tmp";

// ======================= 内部辅助函数 =======================

namespace {

/**
 * @brief 将 CheckpointMetadata 序列化为 JSON 字符串
 */
std::string metadataToJson(const CheckpointMetadata& meta) {
    std::ostringstream oss;
    auto ts = std::chrono::system_clock::to_time_t(meta.timestamp);
    oss << "{"
        << "\"checkpoint_id\":" << meta.checkpoint_id << ","
        << "\"version\":\"" << meta.version << "\","
        << "\"timestamp\":" << ts << ","
        << "\"global_step\":" << meta.global_step << ","
        << "\"loss\":" << meta.loss << ","
        << "\"best_loss\":" << meta.best_loss << ","
        << "\"num_params\":" << meta.num_params << ","
        << "\"total_param_elements\":" << meta.total_param_elements << ","
        << "\"format\":" << static_cast<int>(meta.format) << ","
        << "\"trigger\":" << static_cast<int>(meta.trigger) << ","
        << "\"compression_ratio\":" << meta.compression_ratio
        << "}";
    return oss.str();
}

/**
 * @brief 从 JSON 字符串反序列化 CheckpointMetadata
 */
CheckpointMetadata metadataFromJson(const std::string& json) {
    CheckpointMetadata meta{};
    meta.checkpoint_id = 0;
    meta.version = "unknown";
    meta.timestamp = std::chrono::system_clock::now();
    meta.global_step = 0;
    meta.loss = 0.0f;
    meta.best_loss = std::numeric_limits<float>::max();
    meta.num_params = 0;
    meta.total_param_elements = 0;
    meta.format = CheckpointFormat::CDTF;
    meta.trigger = CheckpointTrigger::Manual;
    meta.compression_ratio = 0;

    // 使用简单的字符串查找解析 JSON 字段
    auto extractInt = [&](const std::string& key) -> uint64_t {
        auto pos = json.find("\"" + key + "\":");
        if (pos == std::string::npos) {
            // 尝试无引号 key（数字 field）
            pos = json.find(key + ":");
            if (pos == std::string::npos) return 0;
        }
        pos = json.find(':', pos) + 1;
        // 跳过空白
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
        // 读数字直到遇到 , 或 }
        char* end = nullptr;
        uint64_t val = std::strtoull(json.c_str() + pos, &end, 10);
        return val;
    };

    auto extractFloat = [&](const std::string& key) -> float {
        auto pos = json.find("\"" + key + "\":");
        if (pos == std::string::npos) {
            pos = json.find(key + ":");
            if (pos == std::string::npos) return 0.0f;
        }
        pos = json.find(':', pos) + 1;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
        char* end = nullptr;
        float val = std::strtof(json.c_str() + pos, &end);
        return val;
    };

    auto extractString = [&](const std::string& key) -> std::string {
        auto keyStr = "\"" + key + "\":\"";
        auto pos = json.find(keyStr);
        if (pos == std::string::npos) return "";
        pos += keyStr.size();
        auto end = json.find('\"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    };

    meta.checkpoint_id = extractInt("checkpoint_id");
    meta.version = extractString("version");

    uint64_t timestamp_sec = extractInt("timestamp");
    if (timestamp_sec > 0) {
        meta.timestamp = std::chrono::system_clock::from_time_t(
            static_cast<std::time_t>(timestamp_sec));
    }

    meta.global_step = extractInt("global_step");
    meta.loss = extractFloat("loss");
    meta.best_loss = extractFloat("best_loss");
    meta.num_params = static_cast<size_t>(extractInt("num_params"));
    meta.total_param_elements = static_cast<size_t>(extractInt("total_param_elements"));
    meta.format = static_cast<CheckpointFormat>(extractInt("format"));
    meta.trigger = static_cast<CheckpointTrigger>(extractInt("trigger"));
    meta.compression_ratio = static_cast<size_t>(extractInt("compression_ratio"));

    return meta;
}

} // anonymous namespace

// ======================= 构造 / 析构 =======================

CheckpointManager::CheckpointManager(CheckpointConfig config)
    : _config(std::move(config))
    , _best_loss_seen(std::numeric_limits<float>::max())
    , _last_save_time(std::chrono::steady_clock::now())
{
    _checkpoint_dir = _config.checkpoint_dir;
    ensureDirectoryExists();
    scanCheckpointDirectory();
}

// ======================= 保存 =======================

uint64_t CheckpointManager::save(
    const std::vector<Tensor*>& params,
    uint64_t global_step,
    float loss,
    float best_loss,
    CheckpointTrigger trigger,
    const std::unordered_map<std::string, std::string>& tags)
{
    std::lock_guard<std::mutex> lock(_mtx);

    auto start_time = std::chrono::steady_clock::now();

    // 生成检查点 ID
    uint64_t checkpoint_id = latestCheckpointId() + 1;
    if (checkpoint_id == 0) checkpoint_id = 1; // 溢出保护

    // 构建元数据
    CheckpointMetadata metadata;
    metadata.checkpoint_id = checkpoint_id;
    metadata.version = "1.0.0";
    metadata.timestamp = std::chrono::system_clock::now();
    metadata.global_step = global_step;
    metadata.loss = loss;
    metadata.best_loss = best_loss;
    metadata.num_params = params.size();
    metadata.total_param_elements = 0;
    metadata.format = _config.format;
    metadata.trigger = trigger;
    metadata.compression_ratio = 0;
    metadata.tags = tags;

    // 计算总参数量
    for (const auto* p : params) {
        if (p) {
            metadata.total_param_elements += p->numel();
        }
    }

    // 序列化参数字节流
    std::vector<uint8_t> param_bytes = serializeParams(params);

    // 序列化元数据为 JSON
    std::string metadata_json = metadataToJson(metadata);
    uint32_t metadata_size = static_cast<uint32_t>(metadata_json.size());

    // 构建文件
    // [64-byte header][metadata_size:4][metadata_json][num_params:4][param1_cdtf][param2_cdtf]...
    std::vector<uint8_t> file_data;
    file_data.reserve(kCheckpointHeaderSize + 4 + metadata_json.size() + 4 + param_bytes.size());

    // --- Header (64 bytes) ---
    file_data.resize(kCheckpointHeaderSize, 0);
    uint32_t magic = kCheckpointMagic;
    uint32_t version = kCheckpointVersion;
    uint32_t flags = 0;
    uint32_t num_params = static_cast<uint32_t>(params.size());

    std::memcpy(file_data.data(), &magic, 4);
    std::memcpy(file_data.data() + 4, &version, 4);
    std::memcpy(file_data.data() + 8, &flags, 4);
    std::memcpy(file_data.data() + 12, &metadata_size, 4);
    std::memcpy(file_data.data() + 16, &num_params, 4);
    // 剩余 44 字节保留（置零）

    // --- Metadata ---
    file_data.insert(file_data.end(), reinterpret_cast<const uint8_t*>(metadata_json.data()),
                     reinterpret_cast<const uint8_t*>(metadata_json.data()) + metadata_json.size());

    // --- Num params (4 bytes) ---
    file_data.insert(file_data.end(),
                     reinterpret_cast<const uint8_t*>(&num_params),
                     reinterpret_cast<const uint8_t*>(&num_params) + 4);

    // --- Param data ---
    file_data.insert(file_data.end(), param_bytes.begin(), param_bytes.end());

    // 原子写入
    std::string path = checkpointPath(checkpoint_id);
    if (!atomicWrite(path, file_data)) {
        _stats.failed_saves++;
        return 0;
    }

    // 更新最佳 loss
    if (loss < _best_loss_seen) {
        _best_loss_seen = loss;
    }

    // 更新统计信息
    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    _stats.total_saves++;
    _stats.current_checkpoint_count++;
    _stats.avg_save_time_ms = (_stats.avg_save_time_ms * (_stats.total_saves - 1) + elapsed_ms)
                              / static_cast<double>(_stats.total_saves);
    _stats.avg_file_size_bytes = (_stats.avg_file_size_bytes * (_stats.total_saves - 1)
                                  + static_cast<double>(file_data.size()))
                                 / static_cast<double>(_stats.total_saves);

    _last_save_time = std::chrono::steady_clock::now();

    // 保存后检查是否需要清理旧检查点（已持有锁，使用 unlocked 版本）
    if (_stats.current_checkpoint_count > _config.max_checkpoints) {
        pruneOldCheckpointsUnlocked();
    }

    return checkpoint_id;
}

uint64_t CheckpointManager::saveWithOptimizerState(
    const std::vector<Tensor*>& params,
    const std::vector<uint8_t>& optimizer_state,
    uint64_t global_step,
    float loss,
    float best_loss,
    CheckpointTrigger trigger,
    const std::unordered_map<std::string, std::string>& tags)
{
    std::lock_guard<std::mutex> lock(_mtx);

    auto start_time = std::chrono::steady_clock::now();

    // 生成检查点 ID
    uint64_t checkpoint_id = latestCheckpointId() + 1;
    if (checkpoint_id == 0) checkpoint_id = 1;

    // 构建元数据
    CheckpointMetadata metadata;
    metadata.checkpoint_id = checkpoint_id;
    metadata.version = "1.0.0";
    metadata.timestamp = std::chrono::system_clock::now();
    metadata.global_step = global_step;
    metadata.loss = loss;
    metadata.best_loss = best_loss;
    metadata.num_params = params.size();
    metadata.total_param_elements = 0;
    metadata.format = _config.format;
    metadata.trigger = trigger;
    metadata.compression_ratio = 0;
    metadata.tags = tags;

    for (const auto* p : params) {
        if (p) {
            metadata.total_param_elements += p->numel();
        }
    }

    // 序列化参数
    std::vector<uint8_t> param_bytes = serializeParams(params);

    // 序列化元数据
    std::string metadata_json = metadataToJson(metadata);
    uint32_t metadata_size = static_cast<uint32_t>(metadata_json.size());

    // 构建文件
    // [64-byte header][metadata_size:4][metadata_json][num_params:4][param_cdtf...][opt_state_size:4][opt_state...]
    std::vector<uint8_t> file_data;
    size_t total_size = kCheckpointHeaderSize + 4 + metadata_json.size() + 4
                        + param_bytes.size() + 4 + optimizer_state.size();
    file_data.reserve(total_size);

    // --- Header (64 bytes) ---
    file_data.resize(kCheckpointHeaderSize, 0);
    uint32_t magic = kCheckpointMagic;
    uint32_t version = kCheckpointVersion;
    uint32_t flags = kCheckpointFlagHasOptimizerState;
    uint32_t num_params = static_cast<uint32_t>(params.size());

    std::memcpy(file_data.data(), &magic, 4);
    std::memcpy(file_data.data() + 4, &version, 4);
    std::memcpy(file_data.data() + 8, &flags, 4);
    std::memcpy(file_data.data() + 12, &metadata_size, 4);
    std::memcpy(file_data.data() + 16, &num_params, 4);
    // 剩余 44 字节保留

    // --- Metadata ---
    file_data.insert(file_data.end(),
                     reinterpret_cast<const uint8_t*>(metadata_json.data()),
                     reinterpret_cast<const uint8_t*>(metadata_json.data()) + metadata_json.size());

    // --- Num params ---
    file_data.insert(file_data.end(),
                     reinterpret_cast<const uint8_t*>(&num_params),
                     reinterpret_cast<const uint8_t*>(&num_params) + 4);

    // --- Param data ---
    file_data.insert(file_data.end(), param_bytes.begin(), param_bytes.end());

    // --- Optimizer state ---
    uint32_t opt_state_size = static_cast<uint32_t>(optimizer_state.size());
    file_data.insert(file_data.end(),
                     reinterpret_cast<const uint8_t*>(&opt_state_size),
                     reinterpret_cast<const uint8_t*>(&opt_state_size) + 4);
    file_data.insert(file_data.end(), optimizer_state.begin(), optimizer_state.end());

    // 原子写入
    std::string path = checkpointPath(checkpoint_id);
    if (!atomicWrite(path, file_data)) {
        _stats.failed_saves++;
        return 0;
    }

    if (loss < _best_loss_seen) {
        _best_loss_seen = loss;
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    _stats.total_saves++;
    _stats.current_checkpoint_count++;
    _stats.avg_save_time_ms = (_stats.avg_save_time_ms * (_stats.total_saves - 1) + elapsed_ms)
                              / static_cast<double>(_stats.total_saves);
    _stats.avg_file_size_bytes = (_stats.avg_file_size_bytes * (_stats.total_saves - 1)
                                  + static_cast<double>(file_data.size()))
                                 / static_cast<double>(_stats.total_saves);

    _last_save_time = std::chrono::steady_clock::now();

    if (_stats.current_checkpoint_count > _config.max_checkpoints) {
        pruneOldCheckpointsUnlocked();
    }

    return checkpoint_id;
}

// ======================= 加载 =======================

CheckpointMetadata CheckpointManager::load(
    std::vector<Tensor*>& params,
    uint64_t checkpoint_id)
{
    std::lock_guard<std::mutex> lock(_mtx);

    auto start_time = std::chrono::steady_clock::now();

    // 查找检查点路径
    if (checkpoint_id == 0) {
        checkpoint_id = latestCheckpointId();
    }
    if (checkpoint_id == 0) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "没有可用的检查点");
    }

    std::string path = checkpointPath(checkpoint_id);
    if (!std::filesystem::exists(path)) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "检查点文件不存在: " + path);
    }

    // 读取文件
    std::vector<uint8_t> file_data = readFile(path);
    if (file_data.size() < kCheckpointHeaderSize) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "检查点文件格式无效（文件过小）: " + path);
    }

    // 解析 Header
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t flags = 0;
    uint32_t metadata_size = 0;
    uint32_t num_params_file = 0;

    std::memcpy(&magic, file_data.data(), 4);
    std::memcpy(&version, file_data.data() + 4, 4);
    std::memcpy(&flags, file_data.data() + 8, 4);
    std::memcpy(&metadata_size, file_data.data() + 12, 4);
    std::memcpy(&num_params_file, file_data.data() + 16, 4);

    if (magic != kCheckpointMagic) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "检查点文件魔数不匹配");
    }

    // 解析 Metadata
    size_t offset = kCheckpointHeaderSize;
    if (offset + metadata_size > file_data.size()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "检查点文件元数据截断");
    }

    std::string metadata_json(reinterpret_cast<const char*>(file_data.data() + offset), metadata_size);
    offset += metadata_size;

    CheckpointMetadata meta = metadataFromJson(metadata_json);

    // 解析参数
    if (offset + 4 > file_data.size()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "检查点文件参数计数截断");
    }

    // 读取 num_params（冗余，与 header 中的一致）
    uint32_t num_params_read = 0;
    std::memcpy(&num_params_read, file_data.data() + offset, 4);
    offset += 4;

    // 提取参数 CDTF 数据块
    std::vector<uint8_t> param_data(file_data.begin() + static_cast<ptrdiff_t>(offset), file_data.end());

    // 如果存在优化器状态，需要减去末尾的 opt_state_size:4 + opt_state 部分
    if (flags & kCheckpointFlagHasOptimizerState) {
        // 找到最后一个参数的结束位置
        // 我们可以通过尝试解析 CDTF 数据来找到边界，但更简单的方法：
        // 如果有优化器状态，最后 4 字节是 opt_state_size
        if (param_data.size() >= 4) {
            uint32_t opt_state_size = 0;
            std::memcpy(&opt_state_size, param_data.data() + param_data.size() - 4, 4);
            // 检查 opt_state_size 是否合理
            if (opt_state_size < param_data.size()) {
                // 截断优化器状态部分
                size_t actual_param_size = param_data.size() - 4 - opt_state_size;
                if (actual_param_size <= param_data.size()) {
                    param_data.resize(actual_param_size);
                }
            }
        }
    }

    // 反序列化参数
    params.clear();
    deserializeParams(param_data, params);

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    _stats.total_loads++;
    _stats.avg_load_time_ms = (_stats.avg_load_time_ms * (_stats.total_loads - 1) + elapsed_ms)
                              / static_cast<double>(_stats.total_loads);

    return meta;
}

std::vector<uint8_t> CheckpointManager::loadOptimizerState(uint64_t checkpoint_id) {
    std::lock_guard<std::mutex> lock(_mtx);

    if (checkpoint_id == 0) {
        checkpoint_id = latestCheckpointId();
    }
    if (checkpoint_id == 0) {
        return {};
    }

    std::string path = checkpointPath(checkpoint_id);
    if (!std::filesystem::exists(path)) {
        return {};
    }

    std::vector<uint8_t> file_data = readFile(path);
    if (file_data.size() < kCheckpointHeaderSize) {
        return {};
    }

    // 解析 Header
    uint32_t magic = 0;
    uint32_t flags = 0;
    std::memcpy(&magic, file_data.data(), 4);
    std::memcpy(&flags, file_data.data() + 8, 4);

    if (magic != kCheckpointMagic) {
        return {};
    }

    // 检查是否有优化器状态
    if (!(flags & kCheckpointFlagHasOptimizerState)) {
        return {};
    }

    // 解析 metadata_size 确定参数数据起始位置
    uint32_t metadata_size = 0;
    uint32_t num_params_file = 0;
    std::memcpy(&metadata_size, file_data.data() + 12, 4);
    std::memcpy(&num_params_file, file_data.data() + 16, 4);

    // 跳过 header + metadata + num_params 字段
    size_t offset = kCheckpointHeaderSize + metadata_size + 4;

    // 解析所有参数 CDTF 块，找到参数数据的末尾
    // 需要遍历所有参数
    for (uint32_t i = 0; i < num_params_file; ++i) {
        if (offset + 4 > file_data.size()) {
            return {};
        }
        uint32_t cdtf_size = 0;
        std::memcpy(&cdtf_size, file_data.data() + offset, 4);
        offset += 4 + cdtf_size;
    }

    // 此时 offset 指向 opt_state_size
    if (offset + 4 > file_data.size()) {
        return {};
    }

    uint32_t opt_state_size = 0;
    std::memcpy(&opt_state_size, file_data.data() + offset, 4);
    offset += 4;

    if (offset + opt_state_size > file_data.size()) {
        return {};
    }

    std::vector<uint8_t> opt_state(file_data.begin() + static_cast<ptrdiff_t>(offset),
                                   file_data.begin() + static_cast<ptrdiff_t>(offset) + opt_state_size);
    return opt_state;
}

CheckpointMetadata CheckpointManager::loadMetadata(uint64_t checkpoint_id) const {
    std::lock_guard<std::mutex> lock(_mtx);

    if (checkpoint_id == 0) {
        checkpoint_id = latestCheckpointId();
    }
    if (checkpoint_id == 0) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "没有可用的检查点");
    }

    std::string path = checkpointPath(checkpoint_id);
    if (!std::filesystem::exists(path)) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "检查点文件不存在: " + path);
    }

    std::vector<uint8_t> file_data = readFile(path);
    if (file_data.size() < kCheckpointHeaderSize) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "检查点文件格式无效（文件过小）: " + path);
    }

    uint32_t magic = 0;
    uint32_t metadata_size = 0;
    std::memcpy(&magic, file_data.data(), 4);
    std::memcpy(&metadata_size, file_data.data() + 12, 4);

    if (magic != kCheckpointMagic) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "检查点文件魔数不匹配");
    }

    if (kCheckpointHeaderSize + metadata_size > file_data.size()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "检查点文件元数据截断");
    }

    std::string metadata_json(
        reinterpret_cast<const char*>(file_data.data() + kCheckpointHeaderSize),
        metadata_size);

    return metadataFromJson(metadata_json);
}

// ======================= 管理 =======================

bool CheckpointManager::remove(uint64_t checkpoint_id) {
    std::lock_guard<std::mutex> lock(_mtx);

    std::string path = checkpointPath(checkpoint_id);
    if (!std::filesystem::exists(path)) {
        return false;
    }

    std::error_code ec;
    bool success = std::filesystem::remove(path, ec);
    if (success) {
        _stats.total_deleted++;
        if (_stats.current_checkpoint_count > 0) {
            _stats.current_checkpoint_count--;
        }
    }
    return success;
}

size_t CheckpointManager::pruneOldCheckpoints() {
    std::lock_guard<std::mutex> lock(_mtx);
    return pruneOldCheckpointsUnlocked();
}

size_t CheckpointManager::pruneOldCheckpointsUnlocked() {
    // 调用方必须已持有 _mtx 锁
    std::vector<std::pair<uint64_t, std::string>> ckpt_list;
    if (std::filesystem::exists(_checkpoint_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(_checkpoint_dir)) {
            if (!entry.is_regular_file()) continue;
            auto p = entry.path();
            if (p.extension() != ".ckpt") continue;
            std::string fname = p.filename().string();
            if (fname.size() > 12 && fname.substr(0, 11) == "checkpoint_") {
                try {
                    std::string num_str = fname.substr(11, fname.size() - 16);
                    uint64_t id = std::stoull(num_str);
                    ckpt_list.emplace_back(id, p.string());
                } catch (...) {
                    continue;
                }
            }
        }
    }

    if (ckpt_list.size() <= _config.max_checkpoints) {
        return 0;
    }

    // 按 ID 升序排列（最旧的在前）
    std::sort(ckpt_list.begin(), ckpt_list.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    size_t to_remove = ckpt_list.size() - _config.max_checkpoints;
    size_t removed = 0;

    for (size_t i = 0; i < to_remove; ++i) {
        std::error_code ec;
        if (std::filesystem::remove(ckpt_list[i].second, ec)) {
            removed++;
            _stats.total_deleted++;
            if (_stats.current_checkpoint_count > 0) {
                _stats.current_checkpoint_count--;
            }
        }
    }

    return removed;
}

std::vector<CheckpointEntry> CheckpointManager::listCheckpoints() const {
    std::lock_guard<std::mutex> lock(_mtx);

    std::vector<CheckpointEntry> entries;

    if (!std::filesystem::exists(_checkpoint_dir)) {
        return entries;
    }

    for (const auto& entry : std::filesystem::directory_iterator(_checkpoint_dir)) {
        if (!entry.is_regular_file()) continue;
        auto path = entry.path();
        if (path.extension() != ".ckpt") continue;

        // 从文件名解析检查点 ID
        std::string filename = path.filename().string();
        // 格式: checkpoint_XXXXX.ckpt
        uint64_t id = 0;
        if (filename.size() > 12 && filename.substr(0, 11) == "checkpoint_") {
            std::string num_str = filename.substr(11, filename.size() - 16); // 去掉 .ckpt
            try {
                id = std::stoull(num_str);
            } catch (...) {
                continue;
            }
        }

        CheckpointEntry cp_entry;
        cp_entry.id = id;
        cp_entry.path = path.string();
        cp_entry.is_valid = false;
        cp_entry.file_size_bytes = static_cast<size_t>(entry.file_size());

        // 尝试解析元数据（内联实现，避免与 loadMetadata 的锁冲突）
        try {
            std::vector<uint8_t> file_data = readFile(path.string());
            if (file_data.size() >= kCheckpointHeaderSize) {
                uint32_t magic = 0;
                uint32_t meta_size = 0;
                std::memcpy(&magic, file_data.data(), 4);
                std::memcpy(&meta_size, file_data.data() + 12, 4);
                if (magic == kCheckpointMagic &&
                    kCheckpointHeaderSize + meta_size <= file_data.size()) {
                    std::string mjson(
                        reinterpret_cast<const char*>(file_data.data() + kCheckpointHeaderSize),
                        meta_size);
                    cp_entry.metadata = metadataFromJson(mjson);
                    cp_entry.is_valid = true;
                }
            }
        } catch (...) {
            cp_entry.is_valid = false;
        }

        entries.push_back(cp_entry);
    }

    // 按 ID 降序排序
    std::sort(entries.begin(), entries.end(),
        [](const CheckpointEntry& a, const CheckpointEntry& b) {
            return a.id > b.id;
        });

    return entries;
}

uint64_t CheckpointManager::latestCheckpointId() const {
    uint64_t max_id = 0;
    if (!std::filesystem::exists(_checkpoint_dir)) {
        return 0;
    }

    for (const auto& entry : std::filesystem::directory_iterator(_checkpoint_dir)) {
        if (!entry.is_regular_file()) continue;
        auto path = entry.path();
        if (path.extension() != ".ckpt") continue;

        std::string filename = path.filename().string();
        if (filename.size() > 12 && filename.substr(0, 11) == "checkpoint_") {
            std::string num_str = filename.substr(11, filename.size() - 16);
            try {
                uint64_t id = std::stoull(num_str);
                if (id > max_id) max_id = id;
            } catch (...) {
                continue;
            }
        }
    }

    return max_id;
}

bool CheckpointManager::needsSave(uint64_t current_step, float current_loss,
                                   float best_loss) const {
    std::lock_guard<std::mutex> lock(_mtx);

    switch (_config.trigger) {
        case CheckpointTrigger::StepInterval:
            if (_config.save_interval_steps == 0) return false;
            return (current_step % _config.save_interval_steps == 0) && current_step > 0;

        case CheckpointTrigger::TimeInterval: {
            if (_config.save_interval_seconds <= 0.0f) return false;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<float>(now - _last_save_time).count();
            return elapsed >= _config.save_interval_seconds;
        }

        case CheckpointTrigger::BestLoss:
            if (!_config.save_on_best_loss) return false;
            return current_loss < best_loss;

        case CheckpointTrigger::Manual:
            return false;

        default:
            return false;
    }
}

// ======================= 私有辅助方法 =======================

void CheckpointManager::ensureDirectoryExists() {
    if (_checkpoint_dir.empty()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "检查点目录路径为空");
    }

    std::error_code ec;
    if (!std::filesystem::exists(_checkpoint_dir)) {
        if (!std::filesystem::create_directories(_checkpoint_dir, ec)) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                         "无法创建检查点目录: " + _checkpoint_dir
                                         + " (错误: " + ec.message() + ")");
        }
    }
}

std::string CheckpointManager::checkpointPath(uint64_t checkpoint_id) const {
    char buf[256];
    int n = std::snprintf(buf, sizeof(buf), "%s/checkpoint_%05" PRIu64 ".ckpt",
                          _checkpoint_dir.c_str(), checkpoint_id);
    if (n < 0 || static_cast<size_t>(n) >= sizeof(buf)) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "检查点路径生成失败");
    }
    return std::string(buf);
}

std::string CheckpointManager::tempPath(uint64_t checkpoint_id) const {
    return checkpointPath(checkpoint_id) + kTempSuffix;
}

std::vector<uint8_t> CheckpointManager::serializeParams(
    const std::vector<Tensor*>& params) const
{
    std::vector<uint8_t> result;
    for (const auto* p : params) {
        if (!p) continue;
        std::vector<uint8_t> cdtf_bytes = tensorToBytes(*p);
        uint32_t cdtf_size = static_cast<uint32_t>(cdtf_bytes.size());

        // [4 bytes: cdtf_size][cdtf_size bytes: CDTF data]
        result.insert(result.end(),
                      reinterpret_cast<const uint8_t*>(&cdtf_size),
                      reinterpret_cast<const uint8_t*>(&cdtf_size) + 4);
        result.insert(result.end(), cdtf_bytes.begin(), cdtf_bytes.end());
    }
    return result;
}

void CheckpointManager::deserializeParams(
    const std::vector<uint8_t>& data,
    std::vector<Tensor*>& params) const
{
    size_t offset = 0;
    while (offset + 4 <= data.size()) {
        uint32_t cdtf_size = 0;
        std::memcpy(&cdtf_size, data.data() + offset, 4);
        offset += 4;

        if (offset + cdtf_size > data.size()) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                         "检查点参数数据截断");
        }

        std::vector<uint8_t> cdtf_data(data.begin() + static_cast<ptrdiff_t>(offset),
                                       data.begin() + static_cast<ptrdiff_t>(offset) + cdtf_size);
        offset += cdtf_size;

        Tensor* t = new Tensor(tensorFromBytes(cdtf_data));
        params.push_back(t);
    }
}

std::vector<uint8_t> CheckpointManager::tensorToBytes(const Tensor& tensor) const {
    // 如果有压缩回调，使用压缩回调
    if (_compress_cb) {
        return _compress_cb(tensor);
    }
    // 默认使用 CDTF 序列化
    return CDTF::serialize(tensor);
}

Tensor CheckpointManager::tensorFromBytes(const std::vector<uint8_t>& data) const {
    // 如果有解压回调，使用解压回调
    if (_decompress_cb) {
        return _decompress_cb(data);
    }
    // 默认使用 CDTF 反序列化
    return CDTF::deserialize(data);
}

bool CheckpointManager::atomicWrite(const std::string& path,
                                     const std::vector<uint8_t>& data) {
    if (!_config.atomic_writes) {
        // 非原子模式：直接写入
        std::string tmp = path;
        std::ofstream ofs(tmp, std::ios::binary);
        if (!ofs) return false;
        ofs.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size()));
        return ofs.good();
    }

    // 原子模式：先写入 .tmp 文件，再 rename
    std::string tmp_path = path + kTempSuffix;
    {
        std::ofstream ofs(tmp_path, std::ios::binary);
        if (!ofs) return false;
        ofs.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size()));
        if (!ofs.good()) {
            ofs.close();
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
    }

    // rename .tmp -> target
    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    return !ec;
}

std::vector<uint8_t> CheckpointManager::readFile(const std::string& path) const {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "无法打开文件: " + path);
    }

    std::streamsize size = ifs.tellg();
    if (size < 0) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "无法获取文件大小: " + path);
    }

    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (!ifs.read(reinterpret_cast<char*>(buffer.data()), size)) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                                     "文件读取失败: " + path);
    }

    return buffer;
}

void CheckpointManager::scanCheckpointDirectory() {
    _stats.current_checkpoint_count = 0;

    if (!std::filesystem::exists(_checkpoint_dir)) {
        return;
    }

    for (const auto& entry : std::filesystem::directory_iterator(_checkpoint_dir)) {
        if (!entry.is_regular_file()) continue;
        auto path = entry.path();
        if (path.extension() == ".ckpt") {
            _stats.current_checkpoint_count++;
        }
    }
}

} // namespace distributed
} // namespace ct