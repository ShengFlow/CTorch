/**
 * @file CheckpointManager.h
 * @brief 分布式检查点管理器 — 模型参数和优化器状态持久化
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details 本文件实现 CheckpointManager，负责分布式训练中的
 *          模型参数和优化器状态的检查点保存与恢复。
 *
 *          设计原则：
 *          1. 原子写入：使用 write-then-rename 确保检查点完整性
 *          2. 版本管理：保留多个版本，支持回滚
 *          3. 压缩存储：使用 CDTF 格式压缩参数和梯度
 *          4. 元数据管理：记录训练步数、loss、时间戳等
 *          5. 故障恢复：从最近的有效检查点恢复
 *
 *          本模块与所有其他模块解耦：
 *          CheckpointManager 只操作 Tensor 和 CRDTState，
 *          不依赖任何分布式组件。
 */

#ifndef CTORCH_DISTRIBUTED_CHECKPOINT_MANAGER_H
#define CTORCH_DISTRIBUTED_CHECKPOINT_MANAGER_H

#include "Ctools.h"
#include "CtorchError.h"
#include "Tensor.h"

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <cstring>

namespace ct {
namespace distributed {

/**
 * @brief 检查点格式枚举
 */
enum class CheckpointFormat : uint8_t {
    CDTF,        ///< CTorch Distributed Tensor Format
    Raw,         ///< 原始二进制格式
    Compressed,  ///< 使用 EntropyAwareCompressor 压缩
};

/**
 * @brief 检查点保存触发策略
 */
enum class CheckpointTrigger : uint8_t {
    StepInterval,     ///< 按步数间隔保存
    TimeInterval,     ///< 按时间间隔保存
    BestLoss,         ///< 达到最佳 loss 时保存
    Manual,           ///< 手动触发
};

/**
 * @struct CheckpointMetadata
 * @brief 检查点元数据
 */
struct CheckpointMetadata {
    uint64_t checkpoint_id;                  ///< 检查点 ID
    std::string version;                     ///< CTorch 版本
    std::chrono::system_clock::time_point timestamp; ///< 创建时间
    uint64_t global_step;                    ///< 全局训练步数
    float loss;                              ///< 当前 loss
    float best_loss;                         ///< 最佳 loss
    size_t num_params;                       ///< 参数数量
    size_t total_param_elements;             ///< 总参数元素数
    CheckpointFormat format;                 ///< 存储格式
    CheckpointTrigger trigger;               ///< 触发策略
    size_t compression_ratio;                ///< 压缩比 (压缩后/原始)
    std::unordered_map<std::string, std::string> tags; ///< 用户自定义标签
};

/**
 * @struct CheckpointEntry
 * @brief 检查点条目
 */
struct CheckpointEntry {
    uint64_t id;                             ///< 检查点 ID
    std::string path;                        ///< 文件路径
    CheckpointMetadata metadata;             ///< 元数据
    bool is_valid;                           ///< 是否有效
    size_t file_size_bytes;                  ///< 文件大小
};

/**
 * @struct CheckpointConfig
 * @brief 检查点配置
 */
struct CheckpointConfig {
    std::string checkpoint_dir;              ///< 检查点存储目录
    CheckpointFormat format;                 ///< 存储格式
    CheckpointTrigger trigger;               ///< 触发策略
    size_t save_interval_steps;              ///< 步数间隔
    float save_interval_seconds;             ///< 时间间隔 (秒)
    size_t max_checkpoints;                  ///< 最大保留数
    bool save_optimizer_state;               ///< 是否保存优化器状态
    bool save_on_best_loss;                  ///< 是否在最佳 loss 时保存
    bool enable_compression;                 ///< 是否启用压缩
    bool atomic_writes;                      ///< 是否使用原子写入

    static CheckpointConfig defaultConfig() {
        return CheckpointConfig{
            "./checkpoints",                // checkpoint_dir
            CheckpointFormat::CDTF,         // format
            CheckpointTrigger::StepInterval, // trigger
            1000,                           // save_interval_steps
            300.0f,                         // save_interval_seconds (5min)
            5,                              // max_checkpoints
            true,                           // save_optimizer_state
            true,                           // save_on_best_loss
            true,                           // enable_compression
            true,                           // atomic_writes
        };
    }
};

/**
 * @struct CheckpointStats
 * @brief 检查点统计信息
 */
struct CheckpointStats {
    size_t total_saves;                      ///< 总保存次数
    size_t total_loads;                      ///< 总加载次数
    size_t total_deleted;                    ///< 总删除次数
    size_t failed_saves;                     ///< 保存失败次数
    size_t failed_loads;                     ///< 加载失败次数
    double avg_save_time_ms;                 ///< 平均保存时间 (ms)
    double avg_load_time_ms;                 ///< 平均加载时间 (ms)
    double avg_file_size_bytes;              ///< 平均文件大小 (byte)
    size_t current_checkpoint_count;         ///< 当前检查点数量

    void reset() {
        total_saves = 0;
        total_loads = 0;
        total_deleted = 0;
        failed_saves = 0;
        failed_loads = 0;
        avg_save_time_ms = 0.0;
        avg_load_time_ms = 0.0;
        avg_file_size_bytes = 0.0;
        current_checkpoint_count = 0;
    }
};

/**
 * @class CheckpointManager
 * @brief 分布式检查点管理器
 *
 * 负责模型参数和优化器状态的检查点管理：
 * 1. 保存：将模型参数保存为 CDTF 格式文件
 * 2. 加载：从检查点文件恢复模型参数
 * 3. 版本管理：自动清理旧检查点
 * 4. 原子写入：保障数据完整性
 * 5. 元数据索引：快速查找检查点
 *
 * 文件结构：
 *   checkpoint_dir/
 *     ├── checkpoint_00001.ckpt   # 检查点文件
 *     ├── checkpoint_00002.ckpt
 *     ├── ...
 *     └── checkpoint_index.json   # 索引文件
 *
 * 检查点文件格式：
 *   [Header 64 bytes]  [Metadata]  [Param 1]  [Param 2]  ...  [CRDT State]
 *   其中每个 Param 使用 CDTF 格式序列化。
 */
class CheckpointManager {
public:
    /**
     * @brief 构造检查点管理器
     * @param config 检查点配置
     */
    explicit CheckpointManager(
        CheckpointConfig config = CheckpointConfig::defaultConfig());

    ~CheckpointManager() = default;

    // ======================= 保存 =======================

    /**
     * @brief 保存检查点
     * @param params 模型参数列表
     * @param global_step 当前全局步数
     * @param loss 当前 loss 值
     * @param best_loss 最佳 loss 值
     * @param trigger 触发策略
     * @param tags 用户自定义标签（可选）
     * @return 检查点 ID（失败返回 0）
     *
     * 使用原子写入：先写入临时文件，再 rename 到目标文件。
     * 如果启用了压缩，每个参数张量使用 CDTF 格式压缩。
     */
    uint64_t save(const std::vector<Tensor*>& params,
                   uint64_t global_step,
                   float loss,
                   float best_loss = std::numeric_limits<float>::max(),
                   CheckpointTrigger trigger = CheckpointTrigger::Manual,
                   const std::unordered_map<std::string, std::string>& tags = {});

    /**
     * @brief 保存检查点（含优化器 CRDT 状态）
     * @param params 模型参数列表
     * @param optimizer_state 优化器状态字节流
     * @param global_step 当前全局步数
     * @param loss 当前 loss 值
     * @param best_loss 最佳 loss 值
     * @param trigger 触发策略
     * @param tags 用户自定义标签
     * @return 检查点 ID
     */
    uint64_t saveWithOptimizerState(
        const std::vector<Tensor*>& params,
        const std::vector<uint8_t>& optimizer_state,
        uint64_t global_step,
        float loss,
        float best_loss = std::numeric_limits<float>::max(),
        CheckpointTrigger trigger = CheckpointTrigger::Manual,
        const std::unordered_map<std::string, std::string>& tags = {});

    // ======================= 加载 =======================

    /**
     * @brief 从检查点加载模型参数
     * @param params 模型参数列表（输出参数，将被填充）
     * @param checkpoint_id 检查点 ID（0 = 最新）
     * @return 加载的检查点元数据
     * @throws CtorchError 如果加载失败或格式无效
     */
    CheckpointMetadata load(std::vector<Tensor*>& params,
                             uint64_t checkpoint_id = 0);

    /**
     * @brief 加载检查点中的优化器状态
     * @param checkpoint_id 检查点 ID（0 = 最新）
     * @return 优化器状态字节流（如果未保存则返回空）
     */
    std::vector<uint8_t> loadOptimizerState(uint64_t checkpoint_id = 0);

    /**
     * @brief 加载检查点元数据（不加载参数数据）
     * @param checkpoint_id 检查点 ID（0 = 最新）
     * @return 检查点元数据
     * @throws CtorchError 如果检查点不存在
     */
    CheckpointMetadata loadMetadata(uint64_t checkpoint_id = 0) const;

    // ======================= 管理 =======================

    /**
     * @brief 删除检查点
     * @param checkpoint_id 检查点 ID
     * @return true 如果删除成功
     */
    bool remove(uint64_t checkpoint_id);

    /**
     * @brief 清理旧检查点，只保留最新的 max_checkpoints 个
     * @return 删除的检查点数量
     */
    size_t pruneOldCheckpoints();

    /**
     * @brief 获取所有检查点列表
     * @return 检查点条目列表，按 ID 降序
     */
    std::vector<CheckpointEntry> listCheckpoints() const;

    /**
     * @brief 获取最新的检查点 ID
     * @return 最新检查点 ID（如果没有则返回 0）
     */
    uint64_t latestCheckpointId() const;

    /**
     * @brief 检查是否需要保存（根据配置的触发策略）
     * @param current_step 当前步数
     * @param current_loss 当前 loss
     * @param best_loss 最佳 loss
     * @return true 如果需要保存
     */
    bool needsSave(uint64_t current_step, float current_loss,
                    float best_loss) const;

    // ======================= 配置 =======================

    /**
     * @brief 设置检查点配置
     * @param config 检查点配置
     */
    void setConfig(const CheckpointConfig& config) { _config = config; }

    /**
     * @brief 获取当前检查点配置
     * @return 检查点配置
     */
    const CheckpointConfig& config() const { return _config; }

    /**
     * @brief 设置压缩回调（用于压缩/解压张量数据）
     * @param compress_cb 压缩回调
     * @param decompress_cb 解压回调
     */
    void setCompressionCallbacks(
        std::function<std::vector<uint8_t>(const Tensor&)> compress_cb,
        std::function<Tensor(const std::vector<uint8_t>&)> decompress_cb) {
        _compress_cb = std::move(compress_cb);
        _decompress_cb = std::move(decompress_cb);
    }

    // ======================= 统计信息 =======================

    /**
     * @brief 获取检查点统计信息
     * @return 统计信息
     */
    CheckpointStats stats() const { return _stats; }

    /**
     * @brief 重置统计信息
     */
    void resetStats() { _stats.reset(); }

private:
    CheckpointConfig _config;
    mutable std::mutex _mtx;
    CheckpointStats _stats;

    // 检查点文件路径
    std::string _checkpoint_dir;

    // 压缩回调
    std::function<std::vector<uint8_t>(const Tensor&)> _compress_cb;
    std::function<Tensor(const std::vector<uint8_t>&)> _decompress_cb;

    // 最佳 loss 跟踪
    float _best_loss_seen;

    // 上次保存时间
    std::chrono::steady_clock::time_point _last_save_time;

    /**
     * @brief 确保检查点目录存在
     * @throws CtorchError 如果创建失败
     */
    void ensureDirectoryExists();

    /**
     * @brief 生成检查点文件名
     * @param checkpoint_id 检查点 ID
     * @return 文件路径
     */
    std::string checkpointPath(uint64_t checkpoint_id) const;

    /**
     * @brief 生成临时文件名
     * @param checkpoint_id 检查点 ID
     * @return 临时文件路径
     */
    std::string tempPath(uint64_t checkpoint_id) const;

    /**
     * @brief 序列化参数列表为字节流
     * @param params 参数列表
     * @return 序列化后的字节流
     */
    std::vector<uint8_t> serializeParams(
        const std::vector<Tensor*>& params) const;

    /**
     * @brief 从字节流反序列化参数
     * @param data 字节流
     * @param params 输出参数列表
     */
    void deserializeParams(const std::vector<uint8_t>& data,
                            std::vector<Tensor*>& params) const;

    /**
     * @brief 将参数张量序列化为 CDTF 字节流
     * @param tensor 张量
     * @return CDTF 字节流
     */
    std::vector<uint8_t> tensorToBytes(const Tensor& tensor) const;

    /**
     * @brief 从 CDTF 字节流反序列化为张量
     * @param data CDTF 字节流
     * @return 张量
     */
    Tensor tensorFromBytes(const std::vector<uint8_t>& data) const;

    /**
     * @brief 原子写入文件
     * @param path 目标路径
     * @param data 数据
     * @return true 如果写入成功
     *
     * 先写入 .tmp 文件，然后 rename 到目标路径。
     */
    bool atomicWrite(const std::string& path,
                      const std::vector<uint8_t>& data);

    /**
     * @brief 读取文件内容
     * @param path 文件路径
     * @return 文件内容
     * @throws CtorchError 如果读取失败
     */
    std::vector<uint8_t> readFile(const std::string& path) const;

    /**
     * @brief 扫描检查点目录，重建索引
     */
    void scanCheckpointDirectory();

    /**
     * @brief 清理旧检查点（不持有锁的版本，调用方需已锁定 _mtx）
     * @return 删除的检查点数量
     */
    size_t pruneOldCheckpointsUnlocked();
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_CHECKPOINT_MANAGER_H