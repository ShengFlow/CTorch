/**
 * @file Ctools.cpp
 * @brief Ctorch 通用工具类实现
 * @author GhostFace
 * @date 2025/12/21
 */

#include "Ctools.h"

// ======================= 辅助函数实现 =======================

/**
 * @brief 将DeviceType枚举转换为ErrorPlatform枚举
 * @param device_type 设备类型枚举值
 * @return 对应的ErrorPlatform枚举值
 */
ErrorPlatform DeviceTypeToErrorPlatform(const DeviceType device_type) {
    return static_cast<ErrorPlatform>(static_cast<uint32_t>(device_type));
}

/**
 * @brief 计算错误码
 * @param level 错误级别
 * @param platform 错误平台
 * @param type 错误类型
 * @return 计算得到的32位错误码
 */
uint32_t computeCode(ErrorLevel level, ErrorPlatform platform, ErrorType type) {
    uint32_t code = 0;
    code |= (static_cast<uint32_t>(platform) << 24); // 平台：高8位
    code |= (static_cast<uint32_t>(type) << 16);    // 类型：次8位
    code |= (static_cast<uint32_t>(level) << 8);    // 级别：次8位
    // 最后8位保留（可用于扩展具体场景码）
    return code;
}

/**
* @brief 返回人类可读的当前时间，精确到毫秒（跨平台、线程安全）
* @return 格式化字符串，示例："2026-02-17 00:00:00.123"
*/
std::string getFormattedTimeMs() {
    using namespace std::chrono;

    // 获取当前时间（精度：ms）
    const auto now = system_clock::now();
    const auto now_ms = time_point_cast<milliseconds>(now);
    const uint64_t ms = now_ms.time_since_epoch().count() % 1000; // 提取毫秒位（0-999）

    // 转换为秒级time_t，用于格式化日期时间
    const std::time_t now_t = system_clock::to_time_t(now);

    // 线程安全的时间解析（避免localtime线程不安全问题）
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &now_t); // Windows线程安全版本
#else
    localtime_r(&now_t, &tm_buf); // Linux/macOS线程安全版本
#endif

    // 4. 格式化字符串（补零到3位毫秒）
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
        << "." << std::setw(3) << std::setfill('0') << ms;

    return oss.str();
}

/**
* @brief 返回当前时间的毫秒级时间戳（跨平台）
* @return uint64_t类型的时间戳，示例：1749984000123
*/
uint64_t getTimestampMs() {
    using namespace std::chrono;
    // system_clock：系统墙钟时间（适配日志/统计场景）
    // steady_clock：单调时钟（适合耗时统计，替换此处即可）
    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()
    ).count();
}
