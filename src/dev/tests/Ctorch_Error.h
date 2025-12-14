//
// Created by GhostFace on 2025/12/13.
//

#ifndef CTORCH_ERROR_H
#define CTORCH_ERROR_H
#include <cstring>
#include <chrono>   // 核心时间库（C++11+）
#include <cstdint>  // uint64_t（固定宽度整数，存储时间戳）
#include <ctime>    // 时间格式化
#include <string>   // 格式化时间字符串
#include <sstream>  // 字符串流（拼接格式化时间）
#include <iomanip>  // std::put_time、std::setw（格式化工具）
#include <stdint.h>  // uint32_t 类型定义
#include <inttypes.h>// PRIu32 等格式化宏

enum class ErrorLevel {
    TRACE = 0,   // 细粒度调试（如kernel启动参数）
    DEBUG = 1,   // 开发调试（如数据拷贝状态）
    INFO = 2,    // 运行信息（如平台检测结果）
    WARN = 3,    // 非致命问题（如CUDA版本兼容提示）
    ERROR = 4,   // 功能异常（如kernel执行失败）
    FATAL = 5    // 崩溃级错误（如内存分配失败）
};

// Error发生的设备
enum class ErrorPlatform {
    kCPU = 0,
    kCUDA = 1,
    kMPS = 2,
    kAMX = 3,
    kUNKNOWN = 4,
};

// 错误类型
enum class ErrorType {
    UNKNOWN = 0,          // 未知错误
    MEMORY = 1,           // 内存相关（分配/释放/拷贝失败）
    DIMENSION = 2,        // 维度相关（不匹配/越界/空维度）
    DEVICE_COMPAT = 3,    // 设备兼容（跨设备运算/设备不支持/设备初始化失败）
    DATATYPE = 4,         // 数据类型（不匹配/转换失败/不支持的类型）
    KERNEL_LAUNCH = 5,    // 内核调用（CUDA/MPS/AMX内核启动失败/执行超时）
    TENSOR_STATE = 6,     // Tensor状态（未初始化/只读/已释放）
    PLATFORM_API = 7      // 平台API调用失败（如cudaGetDevice/MPSGraph创建失败）
};

class Ctorch_Error {
    // 辅助：将枚举转为字符串名称（便于日志可读性）
    static std::string getLevelName(ErrorLevel level) {
        switch (level) {
            case ErrorLevel::TRACE: return "TRACE";
            case ErrorLevel::DEBUG: return "DEBUG";
            case ErrorLevel::INFO: return "INFO";
            case ErrorLevel::WARN: return "WARN";
            case ErrorLevel::ERROR: return "ERROR";
            case ErrorLevel::FATAL: return "FATAL";
            default: return "UNKNOWN";
        }
    }
    static std::string getPlatformName(ErrorPlatform platform) {
        switch (platform) {
            case ErrorPlatform::kUNKNOWN: return "UNKNOWN";
            case ErrorPlatform::kCPU: return "CPU";
            case ErrorPlatform::kCUDA: return "CUDA";
            case ErrorPlatform::kMPS: return "MPS";
            case ErrorPlatform::kAMX: return "AMX";
            default: return "UNKNOWN";
        }
    }
    static std::string getTypeName(ErrorType type) {
        switch (type) {
            case ErrorType::UNKNOWN:      return "UNKNOWN";
            case ErrorType::MEMORY:       return "MEMORY";
            case ErrorType::DIMENSION:    return "DIMENSION";
            case ErrorType::DEVICE_COMPAT:return "DEVICE_COMPAT";
            case ErrorType::DATATYPE:     return "DATATYPE";
            case ErrorType::KERNEL_LAUNCH:return "KERNEL_LAUNCH";
            case ErrorType::TENSOR_STATE: return "TENSOR_STATE";
            case ErrorType::PLATFORM_API: return "PLATFORM_API";
            default: return "UNKNOWN";
        }
    }

    static uint8_t getLevelCode(ErrorLevel level) {
        return static_cast<uint8_t>(level);
    }
    static uint8_t getPlatformCode(ErrorPlatform platform) {
        return static_cast<uint8_t>(platform);
    }
    static uint8_t getTypeCode(ErrorType type) {
        return static_cast<uint8_t>(type);
    }

    static uint32_t computeCode(ErrorLevel level,ErrorPlatform platform,ErrorType type) {
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
    static std::string getFormattedTimeMs() {
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
    static uint64_t getTimestampMs() {
        using namespace std::chrono;
        // system_clock：系统墙钟时间（适配日志/统计场景）
        // steady_clock：单调时钟（适合耗时统计，替换此处即可）
        return duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()
        ).count();
    }
public: static void log(ErrorLevel level,ErrorPlatform platform,ErrorType type,std::string msg) {
        uint32_t error_code = computeCode(level,platform,type);
        printf("[%s][%s %" PRIu64 "] [ERROR_CODE:0x%" PRIX32 "] [PLATFORM:%s] [TYPE:%s] %s\n",
            getLevelName(level).c_str(),
            getFormattedTimeMs().c_str(),
            getTimestampMs(),
            error_code,
            getPlatformName(platform).c_str(),
            getTypeName(type).c_str(),
            msg.c_str());
    }
    static void info(ErrorPlatform platform,std::string msg) {
    printf("[INFO][%s %" PRIu64 "] [PLATFORM:%s] %s\n",
        getFormattedTimeMs().c_str(),
        getTimestampMs(),
        getPlatformName(platform).c_str(),
        msg.c_str());
    }
};
#endif //CTORCH_ERROR_H
