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
#include <mutex>
#include <thread>
#ifdef __CUDACC__
    #include <cuda_runtime.h>
#endif

#if defined(__APPLE__) || defined(__linux__)
    #include <sys/utsname.h>
#endif

#ifdef __linux__
    #include <fstream>
#endif


// 颜色宏
#define ESC_START       "\033["         // 标准开头
#define ESC_END         "\033[0m"       // 标准结束
#define COLOR_TRACE     "38;5;246m"     // TRACE：灰色（低优先级）
#define COLOR_DEBUG     "36;1m"         // DEBUG：青蓝色加粗（开发调试）
#define COLOR_INFO      "32;1m"         // INFO：绿色加粗（正常信息，去掉下划线）
#define COLOR_WARN      "33;1m"         // WARN：黄色加粗（警告）
#define COLOR_ERR       "31;1m"         // ERROR：红色加粗（错误）
#define COLOR_FATAL     "31;5;1m"       // FATAL：亮红+闪烁+加粗（致命错误）
#define COLOR_ALERT     "37;1m"         // 保留：欢迎信息用（白色加粗）

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
    kGENERAL = 5,
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

// 输出详细级别
enum class PrintLevel {
    MINIUM = 0,  // 仅大于等于INFO级别
    MEDIUM = 1,  // 仅大于等于DEBUG级别
    FULL = 2,    // 全部级别输出
};

// 辅助函数

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


/**
 * @param ms 以毫秒为单位的时间
 * @brief 暂停 ms 毫秒的程序执行
 */
inline void ctorch_sleep(uint64_t ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// 该类为单例模式，不允许将构造函数公开
class Ctorch_Stats {
private:
    Ctorch_Stats() = default;
    // 禁止拷贝构造：防止通过“实例拷贝”创建新对象
    Ctorch_Stats(const Ctorch_Stats&);
    // 禁止赋值重载：防止通过“赋值”创建新对象
    Ctorch_Stats& operator=(const Ctorch_Stats&) = delete;
    uint64_t error_count = 0;
    uint64_t warn_count = 0;
    uint64_t fatal_count = 0;

    bool if_first = true;
    std::mutex mutex_;
    static void welCome(){
        // printf(ESC_START COLOR_ALERT);
        printf("============================================================\n");
        printf(" $$$$$$\\  $$$$$$$$\\  $$$$$$\\  $$$$$$$\\   $$$$$$\\  $$\\   $$\\\n");
        printf("$$  __$$\\ \\__$$  __|$$  __$$\\ $$  __$$\\ $$  __$$\\ $$ |  $$ |\n");
        printf("$$ /  \\__|   $$ |   $$ /  $$ |$$ |  $$ |$$ /  \\__|$$ |  $$ |\n");
        printf("$$ |         $$ |   $$ |  $$ |$$$$$$$  |$$ |      $$$$$$$$ |\n");
        printf("$$ |         $$ |   $$ |  $$ |$$  __$$< $$ |      $$  __$$ |\n");
        printf("$$ |  $$\\    $$ |   $$ |  $$ |$$ |  $$ |$$ |  $$\\ $$ |  $$ |\n");
        printf("\\$$$$$$  |   $$ |    $$$$$$  |$$ |  $$ |\\$$$$$$  |$$ |  $$ |\n");
        printf(" \\______/    \\__|    \\______/ \\__|  \\__| \\______/ \\__|  \\__|\n");
        printf("============================================================\n");
        printf("Version RC Public 1.0\n");
        ctorch_sleep(500);
        // printf(ESC_END);
#ifdef __CUDACC__
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string cudaInfo = std::string(prop.name) + " (Compute Capability: " +
                           std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";
    printf("| %-10s %-48s \n", "CUDA:", cudaInfo.c_str());
#else
    printf("| %-10s %-50s \n", "CUDA:", "Not Found (仅支持CPU/MPS/AMX)");
#endif

    // C++标准（修复宽度不一致问题）
    printf("| %-10s C++%-45d\n", "C++:", __cplusplus / 100 - 1997);

    // 编译器信息（动态获取）
    std::string compilerInfo;
#ifdef __GNUC__
    compilerInfo = "GCC " + std::to_string(__GNUC__) + "." +
                   std::to_string(__GNUC_MINOR__) + "." +
                   std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(__clang__)
    compilerInfo = "Clang " + std::to_string(__clang_major__) + "." +
                   std::to_string(__clang_minor__) + "." +
                   std::to_string(__clang_patchlevel__);
#elif defined(_MSC_VER)
    compilerInfo = "MSVC " + std::to_string(_MSC_VER);
#else
    compilerInfo = "Unknown Compiler";
#endif
    printf("| %-10s %-48s \n", "Compiler:", compilerInfo.c_str());

    // 构建时间
    std::string buildTime = std::string(__DATE__) + " " + __TIME__;
    printf("| %-10s %-48s \n", "Build:", buildTime.c_str());

    // 系统信息（动态获取，不再写死）
    std::string systemInfo;
#ifdef __APPLE__
    struct utsname un;
    if (uname(&un) == 0) {
        systemInfo = std::string("macOS (Kernel ") + un.release + ")";
    } else {
        systemInfo = "macOS (Unknown version)";
    }
#elif defined(__linux__)
    // 优先读取/etc/os-release获取发行版名称
    std::ifstream osRelease("/etc/os-release");
    if (osRelease.is_open()) {
        std::string line;
        while (std::getline(osRelease, line)) {
            if (line.find("PRETTY_NAME=") == 0) {
                systemInfo = line.substr(12);
                // 移除可能存在的引号
                if (!systemInfo.empty() && systemInfo.front() == '"')
                    systemInfo.erase(0, 1);
                if (!systemInfo.empty() && systemInfo.back() == '"')
                    systemInfo.pop_back();
                break;
            }
        }
        osRelease.close();
    }

    // 如果无法从os-release获取，则使用uname
    if (systemInfo.empty()) {
        struct utsname un;
        if (uname(&un) == 0) {
            systemInfo = std::string(un.sysname) + " " + un.release;
        } else {
            systemInfo = "Linux (Unknown version)";
        }
    }
#elif defined(_WIN32)
    systemInfo = "Windows";
    // 如需更详细的Windows版本，可使用GetVersionExW或RtlGetVersion
#else
    systemInfo = "Unknown System";
#endif
    printf("| %-10s %-48s \n", "System:", systemInfo.c_str());
    }
public:
    // 调试级别
    PrintLevel level = PrintLevel::FULL;
    // 开始时间
    uint64_t start = 0;

    static Ctorch_Stats& getInstance() {
        static Ctorch_Stats instance_;  // 这里利用了一个巧妙的C++特性，保证全局只会实例化一个instance_
        if (instance_.if_first) {
            instance_.welCome();
            instance_.if_first = false;
            instance_.start = getTimestampMs();
        }
        return instance_;
    }
    static void incrError() {
        Ctorch_Stats& inst = getInstance();
        std::lock_guard<std::mutex> lock(inst.mutex_);
        inst.error_count++;
    }
    static uint64_t getTotalError() {
        Ctorch_Stats& inst = getInstance();
        std::lock_guard<std::mutex> lock(inst.mutex_);
        return inst.error_count;
    }
     static void incrWarn() {
        Ctorch_Stats& inst = getInstance();
        std::lock_guard<std::mutex> lock(inst.mutex_);
        inst.warn_count++;
    }
    static uint64_t getTotalWarn() {
        Ctorch_Stats& inst = getInstance();
        std::lock_guard<std::mutex> lock(inst.mutex_);
        return inst.warn_count;
    }
     static void incrFatal() {
        Ctorch_Stats& inst = getInstance();
        std::lock_guard<std::mutex> lock(inst.mutex_);
        inst.fatal_count++;
    }
    static uint64_t getTotalFatal() {
        Ctorch_Stats& inst = getInstance();
        std::lock_guard<std::mutex> lock(inst.mutex_);
        return inst.fatal_count;
    }

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
            case ErrorPlatform::kGENERAL: return "GENERAL";
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
     static std::string getPrintLevelName(PrintLevel level) {
        switch (level) {
            case PrintLevel::MINIUM:      return "MINIUM";
            case PrintLevel::MEDIUM:     return "MEDIUM";
            case PrintLevel::FULL:        return "FULL";
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

public: static void log(ErrorLevel level,ErrorPlatform platform,ErrorType type,const std::string msg) {
    Ctorch_Stats& inst = Ctorch_Stats::getInstance();
    if (inst.level==PrintLevel::MEDIUM&&level==ErrorLevel::TRACE) return;
    if (inst.level==PrintLevel::MINIUM&&(level==ErrorLevel::DEBUG||level==ErrorLevel::TRACE)) return ;
        uint32_t error_code = computeCode(level, platform, type);
        switch (level) {
            case ErrorLevel::TRACE: {
                printf(ESC_START COLOR_TRACE);
                break;
            }
            case ErrorLevel::INFO: {
                printf(ESC_START COLOR_INFO);
                break;
            }
            case ErrorLevel::DEBUG: {
                printf(ESC_START COLOR_DEBUG);
                break;
            }
            case ErrorLevel::WARN: {
                printf(ESC_START COLOR_WARN);
                break;
            }
            case ErrorLevel::ERROR: {
                printf(ESC_START COLOR_ERR);
                break;
            }
            case ErrorLevel::FATAL: {
                printf(ESC_START COLOR_FATAL);
                break;
            }
            default: {
                // 未知级别用默认色
                printf(ESC_START);
                break;
            }
        }
        printf("[%s] [%s %" PRIu64 "] [ERROR_CODE:0x%" PRIX32 "] [PLATFORM:%s] [TYPE:%s] %s\n",
               getLevelName(level).c_str(),
               getFormattedTimeMs().c_str(),
               getTimestampMs(),
               error_code,
               getPlatformName(platform).c_str(),
               getTypeName(type).c_str(),
               msg.c_str());
        printf(ESC_END);
        if (level == ErrorLevel::ERROR) {
            inst.incrError();
        } else if (level == ErrorLevel::WARN) {
            inst.incrWarn();
        } else if (level == ErrorLevel::FATAL) {
            inst.incrFatal();
        }
    }

    static void info(ErrorPlatform platform,std::string msg) {
        printf(ESC_START COLOR_INFO);
        printf("[INFO]" ESC_END "  [%s %" PRIu64 "] [PLATFORM:%s] %s\n",
               getFormattedTimeMs().c_str(),
               getTimestampMs(),
               getPlatformName(platform).c_str(),
               msg.c_str());
        printf(ESC_END);
    }

    static void stats() {
        Ctorch_Stats &inst = Ctorch_Stats::getInstance();
        printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "Total Error: %" PRIu64 "\n", inst.getTotalError());
        printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "Total WARN: %" PRIu64 "\n", inst.getTotalWarn());
        printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "Total FATAL: %" PRIu64 "\n", inst.getTotalFatal());
        printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "Total Time: %" PRIu64 "ms\n", getTimestampMs()-inst.start);
    }

    static void setPrintLevel(PrintLevel level) {
        Ctorch_Stats &inst = Ctorch_Stats::getInstance();
        inst.level = level;
        printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "[%s %" PRIu64 "] Set Print Level = %s\n", getFormattedTimeMs().c_str(), getTimestampMs(),
               getPrintLevelName(level).c_str());
    }

    static void trace(ErrorPlatform platform,std::string msg) {
        Ctorch_Stats& inst = Ctorch_Stats::getInstance();
        if (inst.level==PrintLevel::MINIUM||inst.level==PrintLevel::MEDIUM) return ;
        printf(ESC_START COLOR_DEBUG);
        printf("[TRACE]" ESC_END "  [%s %" PRIu64 "] [PLATFORM:%s] %s\n",
               getFormattedTimeMs().c_str(),
               getTimestampMs(),
               getPlatformName(platform).c_str(),
               msg.c_str());
        printf(ESC_END);
    }
};
// TODO: 优化Tensor.cpp
// TODO: 编写API文档
#endif //CTORCH_ERROR_H
