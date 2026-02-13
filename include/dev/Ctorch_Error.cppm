module; // 全局模块片段

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

export module Ctorch_Error;
import Ctools;

export {

/**
 * @class Ctorch_Stats
 * @brief Ctorch统计信息类，采用单例模式设计
 * @details 用于统计错误、警告和致命错误的数量，并提供程序运行时间统计
 */
class Ctorch_Stats {
private:
    /**
     * @brief 私有构造函数，防止外部实例化
     */
    Ctorch_Stats() = default;
    /**
     * @brief 禁止拷贝构造函数，防止通过实例拷贝创建新对象
     */
    Ctorch_Stats(const Ctorch_Stats&);
    /**
     * @brief 禁止赋值操作符重载，防止通过赋值创建新对象
     */
    Ctorch_Stats& operator=(const Ctorch_Stats&) = delete;
    /**
     * @brief 错误计数
     */
    uint64_t error_count = 0;
    /**
     * @brief 警告计数
     */
    uint64_t warn_count = 0;
    /**
     * @brief 致命错误计数
     */
    uint64_t fatal_count = 0;

    /**
     * @brief 是否为首次启动标志
     */
    bool if_first = true;
    /**
     * @brief 互斥锁，用于线程安全操作
     */
    std::mutex mutex_;
    /**
     * @brief 打印欢迎信息
     * @details 打印Ctorch的欢迎界面，包括版本信息、CUDA信息、C++标准、编译器信息、构建时间和系统信息
     */
    static void welCome(){
        // printf(ESC_START COLOR_ALERT);
        printf("============================================================\n");
        printf(" $$$$$$\\  $$$$$$$$\\  $$$$$$\\  $$$$$$$\\   $$$$$$\\  $$\\   $$\\\n");
        printf("$$  __$$\\ \__$$  __|$$  __$$\\ $$  __$$\\ $$  __$$\\ $$ |  $$ |\n");
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
    printf("| %-10s C++%-45ld\n", "C++:", __cplusplus / 100 - 1997);

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

    // 系统信息
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
    /**
     * @brief 打印级别
     */
    PrintLevel level = PrintLevel::FULL;
    /**
     * @brief 程序开始时间戳
     */
    uint64_t start = 0;

    /**
     * @brief 获取单例实例
     * @return Ctorch_Stats的引用
     */
    static Ctorch_Stats& getInstance() {
        static Ctorch_Stats instance_;  // 这里利用了一个巧妙的C++特性，保证全局只会实例化一个instance_
        if (instance_.if_first) {
            instance_.welCome();
            instance_.if_first = false;
            instance_.start = getTimestampMs();
        }
        return instance_;
    }
    /**
     * @brief 增加错误计数
     */
    static void incrError() {
        Ctorch_Stats& inst = getInstance();
        std::lock_guard<std::mutex> lock(inst.mutex_);
        inst.error_count++;
    }
    /**
     * @brief 获取总错误数
     * @return 错误总数
     */
    static uint64_t getTotalError() {
        Ctorch_Stats& inst = getInstance();
        std::lock_guard<std::mutex> lock(inst.mutex_);
        return inst.error_count;
    }
    /**
     * @brief 增加警告计数
     */
    static void incrWarn() {
        Ctorch_Stats& inst = getInstance();
        std::lock_guard<std::mutex> lock(inst.mutex_);
        inst.warn_count++;
    }
    /**
     * @brief 获取总警告数
     * @return 警告总数
     */
    static uint64_t getTotalWarn() {
        Ctorch_Stats& inst = getInstance();
        std::lock_guard<std::mutex> lock(inst.mutex_);
        return inst.warn_count;
    }
    /**
     * @brief 增加致命错误计数
     */
    static void incrFatal() {
        Ctorch_Stats& inst = getInstance();
        std::lock_guard<std::mutex> lock(inst.mutex_);
        inst.fatal_count++;
    }
    /**
     * @brief 获取总致命错误数
     * @return 致命错误总数
     */
    static uint64_t getTotalFatal() {
        Ctorch_Stats& inst = getInstance();
        std::lock_guard<std::mutex> lock(inst.mutex_);
        return inst.fatal_count;
    }

};

/**
 * @class Ctorch_Error
 * @brief Ctorch错误处理类
 * @details 用于记录和管理CTorch中的各种错误信息，包括日志记录、错误统计和异常抛出
 */
class Ctorch_Error {
    /**
     * @brief 将ErrorLevel枚举转换为字符串名称
     * @param level 错误级别枚举值
     * @return 对应的字符串名称
     */
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
    /**
     * @brief 将ErrorPlatform枚举转换为字符串名称
     * @param platform 错误平台枚举值
     * @return 对应的字符串名称
     */
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
    /**
     * @brief 将ErrorType枚举转换为字符串名称
     * @param type 错误类型枚举值
     * @return 对应的字符串名称
     */
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
    /**
     * @brief 将PrintLevel枚举转换为字符串名称
     * @param level 打印级别枚举值
     * @return 对应的字符串名称
     */
    static std::string getPrintLevelName(PrintLevel level) {
        switch (level) {
            case PrintLevel::MINIUM:      return "MINIUM";
            case PrintLevel::MEDIUM:     return "MEDIUM";
            case PrintLevel::FULL:        return "FULL";
            default: return "UNKNOWN";
        }
    }

    /**
     * @brief 获取ErrorLevel枚举的数值代码
     * @param level 错误级别枚举值
     * @return 对应的数值代码
     */
    static uint8_t getLevelCode(ErrorLevel level) {
        return static_cast<uint8_t>(level);
    }
    /**
     * @brief 获取ErrorPlatform枚举的数值代码
     * @param platform 错误平台枚举值
     * @return 对应的数值代码
     */
    static uint8_t getPlatformCode(ErrorPlatform platform) {
        return static_cast<uint8_t>(platform);
    }
    /**
     * @brief 获取ErrorType枚举的数值代码
     * @param type 错误类型枚举值
     * @return 对应的数值代码
     */
    static uint8_t getTypeCode(ErrorType type) {
        return static_cast<uint8_t>(type);
    }

public: 
    /**
     * @brief 记录日志信息
     * @param level 错误级别
     * @param platform 错误平台
     * @param type 错误类型
     * @param msg 错误信息
     */
    static void log(ErrorLevel level, ErrorPlatform platform, ErrorType type, const std::string msg) {
        Ctorch_Stats& inst = Ctorch_Stats::getInstance();
        if (inst.level == PrintLevel::MEDIUM && level == ErrorLevel::TRACE) return;
        if (inst.level == PrintLevel::MINIUM && (level == ErrorLevel::DEBUG || level == ErrorLevel::TRACE)) return ;
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

    /**
     * @brief 记录INFO级别信息
     * @param platform 错误平台
     * @param msg 信息内容
     */
    static void info(ErrorPlatform platform, std::string msg) {
        printf(ESC_START COLOR_INFO);
        printf("[INFO]" ESC_END "  [%s %" PRIu64 "] [PLATFORM:%s] %s\n",
               getFormattedTimeMs().c_str(),
               getTimestampMs(),
               getPlatformName(platform).c_str(),
               msg.c_str());
        printf(ESC_END);
    }

    /**
     * @brief 打印统计信息
     * @details 打印总错误数、总警告数、总致命错误数和总运行时间
     */
    static void stats() {
        Ctorch_Stats &inst = Ctorch_Stats::getInstance();
        printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "Total Error: %" PRIu64 "\n", inst.getTotalError());
        printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "Total WARN: %" PRIu64 "\n", inst.getTotalWarn());
        printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "Total FATAL: %" PRIu64 "\n", inst.getTotalFatal());
        printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "Total Time: %" PRIu64 "ms\n", getTimestampMs() - inst.start);
    }

    /**
     * @brief 设置打印级别
     * @param level 打印级别
     */
    static void setPrintLevel(PrintLevel level) {
        Ctorch_Stats &inst = Ctorch_Stats::getInstance();
        inst.level = level;
        printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "[%s %" PRIu64 "] Set Print Level = %s\n", getFormattedTimeMs().c_str(), getTimestampMs(),
               getPrintLevelName(level).c_str());
    }

    /**
     * @brief 记录TRACE级别信息
     * @param platform 错误平台
     * @param msg 信息内容
     */
    static void trace(ErrorPlatform platform, std::string msg) {
        Ctorch_Stats& inst = Ctorch_Stats::getInstance();
        if (inst.level == PrintLevel::MINIUM || inst.level == PrintLevel::MEDIUM) return ;
        printf(ESC_START COLOR_DEBUG);
        printf("[TRACE]" ESC_END "  [%s %" PRIu64 "] [PLATFORM:%s] %s\n",
               getFormattedTimeMs().c_str(),
               getTimestampMs(),
               getPlatformName(platform).c_str(),
               msg.c_str());
        printf(ESC_END);
    }
    
    /**
     * @brief 便捷方法：快速记录ERROR级别错误
     * @param platform 错误平台
     * @param type 错误类型
     * @param msg 错误信息
     */
    static void error(ErrorPlatform platform, ErrorType type, const std::string& msg) {
        log(ErrorLevel::ERROR, platform, type, msg);
    }
    
    /**
     * @brief 便捷方法：快速记录WARN级别错误
     * @param platform 错误平台
     * @param type 错误类型
     * @param msg 错误信息
     */
    static void warn(ErrorPlatform platform, ErrorType type, const std::string& msg) {
        log(ErrorLevel::WARN, platform, type, msg);
    }
    
    /**
     * @brief 便捷方法：快速记录FATAL级别错误
     * @param platform 错误平台
     * @param type 错误类型
     * @param msg 错误信息
     */
    static void fatal(ErrorPlatform platform, ErrorType type, const std::string& msg) {
        log(ErrorLevel::FATAL, platform, type, msg);
    }
    
    /**
     * @brief 便捷方法：快速记录INFO级别信息
     * @param platform 错误平台
     * @param msg 信息内容
     */
    static void info(ErrorPlatform platform, const std::string& msg) {
        log(ErrorLevel::INFO, platform, ErrorType::UNKNOWN, msg);
    }
    
    /**
     * @brief 便捷方法：快速记录DEBUG级别信息
     * @param platform 错误平台
     * @param msg 信息内容
     */
    static void debug(ErrorPlatform platform, const std::string& msg) {
        log(ErrorLevel::DEBUG, platform, ErrorType::UNKNOWN, msg);
    }
    
    /**
     * @brief 抛出异常方法：记录错误并抛出异常
     * @param platform 错误平台
     * @param type 错误类型
     * @param msg 错误信息
     * @throw std::runtime_error 抛出运行时异常
     */
    static void throwException(ErrorPlatform platform, ErrorType type, const std::string& msg) {
        // 首先记录错误
        log(ErrorLevel::ERROR, platform, type, msg);
        // 然后抛出异常
        throw std::runtime_error(msg);
    }
    
    /**
     * @brief 抛出致命异常方法：记录致命错误并抛出异常
     * @param platform 错误平台
     * @param type 错误类型
     * @param msg 错误信息
     * @throw std::runtime_error 抛出运行时异常
     */
    static void throwFatalException(ErrorPlatform platform, ErrorType type, const std::string& msg) {
        // 首先记录致命错误
        log(ErrorLevel::FATAL, platform, type, msg);
        // 然后抛出异常
        throw std::runtime_error(msg);
    }
};

}
