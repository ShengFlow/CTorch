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
#include <vector>   // 向量容器
#include <memory>   // 智能指针
#include <iostream> // 输入输出流

export module Ctools;

export {

/**
 * @def ESC_START
 * @brief 终端颜色转义序列的标准开头
 */
#define ESC_START       "\033["

/**
 * @def ESC_END
 * @brief 终端颜色转义序列的标准结束
 */
#define ESC_END         "\033[0m"

/**
 * @def COLOR_TRACE
 * @brief TRACE级别的终端颜色：灰色（低优先级）
 */
#define COLOR_TRACE     "38;5;246m"

/**
 * @def COLOR_DEBUG
 * @brief DEBUG级别的终端颜色：青蓝色加粗（开发调试）
 */
#define COLOR_DEBUG     "36;1m"

/**
 * @def COLOR_INFO
 * @brief INFO级别的终端颜色：绿色加粗（正常信息）
 */
#define COLOR_INFO      "32;1m"

/**
 * @def COLOR_WARN
 * @brief WARN级别的终端颜色：黄色加粗（警告）
 */
#define COLOR_WARN      "33;1m"

/**
 * @def COLOR_ERR
 * @brief ERROR级别的终端颜色：红色加粗（错误）
 */
#define COLOR_ERR       "31;1m"

/**
 * @def COLOR_FATAL
 * @brief FATAL级别的终端颜色：亮红+闪烁+加粗（致命错误）
 */
#define COLOR_FATAL     "31;5;1m"

/**
 * @def COLOR_ALERT
 * @brief 保留：欢迎信息用的终端颜色（白色加粗）
 */
#define COLOR_ALERT     "37;1m"

/**
 * @enum ErrorLevel
 * @brief 错误级别枚举，用于表示不同严重程度的错误
 */
enum class ErrorLevel {
    TRACE = 0,   ///< 细粒度调试（如kernel启动参数）
    DEBUG = 1,   ///< 开发调试（如数据拷贝状态）
    INFO = 2,    ///< 运行信息（如平台检测结果）
    WARN = 3,    ///< 非致命问题（如CUDA版本兼容提示）
    ERROR = 4,   ///< 功能异常（如kernel执行失败）
    FATAL = 5    ///< 崩溃级错误（如内存分配失败）
};

/**
 * @enum ErrorPlatform
 * @brief 错误发生的设备枚举
 */
enum class ErrorPlatform {
    kCPU = 0,      ///< CPU设备
    kCUDA = 1,     ///< CUDA设备
    kMPS = 2,      ///< MPS设备
    kAMX = 3,      ///< AMX设备
    kUNKNOWN = 4,  ///< 未知设备
    kGENERAL = 5,  ///< 通用设备
};

/**
 * @enum ErrorType
 * @brief 错误类型枚举，用于分类不同类型的错误
 */
enum class ErrorType {
    UNKNOWN = 0,          ///< 未知错误
    MEMORY = 1,           ///< 内存相关（分配/释放/拷贝失败）
    DIMENSION = 2,        ///< 维度相关（不匹配/越界/空维度）
    DEVICE_COMPAT = 3,    ///< 设备兼容（跨设备运算/设备不支持/设备初始化失败）
    DATATYPE = 4,         ///< 数据类型（不匹配/转换失败/不支持的类型）
    KERNEL_LAUNCH = 5,    ///< 内核调用（CUDA/MPS/AMX内核启动失败/执行超时）
    TENSOR_STATE = 6,     ///< Tensor状态（未初始化/只读/已释放）
    PLATFORM_API = 7      ///< 平台API调用失败（如cudaGetDevice/MPSGraph创建失败）
};

/**
 * @enum PrintLevel
 * @brief 输出详细级别枚举
 */
enum class PrintLevel {
    MINIUM = 0,  ///< 仅大于等于INFO级别
    MEDIUM = 1,  ///< 仅大于等于DEBUG级别
    FULL = 2,    ///< 全部级别输出
};

// ==================== 统一矩阵乘法接口 ====================

/**
 * @enum MatMulStrategy
 * @brief 矩阵乘法算法选择策略枚举
 */
enum class MatMulStrategy {
    AUTO,           ///< 自动选择
    NAIVE,          ///< 朴素算法
    BLOCKED,        ///< 分块优化
    STRASSEN,       ///< Strassen递归算法
    OPTIMIZED       ///< 最优算法组合
};

/**
 * @enum DeviceType
 * @brief 设备类型枚举，定义张量存储的位置
 */
enum class DeviceType {
    kCPU = 0,      ///< CPU设备
    kCUDA = 1,     ///< CUDA设备
    kMPS = 2,      ///< MPS设备
    kAMX = 3,      ///< AMX设备
    kUNKNOWN = 4,  ///< 未知设备
    kGENERAL = 5,  ///< 通用设备
};

/**
 * @enum DType
 * @brief 数据类型枚举，定义张量元素的类型
 */
enum class DType {
   kFloat,  ///< 32位浮点数 (torch.float32)
   kDouble, ///< 64位浮点数 (torch.float64)
   kInt,    ///< 32位整数 (torch.int32)
   kLong,   ///< 64位整数 (torch.int64)
   kBool,   ///< 布尔值 (torch.bool)
};

/**
 * @enum op
 * @brief 自动微分操作符枚举
 */
enum class op{
   // 基本运算
   Add,        ///< 加
   Sub,        ///< 减
   Neg,        ///< 负号
   Mul,        ///< 乘
   Div,        ///< 除
   MatMul,     ///< 矩阵乘法
   Dot,        ///< 点乘
   Cos,        ///< 余弦
   Sin,        ///< 正弦

   // 卷积操作
   Conv,       ///< 卷积
   Pool,       ///< 池化

   // 激活函数
   ReLU,       ///< 线性整流函数
   Tanh,       ///< 双曲正切函数
   Sigmoid,    ///< Sigmoid函数
   Softmax,    ///< Softmax函数

   // 激活函数变种
   LReLU,      ///< 渗漏线性整流函数
   PReLU,      ///< 参数化线性整流函数

   // 损失函数
   MSE,        ///< 均方误差
   MAE,        ///< 平均绝对误差
   CE,         ///< 交叉熵损失
   BCE,        ///< 二元交叉熵损失

   // 其他操作
   Sum,        ///< 求和
};

/**
 * @struct MatMulConfig
 * @brief 矩阵乘法性能配置结构体
 */
struct MatMulConfig {
    static constexpr size_t BLOCK_SIZE_THRESHOLD = 64;      ///< 分块大小阈值
    static constexpr size_t STRASSEN_THRESHOLD = 128;        ///< Strassen算法阈值
    static constexpr size_t SMALL_MATRIX_THRESHOLD = 32;     ///< 小矩阵阈值（使用朴素算法）
    static constexpr bool ENABLE_PROFILING = true;           ///< 是否启用性能分析
    static constexpr bool ENABLE_CACHE_OPTIMIZATION = true; ///< 是否启用缓存优化
};

/**
 * @struct BroadCastResult
 * @brief 广播结果结构体，用于存储张量广播后的信息
 */
struct BroadCastResult {
   std::vector<size_t> logicShape;    ///< 广播后的逻辑形状
   std::vector<size_t> logicStridesA; ///< 张量A的逻辑步幅
   std::vector<size_t> logicStridesB; ///< 张量B的逻辑步幅
};

// ======================= 辅助函数 =======================

/**
 * @brief 将 DeviceType 枚举转换为对应的 ErrorPlatform 枚举
 * @details 基于两个枚举成员名称、取值完全一致的特性，直接进行静态类型转换，
 * 转换效率为 O(1)，无额外性能开销，适用于当前枚举完全匹配的场景。
 * @param device_type 输入的设备类型枚举（DeviceType）
 * @return ErrorPlatform 对应的错误平台枚举，与输入 DeviceType 一一映射
 */
ErrorPlatform DeviceTypeToErrorPlatform(const DeviceType device_type);

/**
 * @brief 将ErrorPlatform转换为DeviceType
 * @param platforms 错误平台枚举
 * @return 对应的设备类型枚举
 */
constexpr inline DeviceType platform(ErrorPlatform platforms) {
    return static_cast<DeviceType>(static_cast<uint32_t>(platforms));
}

/**
 * @brief 将数据类型转换为字符串表示
 * @param dtype 数据类型枚举
 * @return 数据类型的字符串表示
 */
constexpr const char* dtypeToString(DType dtype) {
    switch (dtype) {
        case DType::kFloat:  return "float32";
        case DType::kDouble: return "float64";
        case DType::kInt:    return "int32";
        case DType::kLong:   return "int64";
        case DType::kBool:   return "bool";
        default:             return "unknown";
    }
}

/**
 * @brief 获取数据类型的字节大小
 * @param dtype 数据类型枚举
 * @return 数据类型的字节大小
 * @throw std::invalid_argument 如果数据类型未知
 */
constexpr size_t dtypeSize(DType dtype) {
    switch (dtype) {
        case DType::kFloat:  return sizeof(float);
        case DType::kDouble: return sizeof(double);
        case DType::kInt:    return sizeof(int32_t);
        case DType::kLong:   return sizeof(int64_t);
        case DType::kBool:   return sizeof(bool);
        default: 
            // 注意：这里不能直接调用Ctorch_Error::log，因为会导致循环依赖
            // 改为抛出标准异常，在调用处捕获并转换为Ctorch_Error
            throw std::invalid_argument("未知的数据类型");
    }
}

/**
 * @brief 快速计算两个整数的最小值
 * @param a 第一个整数
 * @param b 第二个整数
 * @return 两个整数中的最小值
 */
inline int minx(int a, int b){
   int diff = b - a;
   return a + (diff & (diff >> 31));
}

/**
 * @brief 计算错误码
 * @param level 错误级别
 * @param platform 错误平台
 * @param type 错误类型
 * @return 32位错误码
 */
uint32_t computeCode(ErrorLevel level, ErrorPlatform platform, ErrorType type);

/**
* @brief 返回人类可读的当前时间，精确到毫秒（跨平台、线程安全）
* @return 格式化字符串，示例："2026-02-17 00:00:00.123"
*/
std::string getFormattedTimeMs();

/**
* @brief 返回当前时间的毫秒级时间戳（跨平台）
* @return uint64_t类型的时间戳，示例：1749984000123
*/
uint64_t getTimestampMs();

/**
 * @param ms 以毫秒为单位的时间
 * @brief 暂停 ms 毫秒的程序执行
 */
inline void ctorch_sleep(uint64_t ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

}

// 实现部分

uint32_t computeCode(ErrorLevel level, ErrorPlatform platform, ErrorType type) {
    return (static_cast<uint32_t>(level) << 24) | 
           (static_cast<uint32_t>(platform) << 16) | 
           static_cast<uint32_t>(type);
}

std::string getFormattedTimeMs() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") 
       << "." << std::setw(3) << std::setfill('0') << ms.count();
    return ss.str();
}

uint64_t getTimestampMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

ErrorPlatform DeviceTypeToErrorPlatform(const DeviceType device_type) {
    return static_cast<ErrorPlatform>(static_cast<uint32_t>(device_type));
}
