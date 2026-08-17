/**
 * @file C3Config.h
 * @generation SHARED 跨代开关配置（所有 C3_* env var 集中于此）
 * @brief C3 子功能统一开关体系
 * @details 集中管理 C3 各子功能的开启/关闭，替代散落的 getenv 与硬编码宏。
 *          每个子功能支持两级控制：
 *          1. 编译期硬开关：由 CMake option 生成对应宏（如 CT_C3_DISABLE_SINGLE_KERNEL 生成
 *             C3_DISABLE_SINGLE_KERNEL_PP），编译期关闭后无法在运行时重新开启。
 *          2. 运行时软开关：通过环境变量（如 C3_DISABLE_SINGLE_KERNEL=1）在运行时关闭。
 *
 *          查询接口采用"编译期宏 OR 运行时 env"短路：只要任一层级关闭即视为关闭。
 *          所有查询结果缓存为 static，避免热路径重复 getenv / 分支开销。
 *
 * 子功能清单（对应宏 / env / 查询接口）：
 *  - 单 kernel hotpath 注入   : CT_C3_DISABLE_SINGLE_KERNEL / C3_DISABLE_SINGLE_KERNEL / singleKernelInjectionEnabled()
 *  - 区域融合 (Region Fusion) : CT_C3_DISABLE_REGION_FUSION / C3_DISABLE_REGION_FUSION / regionFusionEnabled()
 *  - 后向融合 (Backward)      : CT_C3_DISABLE_BACKWARD       / C3_DISABLE_BACKWARD       / backwardFusionEnabled()
 *  - 热路径检测/编译触发       : CT_C3_DISABLE_HOTPATH        / C3_DISABLE_HOTPATH         / hotPathTrackingEnabled()
 *
 * @date 2026/8/7
 */

#ifndef CTORCH_C3_C3_CONFIG_H
#define CTORCH_C3_C3_CONFIG_H

#include <cstdlib>

namespace ct {
namespace c3 {

namespace detail {

/// 读取 env 布尔开关：值为 "1" 视为开启（返回 true），其余视为关闭
inline bool envFlag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1';
}

} // namespace detail

// ======================= 单 kernel hotpath 注入 =======================
/// 查询单 kernel hotpath 注入是否启用
/// （编译期 CT_C3_DISABLE_SINGLE_KERNEL 宏关闭，或运行时 C3_DISABLE_SINGLE_KERNEL=1，均禁用）
inline bool singleKernelInjectionEnabled() {
#ifdef CT_C3_DISABLE_SINGLE_KERNEL
    return false;
#else
    static const bool enabled = !detail::envFlag("C3_DISABLE_SINGLE_KERNEL");
    return enabled;
#endif
}

// ======================= 区域融合 (Region Fusion) =======================
/// 查询区域融合是否启用
/// （编译期 CT_C3_DISABLE_REGION_FUSION 宏关闭，或运行时 C3_DISABLE_REGION_FUSION=1，均禁用）
inline bool regionFusionEnabled() {
#ifdef CT_C3_DISABLE_REGION_FUSION
    return false;
#else
    static const bool enabled = !detail::envFlag("C3_DISABLE_REGION_FUSION");
    return enabled;
#endif
}

// ======================= 后向融合 (Backward) =======================
/// 查询后向融合是否启用
/// （编译期 CT_C3_DISABLE_BACKWARD 宏关闭，或运行时 C3_DISABLE_BACKWARD=1，均禁用）
inline bool backwardFusionEnabled() {
#ifdef CT_C3_DISABLE_BACKWARD
    return false;
#else
    static const bool enabled = !detail::envFlag("C3_DISABLE_BACKWARD");
    return enabled;
#endif
}

// ======================= 热路径检测/编译触发 =======================
/// 查询热路径检测与编译触发是否启用
/// （编译期 CT_C3_DISABLE_HOTPATH 宏关闭，或运行时 C3_DISABLE_HOTPATH=1，均禁用）
inline bool hotPathTrackingEnabled() {
#ifdef CT_C3_DISABLE_HOTPATH
    return false;
#else
    static const bool enabled = !detail::envFlag("C3_DISABLE_HOTPATH");
    return enabled;
#endif
}

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_C3_CONFIG_H