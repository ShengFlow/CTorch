/**
 * @file C3Cleanup.h
 * @generation SHARED 跨代退出清理
 * @brief C3 退出清理公共 helper
 * @details 统一 C3 端到端测试/程序的退出清理序列，确保所有 CompiledKernel / LLVM
 *          module 在静态析构前释放，避免退出时 recursive_mutex / removeModule 崩溃。
 *          所有 C3 测试（test_c3_mnist_train、test_region_fusion_auto 等）在 main
 *          返回前都应调用 ct::c3::shutdownAll()。
 * @date 2026/8/7
 */

#ifndef CTORCH_C3_C3CLEANUP_H
#define CTORCH_C3_C3CLEANUP_H

#include "C3Engine.h"
#include "C3HotPathManager.h"
#include "RegionFusion.h"

namespace ct {
namespace c3 {

/**
 * @brief 统一的 C3 退出清理序列
 * @details 清理顺序固定为：
 *          1. C3HotPathManager::shutdown()    —— 停止热路径检测/后台编译
 *          2. C3Engine::shutdown()            —— 等待所有后台编译完成并回收线程
 *          3. C3Engine::clearCache()          —— 清空内存缓存
 *          4. RegionFusionRegistry::clear()   —— 释放区域融合注册表
 *          5. C3KernelRegistry::uninstallAll()—— 卸载所有注入的 kernel
 *
 *          顺序不可随意调整：必须先停止后台任务并清空缓存，再释放各注册表，
 *          确保所有 CompiledKernel / LLVM module 在静态析构前释放。
 */
inline void shutdownAll() {
    C3HotPathManager::instance().shutdown();
    C3Engine::getInstance().shutdown();
    C3Engine::getInstance().clearCache();
    RegionFusionRegistry::getInstance().clear();
    C3KernelRegistry::getInstance().uninstallAll();
}

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_C3CLEANUP_H