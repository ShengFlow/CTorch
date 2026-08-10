// (SECURITY_SENSITIVE) CTorch Storage::Deleter exit-time UAF PoC
// 本地沙箱运行，仅用于验证漏洞存在与修复有效性。
// 禁止用于未授权系统或生产环境。
//
// 触发原理（历史）：
//   Storage::Deleter 在析构时调用 AllocatorManager::getInstance().getAllocator(_device)。
//   AllocatorManager 是函数内 Meyers 单例。若某个全局/静态 Storage 在 main 中首次触发
//   AllocatorManager 的构造，则退出时 AllocatorManager 会在该 Storage 之前被析构；
//   Storage 析构时再次访问 AllocatorManager 的内部成员（mutex/unordered_map），
//   形成 static destruction order fiasco / use-after-free。
//
// 修复后状态（2026-08-02 验证）：
//   Storage::Deleter 持有 std::shared_ptr<DeviceAllocator>，不再依赖 Meyers 单例的
//   运行时查找。shared_ptr 强引用保证 allocator 在 Deleter 调用 deallocate() 前存活，
//   退出顺序：g_storage 析构 → Deleter 调用 _allocator->deallocate（allocator 仍存活）
//   → AllocatorManager 单例最后析构。
//
//   验证方法：编译并运行本 PoC。修复后 PoC 正常退出（exit code 0），无 SIGSEGV/SIGABRT。
//   若未来有人将 Deleter 改回 "getInstance().getAllocator()"，本 PoC 会再次失败，起到
//   回归门禁作用。

#include "Storage.h"
#include <iostream>

// 全局 Storage：在 main 之前完成默认构造。
Storage g_storage;

int main() {
    std::cout << "[PoC] assigning global storage..." << std::endl;

    // 赋值操作构造临时 Storage，临时 Storage 的构造函数首次触发 AllocatorManager 单例构造。
    g_storage = Storage(1024, DType::kFloat, DeviceType::kCPU);

    std::cout << "[PoC] assignment done, g_storage.size()=" << g_storage.size() << std::endl;
    return 0;
    // 退出阶段（reverse order of construction）：
    // 1. AllocatorManager（函数内 static，在 main 中构造）先被析构。
    // 2. g_storage（namespace static，main 前构造）后被析构，其 Deleter 访问已析构的 AllocatorManager。
}
