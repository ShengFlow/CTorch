#include "Tensor.h"
#include "Ctorch_Error.h"
#include <chrono>

// 简单的调度器功能测试
void test_scheduler_functionality() {
    // 创建上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::FULL);
    
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "测试调度器功能...");
    
    // 测试1：基本标量加法
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "\n1. 基本标量加法测试:");
    Tensor a(2.0f);
    Tensor b(3.0f);
    Tensor c = a + b; // 使用调度器执行加法
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "a + b = " + std::to_string(c.item<float>()) + " (预期: 5.0)");
    
    // 测试2：简单张量加法
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "\n2. 简单张量加法测试:");
    Tensor d({2.0f, 3.0f});
    Tensor e({4.0f, 5.0f});
    Tensor f = d + e; // 使用调度器执行加法
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "d + e = [" + std::to_string(f.data<float>()[0]) + ", " + std::to_string(f.data<float>()[1]) + "] (预期: [6.0, 8.0])");
    
    // 测试3：带自动微分的加法
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "\n3. 带自动微分的加法测试:");
    Tensor x(1.0f);
    Tensor y(2.0f);
    x.requires_grad(true);
    y.requires_grad(true);
    Tensor z = x + y;
    backward(z);
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "z = x + y = " + std::to_string(z.item<float>()) + " (预期: 3.0)");
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "dz/dx = " + std::to_string(grad(x).item<float>()) + " (预期: 1.0)");
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "dz/dy = " + std::to_string(grad(y).item<float>()) + " (预期: 1.0)");
    
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "\n✅ 调度器功能测试完成!");
}

// 性能测试：大量加法操作
void test_add_performance() {
    // 创建上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::FULL);
    
    // 测试配置
    const size_t num_ops = 1000;
    const size_t tensor_size = 100; // 100个元素的张量
    
    // 创建测试张量
    Tensor a(tensor_size);
    Tensor b(tensor_size);
    
    // 初始化张量数据
    for (size_t i = 0; i < tensor_size; ++i) {
        // 使用公共接口初始化数据
        a.data<float>()[i] = static_cast<float>(i);
        b.data<float>()[i] = static_cast<float>(i * 2);
    }
    
    // 性能测试
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "\n\n测试大量加法操作性能...");
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "操作次数: " + std::to_string(num_ops));
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "张量大小: " + std::to_string(tensor_size) + " 元素");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_ops; ++i) {
        Tensor c = a + b; // 使用调度器执行加法操作
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "总执行时间: " + std::to_string(duration) + " ms");
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "每次操作平均时间: " + std::to_string(static_cast<double>(duration) / num_ops) + " ms");
    if (duration > 0) {
        Ctorch_Error::trace(ErrorPlatform::kGENERAL, "每秒操作数: " + std::to_string(static_cast<uint64_t>(static_cast<double>(num_ops) / duration * 1000)) + " ops/s");
    }
}

// 测试不同大小张量的加法性能
void test_add_performance_by_size() {
    // 创建上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::FULL);
    
    // 测试不同大小的张量
    const size_t sizes[] = {10, 50, 100, 200, 500};
    const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const size_t num_ops = 100;
    
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "\n\n测试不同大小张量的加法性能...");
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "操作次数: " + std::to_string(num_ops));
    Ctorch_Error::trace(ErrorPlatform::kGENERAL, "\n尺寸\t总时间(ms)\t平均时间(ms)\t每秒操作数(ops/s)\t每秒处理元素数(elem/s)");
    
    for (size_t i = 0; i < num_sizes; ++i) {
        size_t size = sizes[i];
        
        // 创建测试张量
        Tensor a(size);
        Tensor b(size);
        
        // 初始化张量数据
        for (size_t j = 0; j < size; ++j) {
            // 使用公共接口初始化数据
            a.data<float>()[j] = static_cast<float>(j);
            b.data<float>()[j] = static_cast<float>(j * 2);
        }
        
        // 性能测试
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t j = 0; j < num_ops; ++j) {
            Tensor c = a + b; // 使用调度器执行加法操作
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        double avg_time = static_cast<double>(duration) / num_ops;
        uint64_t ops_per_sec = duration > 0 ? static_cast<uint64_t>(static_cast<double>(num_ops) / duration * 1000) : 0;
        uint64_t elems_per_sec = ops_per_sec * size;
        
        Ctorch_Error::trace(ErrorPlatform::kGENERAL, std::to_string(size) + "\t" + std::to_string(duration) + "\t\t" + std::to_string(avg_time) + "\t\t" + std::to_string(ops_per_sec) + "\t\t" + std::to_string(elems_per_sec));
    }
}

int main() {
    // 测试调度器功能
    test_scheduler_functionality();
    
    // 测试大量加法操作性能
    test_add_performance();
    
    // 测试不同大小张量的加法性能
    test_add_performance_by_size();
    
    return 0;
}