#include "Tensor.h"
#include <iostream>
#include <chrono>

// 简单的调度器功能测试
void test_scheduler_functionality() {
    // 创建上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
    
    std::cout << "测试调度器功能..." << std::endl;
    
    // 测试1：基本标量加法
    std::cout << "\n1. 基本标量加法测试:" << std::endl;
    Tensor a(2.0f);
    Tensor b(3.0f);
    Tensor c = a + b; // 使用调度器执行加法
    std::cout << "a + b = " << c.item<float>() << " (预期: 5.0)" << std::endl;
    
    // 测试2：简单张量加法
    std::cout << "\n2. 简单张量加法测试:" << std::endl;
    Tensor d({2.0f, 3.0f});
    Tensor e({4.0f, 5.0f});
    Tensor f = d + e; // 使用调度器执行加法
    std::cout << "d + e = [" << f.data<float>()[0] << ", " << f.data<float>()[1] << "] (预期: [6.0, 8.0])" << std::endl;
    
    // 测试3：带自动微分的加法
    std::cout << "\n3. 带自动微分的加法测试:" << std::endl;
    Tensor x(1.0f);
    Tensor y(2.0f);
    x.requires_grad(true);
    y.requires_grad(true);
    Tensor z = x + y;
    backward(z);
    std::cout << "z = x + y = " << z.item<float>() << " (预期: 3.0)" << std::endl;
    std::cout << "dz/dx = " << grad(x).item<float>() << " (预期: 1.0)" << std::endl;
    std::cout << "dz/dy = " << grad(y).item<float>() << " (预期: 1.0)" << std::endl;
    
    std::cout << "\n✅ 调度器功能测试完成!" << std::endl;
}

// 性能测试：大量加法操作
void test_add_performance() {
    // 创建上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
    
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
    std::cout << "\n\n测试大量加法操作性能..." << std::endl;
    std::cout << "操作次数: " << num_ops << std::endl;
    std::cout << "张量大小: " << tensor_size << " 元素" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_ops; ++i) {
        Tensor c = a + b; // 使用调度器执行加法操作
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "总执行时间: " << duration << " ms" << std::endl;
    std::cout << "每次操作平均时间: " << static_cast<double>(duration) / num_ops << " ms" << std::endl;
    if (duration > 0) {
        std::cout << "每秒操作数: " << static_cast<uint64_t>(static_cast<double>(num_ops) / duration * 1000) << " ops/s" << std::endl;
    }
}

// 测试不同大小张量的加法性能
void test_add_performance_by_size() {
    // 创建上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
    
    // 测试不同大小的张量
    const size_t sizes[] = {10, 50, 100, 200, 500};
    const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const size_t num_ops = 100;
    
    std::cout << "\n\n测试不同大小张量的加法性能..." << std::endl;
    std::cout << "操作次数: " << num_ops << std::endl;
    std::cout << "\n尺寸\t总时间(ms)\t平均时间(ms)\t每秒操作数(ops/s)\t每秒处理元素数(elem/s)" << std::endl;
    
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
        
        std::cout << size << "\t" << duration << "\t\t" << avg_time << "\t\t" << ops_per_sec << "\t\t" << elems_per_sec << std::endl;
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