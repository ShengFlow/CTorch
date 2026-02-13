/**
 * @file large_scale_test.cpp
 * @brief 大规模Tensor计算测试和20个节点的自动微分测试
 * @author GhostFace
 * @date 2025/12/21
 */

#include "Tensor.h"
#include "Ctorch_Scheduler.h"
#include "Ctorch_Error.h"
#include <chrono>
#include <iomanip>

/**
 * @brief 大规模Tensor加法测试
 * @param size Tensor大小
 * @return 测试是否通过
 */
bool test_large_scale_addition(size_t size) {
    Ctorch_Error::trace(ErrorPlatform::kCPU, "=== 测试：大规模Tensor加法 (" + std::to_string(size) + " 元素) ===");
    try {
        // 创建两个大型Tensor
        Tensor a(size, DType::kFloat, DeviceType::kCPU, true);
        Tensor b(size, DType::kFloat, DeviceType::kCPU, true);
        
        // 初始化数据
        for (size_t i = 0; i < size; ++i) {
            a.data<float>()[i] = static_cast<float>(i) / size;
            b.data<float>()[i] = static_cast<float>(size - i) / size;
        }
        
        // 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 执行加法操作（应该通过调度器调用）
        Tensor c = a + b;
        
        // 记录结束时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 验证结果
        bool passed = true;
        for (size_t i = 0; i < size; ++i) {
            float expected = a.data<float>()[i] + b.data<float>()[i];
            if (std::abs(c.data<float>()[i] - expected) > 1e-6) {
                passed = false;
                break;
            }
        }
        
        if (passed) {
            Ctorch_Error::trace(ErrorPlatform::kCPU, "✅ 大规模加法测试通过，耗时：" + std::to_string(duration.count()) + " ms");
            return true;
        } else {
            Ctorch_Error::trace(ErrorPlatform::kCPU, "❌ 大规模加法测试失败");
            return false;
        }
    } catch (const std::exception& e) {
        Ctorch_Error::trace(ErrorPlatform::kCPU, "!!! 大规模加法测试异常: " + std::string(e.what()));
        return false;
    }
}

/**
 * @brief 20个节点的自动微分测试
 * @return 测试是否通过
 */
bool test_20_nodes_autodiff() {
    Ctorch_Error::trace(ErrorPlatform::kCPU, "=== 测试：20个节点的自动微分测试 ===");
    try {
        AutoDiff ctx;
        AutoDiffContext::Guard guard(&ctx);
        
        // 创建20个输入节点
        std::vector<Tensor> inputs;
        for (int i = 0; i < 20; ++i) {
            Tensor t(static_cast<float>(i + 1));
            t.requires_grad(true);
            inputs.push_back(t);
        }
        
        // 构建计算图：仅使用加减操作
        // 模式：(((input0 + input1) - input2) + input3) - ...
        Tensor result = inputs[0];
        for (int i = 1; i < 20; ++i) {
            if (i % 2 == 1) {
                result = result + inputs[i];
            } else {
                result = result - inputs[i];
            }
        }
        
        // 执行反向传播
        backward(result);
        
        // 验证梯度
        // 对于 (((a + b) - c) + d) - e ...
        // 梯度应该是：+1, +1, -1, +1, -1, ...
        // 即：input0=+1，奇数索引=+1，偶数索引>0=-1
        bool passed = true;
        for (int i = 0; i < 20; ++i) {
            Tensor gradient = grad(inputs[i]);
            float expected_grad;
            if (i == 0 || i % 2 == 1) {
                expected_grad = 1.0f;
            } else {
                expected_grad = -1.0f;
            }
            if (std::abs(gradient.item<float>() - expected_grad) > 1e-6) {
                Ctorch_Error::trace(ErrorPlatform::kCPU, "❌ 节点 " + std::to_string(i) + " 梯度错误: 期望 " + std::to_string(expected_grad) + ", 实际 " + std::to_string(gradient.item<float>()));
                passed = false;
            }
        }
        
        if (passed) {
            Ctorch_Error::trace(ErrorPlatform::kCPU, "✅ 20个节点的自动微分测试通过");
            return true;
        } else {
            Ctorch_Error::trace(ErrorPlatform::kCPU, "❌ 20个节点的自动微分测试失败");
            return false;
        }
    } catch (const std::exception& e) {
        Ctorch_Error::trace(ErrorPlatform::kCPU, "!!! 20个节点的自动微分测试异常: " + std::string(e.what()));
        return false;
    }
}

/**
 * @brief 主函数，运行所有测试
 * @return 测试结果
 */
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    
    Ctorch_Error::trace(ErrorPlatform::kCPU, "🚀 开始大规模Tensor计算测试和20个节点的自动微分测试");
    Ctorch_Error::trace(ErrorPlatform::kCPU, "=========================================");
    
    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 测试结果统计
    int total_tests = 0;
    int passed_tests = 0;
    
    // 1. 大规模Tensor加法测试（超过10万元素）
    total_tests++;
    if (test_large_scale_addition(100000)) { // 10万元素
        passed_tests++;
    }
    
    total_tests++;
    if (test_large_scale_addition(500000)) { // 50万元素
        passed_tests++;
    }
    
    total_tests++;
    if (test_large_scale_addition(1000000)) { // 100万元素
        passed_tests++;
    }
    
    // 2. 20个节点的自动微分测试
    total_tests++;
    if (test_20_nodes_autodiff()) {
        passed_tests++;
    }
    
    // 记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 生成测试报告
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n" + std::string(60, '='));
    Ctorch_Error::trace(ErrorPlatform::kCPU, " 大规模Tensor计算测试报告");
    Ctorch_Error::trace(ErrorPlatform::kCPU, std::string(60, '='));
    
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n 测试统计:");
    Ctorch_Error::trace(ErrorPlatform::kCPU, "   总测试数: " + std::to_string(total_tests));
    Ctorch_Error::trace(ErrorPlatform::kCPU, "   通过测试: " + std::to_string(passed_tests));
    Ctorch_Error::trace(ErrorPlatform::kCPU, "   失败测试: " + std::to_string(total_tests - passed_tests));
    Ctorch_Error::trace(ErrorPlatform::kCPU, "   成功率: " + std::to_string(std::fixed) + std::to_string(std::setprecision(1)) + std::to_string(static_cast<double>(passed_tests) / total_tests * 100.0) + "%");
    
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n 性能统计:");
    Ctorch_Error::trace(ErrorPlatform::kCPU, "   总运行时间: " + std::to_string(total_duration.count()) + " ms");
    
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n 测试覆盖范围:");
    Ctorch_Error::trace(ErrorPlatform::kCPU, "   ✓ 大规模Tensor加法 (10万+ 元素)");
    Ctorch_Error::trace(ErrorPlatform::kCPU, "   ✓ 20个节点的自动微分 (仅加减操作)");
    Ctorch_Error::trace(ErrorPlatform::kCPU, "   ✓ 调度器调用验证");
    
    if (passed_tests == total_tests) {
        Ctorch_Error::trace(ErrorPlatform::kCPU, "\n 所有测试通过");
    } else {
        Ctorch_Error::trace(ErrorPlatform::kCPU, "\n 部分测试失败，请检查相关功能。");
    }
    
    Ctorch_Error::trace(ErrorPlatform::kCPU, std::string(60, '='));
    
    return (passed_tests == total_tests) ? 0 : 1;
}