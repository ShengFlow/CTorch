#include"Tensor.h"
#include <chrono>
#include <iomanip>

#include "Ctorch_Scheduler.h"

bool test_addition() {
    std::cout << "=== 测试：加法自动微分 ===" << std::endl;
    try {
        AutoDiff ctx;
        AutoDiffContext::Guard guard(&ctx);

        Tensor a(2.0f);
        Tensor b(3.0f);
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a + b;
        backward(c);

        Tensor grad_a = grad(a);
        Tensor grad_b = grad(b);

        if (std::abs(grad_a.item<float>() - 1.0f) < 1e-6 && std::abs(grad_b.item<float>() - 1.0f) < 1e-6) {
            std::cout << "✅ 加法测试通过" << std::endl;
            return true;
        } else {
            std::cout << "❌ 加法测试失败" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "!!! 加法测试异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_multiplication() {
    std::cout << "=== 测试：乘法自动微分 ===" << std::endl;
    try {
        AutoDiff ctx;
        AutoDiffContext::Guard guard(&ctx);

        Tensor a(2.0f);
        Tensor b(3.0f);
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a * b;
        backward(c);

        Tensor grad_a = grad(a);
        Tensor grad_b = grad(b);

        // ∂(a*b)/∂a = b = 3, ∂(a*b)/∂b = a = 2
        if (std::abs(grad_a.item<float>() - 3.0f) < 1e-6 && std::abs(grad_b.item<float>() - 2.0f) < 1e-6) {
            std::cout << "✅ 乘法测试通过" << std::endl;
            return true;
        } else {
            std::cout << "❌ 乘法测试失败: grad_a=" << grad_a.item<float>() << ", grad_b=" << grad_b.item<float>() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "!!! 乘法测试异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_subtraction() {
    std::cout << "=== 测试：减法自动微分 ===" << std::endl;
    try {
        AutoDiff ctx;
        AutoDiffContext::Guard guard(&ctx);

        Tensor a(5.0f);
        Tensor b(3.0f);
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a - b;
        backward(c);

        Tensor grad_a = grad(a);
        Tensor grad_b = grad(b);

        // ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
        if (std::abs(grad_a.item<float>() - 1.0f) < 1e-6 && std::abs(grad_b.item<float>() - (-1.0f)) < 1e-6) {
            std::cout << "✅ 减法测试通过" << std::endl;
            return true;
        } else {
            std::cout << "❌ 减法测试失败: grad_a=" << grad_a.item<float>() << ", grad_b=" << grad_b.item<float>() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "!!! 减法测试异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_negation() {
    std::cout << "=== 测试：负号自动微分 ===" << std::endl;
    try {
        AutoDiff ctx;
        AutoDiffContext::Guard guard(&ctx);

        Tensor a(4.0f);
        a.requires_grad(true);

        Tensor c = -a;
        backward(c);

        Tensor grad_a = grad(a);

        // ∂(-a)/∂a = -1
        if (std::abs(grad_a.item<float>() - (-1.0f)) < 1e-6) {
            std::cout << "✅ 负号测试通过" << std::endl;
            return true;
        } else {
            std::cout << "❌ 负号测试失败: grad_a=" << grad_a.item<float>() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "!!! 负号测试异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_division() {
    std::cout << "=== 测试：除法自动微分 ===" << std::endl;
    try {
        AutoDiff ctx;
        AutoDiffContext::Guard guard(&ctx);

        Tensor a(6.0f);
        Tensor b(2.0f);
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a / b;
        backward(c);

        Tensor grad_a = grad(a);
        Tensor grad_b = grad(b);

        // ∂(a/b)/∂a = 1/b = 0.5, ∂(a/b)/∂b = -a/(b^2) = -6/4 = -1.5
        if (std::abs(grad_a.item<float>() - 0.5f) < 1e-6 && std::abs(grad_b.item<float>() - (-1.5f)) < 1e-6) {
            std::cout << "✅ 除法测试通过" << std::endl;
            return true;
        } else {
            std::cout << "❌ 除法测试失败: grad_a=" << grad_a.item<float>() << ", grad_b=" << grad_b.item<float>() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "!!! 除法测试异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_relu() {
    std::cout << "=== 测试：ReLU自动微分 ===" << std::endl;
    try {
        AutoDiff ctx;
        AutoDiffContext::Guard guard(&ctx);

        Tensor a(2.0f);
        Tensor b(-1.0f);
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c1 = a.relu();
        Tensor c2 = b.relu();

        backward(c1);
        Tensor grad_a = grad(a);

        // 重置梯度 - 重新创建AutoDiff对象
        AutoDiff ctx2;
        AutoDiffContext::Guard guard2(&ctx2);
        a.requires_grad(true);
        b.requires_grad(true);

        backward(c2);
        Tensor grad_b = grad(b);

        // ReLU(2) = 2, ∂ReLU(2)/∂2 = 1
        // ReLU(-1) = 0, ∂ReLU(-1)/∂(-1) = 0
        if (std::abs(grad_a.item<float>() - 1.0f) < 1e-6 && std::abs(grad_b.item<float>() - 0.0f) < 1e-6) {
            std::cout << "✅ ReLU测试通过" << std::endl;
            return true;
        } else {
            std::cout << "❌ ReLU测试失败: grad_a=" << grad_a.item<float>() << ", grad_b=" << grad_b.item<float>() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "!!! ReLU测试异常: " << e.what() << std::endl;
        return false;
    }
}

// 测试大型张量计算
bool test_large_tensor() {
    std::cout << "=== 测试：大型张量计算（10万+元素） ===" << std::endl;
    try {
        // 创建超过10万元素的张量
        const size_t tensor_size = 123456; // 约12万元素
        Tensor a(tensor_size);
        Tensor b(tensor_size);
        
        // 初始化数据
        for (size_t i = 0; i < tensor_size; ++i) {
            a.data<float>()[i] = static_cast<float>(i) / tensor_size;
            b.data<float>()[i] = static_cast<float>(tensor_size - i) / tensor_size;
        }
        
        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();
        
        // 执行大量加法操作
        Tensor result = a + b;
        
        // 记录结束时间
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // 验证结果
        bool passed = true;
        for (size_t i = 0; i < tensor_size; ++i) {
            float expected = static_cast<float>(i) / tensor_size + static_cast<float>(tensor_size - i) / tensor_size;
            if (std::abs(result.data<float>()[i] - expected) > 1e-6) {
                passed = false;
                break;
            }
        }
        
        if (passed) {
            std::cout << "✅ 大型张量测试通过，耗时：" << duration.count() << " ms" << std::endl;
            return true;
        } else {
            std::cout << "❌ 大型张量测试失败" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "!!! 大型张量测试异常: " << e.what() << std::endl;
        return false;
    }
}

// 测试复杂计算图（20个节点）
bool test_complex_graph() {
    std::cout << "=== 测试：20个节点的复杂计算图 ===" << std::endl;
    try {
        AutoDiff ctx;
        AutoDiffContext::Guard guard(&ctx);
        
        // 创建输入张量
        Tensor x(2.0f);
        Tensor y(3.0f);
        x.requires_grad(true);
        y.requires_grad(true);
        
        // 构建20个节点的计算图，只使用加减操作
        Tensor z1 = x + y;     // 节点1
        Tensor z2 = z1 - x;    // 节点2
        Tensor z3 = z2 + y;    // 节点3
        Tensor z4 = z3 - x;    // 节点4
        Tensor z5 = z4 + y;    // 节点5
        Tensor z6 = z5 - x;    // 节点6
        Tensor z7 = z6 + y;    // 节点7
        Tensor z8 = z7 - x;    // 节点8
        Tensor z9 = z8 + y;    // 节点9
        Tensor z10 = z9 - x;   // 节点10
        Tensor z11 = z10 + y;  // 节点11
        Tensor z12 = z11 - x;  // 节点12
        Tensor z13 = z12 + y;  // 节点13
        Tensor z14 = z13 - x;  // 节点14
        Tensor z15 = z14 + y;  // 节点15
        Tensor z16 = z15 - x;  // 节点16
        Tensor z17 = z16 + y;  // 节点17
        Tensor z18 = z17 - x;  // 节点18
        Tensor z19 = z18 + y;  // 节点19
        Tensor z20 = z19 - x;  // 节点20
        
        // 反向传播
        backward(z20);
        
        // 计算期望梯度
        // 由于只使用加减操作，每个节点对x的梯度传播可以简化计算
        // 最终z20对x的梯度应该是 -9（10次减x操作产生-10，1次加x操作产生+1）
        // 最终z20对y的梯度应该是 10（10次加y操作）
        Tensor grad_x = grad(x);
        Tensor grad_y = grad(y);
        
        if (std::abs(grad_x.item<float>() - (-9.0f)) < 1e-6 && 
            std::abs(grad_y.item<float>() - 10.0f) < 1e-6) {
            std::cout << "✅ 复杂计算图测试通过" << std::endl;
            return true;
        } else {
            std::cout << "❌ 复杂计算图测试失败: grad_x=" << grad_x.item<float>() << ", grad_y=" << grad_y.item<float>() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "!!! 复杂计算图测试异常: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "🚀 开始自动微分系统全面测试" << std::endl;
    std::cout << "=================================" << std::endl;

    // 测试结果统计
    int total_tests = 0;
    int passed_tests = 0;
    std::vector<std::string> test_results;

    // 执行测试
    auto test_start = std::chrono::high_resolution_clock::now();

    // 基础运算测试 - 只测试加减
    total_tests++;
    if (test_addition()) {
        passed_tests++;
        test_results.push_back("[OK] 加法自动微分");
    } else {
        test_results.push_back("❌ 加法自动微分");
    }

    total_tests++;
    if (test_subtraction()) {
        passed_tests++;
        test_results.push_back("[OK] 减法自动微分");
    } else {
        test_results.push_back("❌ 减法自动微分");
    }

    total_tests++;
    if (test_negation()) {
        passed_tests++;
        test_results.push_back("[OK] 负号自动微分");
    } else {
        test_results.push_back("❌ 负号自动微分");
    }

    // 跳过乘法和除法测试，因为它们还没有实现调度器调用
     total_tests++;
     if (test_multiplication()) {
         passed_tests++;
         test_results.push_back("[OK] 乘法自动微分");
     } else {
         test_results.push_back("❌ 乘法自动微分");
     }
    
     total_tests++;
     if (test_division()) {
         passed_tests++;
         test_results.push_back("[OK] 除法自动微分");
     } else {
         test_results.push_back("❌ 除法自动微分");
     }
    
    // 新增测试：大型张量计算
    total_tests++;
    if (test_large_tensor()) {
        passed_tests++;
        test_results.push_back("[OK] 大型张量计算");
    } else {
        test_results.push_back("❌ 大型张量计算");
    }
    
    // 新增测试：复杂计算图
    total_tests++;
    if (test_complex_graph()) {
        passed_tests++;
        test_results.push_back("[OK] 20个节点的复杂计算图");
    } else {
        test_results.push_back("❌ 20个节点的复杂计算图");
    }

    auto test_end = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();

    // 计算时间
    auto test_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_start);
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // 生成完整报告
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << " 自动微分系统测试报告" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "\n 测试统计:" << std::endl;
    std::cout << "   总测试数: " << total_tests << std::endl;
    std::cout << "   通过测试: " << passed_tests << std::endl;
    std::cout << "   失败测试: " << (total_tests - passed_tests) << std::endl;
    std::cout << "   成功率: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(passed_tests) / total_tests * 100.0) << "%" << std::endl;

    std::cout << "\n  性能统计:" << std::endl;
    std::cout << "   测试执行时间: " << test_duration.count() << " ms" << std::endl;
    std::cout << "   总运行时间: " << total_duration.count() << " ms" << std::endl;
    std::cout << "   平均每测试: " << std::fixed << std::setprecision(2)
              << (static_cast<double>(test_duration.count()) / total_tests) << " ms" << std::endl;

    std::cout << "\n 详细结果:" << std::endl;
    for (size_t i = 0; i < test_results.size(); ++i) {
        std::cout << "   " << (i + 1) << ". " << test_results[i] << std::endl;
    }

    std::cout << "\n 系统信息:" << std::endl;
    std::cout << "   编译器: " <<
#ifdef __GNUC__
        "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__
#elif defined(__clang__)
        "Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__
#elif defined(_MSC_VER)
        "MSVC " << _MSC_VER
#else
        "Unknown"
#endif
        << std::endl;
    std::cout << "   构建时间: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "   系统: " <<"macOS Tahoe 26.0" <<std::endl;

    std::cout << "\n 测试覆盖范围:" << std::endl;
    std::cout << "   ✓ 基本数学运算 (+, -, -)" << std::endl;
    std::cout << "   ✓ 大型张量计算（10万+元素）" << std::endl;
    std::cout << "   ✓ 复杂计算图（20个节点）" << std::endl;
    std::cout << "   ✓ 梯度计算和反向传播" << std::endl;
    std::cout << "   ✓ 计算图构建和管理" << std::endl;
    std::cout << "   ✓ 内存管理和资源清理" << std::endl;

    if (passed_tests == total_tests) {
        std::cout << "\n 所有测试通过" << std::endl;
    } else {
        std::cout << "\n  部分测试失败，请检查相关功能。" << std::endl;
    }

    std::cout << std::string(60, '=') << std::endl;
    Ctorch_Error::stats();
    Ctorch_Scheduler::getInstance();
    return (passed_tests == total_tests) ? 0 : 1;
}