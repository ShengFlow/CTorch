# Ctorch：轻量级深度学习框架

@mainpage Ctorch 文档首页
@brief 高性能 C++ ML框架
@date 2025-12-21
@version RC v1.0

## 简介

Ctorch 是一个轻量级、高性能的 C++ 深度学习框架，专注于自动微分和张量运算。它提供了简洁易用的 API，支持多种算子和自动微分，适合科研和生产环境使用。

## 主要特性

-  自动微分（前向模式和反向模式）
-  高性能张量运算
-  支持多种算子（+、-、*、/、ReLU等）
-  计算图优化
-  内存高效管理
-  跨平台支持

## 快速开始

### 安装

```bash
git clone https://github.com/Beapoe/CTorch.git
cd CTorch/src/dev
mkdir build
cd build
cmake ..
make
```

### 基本使用

```cpp
#include "Tensor.h"
#include "AutoDiff.h"

int main() {
    // 设置输出级别为最详细
    Ctorch_Error::setPrintLevel(PrintLevel::FULL);
    // 创建自动微分上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);

    // 创建需要梯度的张量
    Tensor a(2.0f);
    Tensor b(3.0f);
    a.requires_grad(true);
    b.requires_grad(true);

    // 执行正向计算
    Tensor c = a + b;

    // 反向传播
    backward(c);

    // 获取梯度
    Tensor grad_a = grad(a);
    Tensor grad_b = grad(b);

    // 输出结果
    std::cout << "a = " << a.item<float>() << std::endl;
    std::cout << "b = " << b.item<float>() << std::endl;
    std::cout << "c = a + b = " << c.item<float>() << std::endl;
    std::cout << "∂c/∂a = " << grad_a.item<float>() << std::endl;
    std::cout << "∂c/∂b = " << grad_b.item<float>() << std::endl;
    
    // 获取统计信息，包括Error统计、Warn统计、计时等
    Ctorch_Error::stats();
    return 0;
}
```

## 核心概念

### 张量 (Tensor)
张量是库中的基本数据结构，支持标量、向量、矩阵及更高维数组。

### 自动微分上下文 (AutoDiff Context)
管理计算图的创建和反向传播过程，每个计算图需要一个上下文。

### 计算图 (Computation Graph)
记录张量操作的依赖关系，用于反向传播计算梯度。

### 梯度 (Gradient)
张量操作的导数，通过反向传播算法计算。

## API 文档

- [张量 (Tensor)](./class_tensor.html)
- [自动微分 (AutoDiff)](./class_auto_diff.html)
- [自动微分上下文 (AutoDiffContext)](./class_auto_diff_context.html)
- [错误处理 (Ctorch_Error)](./class_ctorch___error.html)
- [调度器 (Ctorch_Scheduler)](./class_ctorch___scheduler.html)
- [存储 (Storage)](./class_storage.html)

## 示例

- [基本测试](./test_8cpp.html)
- [激活函数和损失函数测试](./test__activation__loss_8cpp.html)
- [自动微分综合测试](./test__autodiff__comprehensive_8cpp.html)
- [线性回归测试](./test__linear__regression_8cpp.html)
- [张量梯度测试](./test__tensor__grad_8cpp.html)

## 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 仓库
2. 创建特性分支
3. 提交更改
4. 创建 Pull Request

## 许可证

MIT License

## 联系我们

- 项目地址：https://github.com/Beapoe/CTorch
- QQ：1113109729 2713906889
- Email：cyf31415@yeah.net beapoe1024@163.com