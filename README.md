# Ctorch - 一个轻量级 C++ 深度学习框架

<picture>
  <source srcset="images/logo-dark.png" media="(prefers-color-scheme: dark)">
  <img src="images/logo.png" alt="Ctorch Logo">
</picture>

## 项目简介

Ctorch 是一个轻量级 C++ 深度学习框架，用现代 C++ 实现。项目由**笙歌@ShengFlow团队**开发，目标是创建一个类似 PyTorch 的接口，让 C++ 开发者也能享受简单直观的深度学习体验。目前项目处于 RC1 版本开发阶段，已实现核心架构和基础功能。

```cpp
// 简单示例：创建张量和自动微分
#include "Tensor.h"

int main() {
    // 创建张量
    Tensor a({1.0, 2.0, 3.0});
    Tensor b({4.0, 5.0, 6.0});

    // 自动微分计算
    a.requires_grad(true);
    b.requires_grad(true);
    
    Tensor c = a * b;
    Tensor d = c.sum();
    d.backward();

    std::cout << "a的梯度: " << a << std::endl;
    std::cout << "b的梯度: " << b << std::endl;
    return 0;
}
```

## 为什么选择 Ctorch？

作为一个轻量级深度学习框架，Ctorch 具有以下优势：

- **简洁易用**：提供类似 PyTorch 的直观接口，降低 C++ 深度学习开发的门槛
- **高效灵活**：核心计算优化，支持多设备扩展
- **模块化设计**：清晰的代码结构，易于理解和扩展
- **自动微分**：完整的自动微分系统，支持复杂计算图
- **现代 C++**：使用现代 C++ 特性，代码风格简洁优雅

## 已实现功能

### 核心特性

✅ 多维张量（Tensor）支持  
✅ 自动微分（Autograd）系统  
✅ 算子调度器（Scheduler）  
✅ 多设备支持接口（预留）  
✅ 统一的算子注册和调用机制

### 基础算子

✅ 加法（Add）  
✅ 减法（Sub）  
✅ 乘法（Mul）  
✅ 除法（Div）  
✅ 矩阵乘法（MatMul）  
✅ 一元负号（Neg）

### 张量运算示例

```cpp
// 基本运算
Tensor t(1.0);             // 创建标量张量
Tensor t1({1.0, 2.0, 3.0}); // 创建 1D 张量
Tensor t2(ShapeTag{}, {2, 2}); // 创建 2x2 张量（默认零初始化）

// 算术运算
Tensor t3 = t1 + 5;         // 广播加法
Tensor t4 = t1 * t1;        // 元素级乘法
Tensor t5 = t2.matmul(t2);  // 矩阵乘法

// 激活函数
Tensor t6 = t5.relu();      // ReLU 激活

// 自动微分完整流程
Tensor x({1.0, 2.0, 3.0});
x.requires_grad(true);       // 开启自动微分

Tensor y = x * x;
Tensor z = y.sum();

z.backward();                // 反向传播

// 此时 x 已包含梯度信息
```

## 安装与使用

### 依赖项

- C++17 或更高版本
- CMake 3.12+

### 构建步骤

```bash
git clone https://github.com/Beapoe/CTorch.git
cd CTorch/src/dev
mkdir build
cd build
cmake ..
make
```

### 运行测试

```bash
./Ctorch_test
```

### 集成到自己的项目

1. 将 Ctorch 的头文件和库文件添加到你的项目中
2. 在 CMakeLists.txt 中添加依赖
3. 包含必要的头文件并使用 Ctorch API

## 项目结构

```
Ctorch/
├── docs/              # 文档
│   ├── html/          # API 文档（Doxygen 生成）
│   │   └── index.html # API 文档入口文件
│   └── API_Guide.md   # API 使用指南
├── images/            # 图片资源
├── include/           # 头文件（开发版）
├── src/               # 源代码
│   └── dev/           # 开发目录
│       ├── kernels/   # 算子实现
│       │   ├── kernels.h        # 算子统一声明
│       │   └── CPU-BASIC/       # CPU 基础实现
│       │       ├── Add_BASIC_kernel.cpp
│       │       ├── Sub_BASIC_kernel.cpp
│       │       ├── Mul_BASIC_kernel.cpp
│       │       ├── Div_BASIC_kernel.cpp
│       │       ├── MatMul_BASIC_kernel.cpp
│       │       └── Neg_BASIC_kernel.cpp
│       ├── tests/     # 测试代码
│       ├── Tensor.cpp # 张量实现
│       ├── Tensor.h   # 张量头文件
│       ├── AutoDiff.cpp # 自动微分实现
│       ├── AutoDiff.h   # 自动微分头文件
│       ├── Ctorch_Scheduler.h # 调度器头文件
│       └── CMakeLists.txt # 构建配置
├── README.md          # 项目说明
└── LICENSE            # 许可证文件
```

## API 文档

完整的 API 文档已通过 Doxygen 生成，包含所有类、方法和函数的详细说明。

### 如何查看 API 文档

1. 打开项目根目录下的 `docs/html/index.html` 文件
2. 在浏览器中查看完整的 API 文档
3. 文档包含以下主要部分：
        - 类层次结构
        - Tensor 类的详细说明
        - 自动微分系统
        - 算子和函数
        - 示例代码

### 关键 API 快速参考

- **Tensor 构造函数**：支持标量、初始化列表、形状（使用ShapeTag）等多种方式创建张量
- **自动微分**：`requires_grad(true)` 开启梯度追踪，`backward()` 执行反向传播
- **算术运算**：支持 `+`, `-`, `*`, `/` 等运算符重载
- **矩阵操作**：`matmul()` 方法执行矩阵乘法
- **激活函数**：`relu()` 等方法执行激活操作

## 未来计划

### 短期目标（RC1 到正式版）

- [ ] 完善常用激活函数（Sigmoid、Tanh、Softmax 等）
- [ ] 实现基础损失函数（MSE、CrossEntropy 等）
- [ ] 完善自动微分系统的正确性验证
- [ ] 建立完整的测试框架
- [ ] 优化核心算子的性能

### 中期目标

- [ ] 实现神经网络模块（Linear、Conv2d 等）
- [ ] 支持 CUDA 设备加速
- [ ] 实现优化器（SGD、Adam 等）
- [ ] 提供数据加载和预处理工具

### 长期目标

- [ ] 支持 ONNX 模型导入/导出
- [ ] 实现分布式训练支持
- [ ] 移动端部署优化
- [ ] 构建完整的深度学习生态系统

## 贡献指南

我们欢迎任何形式的贡献！如果你想参与 Ctorch 的开发：

1. **报告问题**：在 GitHub 上提交 Issue，描述你遇到的问题
2. **代码贡献**：Fork 仓库，修改代码，然后提交 Pull Request
3. **文档完善**：帮助改进文档和示例
4. **功能建议**：提出新功能或改进建议

请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细的贡献流程。

## 许可证

本项目采用 **MIT 许可证** - 详情见 [LICENSE](LICENSE) 文件。

## 联系方式

如果你有任何问题或建议，欢迎联系我们：
- QQ:1113109729,2713906889
- 提交issue
- 邮件：ctorch1024@163.com、cyf31415@yeah.net
- 抖音私信：@徽宗手写

---

> "遇事不决，可问春风，春风不语，即随本心."
>
> —— 烽火戏诸侯《剑来》

[![GitHub Stars](https://img.shields.io/github/stars/Beapoe/CTorch?style=social)](https://github.com/Beapoe/CTorch)