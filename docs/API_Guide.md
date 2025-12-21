# Ctorch 使用指南

## 1. 快速开始

### 1.1 安装
目前库采用CMake构建，可通过以下步骤编译：

```bash
mkdir -p build
cd build
cmake ..
make
```

### 1.2 基本示例

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

## 2. 核心概念

### 2.1 张量 (Tensor)
张量是库中的基本数据结构，支持标量、向量、矩阵及更高维数组。

### 2.2 自动微分上下文 (AutoDiff Context)
管理计算图的创建和反向传播过程，每个计算图需要一个上下文。

### 2.3 计算图 (Computation Graph)
记录张量操作的依赖关系，用于反向传播计算梯度。

### 2.4 梯度 (Gradient)
张量操作的导数，通过反向传播算法计算。

## 3. 核心 API 使用指南

### 3.1 张量创建

#### 3.1.1 标量张量
```cpp
// 创建标量张量
Tensor a(2.0f);          // float 标量
Tensor b(3.0);           // double 标量
Tensor c(4);             // int 标量
```

#### 3.1.2 多维张量
```cpp
// 创建指定形状的张量（当前版本需手动初始化数据）
Tensor a(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);

// 访问和修改张量数据
float* data = a.data<float>();
for (size_t i = 0; i < a.numel(); ++i) {
    data[i] = static_cast<float>(i);
}
```

### 3.2 基本运算

#### 3.2.1 算术运算
```cpp
Tensor a(2.0f);
Tensor b(3.0f);

// 加法
Tensor c = a + b;          // c = 5.0f

// 减法
Tensor d = a - b;          // d = -1.0f

// 乘法
Tensor e = a * b;          // e = 6.0f

// 除法
Tensor f = a / b;          // f = 0.666...

// 负号
Tensor g = -a;             // g = -2.0f
```

#### 3.2.2 激活函数
```cpp
Tensor a(-1.0f);
Tensor b(2.0f);

// ReLU 激活函数
Tensor relu_a = a.relu();  // relu_a = 0.0f
Tensor relu_b = b.relu();  // relu_b = 2.0f
```

### 3.3 梯度计算

#### 3.3.1 设置梯度需求
```cpp
Tensor a(2.0f);
a.requires_grad(true);  // 标记需要计算梯度
```

#### 3.3.2 反向传播
```cpp
// 简单反向传播（适用于标量输出）
backward(loss);

// 带梯度输出的反向传播（适用于张量输出）
Tensor grad_output(1.0f);
backward(loss, grad_output);
```

#### 3.3.3 获取梯度
```cpp
Tensor grad_a = grad(a);  // 获取张量 a 的梯度
float grad_value = grad_a.item<float>();  // 转换为标量值
```

## 4. 高级用法

### 4.1 复杂计算图

```cpp
AutoDiff ctx;
AutoDiffContext::Guard guard(&ctx);

// 创建输入张量
Tensor x(2.0f);
Tensor y(3.0f);
x.requires_grad(true);
y.requires_grad(true);

// 构建复杂计算图
Tensor z1 = x + y;
Tensor z2 = z1 * x;
Tensor z3 = z2 / y;

// 反向传播
backward(z3);

// 获取梯度
Tensor grad_x = grad(x);
Tensor grad_y = grad(y);
```

### 4.2 大型张量运算

```cpp
// 创建大型张量（100x100）
Tensor a(ShapeTag{}, {100, 100}, DType::kFloat, DeviceType::kCPU);
Tensor b(ShapeTag{}, {100, 100}, DType::kFloat, DeviceType::kCPU);

// 初始化数据
float* data_a = a.data<float>();
float* data_b = b.data<float>();
for (size_t i = 0; i < a.numel(); ++i) {
    data_a[i] = static_cast<float>(i) / a.numel();
    data_b[i] = static_cast<float>(a.numel() - i) / a.numel();
}

// 执行加法运算
Tensor c = a + b;
```

### 4.3 矩阵运算

```cpp
// 创建矩阵
Tensor a(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
Tensor b(ShapeTag{}, {3, 2}, DType::kFloat, DeviceType::kCPU);

// 矩阵乘法
Tensor c = matMul(a, b);

// 矩阵转置
Tensor a_t = a.t();  // 等价于 a.transpose(0, 1)
```

## 5. 常见问题

### 5.1 如何判断张量是否需要梯度？

```cpp
bool requires_grad = tensor.requires_grad();
```

### 5.2 如何查看张量的形状？

```cpp
const std::vector<size_t>& shape = tensor.shape();
std::cout << "Shape: [";
for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << shape[i];
}
std::cout << "]" << std::endl;
```

### 5.3 如何获取张量元素数量？

```cpp
size_t numel = tensor.numel();
```

### 5.4 如何处理梯度爆炸或梯度消失？

- 可以尝试使用不同的激活函数
- 调整学习率
- 考虑使用梯度裁剪等技术

## 6. 支持的算子

| 算子类型 | 支持的操作 |
|---------|------------|
| 算术运算 | `+`, `-`, `*`, `/`, `-`(负号) |
| 激活函数 | `relu()` |
| 矩阵运算 | `matMul()`, `transpose()`, `t()` |
| 张量操作 | `reshape()`, `broadcast_to()`, `clone()`, `to()` |

## 7. 错误处理

库中的错误信息已本地化，当出现错误时，会输出中文错误提示，例如：
- "无效维度"
- "期望float数据类型"
- "索引越界"
- "转置仅支持2D张量"

## 8. 性能优化建议

1. 对于频繁的张量操作，尽量复用张量对象
2. 大型张量运算时，考虑批量处理
3. 不需要梯度的张量，避免设置 `requires_grad(true)`
4. 复杂计算图中，合理使用 `set_retain_graph()` 控制计算图的保留

## 9. 版本信息

当前版本：RC Public 1.0
- 支持基本算术运算的自动微分
- 支持标量和多维张量
- 支持CPU、CUDA、MPS、AMX设备

---

通过以上指南，您可以快速上手使用 Ctorch-Test 自动微分库进行各种张量运算和自动微分计算。随着库的不断完善，将支持更多的算子和设备类型。