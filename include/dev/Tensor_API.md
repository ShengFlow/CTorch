### Tensor 库 API 文档 (Tensor.h(未审核))

#### **1. 枚举类型**
##### **`DeviceType`**
*定义张量数据存储位置*
- `kCPU`: 主存储器 (RAM)
- `kCUDA`: NVIDIA GPU (暂未实现)
- `kMPS`: Apple Silicon GPU (暂未实现)

##### **`DType`**
*定义张量元素数据类型*
- `kFloat`: 32位浮点数 (对应 torch.float32)
- `kDouble`: 64位浮点数 (对应 torch.float64)
- `kInt`: 32位整数 (对应 torch.int32)
- `kLong`: 64位整数 (对应 torch.int64)
- `kBool`: 布尔值

---

#### **2. 辅助函数**
##### **`dtypeToString(DType dtype) -> const char*`**
*将数据类型转为可读字符串*
```cpp
dtypeToString(DType::kFloat);  // 返回 "float32"
```

##### **`dtypeSize(DType dtype) -> size_t`**
*获取数据类型字节大小*
```cpp
dtypeSize(DType::kLong);  // 返回 sizeof(int64_t)，即8
```

---

#### **3. Storage 类**
*管理张量原始内存*

##### *构造函数*
```cpp
// 分配未初始化内存
Storage(size_t size, DType dtype, DeviceType device = DeviceType::kCPU);

// 从现有数据复制
template <typename T>
Storage(const T* data, size_t size, DType dtype, DeviceType device = DeviceType::kCPU);
```

##### *核心方法*
| 方法 | 描述 | 示例 |
|------|------|------|
| `data<T>() -> T*` | 获取类型化数据指针 | `float* ptr = storage.data<float>();` |
| `size() -> size_t` | 获取元素数量 | `size_t n = storage.size();` |
| `dtype() -> DType` | 获取数据类型 | `DType t = storage.dtype();` |
| `device() -> DeviceType` | 获取存储设备 | `DeviceType d = storage.device();` |
| `clone() -> Storage` | 创建深拷贝 | `Storage copy = storage.clone();` |
| `empty() -> bool` | 检查是否为空 | `if (storage.empty()) ...` |

---

#### **4. Tensor 类**
*实现多维张量操作*

##### *构造函数*
```cpp
Tensor();                          // 创建空张量
Tensor(float value);               // 创建标量张量
Tensor({1.0f, 2.0f, 3.0f});       // 创建一维浮点张量
Tensor({true, false, true});      // 创建一维布尔张量
Tensor(ShapeTag{}, {2, 3});       // 创建2x3未初始化张量
```

##### *核心属性*
| 方法 | 返回值 | 描述 |
|------|--------|------|
| `dim() -> size_t` | 维度数量 | `tensor.dim()` |
| `sizes() -> vector<size_t>` | 形状向量 | `auto shape = tensor.sizes();` |
| `numel() -> size_t` | 元素总数 | `size_t n = tensor.numel();` |
| `dtype() -> DType` | 数据类型 | `DType t = tensor.dtype();` |
| `device() -> DeviceType` | 设备类型 | `DeviceType d = tensor.device();` |

##### *数据访问*
```cpp
// 一维访问
float val = tensor[2];          // 获取第3个元素

// 多维访问
double d = tensor({1, 2, 3});   // 获取[1][2][3]位置元素

// 标量访问
float scalar = tensor.item();   // 0维张量取值

// 原始数据指针
float* data = tensor.data();    // 获取类型化指针
```

##### *张量操作*
| 方法 | 描述 | 示例 |
|------|------|------|
| `clone() -> Tensor` | 创建深拷贝 | `Tensor copy = tensor.clone();` |
| `view(vector<size_t>) -> Tensor` | 改变形状 (零拷贝) | `tensor.view({6});` |
| `transpose() -> Tensor` | 转置最后两维度 | `tensor.transpose();` |
| `operator+` | 逐元素加法 | `t3 = t1 + t2;` |
| `operator==` | 深度相等比较 | `if (t1 == t2) ...` |

##### *输出与调试*
```cpp
tensor.print();  // 打印格式化张量
/* 输出示例:
Tensor(shape=[2,3], dtype=float32, device=cpu)
[[1.00, 2.00, 3.00],
 [4.00, 5.00, 6.00]]
*/

std::string s = tensor.toString();  // 获取字符串表示
```

---

### **使用示例**

#### 1. **基础张量操作**
```cpp
// 创建3x2浮点张量
Tensor t(ShapeTag{}, {3, 2}, DType::kFloat);

// 初始化值
t({0,0}) = 1.5f; t({0,1}) = 2.5f;
t({1,0}) = 3.1f; t({1,1}) = 4.9f;
t({2,0}) = 5.0f; t({2,1}) = 6.0f;

t.print();  // 打印张量
```

#### 2. **张量运算**
```cpp
Tensor A({1.0f, 2.0f, 3.0f});
Tensor B({4.0f, 5.0f, 6.0f});

Tensor C = A + B;  // 逐元素加法
C.print();         // 输出: [5.0, 7.0, 9.0]

Tensor D = A.view({3, 1});  // 重塑为3x1
D.print();
```

#### 3. **高级功能**
```cpp
// 创建布尔张量
Tensor flags{true, false, true, false};

// 矩阵转置
Tensor matrix(ShapeTag{}, {2, 3});
// ... 初始化矩阵 ...
Tensor transposed = matrix.transpose();

// 张量比较
Tensor X({1.0f, 2.0f});
Tensor Y({1.0f, 2.0f});
if (X == Y) {
    std::cout << "张量相等\n";
}
```

---

### **内存管理说明**
1. **零拷贝视图**:
   - `view()` 操作共享底层存储
   - 修改视图会影响原始张量
   ```cpp
   Tensor original({1,2,3,4});
   Tensor view = original.view({2,2});
   view({0,1}) = 10; // 也会修改original
   ```

2. **深拷贝**:
   - 使用 `clone()` 创建独立副本
   ```cpp
   Tensor copy = original.clone();
   copy[0] = 100; // 不影响original
   ```

3. **自动内存管理**:
   - 使用 `shared_ptr` 自动释放内存
   - 最后一个引用离开作用域时释放存储

4. **类型安全**:
   - 所有数据访问进行运行时类型检查
   - 类型不匹配抛出 `runtime_error`
   ```cpp
   Tensor int_tensor(ShapeTag{}, {3}, DType::kInt);
   float* p = int_tensor.data<float>(); // 抛出异常!
   ```

> **设备支持**：当前仅实现CPU支持，CUDA/MPS为预留接口

---
