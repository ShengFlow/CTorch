
### **CONTRIBUTING.md**
# 欢迎贡献 CTorch！

感谢您对 CTorch 的关注！无论您是学生、开发者还是教育工作者，您的参与都将推动项目发展。  
本指南将帮助您快速上手贡献流程，请根据您的角色选择路径：

---

## 🧠 我能贡献什么？

###  **学生（深度学习系统学习者）**
- **文档优化**：修复教程错漏、补充代码注释
- **示例改进**：优化示例代码可读性（如添加更多注释）
- **新手任务**：处理标记为 `good-first-issue` 的简单问题（如文档链接修复）
- **学习笔记**：撰写技术解析（如“CTorch 张量内存管理机制”）

###  **开发者（有 C++ 经验）**
- **核心功能**：实现算子/内存管理优化（标记为 `enhancement` 的 Issue）
- **性能调优**：改进计算图执行效率或 GPU 利用率
- **测试覆盖**：添加单元测试（`tests/` 目录）或基准测试
- **代码审查**：参与 Pull Request 技术评审

###  **教育工作者（教学场景）**
- **课程设计**：提供基于 CTorch 的实验课大纲（如“手写识别实践”）
- **教学案例**：开发可运行的教学示例（需包含数据集+步骤说明）
- **习题设计**：在 `examples/challenges/` 添加编程挑战任务
- **教程录制**：制作 CTorch 功能演示视频（需开源许可）

---

## 🚀 贡献流程（通用步骤）

### 1. **准备环境**
```bash
git clone https://github.com/Beapoe/CTorch.git
cd CTorch
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=<你的 LibTorch 路径>  # 需提前下载 LibTorch
make -j4
```

> 详细依赖见 [INSTALL.md](INSTALL.md)，需确保 **C++17** 兼容 。

### 2. **选择任务**
- 查看 [Issues](https://github.com/Beapoe/CTorch/issues) 中标记：
  - `good-first-issue`（学生友好）
  - `help-wanted`（开发者优先）
  - `teaching-example`（教育工作者）
- **重要**：在 Issue 下留言申领任务，避免重复工作 。

### 3. **提交代码**
```bash
git checkout -b feat/your-feature-name   # 分支命名示例：feat/add-conv2d
# 编写代码/文档...
git commit -m "feat: 添加卷积层支持"     # 遵循 [Conventional Commits](https://www.conventionalcommits.org)
git push origin feat/your-feature-name
```

### 4. **发起 Pull Request**
- **描述清晰**：说明解决的问题、方案设计、测试结果
- **关联 Issue**：在描述中添加 `Closes #123`
- **通过 CI**：确保所有测试通过（GPU 测试需标注环境）

---

## ⚙️ 质量与规范

### 代码要求
- **C++ 风格**：遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
  - 使用 `clang-format` 格式化（配置文件见 `.clang-format`）
- **测试覆盖**：
  - 新增 C++ 代码需包含单元测试（`tests/` 目录）
  - 接口变更需更新 API 文档（`docs/api/`）
- **模块化**：避免深层继承，优先组合模式 。
- **所有的类（class）名、函数名均小写，如果过长，使用驼峰命名规范**
- **所有的宏、常量均全大写，并在前面加下划线**
- **中间变量均小写，过长在中间加下划线**