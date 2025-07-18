
### **CONTRIBUTING.md**
# 欢迎贡献 CTorch！

感谢您对 CTorch 的关注！您的参与将大大推动项目发展。  

## 🚀 贡献流程（通用步骤）

### 1. **准备环境**
```bash
git clone https://github.com/Beapoe/CTorch.git
cd CTorch
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=<你的 LibTorch 路径>  # 需提前下载 LibTorch
make -j4
```

### 2. **选择任务**
- 查看 [Issues](https://github.com/Beapoe/CTorch/issues)
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
- **关联 Issue**：在描述中添加 `Closes #XXX`
- **通过 CI**：确保所有测试通过

### 5.**.等待PR通过**
- 待PR审核通过后，您的贡献便可被合并至项目中
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