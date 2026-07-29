
### **CONTRIBUTING.md**
# 欢迎贡献 CTorch！

感谢您对 CTorch 的关注！您的参与将大大推动项目发展。  

## 🚀 贡献流程（通用步骤）

### 1. **准备环境**
```bash
git clone https://github.com/ShengFlow/CTorch.git
cd CTorch
mkdir build && cd build
cmake .. 
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