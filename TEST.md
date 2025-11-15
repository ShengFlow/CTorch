# 测试相关文件说明

### 测试组列表("group:"后可加)
| 测试组名称   | 说明            |  
|---------|---------------|  
| Storage | 存储Storage相关测试 |  


### 测试用例列表("single:"后可加)
| 测试组     | 测试用例名称                  | 说明              |  
|---------|-------------------------|-----------------|  
| Storage | Storage-BasicAttributes | 存储Storage基础属性测试 |  
| Storage | Storage-Initialization  | 存储Storage初始化测试  |  


**说明**：测试用例定义在 `src/dev/tests/` 目录下，每个子目录（如 `Storage`）对应一个测试组，目录内的 `.cpp` 文件为具体测试用例（名称去除 `.cpp` 后缀）。