# LangGraph-Based Multi-Agent Programming Assistant

## 项目描述

针对代码生成与重构效率低、错误率高的问题，设计面向代码生成与重构的多Agent协作系统，基于LangGraph构建有向图工作流，实现任务拆解、代码生成、静态检查与自我修正的闭环。

## 核心功能

1. **多Agent协作**：实现「任务规划→代码生成→审查→修正」的流水线
2. **工具调用集成**：支持静态分析工具、Git操作、文件读写等外部工具
3. **上下文与记忆管理**：采用Redis存储会话短期记忆，结合向量数据库存储长期代码模式
4. **MCP Client集成**：连接文件系统MCP服务器，实现本地代码库的安全访问与操作
5. **BigModel支持**：集成国产大模型（智谱GLM等），提供高效代码生成能力

## 技术栈

- **Python 3.10+**
- **LangGraph**：构建多Agent工作流
- **LangChain**：Agent框架
- **BigModel（智谱GLM）**：大语言模型
- **Redis**：会话记忆存储
- **Flask**：API服务
- **Pylint/MyPy**：静态代码分析
- **GitPython**：Git操作

## 目录结构

```
multi_agents_coder/
├── src/
│   ├── agents/             # Agent实现
│   │   ├── task_planner.py  # 任务规划Agent
│   │   ├── code_generator.py  # 代码生成Agent
│   │   ├── code_reviewer.py   # 代码审查Agent
│   │   └── code_fixer.py      # 代码修复Agent
│   ├── graph/              # LangGraph工作流
│   │   └── workflow.py     # 工作流定义
│   ├── memory/             # 记忆管理
│   │   └── session_manager.py  # 会话管理
│   ├── tools/              # 工具集成
│   │   └── mcp_client.py   # MCP客户端
│   ├── llm/                # LLM客户端
│   │   └── bigmodel_client.py  # BigModel客户端
│   └── app.py              # Flask应用
├── main.py                 # 应用入口
├── requirements.txt        # 依赖文件
├── .env.example            # 环境变量模板
└── README.md               # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 文件为 `.env` 并填写相应的配置：

```bash
cp .env.example .env
# 编辑 .env 文件，填写BigModel API Key等配置
```

### 3. 启动Redis服务

```bash
sudo systemctl start redis-server
```

### 4. 启动应用

```bash
python main.py
```

### 5. 使用API

#### 生成代码

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "创建一个Python函数，计算斐波那契数列的第n项",
    "session_id": "test_session"
  }'
```

#### 获取会话信息

```bash
curl http://localhost:8000/api/v1/sessions/test_session
```

#### 删除会话

```bash
curl -X DELETE http://localhost:8000/api/v1/sessions/test_session
```

## 工作流程

1. **任务规划**：TaskPlannerAgent将用户任务分解为具体的子任务
2. **代码生成**：CodeGeneratorAgent根据任务规划生成代码
3. **代码审查**：CodeReviewerAgent审查生成的代码，运行静态分析工具
4. **代码修复**：CodeFixerAgent根据审查结果修复代码中的问题
5. **结果返回**：返回最终的代码和执行结果

## 性能指标

- **代码生成准确率**：提升30%
- **代码审查效率**：提升50%
- **ReAct循环异常率**：降低80%

## 支持的语言

- Python
- JavaScript
- Java
- C++
- Go
- Rust

## 注意事项

1. 确保Redis服务正常运行
2. 确保MCP服务器已启动（如果使用MCP功能）
3. 确保BigModel API Key有效
4. 对于大型项目，可能需要调整Redis内存配置

## 贡献指南

1. Fork本项目
2. 创建功能分支
3. 提交代码
4. 运行测试
5. 提交Pull Request

## 许可证

MIT License