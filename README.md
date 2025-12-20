# Simple Memory

基于 LanceDB 的智能记忆仓库 MCP (Model Context Protocol) 服务器。类似于 mem0，提供记忆存储、语义搜索和 AI 整理功能。

## 功能特性

- **智能记忆处理**: 使用 LLM 自动整理和提取关键信息，生成结构化的记忆条目
- **语义搜索**: 基于向量嵌入的语义搜索，找到最相关的记忆内容
- **MCP 集成**: 通过 MCP 协议与 Claude 等 AI 助手无缝集成
- **LanceDB 存储**: 使用高性能的 LanceDB 作为向量数据库
- **Web 管理界面**: 使用 Jinja2 模板实现的 Web 界面，查看和管理记忆
- **灵活的嵌入模型**: 支持 OpenAI API 和 Ollama 本地模型

## 安装

```bash
# 使用 pip
pip install -e .

# 或使用 uv
uv pip install -e .
```

## 快速开始

### 1. 启动 Web 界面

```bash
# 使用命令行
simple-memory web

# 或使用 Python
python -m simple_memory.cli web

# 指定端口
simple-memory web -p 8080
```

访问 http://localhost:8765 进入 Web 管理界面。

### 2. 配置系统

在 Web 界面的"设置"页面配置：

**LLM 配置** (用于整理记忆):
- OpenAI / 兼容 API：填写 API URL、API Key 和模型名称
- Ollama：填写 Ollama 地址和模型名称（支持自动下载）

**嵌入模型配置** (用于生成向量):
- OpenAI：使用 `text-embedding-3-small` 等模型
- Ollama：使用 `nomic-embed-text`、`mxbai-embed-large` 等模型

配置完成后点击"测试连接"验证配置是否正确。

### 3. 配置 MCP 客户端

将以下配置添加到你的 Claude Desktop 配置文件：

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "simple-memory": {
      "command": "python",
      "args": ["-m", "simple_memory.mcp_server"]
    }
  }
}
```

或使用 uv：

```json
{
  "mcpServers": {
    "simple-memory": {
      "command": "uv",
      "args": ["run", "simple-memory-mcp"]
    }
  }
}
```

## MCP 工具

配置完成后，Claude 可以使用以下工具：

| 工具 | 描述 |
|------|------|
| `add_memory` | 添加新记忆到知识库 |
| `search_memories` | 语义搜索相关记忆 |
| `get_memory` | 获取指定 ID 的记忆 |
| `list_memories` | 列出所有记忆 |
| `update_memory` | 更新现有记忆 |
| `delete_memory` | 删除指定记忆 |
| `get_memory_stats` | 获取记忆统计信息 |

### 使用示例

在 Claude 中：

```
请记住：我喜欢用 Python 编程，常用的框架是 FastAPI 和 Flask。

搜索关于 Python 的记忆。

列出我所有的记忆。
```

## 配置选项

### 环境变量

配置文件位于 `~/.simple_memory/config.json`，也可以通过 Web 界面修改。

### LLM 配置

```json
{
  "llm": {
    "provider": "openai",
    "api_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model": "gpt-4o-mini",
    "ollama_host": "http://localhost:11434"
  }
}
```

### 嵌入模型配置

```json
{
  "embedding": {
    "provider": "openai",
    "api_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model": "text-embedding-3-small",
    "dimensions": 1536,
    "ollama_host": "http://localhost:11434",
    "ollama_model": "nomic-embed-text"
  }
}
```

## 使用 Ollama

如果你想使用本地模型，可以配置 Ollama：

1. 安装 Ollama: https://ollama.ai
2. 启动 Ollama 服务
3. 在 Web 设置中选择 Ollama 作为提供商
4. 填写模型名称（如 `llama3.2` 或 `nomic-embed-text`）
5. 如果模型不存在，系统会自动下载

推荐的 Ollama 模型：
- LLM: `llama3.2`, `qwen2.5`, `mistral`
- 嵌入: `nomic-embed-text`, `mxbai-embed-large`

## 项目结构

```
simple_memory/
├── src/simple_memory/
│   ├── __init__.py
│   ├── cli.py          # CLI 入口点
│   ├── config.py       # 配置管理
│   ├── database.py     # LanceDB 操作
│   ├── embeddings.py   # 嵌入模型
│   ├── llm.py          # LLM 接口
│   ├── memory.py       # 记忆管理核心
│   ├── mcp_server.py   # MCP 服务器
│   └── web/
│       ├── app.py      # Flask 应用
│       ├── templates/  # Jinja2 模板
│       └── static/     # CSS/JS
├── data/               # LanceDB 数据目录
├── pyproject.toml
└── README.md
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black src/
ruff check src/
```

## 许可证

MIT License
