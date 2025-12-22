# Simple Memory

基于 LanceDB 的智能记忆仓库 MCP (Model Context Protocol) 服务器。类似于 mem0，提供记忆存储、语义搜索和 AI 整理功能。

## 功能特性

- **智能记忆处理**: 使用 LLM 自动整理和提取关键信息，生成中文总结和标签
- **双向量语义搜索**: 同时对原始内容和处理后内容进行向量化，融合搜索提高准确性
- **长文本自动切分**: 超长内容自动按句子边界切分，保持语义完整性
- **MCP 集成**: 支持 stdio 和 SSE 两种传输方式，可本地或远程访问
- **LanceDB 存储**: 使用高性能的 LanceDB 作为向量数据库
- **Web 管理界面**: 使用 Jinja2 模板实现的 Web 界面，查看和管理记忆
- **灵活的模型配置**: 支持 OpenAI API 和 Ollama 本地模型

## 安装

### 方式一：Docker 部署（推荐）

最快速的部署方式，克隆代码后一键启动：

```bash
# 克隆项目
git clone https://github.com/your-repo/simple-memory.git
cd simple-memory

# 启动服务
docker compose up -d

# 查看日志
docker compose logs -f
```

服务启动后：
- Web 管理界面：http://localhost:8765
- MCP SSE 端点：http://localhost:8766/sse

#### 使用 Ollama 本地模型（完全离线）

```bash
# 使用包含 Ollama 的配置启动
docker compose -f docker-compose.ollama.yml up -d

# 首次启动会自动下载模型（qwen2.5:3b 和 nomic-embed-text）
# 可能需要几分钟时间
```

然后在 Web 设置中配置 Ollama：
- LLM 提供商：Ollama
- Ollama Host：`http://ollama:11434`
- 模型：`qwen2.5:3b`
- 嵌入模型：`nomic-embed-text`

#### Docker 环境变量

可以通过环境变量自定义配置：

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  simple-memory:
    environment:
      - SIMPLE_MEMORY_WEB_PORT=8765
      - SIMPLE_MEMORY_SSE_PORT=8766
```

### 方式二：pip 安装

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

**LLM 配置** (用于整理记忆，生成中文总结和标签):
- OpenAI / 兼容 API：填写 API URL、API Key 和模型名称
- Ollama：填写 Ollama 地址和模型名称（支持自动下载）

**嵌入模型配置** (用于生成向量):
- OpenAI：使用 `text-embedding-3-small` 等模型
- Ollama：使用 `nomic-embed-text`、`mxbai-embed-large` 等模型

**搜索配置**:
- 最低相似度阈值：过滤低相似度结果（建议 50-70%）
- 最大距离阈值：向量距离过滤（建议 0.5-1.0）

配置完成后点击"测试连接"验证配置是否正确。

### 3. 配置 MCP 客户端

#### 本地模式 (stdio)

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

#### 远程模式 (SSE)

在服务器上启动 SSE 服务：

```bash
# 启动 SSE MCP 服务器
simple-memory serve -p 8766

# 或
python -m simple_memory.mcp_server --transport sse -p 8766
```

客户端配置：

```json
{
  "mcpServers": {
    "simple-memory": {
      "transport": "sse",
      "url": "http://YOUR_SERVER_IP:8766/sse"
    }
  }
}
```

## MCP 工具

配置完成后，Claude 可以使用以下工具：

| 工具 | 描述 |
|------|------|
| `add_memory` | 添加新记忆（自动中文总结、标签生成、长文本切分） |
| `search_memories` | 双向量融合语义搜索 |
| `get_memory` | 获取指定 ID 的记忆 |
| `list_memories` | 列出所有记忆 |
| `update_memory` | 更新现有记忆 |
| `delete_memory` | 删除指定记忆 |
| `get_memory_stats` | 获取记忆统计信息 |

### 工具参数详情

#### add_memory

```json
{
  "content": "要记忆的内容",
  "user_id": "用户ID（可选，默认 'default'）",
  "metadata": {},
  "chunk_long_text": true  // 是否自动切分长文本
}
```

#### search_memories

```json
{
  "query": "搜索关键词",
  "limit": 5,
  "user_id": "用户ID（可选）",
  "tags": ["标签1", "标签2"],
  "min_similarity": 50,  // 最低相似度阈值 (0-100)
  "fusion_weight": 0.5   // 融合权重：0=仅原文，1=仅总结，0.5=均衡
}
```

### 使用示例

在 Claude 中：

```
请记住：我喜欢用 Python 编程，常用的框架是 FastAPI 和 Flask。

搜索关于 Python 的记忆。

列出我所有的记忆。
```

## 核心特性详解

### 1. 中文总结和标签

系统使用 LLM 将记忆内容整理成中文总结，并提取 3-5 个中文标签：

```
输入：I love programming in Python. My favorite frameworks are FastAPI and Flask.
输出：
  - 总结：喜欢使用 Python 编程，最常用的框架是 FastAPI 和 Flask
  - 标签：["Python", "编程", "FastAPI", "Flask"]
```

### 2. 双向量融合搜索

每条记忆存储两个向量：
- `vector`: 处理后内容（中文总结）的向量
- `content_vector`: 原始内容的向量

搜索时同时查询两个向量，使用加权平均融合结果：

```
融合相似度 = 总结相似度 × weight + 原文相似度 × (1 - weight)
```

搜索结果包含三个相似度分数：
- `similarity`: 融合后的相似度
- `processed_similarity`: 总结内容相似度
- `content_similarity`: 原始内容相似度

### 3. 长文本自动切分

超过 1000 字符的内容会自动切分：
- 按句子边界切分，保持语义完整
- 切分块之间有 200 字符重叠，确保上下文连贯
- 每个块独立存储，带有 chunk 元数据

## 配置选项

### 配置文件

配置文件位于 `~/.simple_memory/config.json`，也可以通过 Web 界面修改。

### 完整配置示例

```json
{
  "llm": {
    "provider": "openai",
    "api_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model": "gpt-4o-mini",
    "ollama_host": "http://localhost:11434"
  },
  "embedding": {
    "provider": "openai",
    "api_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model": "text-embedding-3-small",
    "dimensions": 1536,
    "ollama_host": "http://localhost:11434",
    "ollama_model": "nomic-embed-text"
  },
  "search": {
    "min_similarity": 50.0,
    "distance_threshold": 1.0
  },
  "database": {
    "path": "./data/lancedb",
    "table_name": "memories"
  },
  "web": {
    "host": "0.0.0.0",
    "port": 8765,
    "debug": false
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
- 嵌入: `nomic-embed-text` (768维), `mxbai-embed-large` (1024维)

## 部署说明

### Docker 部署（推荐）

```bash
# 基础部署（使用外部 API）
docker compose up -d

# 完整本地部署（包含 Ollama）
docker compose -f docker-compose.ollama.yml up -d

# 停止服务
docker compose down

# 查看日志
docker compose logs -f simple-memory

# 重新构建镜像
docker compose build --no-cache
```

### 单机部署（非 Docker）

```bash
# 启动 Web 管理界面
simple-memory web

# 在另一个终端启动 MCP SSE 服务（如需远程访问）
simple-memory serve -p 8766
```

### 端口说明

| 端口 | 服务 | 说明 |
|------|------|------|
| 8765 | Web 管理界面 | 配置、查看和管理记忆 |
| 8766 | MCP SSE 服务 | 远程 MCP 客户端连接 |
| 11434 | Ollama | 本地模型服务（可选） |

### 健康检查

```bash
# 检查 MCP SSE 服务状态
curl http://localhost:8766/health

# Docker 容器状态
docker compose ps
```

## 项目结构

```
simple_memory/
├── src/simple_memory/
│   ├── __init__.py
│   ├── cli.py          # CLI 入口点
│   ├── config.py       # 配置管理
│   ├── database.py     # LanceDB 操作（双向量存储）
│   ├── embeddings.py   # 嵌入模型
│   ├── llm.py          # LLM 接口（中文提示词）
│   ├── memory.py       # 记忆管理核心（融合搜索、文本切分）
│   ├── mcp_server.py   # MCP 服务器（stdio + SSE）
│   └── web/
│       ├── app.py      # Flask 应用
│       ├── templates/  # Jinja2 模板
│       └── static/     # CSS/JS
├── data/               # LanceDB 数据目录
├── Dockerfile          # Docker 镜像构建
├── docker-compose.yml  # Docker Compose 配置
├── docker-compose.ollama.yml  # 包含 Ollama 的配置
├── pyproject.toml
└── README.md
```

## 数据迁移

如果从旧版本升级，需要重建数据库以使用新的双向量功能：

```bash
# 备份旧数据（可选）
mv data/lancedb data/lancedb.bak

# 重新启动，系统会创建新的数据库结构
simple-memory web
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
