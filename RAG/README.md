# 企业知识库RAG智能问答系统

## 项目简介
基于RAG技术的企业级智能问答系统，支持PDF/Word/TXT文档上传，实现语义检索和智能问答。

## 功能特性
- 支持多种文档格式上传（PDF、Word、TXT）
- 基于向量数据库的语义检索
- 智能问答，支持上下文理解
- 多源信息融合，提高回答准确性
- 幻觉检测，确保回答可靠性
- RESTful API接口，便于集成
- Streamlit可视化界面

## 快速启动

### 1. 安装依赖
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/Mac
# source .venv/bin/activate

# 升级pip
python -m pip install --upgrade pip setuptools wheel

# 安装依赖
pip install -r requirements-py310.txt
```

### 2. 配置环境变量
复制`.env.example`文件为`.env`，并填写相关配置：

```bash
cp .env.example .env
```

编辑`.env`文件，设置API密钥和模型配置：

```
# ===== Qwen (DashScope OpenAI-compatible) =====
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen2.5-7b-instruct

# 推荐使用 LLM_API_KEY（代码也兼容 DASHSCOPE_API_KEY）
LLM_API_KEY=your_api_key_here

# ===== Embedding model (local path) =====
EMBEDDING_MODEL=./models/bge-m3
```

### 3. 启动服务

#### FastAPI服务
```bash
python main.py
```

服务将在 `http://localhost:8001` 启动，可通过以下端点访问：
- 健康检查：`GET /health`
- 上传文档：`POST /api/upload`
- 查询知识库：`POST /api/query`
- 获取统计信息：`GET /api/stats`
- 重建知识库：`POST /api/rebuild`
- 清空知识库：`DELETE /api/clear`

#### Streamlit界面
```bash
streamlit run streamlit_app.py
```

Streamlit界面将在 `http://localhost:8501` 启动。

## 项目结构

```
├── main.py              # FastAPI主入口
├── rag_llm.py           # RAG核心引擎
├── document_parser.py   # 文档解析器
├── vector_store.py      # 向量存储
├── config.py            # 配置文件
├── streamlit_app.py     # Streamlit界面
├── requirements.txt     # 依赖文件
├── requirements-py310.txt # Python 3.10专用依赖
├── .env.example         # 环境变量示例
├── data/                # 数据目录
│   ├── documents/       # 文档存储
│   └── chroma_db/       # 向量数据库
└── models/              # 模型存储
```

## 技术栈

- **后端框架**：FastAPI、Streamlit
- **RAG核心**：LangChain、ChromaDB
- **嵌入模型**：BGE-M3
- **LLM**：Qwen 2.5 (通过DashScope API)
- **文档处理**：PyPDF2、python-docx
- **向量存储**：ChromaDB

## 使用指南

### 1. 上传文档
- 通过FastAPI接口 `POST /api/upload` 上传文档
- 或通过Streamlit界面上传

### 2. 查询知识库
- 通过FastAPI接口 `POST /api/query` 发送查询
- 或通过Streamlit界面输入问题

### 3. 管理知识库
- 重建知识库：`POST /api/rebuild`
- 清空知识库：`DELETE /api/clear`
- 查看统计信息：`GET /api/stats`

## 注意事项

1. 首次启动时，系统会自动导入 `data/documents` 目录下的文档
2. 请确保 `.env` 文件中的API密钥正确配置
3. 对于大型文档，可能需要较长的处理时间
4. 建议在全新的虚拟环境中安装依赖，避免包冲突

## 许可证

MIT License
