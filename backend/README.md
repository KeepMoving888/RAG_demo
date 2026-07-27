# 企业内部 RAG 知识库 - 后端工作目录说明

本目录包含 FastAPI 后端代码, 主要模块:

- `app/` - 应用主代码
  - `main.py` - 应用入口
  - `config.py` - 配置中心
  - `database.py` - 异步数据库引擎
  - `celery_app.py` - Celery 异步任务实例
  - `metrics.py` - Prometheus 业务指标
  - `core/` - 安全 + 限流 + 请求上下文
  - `models/` - SQLAlchemy ORM
  - `schemas/` - Pydantic 响应模型
  - `api/v1/` - REST 路由
  - `ingestion/` - 文档解析异步流水线
  - `rag/` - 权限隔离 + 混合检索
  - `dialog/` - 多轮对话 + 缓存 + 限流 + 溯源
  - `graphrag/` - GraphRAG 知识图谱增强
  - `llm/` - LLM Provider 抽象层
  - `utils/` - 工具
- `scripts/` - 运维脚本 (init_db / seed_docs / eval_rag / extract_graph)
- `tests/` - 测试
- `data/seed/` - 种子数据

详见根目录 [README.md](../README.md) 与 [docs/](../docs/)。
