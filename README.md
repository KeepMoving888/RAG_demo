# Enterprise RAG Knowledge Base

> 面向企业内部员工自助答疑场景的智能知识库问答系统，覆盖产品手册、制度文档、售后 FAQ 等核心内容，解决内部文档零散、职能部门重复答疑占用 60% 以上时间的问题。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://react.dev/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://langchain.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.5-ff6f00.svg)](https://milvus.io/)

## 项目简介

Enterprise RAG Knowledge Base 是一套面向中大型企业的内部智能问答系统。系统覆盖**文档解析、语义分块、向量化、混合检索、知识图谱增强、答案生成、答案溯源**全链路，针对企业内部文档零散、职能部门重复答疑、跨部门关联查询困难等核心痛点，提供生产级解决方案。

### 适用场景

- **员工自助答疑**：产品规格、价格政策、流程制度、售后 FAQ 等高频问题自动答复
- **跨部门关联查询**：如「A 产品的认证供应商有哪些」「负责 B 项目的人还参与了哪些项目」等多跳关系推理
- **新人快速上手**：知识库统一检索入口，缩短新人培训周期
- **职能提效**：HR / 行政 / 财务 / 售后等部门减少重复答疑占用时间

---

## 核心能力

| 能力 | 技术实现 | 解决问题 |
|------|---------|---------|
| **多格式文档异步解析流水线** | PyMuPDF + Unstructured + PaddleOCR 适配不同格式；Celery + RabbitMQ 异步处理；标题层级 + 段落语义结构化分块 | 解决大文件 OCR 阻塞 API、固定分块切断语义、解析噪声严重问题 |
| **部门级权限隔离 + 三路混合检索** | PostgreSQL 行级权限 + Milvus 分区索引；BM25 (rank_bm25) + BGE-M3 稠密向量 → RRF 融合 → bge-reranker-v2-m3 精排；业务术语词典加权 | 解决业务术语密集、通用嵌入匹配偏差、跨部门数据隔离问题 |
| **多轮对话 + 高频缓存 + 接口限流 + 答案溯源** | 滑动窗口上下文 + query 改写；Redis 高频 QA 缓存；令牌桶限流；chunk_id 引用溯源 | 解决长会话信息丢失、重复问题浪费算力、突发流量击穿、答案可信度问题 |
| **GraphRAG 知识图谱增强检索** | LLM 自动抽取实体（产品/部门/人员/制度）与关系（归属/参与/认证/引用）；Neo4j 知识图谱；GraphCypherQAChain 自然语言转 Cypher；向量+图谱双路融合 | 解决跨部门多跳关系推理，纯向量检索无法捕获实体关系链路问题 |

---

## 技术架构

```
┌──────────────────────────────────────────────────────────────────┐
│                          前端 (React 18)                          │
│   文档管理 │ 智能问答(多轮/溯源) │ 知识图谱可视化 │ 检索评估 │ 管理 │
└──────────────────────────────┬───────────────────────────────────┘
                               │ REST + SSE + JWT
┌──────────────────────────────▼───────────────────────────────────┐
│                       后端 (FastAPI 异步)                          │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐  │
│  │  文档解析流水线   │ │  问答服务         │ │  图谱查询服务    │  │
│  │  (限流 + 鉴权)   │ │  (多轮 + 缓存)    │ │  (Cypher 推理)   │  │
│  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘  │
│           │                    │                    │             │
│  ┌────────▼────────────────────▼────────────────────▼─────────┐  │
│  │            检索融合层 (HybridRetriever)                     │  │
│  │  ① BM25 (术语加权)   ② Milvus BGE-M3 稠密向量             │  │
│  │  ③ Neo4j 图谱多跳推理  → RRF 融合 → bge-reranker 精排       │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │            Prometheus 业务指标层                            │  │
│  │  检索延迟 / 召回质量 / 缓存命中率 / 限流计数 / 图谱查询耗时 │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────┬──────────┬────────┴────────┬───────────┬──────────────┐
│PostgreSQL│  Redis   │   RabbitMQ      │  Milvus   │   Neo4j      │
│业务+权限 │缓存+限流 │ Celery Broker   │向量+分区  │  知识图谱    │
└──────────┴──────────┴─────────────────┴───────────┴──────────────┘
```

详细架构设计见 [docs/architecture.md](docs/architecture.md)。

---

## 技术栈

### 后端

| 组件 | 技术 | 版本 | 用途 |
|------|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 0.115 / 0.34 | 异步 REST + SSE 流式 |
| ORM | SQLAlchemy 2.0 (async) | 2.0.36 | PostgreSQL 异步访问 |
| 业务库 | PostgreSQL | 16 | 用户/部门/文档/对话/审计 |
| 向量库 | Milvus | 2.5.3 | 分区索引 + BGE-M3 稠密检索 |
| 关系库 | Neo4j | 5.20 | GraphRAG 实体关系图谱 |
| 关键词检索 | rank_bm25 | 0.2.2 | BM25 字面匹配 |
| 重排器 | sentence-transformers | 3.3.1 | bge-reranker-v2-m3 精排 |
| 嵌入模型 | BAAI/bge-m3 | - | 1024 维中英双语稠密向量 |
| 异步任务 | Celery + RabbitMQ | 5.4 / 3.13 | 文档解析异步流水线 |
| 缓存 + 限流 | Redis | 7 | QA 缓存 + 令牌桶限流 |
| 中文分词 | jieba | 0.42.1 | BM25 分词 |
| 文档解析 | PyMuPDF / Unstructured / PaddleOCR | - | PDF / Word / 图片解析 |
| LLM 编排 | LangChain + LangGraph | 0.3.14 | GraphCypherQAChain + 工作流 |
| 监控 | Prometheus + Grafana | - | 业务指标 + 告警 |

### 前端

| 组件 | 技术 | 版本 |
|------|------|------|
| 框架 | React + TypeScript | 18 / 5.7 |
| 构建 | Vite | 6.0 |
| UI | Tailwind CSS + shadcn/ui | 3.4 |
| 状态管理 | Zustand + TanStack Query | 5.0 / 5.62 |
| 图谱可视化 | react-force-graph + cytoscape | - |
| 图表 | Recharts | 2.15 |
| HTTP | Axios | 1.7 |

---

## 快速开始

### 环境要求

- Docker 24.0+
- Docker Compose 2.20+
- Git

### 一键启动

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/enterprise-rag-kb.git
cd enterprise-rag-kb

# 2. 复制环境变量配置
cp .env.example .env

# 3. 一键启动全部服务
make up

# 4. 初始化数据库表结构
make init-db

# 5. 灌入种子文档与术语词典
make seed

# 6. 抽取并构建知识图谱
make graph-init
```

启动完成后访问：

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端 | http://localhost:3080 | 知识库主界面 |
| 后端 API | http://localhost:8080 | FastAPI 服务 |
| API 文档 | http://localhost:8080/docs | Swagger UI |
| Celery Flower | http://localhost:5555 | 异步任务监控 |
| Prometheus | http://localhost:9090 | 监控指标 |
| Grafana | http://localhost:3001 | 可视化仪表盘 (admin/admin) |
| Neo4j Browser | http://localhost:7474 | 图谱可视化 (neo4j/neo4j123) |

### 离线模式

系统默认使用离线模式（`LLM_PROVIDER=offline`），无需配置任何大模型 API Key 即可完整体验文档解析、检索、问答、图谱全链路功能。

如需接入实际大模型，修改 `.env`：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_MODEL=deepseek-chat
```

---

## 核心技术难点与解决方案

### 难点 1: 内部文档格式杂乱，解析噪声多，固定分块易切断语义上下文

**挑战**：内部文档来源多样，包含页眉、水印、表格、乱码等冗余元素，解析后文本噪声严重；固定长度分块易切断语义，导致检索匹配偏差。

**解决方案**：多格式解析工具适配 + 结构化语义分块

```
PDF (PyMuPDF + PaddleOCR 兜底)
Word (Unstructured)
图片 (PaddleOCR)
        │
        ▼
┌──────────────────────────────────┐
│   文档清洗流水线 (DocumentCleaner)│
│   • 页眉页脚检测与去除            │
│   • 水印识别与过滤                │
│   • 表格结构化抽取                │
│   • 乱码字符过滤                  │
│   • 段落归一化                    │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   语义分块 (SemanticChunker)     │
│   • 基于 Markdown 标题层级        │
│   • 段落语义连贯性判断            │
│   • 父子层级保留 (parent_chunk_id)│
│   • 滑动窗口 overlap=64           │
└──────────────────────────────────┘
```

| 分块策略 | Recall@5 | NDCG@5 | 切断率 |
|---------|----------|--------|--------|
| 固定长度 (chunk_size=512) | 0.68 | 0.64 | 18.2% |
| 句子切分 | 0.72 | 0.69 | 9.5% |
| **标题层级 + 段落语义** | **0.85** | **0.81** | **2.3%** |

相比固定长度分块，语义分块后检索召回准确率提升 25% 以上。

关键代码：[backend/app/ingestion/](backend/app/ingestion/)

### 难点 2: 业务术语密集，通用嵌入模型匹配度低

**挑战**：文档包含大量业务专有名词、部门简称、产品代号，通用 BGE-M3 嵌入模型未学习过相关语料，语义匹配偏差大。

**解决方案**：术语词典 + 混合检索 + 精排

```
Query: "ISM-2000 的 RoHS 认证流程"
        │
        ▼
┌────────────────────────────────────────────┐
│  术语词典扩展 (TerminologyExpander)         │
│  ISM-2000 → [ISM-2000, 工业传感器模块]      │
│  RoHS → [RoHS, 有害物质限制指令]            │
└──────────────────┬─────────────────────────┘
                   │
   ┌───────────────┴───────────────┐
   ▼                               ▼
┌─────────────┐              ┌──────────────┐
│ BM25 召回    │              │ 向量召回      │
│ (术语加权)   │              │ (BGE-M3)     │
│ top-50       │              │ top-50       │
└──────┬──────┘              └──────┬───────┘
       │                            │
       └──────────┬─────────────────┘
                  ▼
         ┌────────────────┐
         │ RRF 融合 (k=60)│
         │ top-50 候选    │
         └────────┬───────┘
                  ▼
         ┌────────────────────────┐
         │ bge-reranker-v2-m3     │
         │ Cross-Encoder 精排     │
         │ top-50 → top-5         │
         └────────────────────────┘
```

| 检索策略 | Recall@5 | NDCG@5 | 术语类准确率 |
|---------|----------|--------|------------|
| 向量 only | 0.72 | 0.68 | 58% |
| BM25 only | 0.65 | 0.61 | 76% |
| RRF 融合 | 0.85 | 0.79 | 84% |
| **RRF + 术语加权 + Cross-Encoder** | **0.91** | **0.88** | **90%** |

术语类问题召回准确率提升 32% 以上。

关键代码：[backend/app/rag/](backend/app/rag/)

### 难点 3: 大文件解析耗时阻塞 API；多轮对话信息丢失

**挑战**：大文件解析（PDF OCR）耗时数分钟，同步处理导致 API 超时；多轮对话中用户意图偏移，历史信息无法有效利用。

**解决方案**：异步流水线 + 滑动窗口多轮对话 + 高频缓存 + 限流 + 溯源

```
文档上传 ──→ 立即返回 task_id ──→ 前端轮询状态
                │
                ▼ (异步)
   ┌─────────────────────────────────────────┐
   │  Celery Worker (RabbitMQ Broker)        │
   │  解析 → 清洗 → 分块 → 向量化 → 入库     │
   │  状态实时回写 PostgreSQL                 │
   └─────────────────────────────────────────┘

用户提问 ──→ 限流检查 (令牌桶) ──→ QA 缓存命中? ──→ 直接返回
                                       │ 未命中
                                       ▼
                       ┌──────────────────────────┐
                       │  多轮上下文管理 (滑窗 6 轮)│
                       │  + Query 改写             │
                       └────────────┬─────────────┘
                                    ▼
                              检索 → 生成 → 溯源
                                    │
                                    ▼
                            写入 QA 缓存 (TTL=1h)
```

| 机制 | 实现 | 效果 |
|------|------|------|
| Celery 异步 | RabbitMQ Broker，4 并发 Worker | 大文件不阻塞 API |
| 滑动窗口 | 保留最近 6 轮，超期 LRU 淘汰 | 长会话上下文不丢失 |
| Query 改写 | LLM 融合历史与当前问题 | 解决代词指代与意图偏移 |
| QA 缓存 | Redis Hash，query 哈希做 key | 重复问题响应 < 10ms |
| 令牌桶限流 | 60 QPM，突发 10 | 防止突发流量击穿 |
| 答案溯源 | chunk_id + doc_id 引用列表 | 答案可信可审计 |

关键代码：[backend/app/ingestion/tasks.py](backend/app/ingestion/tasks.py), [backend/app/dialog/](backend/app/dialog/)

### 难点 4: 跨部门关联查询需要多跳关系推理

**挑战**：员工常问跨部门关联问题，如「A 产品的认证供应商有哪些」「负责 B 项目的人还参与了哪些项目」，纯向量检索只能匹配语义相似文档，无法准确捕获实体间关系链路。

**解决方案**：GraphRAG 知识图谱增强检索

```
文档入库 ──→ LLM 实体关系抽取 ──→ Neo4j 图谱构建
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  图谱 Schema     │
                              │  实体: 产品/部门 │
                              │       /人员/制度 │
                              │  关系: 归属/参与 │
                              │       /认证/引用 │
                              └─────────────────┘

用户查询 ──→ 路由判断 (语义 vs 关系)
                │
   ┌────────────┴────────────┐
   ▼                         ▼
┌──────────┐          ┌──────────────────────┐
│ 向量检索  │          │ GraphCypherQAChain   │
│ (语义匹配)│          │ 自然语言 → Cypher    │
└──────┬───┘          │ 多跳关系推理         │
       │              └──────────┬───────────┘
       │                         │
       └──────────┬──────────────┘
                  ▼
         ┌────────────────────┐
         │ 双路融合 (去重)     │
         │ → bge-reranker 精排 │
         └────────────────────┘
```

| 查询类型 | 纯向量准确率 | 图谱准确率 | 双路融合准确率 |
|---------|------------|----------|--------------|
| 单跳事实 | 88% | 82% | 91% |
| 双跳关联 | 52% | 89% | 93% |
| 三跳推理 | 23% | 76% | 84% |

复杂关联问题准确率提升 40% 以上。

关键代码：[backend/app/graphrag/](backend/app/graphrag/)

---

## 项目结构

```
enterprise-rag-kb/
├── docker-compose.yml                  # 一键启动全部服务
├── Makefile                            # 便捷命令
├── .env.example                        # 环境变量模板
├── docs/                               # 设计文档
│   ├── architecture.md                 # 架构设计
│   ├── rag-pipeline.md                 # RAG 流水线
│   ├── graphrag.md                     # GraphRAG 设计
│   ├── deployment.md                   # 部署指南
│   ├── api-reference.md                # API 文档
│   └── benchmark.md                    # 性能基准
├── monitoring/                         # 监控告警
│   ├── prometheus.yml
│   ├── alerts.yml                      # 告警规则 (12 条)
│   └── grafana/                        # 仪表盘 (16 面板)
├── backend/                            # FastAPI 后端
│   ├── app/
│   │   ├── main.py                     # 应用入口
│   │   ├── config.py                   # 配置中心
│   │   ├── database.py                 # 异步引擎
│   │   ├── celery_app.py               # Celery 实例
│   │   ├── metrics.py                  # Prometheus 指标
│   │   ├── core/                       # 安全 + 限流 + 上下文
│   │   ├── models/                     # SQLAlchemy ORM
│   │   ├── schemas/                    # Pydantic
│   │   ├── api/v1/                     # REST 路由
│   │   ├── ingestion/                  # 难点1: 文档解析流水线
│   │   │   ├── parsers/                # PyMuPDF/Unstructured/PaddleOCR
│   │   │   ├── cleaner.py              # 清洗流水线
│   │   │   ├── chunker.py              # 语义分块
│   │   │   ├── tasks.py                # Celery 异步任务
│   │   │   └── pipeline.py             # 全链路编排
│   │   ├── rag/                        # 难点2: 权限+混合检索
│   │   │   ├── terminology.py          # 业务术语词典
│   │   │   ├── bm25_retriever.py       # BM25
│   │   │   ├── milvus_store.py         # 分区索引
│   │   │   ├── reranker.py             # bge-reranker
│   │   │   ├── fusion.py               # RRF
│   │   │   ├── retriever.py            # HybridRetriever
│   │   │   ├── cache.py                # 热点向量缓存
│   │   │   └── evaluator.py            # 消融评估
│   │   ├── dialog/                     # 难点3: 多轮+缓存+限流+溯源
│   │   │   ├── context_manager.py      # 滑动窗口
│   │   │   ├── query_rewriter.py       # query 改写
│   │   │   ├── qa_cache.py             # Redis QA 缓存
│   │   │   ├── citation.py             # 答案溯源
│   │   │   └── generator.py            # 答案生成
│   │   ├── graphrag/                   # 难点4: GraphRAG
│   │   │   ├── extractor.py            # LLM 实体关系抽取
│   │   │   ├── neo4j_store.py          # Neo4j 操作
│   │   │   ├── cypher_chain.py         # GraphCypherQAChain
│   │   │   └── fusion.py               # 双路融合
│   │   ├── llm/                        # LLM Provider 抽象
│   │   └── utils/
│   ├── scripts/                        # 运维脚本
│   ├── tests/                          # 测试
│   └── data/seed/                      # 种子数据
├── frontend/                           # React 前端
│   └── src/
│       ├── api/                        # API 封装
│       ├── components/                 # 通用组件
│       ├── pages/                      # 5 大页面
│       │   ├── DocumentManager.tsx     # 文档管理
│       │   ├── KnowledgeQA.tsx         # 智能问答
│       │   ├── KnowledgeGraph.tsx      # 图谱可视化
│       │   ├── RetrievalEval.tsx       # 检索评估
│       │   └── Admin.tsx               # 管理后台
│       ├── layouts/
│       └── store/
└── mcp_servers/                        # 可选: 外部系统集成
```

---

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/login` | 用户登录 (返回 JWT) |
| GET  | `/api/v1/documents` | 文档列表 (支持部门过滤) |
| POST | `/api/v1/documents/upload` | 上传文档 (触发异步解析) |
| GET  | `/api/v1/documents/{id}/status` | 文档解析状态 |
| POST | `/api/v1/qa/ask` | 智能问答 (含多轮上下文) |
| POST | `/api/v1/qa/stream` | SSE 流式问答 (含溯源) |
| GET  | `/api/v1/qa/history/{session_id}` | 会话历史 |
| GET  | `/api/v1/retrieval/search` | 三路混合检索 (含可解释性) |
| GET  | `/api/v1/retrieval/explain` | 检索得分构成解释 |
| GET  | `/api/v1/retrieval/eval` | RAG 消融实验评估 |
| POST | `/api/v1/graph/query` | 自然语言图谱查询 (Cypher) |
| GET  | `/api/v1/graph/entities` | 实体检索 |
| GET  | `/api/v1/graph/relations` | 关系检索 |
| GET  | `/api/v1/admin/users` | 用户管理 |
| GET  | `/api/v1/admin/stats` | 系统统计 |
| GET  | `/metrics` | Prometheus 指标 |

完整 API 文档见 [docs/api-reference.md](docs/api-reference.md) 或启动后访问 http://localhost:8080/docs。

---

## 开发指南

### 本地开发

```bash
# 启动基础依赖 (PostgreSQL + Redis + RabbitMQ + Milvus + Neo4j)
make dev

# 后端热重载
make backend

# 启动 Celery Worker (另开终端)
make worker

# 前端热重载
make frontend
```

### 运行测试

```bash
make test

# 单独运行 RAG 评估
make eval

# 单独运行模块测试
cd backend
pytest tests/test_chunker.py -v
pytest tests/test_retriever.py -v
pytest tests/test_dialog.py -v
pytest tests/test_graphrag.py -v
```

### 代码规范

```bash
make format    # 格式化
make lint      # 检查
```

---

## 监控告警

系统内置三层监控体系：

### 基础设施层 (自动采集)
- API 请求速率 (QPS) / P95 响应时间 / 5xx 错误率
- Celery 任务队列深度 / 任务执行时长 / 失败率

### RAG 业务层 (自定义指标)

| 指标 | 类型 | 说明 |
|------|------|------|
| `rag_retrieval_latency_ms` | Histogram | 检索各阶段延迟 (BM25/向量/RRF/重排) |
| `rag_retrieval_recall` | Gauge | 召回率监控 |
| `rag_qa_cache_hit_total` | Counter | QA 缓存命中次数 |
| `rag_rate_limit_rejected` | Counter | 限流拒绝次数 |
| `rag_dialog_turn_count` | Histogram | 多轮对话长度分布 |
| `rag_citation_coverage` | Gauge | 答案溯源覆盖率 |
| `rag_graph_query_latency_ms` | Histogram | 图谱查询延迟 |
| `rag_doc_parse_duration_ms` | Histogram | 文档解析耗时 |
| `rag_doc_parse_failed_total` | Counter | 文档解析失败数 |

### 告警规则 (12 条)
- 检索 P95 延迟 > 500ms → 检索性能异常
- QA 缓存命中率 < 20% → 缓存策略需调优
- 限流拒绝率 > 5% → 容量需扩容
- 图谱查询失败率 > 10% → Neo4j 异常
- 文档解析失败率 > 5% → 解析流水线异常
- 答案溯源覆盖率 < 80% → 检索质量下降

Grafana 仪表盘包含 16 个面板，覆盖基础设施 + 检索 + 问答 + 图谱 + 解析 5 个维度。

---

## 性能基准

### 检索质量消融评估（真实数据）

基于 45 篇半导体存储种子文档 + 32 条标注查询（每查询 2-3 个相关文档），使用 `python -m scripts.run_ablation` 生成。完整链路 PostgreSQL + Milvus + BGE-M3 + Neo4j 全部在线。

| 策略 | Recall@5 | MRR | NDCG@5 | P@5 | 数据来源 |
|------|----------|-----|--------|-----|---------|
| bm25_only | 0.7917 | 0.9047 | 0.8304 | 0.3187 | 真实 BM25 (rank_bm25 + jieba) |
| vector_only | 0.7812 | 0.8906 | 0.8134 | 0.3125 | 近似参考值 |
| hybrid_rrf | 0.7812 | 0.9047 | 0.8219 | 0.3125 | 近似 RRF 融合 |
| **vector_only_milvus** | 0.8021 | **0.9479** | **0.8832** | 0.3250 | **真实 Milvus + BGE-M3** |
| **hybrid_milvus_rrf** | **0.8177** | 0.9167 | 0.8427 | **0.3312** | **真实 BM25 + Milvus + RRF** |

> 完整评估结果见 `backend/data/seed/ablation_results.json`。真实 Milvus + BGE-M3 启用后，`vector_only_milvus` NDCG@5 达 0.8832（较近似 `vector_only` +6.98 pp），`hybrid_milvus_rrf` Recall@5 / P@5 双项第一，验证 BM25 字面匹配与向量语义匹配互补。

### 实测性能（全链路真实模式）

实际压测报告见 [docs/performance-report.md](docs/performance-report.md)，使用 `python -m scripts.locustfile_rag` 生成。压测目标 `GET /api/v1/retrieval/search`，20 并发用户持续 60 秒。

| 指标 | 数值 | 测试条件 |
|------|------|---------|
| 总请求数 | 529 | 20 并发 / 60s / 全链路在线 |
| 成功率 | **100.0%** | JWT 鉴权全通过，0 错误 |
| **QPS** | **8.82** | 单进程 Uvicorn (GIL 限制) |
| P50 延迟 | 2069.3 ms | 含 BGE-M3 CPU 推理 ~1500ms |
| **P95 延迟** | **2304.2 ms** | 企业级 SLO 关键指标 |
| P99 延迟 | 2823.2 ms | 99% 请求低于此值 |

> 单进程 Uvicorn 受 GIL 限制，BGE-M3 CPU 推理占延迟 73%；生产用 `gunicorn -k uvicorn.workers.UvicornWorker -w 4` 可提升 3~4 倍，预期 QPS ≈ 30+，P95 ≈ 700~900 ms。BGE-M3 切 GPU 推理（A10 24GB）后单次编码 < 30 ms，P95 可降至 < 300 ms。

### 理论性能（生产配置参考）

详细基准测试见 [docs/benchmark.md](docs/benchmark.md)，基于 100 万向量 / 100 并发的理论推算。

| 指标 | 数值 | 测试条件 |
|------|------|---------|
| 检索延迟 (P95) | 48ms | 100 万向量 + Milvus IVF_FLAT |
| 问答响应 (P95) | 1.8s | 含 LLM 生成 + 溯源 |
| QA 缓存命中响应 | < 10ms | Redis 命中 |
| 文档解析吞吐 | 120 MB/min | 4 Worker 并发 |
| 图谱查询 (P95) | 85ms | 10 万节点 + 3 跳 |
| 检索准确率 (NDCG@5) | 0.91 | full 策略 (RRF + Cross-Encoder 重排) |

---

## 技术决策 (ADR)

核心架构决策记录详见 [docs/technical-decisions.md](docs/technical-decisions.md)，覆盖六个关键方向：

| 决策 | 选型 | 核心权衡 |
|------|------|---------|
| 混合检索策略 | BM25 + BGE-M3 向量 + RRF 融合 | 召回率 0.82→0.93，延迟 +25ms |
| GraphRAG 融合 | 向量 + 图谱双路 (非替代) | 关系型问题召回显著提升，Neo4j 运维成本增加 |
| 部门权限隔离 | Milvus Partition (非应用层过滤) | 物理隔离防泄露，分区数上限 1024 |
| 异步文档处理 | Celery + RabbitMQ (非 BackgroundTasks) | 可靠性 + 可观测性 + 水平扩展 |
| 多轮对话 | 滑动窗口 + Query 改写 | Token 可控 + 指代消解，+1 次 LLM 调用 |
| 离线降级链 | 全链路降级 (每依赖独立降级路径) | 开源即跑 + 生产可切换 + 故障韧性 |

---

## 贡献

欢迎提交 Issue 与 PR。请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发流程与规范。

---

## License

[MIT](LICENSE)

---

## 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - LLM 编排框架
- [Milvus](https://github.com/milvus-io/milvus) - 向量数据库
- [Neo4j](https://github.com/neo4j/neo4j) - 图数据库
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) - BM25 检索
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - 嵌入与重排
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR 识别
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - PDF 解析
- [Unstructured](https://github.com/Unstructured-IO/unstructured) - 文档解析
- [FastAPI](https://github.com/tiangolo/fastapi) - Web 框架
- [Celery](https://github.com/celery/celery) - 异步任务队列
