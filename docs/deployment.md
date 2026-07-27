# 部署运维指南

本文档描述企业级 RAG 知识库的部署方式、环境配置、一键部署流程、离线与真实大模型接入、生产环境注意事项、监控告警以及常见问题排查。所有部署操作均通过 Docker Compose 编排，配合 Makefile 封装为简单命令。

## 1. 环境要求

| 项 | 最低要求 | 推荐配置 | 说明 |
|----|----------|----------|------|
| 操作系统 | Linux x86_64 | Ubuntu 22.04 / CentOS 8+ | 亦支持 macOS、Windows + WSL2 |
| Docker | 24.0+ | 25.0+ | `docker --version` 校验 |
| Docker Compose | 2.20+ | 2.24+ | 使用 `docker compose`（v2 语法） |
| CPU | 8 核 | 16 核 | 向量化与 OCR 为 CPU 密集型 |
| 内存 | 16 GB | 32 GB | Milvus 与 Neo4j 对内存敏感 |
| 磁盘 | 50 GB | 100 GB+ SSD | 文档原文件、向量数据、图谱数据 |
| GPU | 可选 | 1 张 T4/A10 | 启用 Cross-Encoder 精排与本地嵌入加速 |

无 GPU 环境下，Cross-Encoder 精排降级为 CPU 推理（延迟升高约 3 倍）或自动跳过（见架构文档降级策略链）。

## 2. 一键部署

项目根目录提供 `Makefile`，将常用操作封装为命令。完整启动流程：

```bash
# 1. 克隆并进入项目
git clone <repo-url> enterprise-rag-kb
cd enterprise-rag-kb

# 2. 复制环境变量模板并按需修改
cp .env.example .env

# 3. 拉起全部服务（前端、后端、Worker、存储、监控）
make up

# 4. 初始化数据库 schema 与向量索引
make init-db

# 5. 导入种子数据（示例文档、术语词典、测试账号）
make seed

# 6. 初始化 Neo4j 图谱 Schema 与约束
make graph-init
```

各 Makefile 目标对应的实际命令：

| 目标 | 作用 | 等价命令 |
|------|------|----------|
| `make up` | 拉起全部容器 | `docker compose up -d` |
| `make down` | 停止并移除容器 | `docker compose down` |
| `make init-db` | 初始化数据库与索引 | `docker compose exec backend python -m app.scripts.init_db` |
| `make seed` | 导入种子数据 | `docker compose exec backend python -m app.scripts.seed` |
| `make graph-init` | 初始化图谱 Schema | `docker compose exec backend python -m app.scripts.graph_init` |
| `make logs` | 查看全部服务日志 | `docker compose logs -f --tail=200` |
| `make ps` | 查看服务状态 | `docker compose ps` |

## 3. .env 配置项详解

所有配置通过项目根目录的 `.env` 文件注入，Compose 自动读取。以下按模块列出关键配置项。

### 3.1 PostgreSQL

```ini
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=rag_kb
POSTGRES_USER=rag
POSTGRES_PASSWORD=change-me-in-production
# pgvector 扩展随初始化脚本自动创建
PGVECTOR_ENABLED=true
```

### 3.2 Redis

```ini
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
# 问答缓存 TTL（秒）
QA_CACHE_TTL=3600
# 向量缓存 TTL（秒）
EMBED_CACHE_TTL=86400
```

### 3.3 Milvus

```ini
MILVUS_HOST=milvus
MILVUS_PORT=19530
MILVUS_COLLECTION=doc_chunks
MILVUS_INDEX_TYPE=IVF_FLAT
MILVUS_INDEX_PARAMS={"nlist":1024}
MILVUS_SEARCH_PARAMS={"nprobe":16}
MILVUS_METRIC_TYPE=COSINE
```

### 3.4 Neo4j

```ini
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=change-me-in-production
NEO4J_DATABASE=neo4j
# 图谱查询超时（秒）
GRAPH_QUERY_TIMEOUT=5
# 单查询返回上限
GRAPH_QUERY_LIMIT=200
```

### 3.5 LLM

```ini
# offline: 不调用外部 API, 仅返回检索结果（无需 API Key）
# deepseek: 接入 DeepSeek
# openai: 接入 OpenAI 兼容接口
LLM_PROVIDER=offline

# DeepSeek 配置（LLM_PROVIDER=deepseek 时生效）
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# 嵌入模型（本地加载）
EMBED_MODEL=BAAI/bge-m3
EMBED_DIMENSION=1024
# 重排模型（本地加载，无 GPU 可置空跳过精排）
RERANK_MODEL=BAA/bge-reranker-v2-m3
```

### 3.6 JWT 与鉴权

```ini
# 生产环境必须更换为高强度随机串（>= 32 字节）
JWT_SECRET_KEY=change-me-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### 3.7 限流与 CORS

```ini
# 限流（令牌桶，每秒补充令牌数 = 容量）
RATE_LIMIT_USER_QPS=20
RATE_LIMIT_IP_QPS=100
RATE_LIMIT_ENABLED=true

# CORS
CORS_ORIGINS=http://localhost:3080,http://127.0.0.1:3080
```

## 4. 离线模式

离线模式（`LLM_PROVIDER=offline`）面向无外网或无大模型 API Key 的环境。该模式下：

- **问答生成**：不调用 LLM，直接将检索 Top-5 chunk 拼接为答案，标注每段来源，不做自然语言改写。
- **Query 改写**：跳过 LLM 改写，仅执行规则化术语扩展（`TerminologyExpander`）。
- **图谱抽取**：入库时跳过 LLM 实体关系抽取，图谱不构建（GraphRAG 第三路召回不可用，自动降级为向量 + BM25 双路）。

离线模式适合内部数据敏感性高、无法调用外部 API 的场景，可完整验证检索链路质量。接入真实大模型后，生成与图谱能力自动启用。

## 5. 接入真实大模型

以 DeepSeek 为例：

```bash
# 编辑 .env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-real-key-here
DEEPSEEK_MODEL=deepseek-chat

# 重启后端与 Worker
make down
make up
```

接入后，问答生成、Query 改写、图谱实体抽取全部走真实大模型。建议首次接入后在评测集上跑一次消融评估（`POST /api/v1/search/eval`），确认检索质量与离线模式一致（生成质量不影响检索指标）。

接入 OpenAI 兼容接口（如自部署的 vLLM）只需将 `LLM_PROVIDER=openai` 并配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。

## 6. 服务端口表

| 服务 | 容器端口 | 宿主映射 | 说明 |
|------|----------|----------|------|
| 前端（Nginx） | 80 | 3080 | React 静态资源 + 反向代理 |
| 后端 API | 8080 | 8080 | FastAPI |
| PostgreSQL | 5432 | 5432 | 业务数据 + pgvector |
| Redis | 6379 | 6379 | 缓存 + 限流 + 会话 |
| Milvus | 19530 | 19530 | 向量检索 |
| Neo4j | 7687 | 7687 | Bolt 协议（图谱查询） |
| Neo4j HTTP | 7474 | 7474 | Neo4j Browser |
| Elasticsearch | 9200 | 9200 | BM25 倒排 |
| RabbitMQ AMQP | 5672 | 5672 | 任务队列 |
| RabbitMQ 管理 | 15672 | 15672 | 管理界面 |
| Prometheus | 9090 | 9090 | 指标采集 |
| Grafana | 3000 | 3001 | 监控面板 |

宿主端口冲突时，修改 `docker-compose.yml` 中对应服务的 `ports` 映射，并同步更新 `.env` 中的连接配置。

## 7. 生产部署注意事项

### 7.1 密钥与密码

- **JWT_SECRET_KEY**：必须更换为 ≥ 32 字节的随机串，建议 `openssl rand -hex 32` 生成，并定期轮换。
- **数据库密码**：`POSTGRES_PASSWORD`、`NEO4J_PASSWORD`、`REDIS_PASSWORD` 全部更换，禁止使用默认值。
- 密钥管理建议接入企业密钥管理服务（如 Vault），`.env` 文件不入版本库（已在 `.gitignore` 中）。

### 7.2 资源限制

在 `docker-compose.yml` 中为每个服务设置 `deploy.resources.limits`，避免单服务异常耗尽宿主资源：

```yaml
services:
  milvus:
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 8G
  neo4j:
    environment:
      - NEO4J_server_memory_heap_max__size=4G
      - NEO4J_server_memory_pagecache_size=2G
  backend:
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 2G
```

### 7.3 日志与数据卷

- 日志卷：所有服务日志持久化到 `./volumes/logs/`，按天滚动，保留 30 天。
- 数据卷：`./volumes/postgres/`、`./volumes/milvus/`、`./volumes/neo4j/`、`./volumes/es/` 挂载到宿主，便于备份与迁移。
- 生产环境建议将数据卷挂载到独立磁盘或网络存储，与系统盘隔离。

### 7.4 备份策略

| 数据 | 备份方式 | 频率 | 保留 |
|------|----------|------|------|
| PostgreSQL | `pg_dump` 全量 + WAL 归档 | 每日全量 + 实时归档 | 30 天 |
| Milvus | 快照 + S3 导出 | 每日 | 7 天 |
| Neo4j | `neo4j-admin dump` | 每日 | 7 天 |
| 文档原文件 | 对象存储版本管理 | 实时 | 永久 |
| Redis | RDB 快照 | 每小时 | 24 份（缓存可重建，非关键） |

备份脚本位于 `scripts/backup/`，通过 cron 定时执行，备份结果上报到 Grafana 告警通道。

## 8. 监控与告警

### 8.1 Prometheus 采集

后端通过 `prometheus-fastapi-instrumentator` 暴露 `/metrics` 端点，Worker 通过 `celery-exporter` 暴露任务指标。Prometheus 配置自动发现各服务并采集。

核心采集指标：

- `http_request_duration_seconds`（API 延迟直方图）
- `rag_retrieval_duration_seconds`（检索延迟，按 retriever 分标签）
- `rag_llm_tokens_total`（LLM Token 消耗）
- `celery_queue_length`（Worker 队列堆积）
- `milvus_search_latency_seconds`（Milvus 检索延迟）
- `neo4j_query_duration_seconds`（图谱查询延迟）
- `rag_cache_hit_total` / `rag_cache_miss_total`（缓存命中率）

### 8.2 Grafana 面板

预置 16 个监控面板，覆盖：

1. API 总览（QPS、延迟 P50/P95/P99、错误率）
2. 问答链路各阶段耗时（限流/缓存/改写/检索/重排/生成/溯源）
3. 检索召回数分布
4. BM25 / 向量 / 图谱三路检索延迟对比
5. Cross-Encoder 精排延迟与吞吐
6. LLM Token 消耗与成本
7. 缓存命中率
8. Worker 队列堆积与任务耗时
9. 文档入库吞吐与失败率
10. Milvus 检索延迟与 QPS
11. Neo4j 查询延迟与慢查询
12. PostgreSQL 连接数与慢查询
13. Redis 命中率与内存占用
14. RabbitMQ 队列深度
15. 降级状态总览（各下游降级/恢复事件）
16. 系统资源（CPU/内存/磁盘/网络）

### 8.3 告警规则

预置 12 条告警规则，告警通过 Grafana 推送到企业 IM 通道：

| 告警 | 触发条件 | 严重级别 |
|------|----------|----------|
| API P95 延迟过高 | `histogram_quantile(0.95, ...) > 3s` 持续 5min | warning |
| API 错误率飙升 | `rate(http_errors[5m]) > 0.05` | critical |
| 检索召回数为 0 比例高 | `rag_zero_recall_ratio > 0.1` 持续 10min | warning |
| LLM 调用失败 | `rag_llm_error_rate > 0.1` 持续 5min | critical |
| Worker 队列堆积 | `celery_queue_length > 1000` 持续 10min | warning |
| Milvus 不可用 | `milvus_up == 0` 持续 1min | critical |
| Neo4j 不可用 | `neo4j_up == 0` 持续 1min | critical |
| Redis 不可用 | `redis_up == 0` 持续 1min | critical |
| 缓存命中率低 | `rag_cache_hit_ratio < 0.3` 持续 30min | warning |
| 数据库连接池耗尽 | `pg_connections / pg_max_connections > 0.8` | warning |
| 磁盘空间不足 | `disk_free / disk_total < 0.15` | critical |
| 任一服务降级触发 | `degradation_active > 0` | warning |

## 9. 常见问题排查

### 9.1 Milvus 启动慢或检索超时

**现象**：`make up` 后 Milvus 容器长时间未就绪，或问答报 Milvus 检索超时。

**排查**：
- Milvus 首次启动需初始化内部存储，预计 1~3 分钟，检查 `docker logs milvus` 是否出现 `Milvus Proxy successfully started`。
- 内存不足会导致 Milvus 频繁 GC，`docker stats` 确认内存使用，必要时调大宿主内存或降低 `MILVUS_INDEX_PARAMS.nlist`。
- 检索超时优先调大 `MILVUS_SEARCH_PARAMS.nprobe`（精度）或调小（速度），默认 16 在 100 万向量下平衡良好。

### 9.2 Neo4j 内存溢出

**现象**：Neo4j 容器 OOM 重启，图谱查询失败。

**排查**：
- 调整 `NEO4J_server_memory_heap_max__size` 与 `NEO4J_server_memory_pagecache_size`，两者之和不超过宿主内存的 70%。
- 大规模图谱（> 1000 万节点）建议升级到因果集群，读查询分流到读副本。
- 慢查询通过 `dbms.listQueries()` 定位，对全图扫描查询补充索引或限制 `LIMIT`。

### 9.3 Celery Worker 不消费任务

**现象**：文档上传后状态长期 `pending`，Worker 日志无消费记录。

**排查**：
- `docker compose ps` 确认 Worker 容器处于 `running`，`docker logs worker` 查看是否有连接 RabbitMQ 失败报错。
- 确认 RabbitMQ 队列存在且任务已入队：访问 `http://localhost:15672`（默认 guest/guest）查看队列深度。
- Worker 并发数过少会导致堆积，调整 `--concurrency`（默认 4），向量化 Worker 建议 `--concurrency=2`（每进程占用内存高）。

### 9.4 端口冲突

**现象**：`make up` 报 `port is already allocated`。

**排查**：
- `netstat -ano | findstr <port>`（Windows）或 `lsof -i:<port>`（Linux）定位占用进程。
- 优先关闭占用进程；无法关闭时修改 `docker-compose.yml` 的宿主映射端口，并同步 `.env` 中客户端连接配置。
- 常见冲突：5432（本机 PostgreSQL）、6379（本机 Redis）、3000（其他服务）。

### 9.5 向量化阶段 OOM

**现象**：Worker 在嵌入阶段被 OOM Kill。

**排查**：
- 降低 `EMBED_BATCH_SIZE`（默认 32），减少单批内存占用。
- BGE-M3 模型本身占用约 2GB 内存，确认 Worker 容器内存限制 ≥ 4GB。
- 大批量重解析时建议分批提交，避免单 Worker 同时处理过多文档。

## 10. 相关文档

- [系统架构设计](./architecture.md)：各服务在架构中的角色。
- [API 接口文档](./api-reference.md)：部署完成后的接口调用方式。
- [性能基准测试](./benchmark.md)：部署规模与性能预期参考。
