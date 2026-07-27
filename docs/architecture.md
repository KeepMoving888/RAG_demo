# 系统架构设计

本文档描述企业级 RAG 知识库问答系统的整体架构、核心设计原则、数据流转、存储职责划分、降级策略以及安全设计。文档面向平台开发与运维人员，便于在二次开发、容量规划与故障定位时建立全局视图。

## 1. 设计目标

- **检索质量优先**：在 100 万级文档分块规模下，Top-5 召回率 ≥ 0.90，端到端问答 P95 延迟 ≤ 1.8s。
- **企业级可治理**：部门级行级权限、全链路审计、可观测性覆盖检索与生成全流程。
- **弹性可降级**：任一外部依赖（LLM、向量库、图库、缓存）不可用时，系统以可预期的退化模式继续提供服务，而非整体不可用。
- **异步解耦**：文档入库重计算路径与在线问答路径完全隔离，重解析任务不阻塞检索请求。

## 2. 整体架构

系统采用四层架构：接入层、API 层、检索融合层、存储层。前端通过 Nginx 反向代理与后端通信，所有跨存储的检索逻辑收敛到检索融合层统一编排。

```
                           ┌─────────────────────────────────────────────┐
                           │                  接入层                       │
                           │   React 18 + TypeScript + Vite + Tailwind    │
                           │            shadcn/ui 组件库                  │
                           │        Nginx 反向代理 (端口 3080)             │
                           └───────────────────┬─────────────────────────┘
                                               │ HTTPS / WebSocket / SSE
                                               ▼
                           ┌─────────────────────────────────────────────┐
                           │                  API 层                       │
                           │              FastAPI (端口 8080)              │
                           │  ┌───────────┐ ┌──────────┐ ┌─────────────┐  │
                           │  │ Auth 中间件│ │ 限流中间件│ │ 审计中间件  │  │
                           │  └───────────┘ └──────────┘ └─────────────┘  │
                           │  ┌──────────────────────────────────────┐   │
                           │  │   路由层 (auth/docs/qa/search/graph)  │   │
                           │  └──────────────────────────────────────┘   │
                           └───────────────────┬─────────────────────────┘
                                               │
                       ┌───────────────────────┼────────────────────────┐
                       │                       │                        │
            文档入库流（异步）          问答流（在线同步）         管理/监控流
                       │                       │                        │
                       ▼                       ▼                        ▼
           ┌───────────────────┐   ┌────────────────────────┐   ┌──────────────┐
           │  RabbitMQ Broker  │   │     检索融合层          │   │  管理后台     │
           │  (任务队列)        │   │  Query Rewriter        │   │  Prometheus  │
           └─────────┬─────────┘   │  Hybrid Retriever      │   │  Grafana     │
                     │             │  RRF Fusion            │   └──────────────┘
                     ▼             │  Cross-Encoder Rerank  │
           ┌───────────────────┐   │  LLM Generator         │
           │ Celery Worker 集群 │   │  Citation Tracer       │
           │ 解析/清洗/分块/向量化│   └───────────┬────────────┘
           └─────────┬─────────┘                 │
                     │                           │
                     ▼                           ▼
           ┌─────────────────────────────────────────────────┐
           │                    存储层                         │
           │  ┌────────────┐ ┌────────┐ ┌───────┐ ┌────────┐ │
           │  │ PostgreSQL │ │ Milvus │ │ Neo4j │ │ Redis  │ │
           │  │ 业务+权限   │ │ 向量库  │ │ 图谱  │ │ 缓存   │ │
           │  │ +审计+pgvector│ │ +分区 │ │       │ │ +限流  │ │
           │  └────────────┘ └────────┘ └───────┘ └────────┘ │
           └─────────────────────────────────────────────────┘
```

## 3. 核心设计原则

### 3.1 异步解耦

文档入库是一个典型的重计算路径：一个 50MB 的 PDF 文件经解析、清洗、分块、向量化后，需要向 Milvus 写入约 800~1200 个向量，同时向 PostgreSQL 写入分块元数据，向 Neo4j 写入抽取的实体关系。整个流程耗时可达数十秒至数分钟。若将其放在 HTTP 请求链路内处理，将导致上传接口长时间占用连接、无法横向扩展。

本系统的处理方式：

- 上传接口仅完成文件落盘（对象存储 / 本地卷）与任务入队，立即返回 `task_id`。
- Celery Worker 从 RabbitMQ 拉取任务，按流水线阶段执行，每阶段结果落库可断点续跑。
- 问答侧独立横向扩展，与入库 Worker 互不抢占资源，Worker 数量与 API 实例数量分别独立伸缩。

### 3.2 分级降级

每个外部依赖都被视为"可能不可用"。检索融合层在调用每个下游时都带有降级开关与超时控制，详见第 6 节降级策略链。

### 3.3 权限隔离

权限控制在两个层级强制执行：

- **API 层**：JWT 解析出 `user_id`、`department_id`、`role`，注入请求上下文。
- **存储层**：检索召回的 chunk 必须携带 `department_id`、`doc_acl`，在融合层统一过滤，确保用户不会看到越权文档的任何片段。这一过滤发生在 RRF 融合之前，避免越权内容影响排序。

### 3.4 可观测性

全链路埋点覆盖文档入库与问答两条主线：

- **指标**：Prometheus 采集检索延迟、召回数、LLM Token 消耗、Worker 队列堆积、各存储 P95 延迟。
- **链路**：每个问答请求生成 `trace_id`，贯穿限流 → 缓存 → 改写 → 检索 → 重排 → 生成 → 溯源，可在 Grafana 中按 `trace_id` 串联。
- **审计**：所有写操作（上传、删除、重解析、权限变更）写入 `audit_log` 表，含操作人、时间、目标、前后值。

## 4. 数据流

### 4.1 文档入库流

```
用户上传
   │
   ▼
[POST /api/v1/documents/upload]  ── 落盘 + 写 documents 表(status=pending) + 入队
   │
   ▼
RabbitMQ (queue: doc.ingest)
   │
   ▼
Celery Worker (pipeline: ingest_document)
   │
   ├─► 1. 解析 Parser
   │       PDF  → PyMuPDF 文本层优先, 无法提取时 PaddleOCR 兜底
   │       Word → Unstructured 解析
   │       图片 → PaddleOCR
   │
   ├─► 2. 清洗 Cleaner
   │       页眉页脚检测、水印过滤、乱码过滤、段落归一化、表格结构化抽取
   │
   ├─► 3. 分块 Chunker
   │       heading stack + 段落语义连贯性 + 父子层级 + overlap=64 二次切分
   │
   ├─► 4. 向量化 Embedder
   │       BGE-M3 (1024 维) 批量嵌入, Redis 向量缓存命中跳过
   │
   ├─► 5. 写入 Milvus (collection + partition=department_id)
   │
   ├─► 6. 写入 BM25 倒排 (Elasticsearch / 内存 BM25 索引)
   │
   ├─► 7. 写入 Neo4j (LLM 抽取实体 + 关系, 去重归一化)
   │
   └─► 8. 更新 documents 表 status=ready, 记录 chunk_count / entity_count
```

入库流水线任一阶段失败都会将 `documents.status` 置为 `failed` 并记录错误堆栈，用户可通过 `GET /api/v1/documents/{id}/status` 查看，并触发 `POST /api/v1/documents/{id}/reparse` 重解析（从失败阶段续跑，已成功的阶段不重复执行）。

### 4.2 问答流

```
[POST /api/v1/qa/ask]
   │
   ▼
1. 限流 (令牌桶, Redis)  ── 超限返回 429
   │
   ▼
2. 缓存查询 (Redis, key = hash(question + context + user_acl))
   │  命中 ──► 直接返回 (附 cache_hit=true), P95 < 10ms
   │
   ▼ 未命中
3. 多轮上下文组装 (最近 N 轮对话, Redis 会话存储)
   │
   ▼
4. Query 改写 (LLM / 规则)
   │  指代消解、术语扩展(TerminologyExpander)、子问题拆分
   │
   ▼
5. 三路混合检索 (并行)
   ├─► Milvus 稠密向量检索 (Top-K=20)
   ├─► BM25 稀疏检索 (Top-K=20, 术语关键词加权)
   └─► Neo4j 图谱检索 (按查询路由决定是否调用)
   │
   ▼
6. ACL 过滤 (按 user.department_id / doc_acl 过滤越权 chunk)
   │
   ▼
7. RRF 融合 (k=60)  ── 合并三路结果, 去重
   │
   ▼
8. Cross-Encoder 精排 (bge-reranker-v2-m3, Top-20 → Top-5)
   │
   ▼
9. LLM 生成 (Prompt 拼接 Top-5 chunk + 溯源元数据)
   │
   ▼
10. 溯源标注 (将生成文本与 chunk_id 对应, 返回 citations[])
    │
    ▼
11. 写缓存 + 写会话历史 + 返回
```

## 5. 存储层职责划分

| 存储 | 角色 | 关键集合/表 | 写入方 | 读取方 |
|------|------|-------------|--------|--------|
| PostgreSQL | 业务数据、权限、审计 | `users` / `departments` / `documents` / `chunks` / `qa_sessions` / `qa_messages` / `feedback` / `audit_log` | API、Worker | API、Worker |
| pgvector | 文档级向量备份、小规模近线检索 | `chunks.embedding` (vector(1024)) | Worker | API（降级时） |
| Milvus | 主向量检索引擎 | `collection=doc_chunks`, `partition=dept_{id}`, `index=IVF_FLAT nlist=1024` | Worker | 检索融合层 |
| Elasticsearch | BM25 倒排索引、全文检索 | `index=doc_chunks_bm25` | Worker | 检索融合层 |
| Neo4j | 知识图谱 | 节点: `Product/Department/Person/Policy`, 关系: `BELONGS_TO/PARTICIPATES/CERTIFIES/REFERENCES` | Worker | GraphRAG 检索 |
| Redis | 多路缓存 + 限流 + 会话 | `qa:cache:*` / `embed:cache:*` / `ratelimit:*` / `session:*` | API | API |
| RabbitMQ | 异步任务 Broker | `doc.ingest` / `doc.reparse` / `graph.extract` | API | Celery Worker |

### 5.1 为何同时保留 pgvector 与 Milvus

pgvector 用于开发期与小规模近线场景（单租户 < 10 万向量），降低部署复杂度；Milvus 作为生产主引擎，承担百万级向量的高吞吐检索。检索融合层通过 `VECTOR_STORE_PROVIDER` 配置切换，业务代码无感知。这一双写设计同时为 Milvus 不可用时的降级路径提供兜底数据源。

## 6. 降级策略链

降级不是"全有或全无"，而是逐级退化。每一级都尽量保留尽可能多的检索能力，仅在确实无法恢复时退回到最简模式。

```
正常态
  │  Milvus 不可用
  ▼
[Level 1] 向量检索降级到 pgvector / 仅 BM25 + 图谱
  │  Cross-Encoder 不可用 (GPU OOM / 模型加载失败)
  ▼
[Level 2] 跳过精排, 直接用 RRF 融合结果 Top-5
  │  Neo4j 不可用
  ▼
[Level 3] 关系类问题回退到向量+BM25, 不走图谱
  │  Redis 不可用
  ▼
[Level 4] 缓存与限流降级为直查 (临时关闭限流需告警), 会话改用 DB 持久化
  │  LLM 不可用 (API Key 失效 / 配额耗尽 / 网络中断)
  ▼
[Level 5] offline 模式: 仅返回检索结果 + 拼接片段, 不做生成
```

降级状态由 `DegradationManager` 统一管理，每次下游调用失败计数达到阈值（默认连续 3 次失败）触发降级，并以指数退避方式探测恢复。降级与恢复事件均写入 `audit_log` 并触发 Grafana 告警。

实现示例（简化）：

```python
class DegradationManager:
    """统一管理各下游的降级状态与探测恢复。"""

    def __init__(self, redis: RedisClient):
        self._redis = redis
        self._threshold = 3  # 连续失败阈值

    async def call_with_fallback(self, key: str, primary, fallback, *args, **kwargs):
        if await self._is_degraded(key):
            return await fallback(*args, **kwargs)
        try:
            result = await primary(*args, **kwargs)
            await self._reset_failures(key)
            return result
        except Exception:
            await self._incr_failures(key)
            if await self._failures(key) >= self._threshold:
                await self._mark_degraded(key, ttl=60)
                # 触发告警
            return await fallback(*args, **kwargs)
```

## 7. 安全设计

### 7.1 JWT 鉴权

- 登录签发 `access_token`（有效期 30 分钟）+ `refresh_token`（有效期 7 天）。
- Token 载荷包含 `user_id`、`department_id`、`role`、`exp`，签名算法 HS256。
- 刷新机制：`access_token` 过期后由前端用 `refresh_token` 静默换取新令牌，避免用户频繁重登。
- 密钥通过 `JWT_SECRET_KEY` 环境变量注入，生产环境必须更换默认值并定期轮换。

### 7.2 部门行级权限

权限模型基于"文档归属部门 + 部门可见范围矩阵"：

- 每个文档入库时绑定 `owner_department_id`，并继承到其所有 chunk。
- `department_visibility` 表记录部门间可见关系（自部门可见、跨部门只读、跨部门不可见）。
- 检索融合层在 RRF 融合前执行 ACL 过滤：`chunk.department_id ∈ visible_departments(user)`。
- 管理员角色（`role=admin`）可豁免 ACL，但所有越权访问仍写入审计日志。

### 7.3 Cypher 注入防护

GraphRAG 接收自然语言查询并生成 Cypher，存在注入风险。防护策略：

- **参数化查询**：所有用户可控输入以 `$param` 形式传入，绝不字符串拼接。
- **白名单校验**：生成的 Cypher 在执行前经过 AST 解析，仅允许 `MATCH / WHERE / RETURN / WITH / LIMIT`，禁止 `CREATE / DELETE / SET / REMOVE / MERGE`。
- **只读用户**：Neo4j 连接使用只读角色，即使防护被绕过也无法写入。
- **超时与行数限制**：单查询 `timeout=5s`，`LIMIT 200` 硬上限。

```python
# 正确：参数化
graph_query = """
MATCH (p:Product)-[:BELONGS_TO]->(d:Department {name: $dept})
RETURN p.name AS product
LIMIT $limit
"""
session.run(graph_query, dept=user_dept, limit=50)

# 错误：字符串拼接（禁止）
# graph_query = f"MATCH ... WHERE d.name = '{user_dept}' ..."
```

### 7.4 令牌桶限流

- 维度：按 `user_id` 与按 `IP` 双维度，任一触发即拒绝。
- 默认配额：普通用户 20 QPS、IP 100 QPS，可在 `RateLimitConfig` 按 role 配置。
- 实现：Redis + Lua 脚本保证原子性，避免并发下桶状态错乱。
- 超限返回 `429 Too Many Requests`，响应头携带 `X-RateLimit-Remaining` 与 `Retry-After`。

```lua
-- 令牌桶 Lua 脚本（简化）：原子地取令牌并返回剩余
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])  -- 每秒补充令牌数
local now = tonumber(ARGV[3])
local key = KEYS[1]
local bucket = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(bucket[1]) or capacity
local ts = tonumber(bucket[2]) or now
tokens = math.min(capacity, tokens + (now - ts) * refill)
local allowed = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
end
redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, 60)
return allowed
```

## 8. 横向扩展策略

| 组件 | 扩展方式 | 注意点 |
|------|----------|--------|
| FastAPI | 无状态，多实例 + Nginx 负载均衡 | 会话与缓存走 Redis，不依赖进程内存 |
| Celery Worker | 按 queue 横向扩容，解析与向量化分属不同队列 | 向量化 Worker 建议绑 GPU 节点 |
| Milvus | 分布式模式独立扩容 querynode / datanode | 单分区数据量建议 < 500 万 |
| PostgreSQL | 主从读写分离，读副本承接检索后元数据回查 | 写主库仅 API 与 Worker |
| Redis | 主从 + 哨兵，或 Cluster | 限流 key 需保证同一 user 落同一分片 |
| Neo4j | 因果集群，读副本承接查询 | 写入仅 Worker，图谱规模 < 1000 万节点时单机足够 |

## 9. 相关文档

- [RAG 检索流水线设计](./rag-pipeline.md)：分块、向量化、混合检索与消融评估细节。
- [GraphRAG 知识图谱增强](./graphrag.md)：图谱 Schema、Cypher 安全与双路融合。
- [部署运维指南](./deployment.md)：环境要求、配置项、监控与故障排查。
- [性能基准测试](./benchmark.md)：延迟、吞吐与消融实验完整数据。
