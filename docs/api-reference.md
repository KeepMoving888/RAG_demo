# API 接口文档

本文档描述企业级 RAG 知识库的 RESTful API 接口。所有接口基于 HTTP，返回 JSON。问答流式接口基于 SSE（Server-Sent Events）。接口版本通过 URL 前缀 `/api/v1` 标识。

## 1. 认证

系统采用 JWT Bearer Token 鉴权。除登录、健康检查、指标采集外，所有接口需在请求头携带有效 Token：

```
Authorization: Bearer <access_token>
```

Token 过期后使用 `refresh_token` 换取新令牌，无需重新登录。

## 2. 通用响应格式

### 2.1 成功响应

```json
{
  "code": 0,
  "message": "ok",
  "data": { }
}
```

`code` 为 0 表示成功，非 0 表示业务错误（错误码见第 8 节）。`data` 为业务数据，结构因接口而异。

### 2.2 分页响应

列表类接口统一使用分页结构：

```json
{
  "code": 0,
  "message": "ok",
  "data": [ ],
  "total": 128,
  "page": 1,
  "page_size": 20,
  "total_pages": 7
}
```

分页参数通过 query string 传递：`?page=1&page_size=20`，默认 `page=1`、`page_size=20`，`page_size` 上限 100。

### 2.3 错误响应

```json
{
  "code": 40101,
  "message": "access token expired",
  "data": null
}
```

## 3. 认证接口

### 3.1 登录

`POST /api/v1/auth/login`

**请求体**

```json
{
  "username": "admin",
  "password": "your-password"
}
```

**响应体**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "bearer",
    "expires_in": 1800,
    "user": {
      "user_id": "u_001",
      "username": "admin",
      "role": "admin",
      "department_id": "dept_root"
    }
  }
}
```

**示例**

```bash
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"your-password"}'
```

### 3.2 刷新令牌

`POST /api/v1/auth/refresh`

**请求体**

```json
{ "refresh_token": "eyJhbGciOiJIUzI1NiIs..." }
```

**响应体**：同登录响应 `data` 结构。

### 3.3 登出

`POST /api/v1/auth/logout`

将当前 `access_token` 加入黑名单，需携带 Token。

## 4. 文档接口

### 4.1 文档列表

`GET /api/v1/documents`

**Query 参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| page | int | 页码，默认 1 |
| page_size | int | 每页条数，默认 20 |
| status | string | 按状态过滤：pending/processing/ready/failed |
| keyword | string | 按文件名模糊搜索 |

**响应体**

```json
{
  "code": 0,
  "message": "ok",
  "data": [
    {
      "document_id": "doc_001",
      "filename": "编码规范.pdf",
      "status": "ready",
      "chunk_count": 124,
      "entity_count": 38,
      "owner_department_id": "dept_rd",
      "created_at": "2026-07-20T10:23:00Z",
      "updated_at": "2026-07-20T10:25:12Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "total_pages": 1
}
```

### 4.2 上传文档

`POST /api/v1/documents/upload`

`multipart/form-data` 上传，支持 PDF / Word / 图片。接口立即返回 `task_id`，解析在 Celery Worker 异步执行。

**请求参数（form-data）**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | 是 | 文件二进制 |
| owner_department_id | string | 是 | 归属部门 ID（用于权限） |

**响应体**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "document_id": "doc_002",
    "task_id": "task_abc123",
    "status": "pending"
  }
}
```

**示例**

```bash
curl -X POST http://localhost:8080/api/v1/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@编码规范.pdf" \
  -F "owner_department_id=dept_rd"
```

### 4.3 文档状态

`GET /api/v1/documents/{document_id}/status`

**响应体**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "document_id": "doc_002",
    "status": "processing",
    "progress": 0.6,
    "stage": "embedding",
    "chunk_count": 124,
    "error": null
  }
}
```

`stage` 取值：`parsing` / `cleaning` / `chunking` / `embedding` / `indexing` / `graph_extracting` / `done`。失败时 `status=failed`，`error` 含错误信息。

### 4.4 删除文档

`DELETE /api/v1/documents/{document_id}`

级联删除该文档在 Milvus、BM25 索引、Neo4j 图谱中的所有关联数据，并移除原文件。

**响应体**

```json
{ "code": 0, "message": "ok", "data": { "document_id": "doc_002", "deleted": true } }
```

### 4.5 重解析

`POST /api/v1/documents/{document_id}/reparse`

从失败阶段续跑，已成功的阶段不重复执行。返回新的 `task_id`。

**响应体**

```json
{
  "code": 0,
  "message": "ok",
  "data": { "document_id": "doc_002", "task_id": "task_def456", "resume_from": "embedding" }
}
```

## 5. 问答接口

### 5.1 同步问答

`POST /api/v1/qa/ask`

**请求体**

```json
{
  "question": "研发部参与了哪些制度的制定？",
  "session_id": "sess_001",
  "top_k": 5,
  "strategy": "auto"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| question | string | 用户问题 |
| session_id | string | 会话 ID，用于多轮上下文；首次可传空，由服务端创建 |
| top_k | int | 返回溯源条数，默认 5 |
| strategy | string | 检索策略：`auto`（路由）/ `vector_only` / `hybrid` / `graph_only`，默认 auto |

**响应体**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "answer": "根据检索到的制度文件，研发部参与了《编码规范》《代码评审制度》《发布流程管理》的制定...",
    "session_id": "sess_001",
    "message_id": "msg_001",
    "citations": [
      {
        "chunk_id": "chk_01023",
        "document_id": "doc_001",
        "filename": "编码规范.pdf",
        "heading_path": ["第二章", "2.1 适用范围"],
        "content": "本规范适用于研发部全体人员...",
        "score": 0.92
      }
    ],
    "retrieval_meta": {
      "strategy_used": "graph_only",
      "vector_hits": 0,
      "bm25_hits": 0,
      "graph_hits": 5,
      "rerank_latency_ms": 0,
      "cache_hit": false
    },
    "trace_id": "trace_9f8e7d"
  }
}
```

### 5.2 流式问答（SSE）

`POST /api/v1/qa/stream`

请求体同 5.1。响应为 SSE 流，逐 token 推送生成内容，最后推送溯源元数据。适用于前端实现打字机效果，降低首字延迟。

**SSE 事件协议**

| event | data 内容 | 说明 |
|-------|-----------|------|
| `token` | `{"text": "根据"}` | 生成文本片段，前端增量拼接 |
| `meta` | `{"citations":[...], "retrieval_meta":{...}}` | 检索溯源元数据，在 token 流结束后推送 |
| `error` | `{"code":50001,"message":"llm unavailable"}` | 流中发生错误 |
| `done` | `{}` | 流结束标记 |

**示例**

```bash
curl -N -X POST http://localhost:8080/api/v1/qa/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是 RTO 标准？","session_id":"sess_001"}'
```

**响应流（示例）**

```
event: token
data: {"text":"RTO"}

event: token
data: {"text":"（恢复时间目标）"}

event: token
data: {"text":"是指系统发生故障后..."}

event: meta
data: {"citations":[{"chunk_id":"chk_01023","filename":"容灾标准.pdf","score":0.94}]}

event: done
data: {}
```

### 5.3 会话历史

`GET /api/v1/qa/sessions/{session_id}/history`

**Query 参数**：`page`、`page_size`

**响应体**

```json
{
  "code": 0,
  "message": "ok",
  "data": [
    {
      "message_id": "msg_001",
      "role": "user",
      "content": "研发部参与了哪些制度的制定？",
      "created_at": "2026-07-22T09:00:00Z"
    },
    {
      "message_id": "msg_002",
      "role": "assistant",
      "content": "根据检索到的制度文件...",
      "citations": [ ],
      "created_at": "2026-07-22T09:00:02Z"
    }
  ],
  "total": 2,
  "page": 1,
  "page_size": 20,
  "total_pages": 1
}
```

### 5.4 会话列表

`GET /api/v1/qa/sessions`

返回当前用户的会话列表，含最近一条消息预览。

### 5.5 问答反馈

`POST /api/v1/qa/feedback`

**请求体**

```json
{
  "message_id": "msg_002",
  "rating": "positive",
  "comment": "答案准确，溯源清晰",
  "tags": ["accurate", "well_cited"]
}
```

`rating` 取值：`positive` / `negative` / `neutral`。反馈用于后续检索质量评估与调优。

### 5.6 归档会话

`POST /api/v1/qa/sessions/{session_id}/archive`

将会话标记为归档，不再出现在默认会话列表，但历史仍可查询。

## 6. 检索接口

### 6.1 检索

`POST /api/v1/search`

直接调用检索融合层，不经过 LLM 生成，用于调试检索质量或集成到其他系统。

**请求体**

```json
{
  "query": "RTO 标准",
  "top_k": 10,
  "strategy": "hybrid",
  "department_filter": ["dept_rd"],
  "with_rerank": true
}
```

**响应体**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "results": [
      {
        "chunk_id": "chk_01023",
        "document_id": "doc_001",
        "filename": "容灾标准.pdf",
        "heading_path": ["第三章", "3.2 RTO 定义"],
        "content": "RTO（恢复时间目标）指...",
        "score": 0.94,
        "retriever": "hybrid"
      }
    ],
    "retrieval_meta": {
      "strategy_used": "hybrid",
      "vector_hits": 8,
      "bm25_hits": 7,
      "graph_hits": 0,
      "rrf_fused": 10,
      "rerank_latency_ms": 14,
      "total_latency_ms": 42
    }
  }
}
```

### 6.2 检索解释

`POST /api/v1/search/explain`

返回三路检索的原始打分与排名，便于排查"为何某结果未召回"。

**响应体（节选）**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "query": "RTO 标准",
    "vector_ranking": [
      { "chunk_id": "chk_01023", "score": 0.94, "rank": 1 }
    ],
    "bm25_ranking": [
      { "chunk_id": "chk_01023", "score": 18.7, "rank": 2 }
    ],
    "graph_ranking": [ ],
    "rrf_scores": [
      { "chunk_id": "chk_01023", "rrf_score": 0.0328, "final_rank": 1 }
    ]
  }
}
```

### 6.3 消融评估

`POST /api/v1/search/eval`

在评测集上对比多种检索策略，返回指标对比。详见 [RAG 流水线文档](./rag-pipeline.md) 第 8 节。

**请求体**

```json
{
  "strategies": ["vector_only", "bm25_only", "rrf", "full", "full_with_terminology"],
  "top_k": 5,
  "dataset": "internal_v3"
}
```

**响应体**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "dataset": "internal_v3",
    "question_count": 500,
    "results": [
      {
        "strategy": "vector_only",
        "recall_at_5": 0.72,
        "mrr": 0.55,
        "ndcg_at_5": 0.61,
        "precision_at_5": 0.48
      },
      {
        "strategy": "full_with_terminology",
        "recall_at_5": 0.91,
        "mrr": 0.76,
        "ndcg_at_5": 0.83,
        "precision_at_5": 0.70
      }
    ]
  }
}
```

## 7. 图谱接口

### 7.1 图谱查询

`POST /api/v1/graph/query`

自然语言查询图谱，内部走 GraphCypherQAChain。仅授权角色可调用。

**请求体**

```json
{ "question": "研发部参与了哪些制度的制定？" }
```

**响应体**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "answer": "研发部参与了《编码规范》《代码评审制度》《发布流程管理》的制定。",
    "cypher": "MATCH (d:Department {name: $dept})-[:PARTICIPATES]->(p:Policy) RETURN p.name AS policy",
    "rows": [
      { "policy": "编码规范" },
      { "policy": "代码评审制度" },
      { "policy": "发布流程管理" }
    ],
    "latency_ms": 78
  }
}
```

### 7.2 实体查询

`GET /api/v1/graph/entities`

**Query 参数**：`type`（Product/Department/Person/Policy）、`name`、`page`、`page_size`

### 7.3 关系查询

`GET /api/v1/graph/relations`

**Query 参数**：`entity_id`、`relation`（关系类型）、`direction`（out/in/both）、`depth`（跳数，1~3）

**响应体**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "start_entity": { "entity_id": "e_001", "type": "Product", "name": "产品A" },
    "paths": [
      {
        "end_entity": { "entity_id": "e_010", "type": "Policy", "name": "准入标准" },
        "hops": [
          { "from": "产品A", "relation": "BELONGS_TO", "to": "研发部" },
          { "from": "研发部", "relation": "PARTICIPATES", "to": "准入标准" }
        ],
        "depth": 2
      }
    ]
  }
}
```

### 7.4 图谱 Schema

`GET /api/v1/graph/schema`

返回当前图谱的节点/关系类型定义与统计，含健康检查信息。

**响应体（节选）**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "node_types": [
      { "type": "Product", "count": 1280, "properties": ["name", "version", "category", "doc_ids"] }
    ],
    "relation_types": [
      { "type": "BELONGS_TO", "count": 1120 }
    ],
    "health": {
      "orphan_nodes": 23,
      "connected_components": 18,
      "largest_component_size": 8421
    }
  }
}
```

## 8. 管理接口

### 8.1 用户管理

`GET /api/v1/admin/users` — 用户列表（分页）
`POST /api/v1/admin/users` — 创建用户
`PATCH /api/v1/admin/users/{user_id}` — 更新用户（角色、部门）
`DELETE /api/v1/admin/users/{user_id}` — 禁用用户

### 8.2 部门管理

`GET /api/v1/admin/departments` — 部门树
`POST /api/v1/admin/departments` — 创建部门
`PUT /api/v1/admin/departments/{dept_id}/visibility` — 配置部门可见范围矩阵

### 8.3 统计

`GET /api/v1/admin/stats`

**响应体**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "documents": { "total": 1280, "ready": 1260, "processing": 12, "failed": 8 },
    "chunks": { "total": 184200 },
    "qa": { "today_count": 3420, "avg_latency_ms": 1450, "cache_hit_rate": 0.42 },
    "graph": { "nodes": 89200, "relations": 142000 },
    "storage": { "milvus_rows": 184200, "es_docs": 184200 }
  }
}
```

## 9. 监控接口

### 9.1 指标

`GET /metrics`

Prometheus 格式文本，无需鉴权（由内网 Prometheus 抓取）。包含 API 延迟、检索延迟、LLM Token、Worker 队列、各存储健康等指标，详见 [部署文档](./deployment.md) 第 8 节。

### 9.2 健康检查

`GET /api/v1/health`

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "status": "healthy",
    "components": {
      "postgres": "up",
      "redis": "up",
      "milvus": "up",
      "neo4j": "up",
      "elasticsearch": "up",
      "rabbitmq": "up"
    },
    "degradation": []
  }
}
```

任一组件不可用时 `status` 为 `degraded`，`degradation` 列出当前降级项。

## 10. 错误码

| HTTP 状态码 | code | 含义 | 说明 |
|-------------|------|------|------|
| 400 | 40001 | bad request | 请求参数错误 |
| 400 | 40002 | validation error | 字段校验失败，`data` 含字段级错误明细 |
| 401 | 40101 | token expired | access_token 过期，需刷新 |
| 401 | 40102 | token invalid | Token 无效或被吊销 |
| 403 | 40301 | forbidden | 无权限访问该资源（部门 ACL 拒绝） |
| 403 | 40302 | cypher rejected | Cypher 校验未通过（含禁用子句） |
| 404 | 40401 | not found | 资源不存在 |
| 429 | 42901 | rate limited | 触发限流，响应头含 `Retry-After` |
| 500 | 50001 | llm unavailable | LLM 调用失败，已降级为 offline 模式 |
| 500 | 50002 | retrieval failed | 检索链路异常 |
| 500 | 50003 | internal error | 未预期的服务端错误 |

## 11. 相关文档

- [系统架构设计](./architecture.md)：接口背后的架构与降级策略。
- [RAG 检索流水线设计](./rag-pipeline.md)：检索与消融评估实现细节。
- [GraphRAG 知识图谱增强](./graphrag.md)：图谱接口的查询编排与安全。
- [部署运维指南](./deployment.md)：服务端口与监控配置。
