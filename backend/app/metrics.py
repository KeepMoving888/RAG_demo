"""
Enterprise RAG Knowledge Base - Prometheus 业务指标

三层指标体系:
1. 基础设施层 (自动采集): QPS / P95 / 5xx
2. RAG 业务层 (自定义): 检索延迟 / 召回率 / 缓存命中 / 限流 / 图谱查询
3. 任务流水线层: 文档解析耗时 / 队列深度 / 失败率
"""

from prometheus_client import Counter, Gauge, Histogram

# ======================== 检索层指标 ========================
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_ms",
    "RAG 检索各阶段延迟 (毫秒)",
    labelnames=["stage"],  # stage: bm25 / vector / rrf / rerank / graph
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000),
)

RETRIEVAL_RECALL = Gauge(
    "rag_retrieval_recall",
    "RAG 检索召回率",
    labelnames=["strategy"],  # strategy: bm25 / vector / hybrid / graph
)

RETRIEVAL_RESULT_COUNT = Histogram(
    "rag_retrieval_result_count",
    "RAG 检索返回结果数",
    buckets=(0, 1, 3, 5, 10, 20, 50),
)

# ======================== 问答层指标 ========================
QA_CACHE_HIT = Counter(
    "rag_qa_cache_hit_total",
    "QA 缓存命中次数",
    labelnames=["result"],  # result: hit / miss
)

QA_RESPONSE_LATENCY = Histogram(
    "rag_qa_response_latency_ms",
    "QA 完整响应延迟 (含 LLM 生成)",
    buckets=(50, 100, 250, 500, 1000, 2000, 5000, 10000),
)

DIALOG_TURN_COUNT = Histogram(
    "rag_dialog_turn_count",
    "多轮对话长度分布",
    buckets=(1, 2, 4, 6, 10, 15, 20),
)

CITATION_COVERAGE = Gauge(
    "rag_citation_coverage",
    "答案溯源覆盖率 (含引用的答案比例)",
)

# ======================== 限流指标 ========================
RATE_LIMIT_REJECTED = Counter(
    "rag_rate_limit_rejected_total",
    "限流拒绝次数",
    labelnames=["endpoint", "user_id"],
)

# ======================== 图谱层指标 ========================
GRAPH_QUERY_LATENCY = Histogram(
    "rag_graph_query_latency_ms",
    "GraphRAG 图谱查询延迟",
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2000),
)

GRAPH_QUERY_RESULT = Counter(
    "rag_graph_query_result_total",
    "图谱查询结果统计",
    labelnames=["status"],  # status: success / empty / failed
)

GRAPH_ENTITIES_EXTRACTED = Counter(
    "rag_graph_entities_extracted_total",
    "实体抽取累计数量",
)

# ======================== 文档解析层指标 ========================
DOC_PARSE_DURATION = Histogram(
    "rag_doc_parse_duration_ms",
    "文档解析耗时",
    labelnames=["format"],  # format: pdf / docx / image / txt
    buckets=(100, 500, 1000, 5000, 10000, 30000, 60000, 300000),
)

DOC_PARSE_RESULT = Counter(
    "rag_doc_parse_result_total",
    "文档解析结果统计",
    labelnames=["status"],  # status: success / failed
)

DOC_CHUNK_COUNT = Histogram(
    "rag_doc_chunk_count",
    "单文档分块数分布",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500),
)

# ======================== Celery 任务层指标 ========================
CELERY_TASK_DURATION = Histogram(
    "rag_celery_task_duration_ms",
    "Celery 任务执行耗时",
    labelnames=["task"],
    buckets=(100, 500, 1000, 5000, 10000, 30000, 60000, 300000),
)

CELERY_TASK_RESULT = Counter(
    "rag_celery_task_result_total",
    "Celery 任务结果统计",
    labelnames=["task", "status"],
)


def record_retrieval_stage(stage: str, latency_ms: float):
    """记录检索阶段延迟"""
    RETRIEVAL_LATENCY.labels(stage=stage).observe(latency_ms)


def record_qa_cache(result: str):
    """记录 QA 缓存命中"""
    QA_CACHE_HIT.labels(result=result).inc()


def record_rate_limit(endpoint: str, user_id: str):
    """记录限流拒绝"""
    RATE_LIMIT_REJECTED.labels(endpoint=endpoint, user_id=user_id).inc()


def record_graph_query(latency_ms: float, status: str):
    """记录图谱查询"""
    GRAPH_QUERY_LATENCY.observe(latency_ms)
    GRAPH_QUERY_RESULT.labels(status=status).inc()


def record_doc_parse(format_: str, duration_ms: float, success: bool):
    """记录文档解析"""
    DOC_PARSE_DURATION.labels(format=format_).observe(duration_ms)
    DOC_PARSE_RESULT.labels(status="success" if success else "failed").inc()
