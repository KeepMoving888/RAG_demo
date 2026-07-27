"""
Enterprise RAG Knowledge Base - Celery 异步任务实例

用途:
- 文档解析异步流水线 (大文件 OCR 不阻塞 API)
- 知识图谱实体关系抽取 (LLM 调用耗时长)
- 向量索引重建 (离线批量)

架构:
- Broker: RabbitMQ (可靠性高, 支持优先级队列)
- Result Backend: Redis (任务状态查询)
- 序列化: JSON (可观测性, 不用 pickle 避免安全风险)
"""

# ======================== Windows 事件循环修复 ========================
# asyncpg 在 Windows 默认 ProactorEventLoop 下会触发
# AttributeError: 'NoneType' object has no attribute 'send'
# 必须在 import celery/asyncpg 之前切换到 SelectorEventLoop
# 参考: https://github.com/MagicStack/asyncpg/issues/515
import sys
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from celery import Celery
from kombu import Queue

from app.config import settings


def _make_celery() -> Celery:
    """创建 Celery 实例"""
    broker = settings.celery_broker_url
    backend = settings.celery_result_backend

    app = Celery(
        "rag_kb",
        broker=broker,
        backend=backend,
        include=[
            "app.ingestion.tasks",
            "app.graphrag.extractor_tasks",
        ],
    )

    app.conf.update(
        # 序列化
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        # 时区
        timezone="Asia/Shanghai",
        enable_utc=False,
        # 任务路由 (按队列分流)
        task_routes={
            "app.ingestion.tasks.parse_document": {"queue": "ingestion"},
            "app.ingestion.tasks.chunk_and_embed": {"queue": "embedding"},
            "app.graphrag.extractor_tasks.extract_entities_task": {"queue": "graphrag"},
        },
        # 队列定义
        task_queues=(
            Queue("ingestion", routing_key="ingestion"),
            Queue("embedding", routing_key="embedding"),
            Queue("graphrag", routing_key="graphrag"),
            Queue("default", routing_key="default"),
        ),
        task_default_queue="default",
        task_default_routing_key="default",
        # 重试策略
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,  # 公平派发, 避免长任务堆积
        # 并发
        worker_concurrency=4,
        worker_max_tasks_per_child=100,  # 防内存泄漏
        # 结果过期
        result_expires=3600,
        # 监控
        task_send_sent_event=True,
        worker_send_task_events=True,
    )

    return app


# 全局 Celery 实例
celery_app = _make_celery()


# ======================== 任务自动发现 ========================
@celery_app.task(name="app.celery_app.ping")
def ping() -> str:
    """健康检查任务"""
    return "pong"
