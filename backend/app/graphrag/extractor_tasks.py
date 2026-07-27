"""
GraphRAG 实体抽取 Celery 异步任务

用途:
    将耗时的 LLM 实体关系抽取 + Neo4j 入库从 API 请求链路中剥离, 投递到
    graphrag 队列异步执行, 避免文档入库接口因图谱构建阻塞.

执行流程:
    1. 从 DB 加载 Document 及其全部 DocumentChunk;
    2. EntityExtractor.extract_from_document 抽取并跨 chunk 去重合并;
    3. neo4j_store.batch_upsert 批量入库;
    4. 上报 record_graph_query(latency, status) 与 GRAPH_ENTITIES_EXTRACTED;
    5. 失败由 tenacity 重试 3 次 (指数退避), 仍失败则抛出供 Celery 死信处理.

并发模型:
    Celery 任务本身是同步函数, 内部通过 asyncio.run 驱动异步抽取逻辑
    (extractor / neo4j_store 均为协程). 每个任务独占一个事件循环, 互不干扰.
"""

import asyncio
import time

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.celery_app import celery_app
from app.metrics import GRAPH_ENTITIES_EXTRACTED, record_graph_query
from app.utils.logger import logger


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 2s, 4s 指数退避
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _extract_with_retry(document_id: int):
    """带 tenacity 重试的抽取+入库 (指数退避 3 次)"""
    from app.graphrag.extractor import EntityExtractor
    from app.graphrag.neo4j_store import neo4j_store

    extractor = EntityExtractor()
    extraction = await extractor.extract_from_document(document_id)
    upsert_result = await neo4j_store.batch_upsert(extraction)
    return extraction, upsert_result


@celery_app.task(
    name="app.graphrag.extractor_tasks.extract_entities_task",
    queue="graphrag",
)
def extract_entities_task(document_id: int) -> dict:
    """异步抽取文档实体关系并入库图谱

    Args:
        document_id: 文档 ID.

    Returns:
        {"entities_added": int, "relations_added": int, "latency_ms": float}
    """
    logger.info("收到图谱抽取任务 document_id={}", document_id)
    start = time.time()

    async def _run():
        extraction, upsert_result = await _extract_with_retry(document_id)
        return extraction, upsert_result

    try:
        extraction, upsert_result = asyncio.run(_run())
        latency_ms = (time.time() - start) * 1000

        # 上报指标: 图谱查询延迟 + 实体抽取计数 (文档级, 避免逐 chunk 重复)
        record_graph_query(latency_ms, status="success")
        GRAPH_ENTITIES_EXTRACTED.inc(len(extraction.entities))

        logger.info(
            "图谱抽取任务完成 document_id={} 实体={} 关系={} 入库={} 耗时={:.0f}ms",
            document_id, len(extraction.entities), len(extraction.relations),
            upsert_result, latency_ms,
        )
        return {
            "entities_added": upsert_result.get("entities_added", 0),
            "relations_added": upsert_result.get("relations_added", 0),
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        record_graph_query(latency_ms, status="failed")
        logger.exception(
            "图谱抽取任务失败 document_id={} 耗时={:.0f}ms: {}",
            document_id, latency_ms, str(e),
        )
        # 抛出后由 Celery 进入失败状态 (重试已在 tenacity 内耗尽)
        raise


__all__ = ["extract_entities_task"]
