"""Celery 异步任务：文档摄入全链路编排。

设计要点
========
1. **全链路编排**：``parse_document`` 串联 解析→清洗→分块→向量化→落库，
   每阶段更新 ``ParseTask.stage`` 与 ``progress``，使前端可实时展示进度。
2. **断点续跑**：拆出 ``chunk_and_embed`` 独立任务，解析完成后若向量化失败
   可单独重跑，避免重复解析大文档。
3. **重试机制**：使用 ``tenacity`` 指数退避重试（最多 3 次），覆盖瞬时故障
   （DB 连接抖动、模型加载超时），不可恢复错误则标记 failed。
4. **指标上报**：每文档上报 ``record_doc_parse``（格式/耗时/成功），每文档
   上报 ``DOC_CHUNK_COUNT``（分块数），供监控系统聚合。
5. **降级**：Milvus 不可用时跳过向量入库（仅落 DB），保证流水线不因向量库
   抖动而失败。
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.celery_app import celery_app
from app.database import db_session
from app.ingestion.chunker import Chunk, SemanticChunker
from app.ingestion.cleaner import DocumentCleaner
from app.ingestion.embedder import embedder
from app.ingestion.parsers.registry import parser_registry
from app.metrics import DOC_CHUNK_COUNT, record_doc_parse
from app.models import Document, DocumentChunk, ParseTask
from app.utils.logger import logger

# Milvus 向量库：rag 子模块负责创建，本模块仅调用；import 失败则降级跳过
try:
    from app.rag.milvus_store import milvus_store  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001 rag 模块尚未就绪时降级
    milvus_store = None  # type: ignore[assignment]

# 阶段定义
STAGE_PARSING = "parsing"
STAGE_CLEANING = "cleaning"
STAGE_CHUNKING = "chunking"
STAGE_EMBEDDING = "embedding"
STAGE_PERSISTING = "persisting"
STAGE_DONE = "done"
STAGE_FAILED = "failed"

# 进度区间：各阶段占据的 progress 区间，使进度条平滑推进
_PROGRESS = {
    STAGE_PARSING: 0.10,
    STAGE_CLEANING: 0.25,
    STAGE_CHUNKING: 0.40,
    STAGE_EMBEDDING: 0.75,
    STAGE_PERSISTING: 0.95,
    STAGE_DONE: 1.00,
}


# ===========================================================================
# Celery 任务入口（同步壳，内部驱动异步逻辑）
# ===========================================================================


@celery_app.task(bind=True, name="app.ingestion.tasks.parse_document", queue="ingestion")
def parse_document(self, document_id: int) -> dict:
    """文档全链路解析任务。

    Args:
        document_id: 待解析文档 ID。

    Returns:
        dict: 包含 document_id、chunk_count、status 的结果摘要。
    """
    logger.info(f"[task] parse_document 启动: document_id={document_id}")
    try:
        result = asyncio.run(_parse_document_async(document_id))
        return result
    except Exception as exc:  # noqa: BLE001 兜底：确保失败被记录
        logger.exception(f"[task] parse_document 最终失败: {exc}")
        asyncio.run(_mark_failed_safe(document_id, str(exc)))
        raise


@celery_app.task(bind=True, name="app.ingestion.tasks.chunk_and_embed", queue="embedding")
def chunk_and_embed(self, document_id: int) -> dict:
    """独立的分块+向量化任务，支持断点续跑。

    适用场景：解析已完成但向量化失败的文档，可仅重跑分块与向量化，
    避免对大文档重复解析。本实现按需重新解析以重建 ParsedDocument
    （因当前未持久化解析中间产物），完成后执行分块→向量化→落库。
    """
    logger.info(f"[task] chunk_and_embed 启动: document_id={document_id}")
    try:
        result = asyncio.run(_chunk_and_embed_async(document_id))
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"[task] chunk_and_embed 最终失败: {exc}")
        asyncio.run(_mark_failed_safe(document_id, str(exc)))
        raise


# ===========================================================================
# 异步实现（带 tenacity 重试）
# ===========================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
async def _parse_document_async(document_id: int) -> dict:
    """解析全链路异步实现。

    流程：加载 Document → 解析 → 清洗 → 分块 → 向量化 → 落 DB+Milvus →
    标记 ready。任一阶段异常经 tenacity 重试后仍失败则标记 failed 并上报指标。
    """
    # 关键修复：Celery solo pool 中 asyncio.run() 每次创建新事件循环，
    # 但 module-level engine 的连接池绑定到旧循环。
    # dispose 后 engine 会按需在当前循环上重建连接池，避免
    # "Future attached to a different loop" 错误。
    from app.database import engine

    await engine.dispose()

    start_ts = time.monotonic()
    doc = await _load_document(document_id)
    file_format = getattr(doc, "file_format", "") or _infer_format(getattr(doc, "file_path", ""))
    department_id = getattr(doc, "department_id", None)

    try:
        # 1. 解析
        await _update_parse_task(document_id, STAGE_PARSING, _PROGRESS[STAGE_PARSING])
        parsed_doc = await _do_parse(doc)

        # 2. 清洗
        await _update_parse_task(document_id, STAGE_CLEANING, _PROGRESS[STAGE_CLEANING])
        cleaner = DocumentCleaner()
        parsed_doc = cleaner.clean(parsed_doc)

        # 3. 分块
        await _update_parse_task(document_id, STAGE_CHUNKING, _PROGRESS[STAGE_CHUNKING])
        chunker = SemanticChunker()
        chunks = chunker.chunk(parsed_doc)
        logger.info(f"document_id={document_id} 产出 {len(chunks)} 个分块")

        # 4. 向量化 + 落库
        await _persist(document_id, doc, chunks)

        # 5. 完成
        await _update_parse_task(document_id, STAGE_DONE, _PROGRESS[STAGE_DONE], status="ready")
        await _update_document_status(document_id, "ready")

        duration_ms = int((time.monotonic() - start_ts) * 1000)
        record_doc_parse(file_format, duration_ms, True)
        DOC_CHUNK_COUNT.observe(len(chunks))
        logger.info(
            f"[task] parse_document 完成: document_id={document_id}, "
            f"chunks={len(chunks)}, duration_ms={duration_ms}"
        )
        return {"document_id": document_id, "chunk_count": len(chunks), "status": "ready"}

    except Exception as exc:
        duration_ms = int((time.monotonic() - start_ts) * 1000)
        record_doc_parse(file_format, duration_ms, False)
        await _mark_failed_safe(document_id, str(exc))
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
async def _chunk_and_embed_async(document_id: int) -> dict:
    """断点续跑：重新解析→分块→向量化→落库。"""
    # 同 _parse_document_async：先 dispose 旧循环上的连接池
    from app.database import engine

    await engine.dispose()

    start_ts = time.monotonic()
    doc = await _load_document(document_id)
    file_format = getattr(doc, "file_format", "") or _infer_format(getattr(doc, "file_path", ""))

    try:
        parsed_doc = await _do_parse(doc)
        cleaner = DocumentCleaner()
        parsed_doc = cleaner.clean(parsed_doc)
        await _update_parse_task(document_id, STAGE_CHUNKING, _PROGRESS[STAGE_CHUNKING])
        chunker = SemanticChunker()
        chunks = chunker.chunk(parsed_doc)
        await _persist(document_id, doc, chunks)
        await _update_parse_task(document_id, STAGE_DONE, _PROGRESS[STAGE_DONE], status="ready")
        await _update_document_status(document_id, "ready")

        duration_ms = int((time.monotonic() - start_ts) * 1000)
        record_doc_parse(file_format, duration_ms, True)
        DOC_CHUNK_COUNT.observe(len(chunks))
        return {"document_id": document_id, "chunk_count": len(chunks), "status": "ready"}
    except Exception as exc:
        duration_ms = int((time.monotonic() - start_ts) * 1000)
        record_doc_parse(file_format, duration_ms, False)
        await _mark_failed_safe(document_id, str(exc))
        raise


# ===========================================================================
# 阶段实现
# ===========================================================================


async def _do_parse(doc: Document) -> Any:
    """调用 ParserRegistry 解析文档。"""
    file_path = getattr(doc, "file_path", "")
    file_format = (getattr(doc, "file_format", "") or _infer_format(file_path)).lower()
    parser = parser_registry.get_parser(file_format)
    parsed = await parser.parse(file_path)
    parsed.metadata.setdefault("document_id", getattr(doc, "id", None))
    return parsed


async def _persist(document_id: int, doc: Document, chunks: list[Chunk]) -> None:
    """向量化 + 写 DocumentChunk 表 + 写 Milvus。

    向量化阶段先于 DB 写入，以便在 DB 落盘时同步写入向量库，保证一致性。
    """
    await _update_parse_task(document_id, STAGE_EMBEDDING, _PROGRESS[STAGE_EMBEDDING])
    contents = [c.content for c in chunks]
    embeddings = await embedder.embed(contents) if contents else []

    await _update_parse_task(document_id, STAGE_PERSISTING, _PROGRESS[STAGE_PERSISTING])
    department_id = getattr(doc, "department_id", None)

    # 写 DocumentChunk 表，并构建 local_index -> db_id 映射用于父子关系
    id_map: dict[int, int] = {}
    async with db_session() as session:
        # 清理旧分块（断点续跑场景）
        await _delete_old_chunks(session, document_id)

        chunk_objs: list[tuple[Chunk, DocumentChunk]] = []
        for local_idx, chunk in enumerate(chunks):
            obj = DocumentChunk(
                document_id=document_id,
                department_id=department_id,
                parent_chunk_id=None,  # 二次回填
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                token_count=chunk.token_count,
                char_count=chunk.char_count,
                heading_path=chunk.heading_path,
                page_number=chunk.page_number,
                metadata_=chunk.metadata,
            )
            session.add(obj)
            chunk_objs.append((chunk, obj))

        await session.flush()  # 获取自增 id

        # 回填父子关系（本地索引 → DB id）
        for local_idx, (chunk, obj) in enumerate(chunk_objs):
            id_map[local_idx] = obj.id
        for chunk, obj in chunk_objs:
            if chunk.parent_chunk_id is not None and chunk.parent_chunk_id in id_map:
                obj.parent_chunk_id = id_map[chunk.parent_chunk_id]

        await session.commit()
        logger.info(f"已写入 {len(chunk_objs)} 个 DocumentChunk (document_id={document_id})")

    # 写 Milvus（partition_key=department_id）；不可用则降级跳过
    if milvus_store is not None and embeddings:
        try:
            await milvus_store.insert_chunks(chunks, embeddings, department_id)
            logger.info(f"已写入 Milvus {len(embeddings)} 条向量 (dept={department_id})")
        except Exception as exc:  # noqa: BLE001 向量库故障不阻断主流程
            logger.warning(f"Milvus 写入失败，已降级跳过向量入库: {exc}")
    elif milvus_store is None:
        logger.info("milvus_store 不可用，跳过向量入库（离线模式）")


# ===========================================================================
# DB 辅助
# ===========================================================================


async def _load_document(document_id: int) -> Document:
    """加载 Document 记录。"""
    from sqlalchemy import select

    async with db_session() as session:
        result = await session.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()
        if doc is None:
            raise ValueError(f"Document 不存在: {document_id}")
        return doc


async def _update_document_status(document_id: int, status: str) -> None:
    from sqlalchemy import update

    async with db_session() as session:
        await session.execute(
            update(Document).where(Document.id == document_id).values(status=status)
        )
        await session.commit()


async def _update_parse_task(
    document_id: int,
    stage: str,
    progress: float,
    status: str | None = None,
    error: str | None = None,
) -> None:
    """更新或创建 ParseTask 进度记录。"""
    from sqlalchemy import select

    async with db_session() as session:
        result = await session.execute(
            select(ParseTask).where(ParseTask.document_id == document_id)
        )
        task = result.scalar_one_or_none()
        values: dict[str, Any] = {"stage": stage, "progress": float(progress)}
        if status is not None:
            values["status"] = status
        if error is not None:
            values["error_message"] = error

        if task is None:
            task = ParseTask(document_id=document_id, **values)
            session.add(task)
        else:
            for k, v in values.items():
                setattr(task, k, v)
        await session.commit()


async def _delete_old_chunks(session: Any, document_id: int) -> None:
    """删除旧分块，支持断点续跑重写。"""
    from sqlalchemy import delete

    await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))


async def _mark_failed_safe(document_id: int, error_message: str) -> None:
    """安全标记失败（吞掉自身异常，确保不掩盖原始错误）。"""
    try:
        await _update_parse_task(
            document_id,
            STAGE_FAILED,
            0.0,
            status="failed",
            error=error_message[:1000],
        )
        await _update_document_status(document_id, "failed")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"标记失败状态时出错（已忽略）: {exc}")


def _infer_format(file_path: str) -> str:
    """从文件路径推断格式扩展名。"""
    return os.path.splitext(file_path)[1].lower().lstrip(".") or "unknown"
