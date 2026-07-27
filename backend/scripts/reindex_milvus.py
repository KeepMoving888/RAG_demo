"""从 DB 读取所有 chunks, 重新写入 Milvus (重建向量索引).

用途:
- 当 Milvus 数据丢失但 DB 完好时, 不必重新解析文档, 直接从 DB 拉取
  chunks -> BGE-M3 embedding -> Milvus insert.
- 也可用于切换 embedding 模型后的全量重建.

用法:
    python -m scripts.reindex_milvus
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from app.database import db_session, engine
from app.models import DocumentChunk, Document
from app.rag.embedder import get_embedder
from app.rag.milvus_store import milvus_store
from app.utils.logger import logger


BATCH_SIZE = 16  # BGE-M3 批大小


async def main() -> int:
    # 1. 初始化 Milvus
    await engine.dispose()
    logger.info("初始化 Milvus Collection...")
    ok = await milvus_store.init_collection()
    if not ok:
        logger.error("Milvus 初始化失败, 退出")
        return 1
    logger.info("Milvus 就绪: partitions={}", milvus_store.partitions)

    # 2. 加载 embedder
    embedder = get_embedder()
    logger.info("预热 BGE-M3 模型...")
    await embedder.embed(["warmup"])
    logger.info("BGE-M3 就绪: loaded={}", embedder.is_loaded)

    # 3. 拉取所有 chunks (按 department_id 分组, 用于 partition 路由)
    async with db_session() as session:
        result = await session.execute(
            select(DocumentChunk, Document.department_id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .order_by(DocumentChunk.id)
        )
        rows = result.all()

    logger.info("DB 中 chunk 总数: {}", len(rows))
    if not rows:
        logger.warning("DB 中无 chunk, 退出")
        return 0

    # 4. 按 department_id 分组 (None / 0 -> _public, others -> dept_<id>)
    groups: dict = {}
    for chunk, dept_id in rows:
        key = dept_id
        groups.setdefault(key, []).append(chunk)

    logger.info("分组: {}", {k: len(v) for k, v in groups.items()})

    # 5. 逐组 embedding + Milvus 插入
    total_inserted = 0
    start_ts = time.monotonic()
    for dept_id, chunks in groups.items():
        logger.info("处理 dept_id={} ({} chunks)...", dept_id, len(chunks))

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            contents = [c.content for c in batch]
            # BGE-M3 批量向量化
            t0 = time.monotonic()
            embeddings = await embedder.embed(contents)
            emb_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "  batch {}/{}: {} 条, embedding 耗时 {:.0f}ms",
                i // BATCH_SIZE + 1,
                (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE,
                len(batch), emb_ms,
            )

            # 构造 chunks dict (与 ingestion tasks 一致的字段)
            chunk_dicts = []
            for c in batch:
                chunk_dicts.append({
                    "chunk_id": str(c.id),  # 用 DB 自增 id 作为 chunk_id
                    "document_id": c.document_id,
                    "content": c.content,
                    "heading_path": c.heading_path or "",
                    "page_number": c.page_number or 0,
                })

            # 写入 Milvus
            t0 = time.monotonic()
            n = await milvus_store.insert_chunks(
                chunk_dicts, embeddings, dept_id
            )
            ins_ms = (time.monotonic() - t0) * 1000
            total_inserted += n
            logger.info(
                "  写入完成: {} 条, 耗时 {:.0f}ms", n, ins_ms,
            )

    total_ms = (time.monotonic() - start_ts) * 1000
    logger.info(
        "全部完成: 共写入 {} 条向量, 总耗时 {:.0f}ms",
        total_inserted, total_ms,
    )

    # 6. 验证
    col = milvus_store._collection
    col.flush()
    logger.info("Milvus num_entities: {}", col.num_entities)
    logger.info("Milvus partitions: {}", [getattr(p, 'name', str(p)) for p in col.partitions])

    await engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
