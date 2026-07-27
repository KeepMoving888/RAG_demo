"""
图谱抽取触发脚本

功能:
1. 遍历所有 ``status="ready"`` 的文档;
2. 触发 ``extract_entities_task.delay(document_id)`` 异步抽取实体关系;
3. 打印抽取进度 (已触发 / 失败 / 总数).

用法:
    python -m scripts.extract_graph
    或: python scripts/extract_graph.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 确保 backend 在 sys.path 中
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from sqlalchemy import select  # noqa: E402

from app.database import AsyncSessionLocal, engine  # noqa: E402
from app.graphrag.extractor_tasks import extract_entities_task  # noqa: E402
from app.models import Document  # noqa: E402
from app.utils.logger import logger  # noqa: E402


async def main() -> int:
    """遍历 ready 文档, 触发图谱抽取任务, 打印进度.

    Returns:
        进程退出码 (0=成功).
    """
    logger.info("=" * 60)
    logger.info("开始触发图谱抽取...")
    logger.info("=" * 60)

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(
                Document.status == "ready",
                Document.is_deleted == False,  # noqa: E712
            ).order_by(Document.id.asc())
        )
        docs = result.scalars().all()

    total = len(docs)
    triggered = 0
    failed = 0
    task_ids: list[str] = []

    logger.info("待抽取文档数: {}", total)

    for i, doc in enumerate(docs, 1):
        try:
            async_result = extract_entities_task.delay(doc.id)
            tid = (
                async_result.id
                if hasattr(async_result, "id")
                else str(async_result)
            )
            task_ids.append(tid)
            triggered += 1
            logger.info(
                "[{}/{}] 已触发: doc_id={} title={!r} task_id={}",
                i, total, doc.id, doc.title[:50], tid,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.warning(
                "[{}/{}] 触发失败: doc_id={} error={}",
                i, total, doc.id, str(exc),
            )

    logger.info("=" * 60)
    logger.info("图谱抽取触发完成:")
    logger.info("  - 总文档数:    {}", total)
    logger.info("  - 成功触发:    {}", triggered)
    logger.info("  - 触发失败:    {}", failed)
    logger.info("  - 任务 ID 数:  {}", len(task_ids))
    logger.info("=" * 60)

    await engine.dispose()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
