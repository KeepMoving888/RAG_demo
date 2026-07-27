"""文档摄入流水线同步入口，供 API 层调用。

设计要点
========
1. **同步入口**：API 层（FastAPI 路由）调用本类提交/查询/重试/撤销解析任务，
   不直接接触 Celery 与 DB 细节，职责清晰。
2. **任务解耦**：``submit`` 仅创建 ParseTask 记录并投递 Celery，立即返回
   ``celery_task_id``，重活由 worker 异步处理，保证 API 响应及时。
3. **状态查询**：``get_status`` 从 ParseTask 表读取阶段、进度、状态，供前端
   实时展示进度条。
4. **失败重试**：``retry`` 重置状态后重新投递任务；``cancel`` 调用 Celery
   revoke 撤销尚未执行的 worker 任务。
"""
from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import select

from app.celery_app import celery_app
from app.database import db_session
from app.ingestion.tasks import parse_document
from app.models import Document, ParseTask
from app.utils.logger import logger


class DocumentPipeline:
    """文档流水线对外入口。

    所有方法为 async，由 API 层在异步上下文中调用。
    """

    # ------------------------------------------------------------------
    # 提交
    # ------------------------------------------------------------------

    async def submit(self, document_id: int) -> str:
        """提交文档解析任务。

        Args:
            document_id: 待解析文档 ID。

        Returns:
            Celery 任务 ID（``celery_task_id``）。

        Raises:
            ValueError: Document 不存在或状态不允许提交。
        """
        doc = await self._load_document(document_id)
        if doc is None:
            raise ValueError(f"Document 不存在: {document_id}")

        # 更新 Document 状态为处理中
        await self._update_document_status(document_id, "processing")
        # 创建/重置 ParseTask
        await self._upsert_parse_task(
            document_id, stage="queued", progress=0.0, status="processing"
        )

        # 投递 Celery 任务
        async_result = parse_document.delay(document_id)
        celery_task_id = async_result.id if hasattr(async_result, "id") else str(async_result)
        # 回填 celery_task_id 便于后续 cancel
        await self._update_parse_task_celery_id(document_id, celery_task_id)

        logger.info(
            f"已提交解析任务: document_id={document_id}, celery_task_id={celery_task_id}"
        )
        return celery_task_id

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    async def get_status(self, document_id: int) -> dict:
        """查询解析状态。

        Returns:
            dict: 含 stage / progress / status / error_message / celery_task_id。
        """
        async with db_session() as session:
            result = await session.execute(
                select(ParseTask).where(ParseTask.document_id == document_id)
            )
            task = result.scalar_one_or_none()
            doc_result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            doc = doc_result.scalar_one_or_none()

        if task is None and doc is None:
            return {"document_id": document_id, "status": "not_found"}

        return {
            "document_id": document_id,
            "stage": getattr(task, "stage", None) if task else None,
            "progress": float(getattr(task, "progress", 0.0)) if task else 0.0,
            "status": getattr(task, "status", None) if task else getattr(doc, "status", None),
            "error_message": getattr(task, "error_message", None) if task else None,
            "celery_task_id": getattr(task, "celery_task_id", None) if task else None,
        }

    # ------------------------------------------------------------------
    # 重试
    # ------------------------------------------------------------------

    async def retry(self, document_id: int) -> str:
        """失败重试：重置状态后重新投递。

        Args:
            document_id: 失败文档 ID。

        Returns:
            新的 Celery 任务 ID。
        """
        doc = await self._load_document(document_id)
        if doc is None:
            raise ValueError(f"Document 不存在: {document_id}")

        await self._update_document_status(document_id, "processing")
        await self._upsert_parse_task(
            document_id, stage="queued", progress=0.0, status="processing", error=None
        )

        async_result = parse_document.delay(document_id)
        celery_task_id = async_result.id if hasattr(async_result, "id") else str(async_result)
        await self._update_parse_task_celery_id(document_id, celery_task_id)
        logger.info(f"已重新投递解析任务: document_id={document_id}, task_id={celery_task_id}")
        return celery_task_id

    # ------------------------------------------------------------------
    # 撤销
    # ------------------------------------------------------------------

    async def cancel(self, document_id: int) -> None:
        """撤销尚未执行的解析任务。

        已在执行的任务以 terminate 方式撤销；撤销后更新状态为 cancelled。
        """
        async with db_session() as session:
            result = await session.execute(
                select(ParseTask).where(ParseTask.document_id == document_id)
            )
            task = result.scalar_one_or_none()
            celery_task_id = getattr(task, "celery_task_id", None) if task else None

        if celery_task_id:
            try:
                celery_app.control.revoke(celery_task_id, terminate=False)
                logger.info(f"已撤销 Celery 任务: {celery_task_id}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"撤销 Celery 任务失败（已忽略）: {exc}")

        await self._update_document_status(document_id, "cancelled")
        await self._upsert_parse_task(
            document_id, stage="cancelled", progress=0.0, status="cancelled"
        )

    # ------------------------------------------------------------------
    # DB 辅助
    # ------------------------------------------------------------------

    @staticmethod
    async def _load_document(document_id: int) -> Optional[Document]:
        async with db_session() as session:
            result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            return result.scalar_one_or_none()

    @staticmethod
    async def _update_document_status(document_id: int, status: str) -> None:
        from sqlalchemy import update

        async with db_session() as session:
            await session.execute(
                update(Document).where(Document.id == document_id).values(status=status)
            )
            await session.commit()

    @staticmethod
    async def _upsert_parse_task(
        document_id: int,
        stage: str,
        progress: float,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        async with db_session() as session:
            result = await session.execute(
                select(ParseTask).where(ParseTask.document_id == document_id)
            )
            task = result.scalar_one_or_none()
            values: dict[str, Any] = {
                "stage": stage,
                "progress": float(progress),
                "status": status,
            }
            if error is not None:
                values["error_message"] = error
            if task is None:
                task = ParseTask(document_id=document_id, **values)
                session.add(task)
            else:
                for k, v in values.items():
                    setattr(task, k, v)
            await session.commit()

    @staticmethod
    async def _update_parse_task_celery_id(document_id: int, celery_task_id: str) -> None:
        async with db_session() as session:
            result = await session.execute(
                select(ParseTask).where(ParseTask.document_id == document_id)
            )
            task = result.scalar_one_or_none()
            if task is not None:
                setattr(task, "celery_task_id", celery_task_id)
                await session.commit()
