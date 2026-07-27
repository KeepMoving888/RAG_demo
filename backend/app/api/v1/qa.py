"""
智能问答 API 路由

提供同步问答 / SSE 流式问答 / 会话历史 / 会话列表 / 用户反馈 / 会话归档接口.

设计要点:
1. 同步问答: 调用 ``AnswerGenerator.generate``, 无 session_id 时自动 ``create_session``;
2. 流式问答: 使用 ``sse_starlette.EventSourceResponse`` 包装 ``generate_stream``,
   每个 token / 元数据帧作为独立 SSE event 推送, 前端可在流末尾解析溯源元数据;
3. 限流: ``/ask`` 与 ``/stream`` 均触发限流, 防止单用户击穿 LLM 配额;
4. 反馈: 1=正确 / -1=错误, 写入 ``QAMessage.feedback`` 供后续评估与微调.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.rate_limit import rate_limit_dependency
from app.core.security import get_current_user
from app.database import get_db
from app.dialog.context_manager import DialogContextManager
from app.dialog.generator import get_generator
from app.models import QAMessage, QASession, User
from app.schemas.common import PaginatedResponse, SuccessResponse
from app.utils.logger import logger

router = APIRouter()


# ======================== Schemas ========================
class AskRequest(BaseModel):
    """同步问答请求"""

    query: str = Field(min_length=1, max_length=2048)
    session_id: str | None = Field(None, description="会话 ID, 缺省自动创建")
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    """问答响应"""

    answer: str
    citations: list[dict[str, Any]] = []
    rewritten_query: str | None = None
    retrieved_chunks: list[dict[str, Any]] = []
    latency_ms: float = 0.0
    cache_hit: bool = False
    answer_source: str = "llm"
    session_id: str
    turn_count: int = 0


class StreamRequest(BaseModel):
    """流式问答请求 (与 AskRequest 同构)"""

    query: str = Field(min_length=1, max_length=2048)
    session_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class MessageOut(BaseModel):
    """消息对外结构"""

    id: int
    session_id: int
    role: str
    user_query: str | None = None
    rewritten_query: str | None = None
    answer: str | None = None
    answer_source: str = "llm"
    latency_ms: int | None = None
    cache_hit: bool = False
    feedback: int | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class SessionOut(BaseModel):
    """会话对外结构"""

    id: int
    user_id: int
    title: str | None = None
    department_id: int | None = None
    turn_count: int = 0
    is_archived: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class FeedbackRequest(BaseModel):
    """用户反馈"""

    feedback: int = Field(..., ge=-1, le=1, description="1=正确, -1=错误, 0=取消")


# ======================== 辅助 ========================
async def _ensure_session(
    context_manager: DialogContextManager,
    user: User,
    session_id: str | None,
) -> str:
    """确保 session_id 有效, 缺省时自动创建."""
    if session_id:
        return session_id
    return await context_manager.create_session(user.id, user.department_id)


async def _resolve_pg_session_id(db: AsyncSession, session_uuid: str, user: User) -> int | None:
    """从 Redis 元数据反查 PostgreSQL QASession 主键.

    若 Redis 不可用或会话不存在, 则回退到按 user_id + title 前缀匹配.
    """
    context_manager = DialogContextManager()
    data = await context_manager._load(session_uuid)  # noqa: SLF001
    if data and data.get("pg_session_id"):
        return int(data["pg_session_id"])

    # 回退: 按 title 前缀匹配
    prefix = f"dialog:{session_uuid[:8]}"
    result = await db.execute(
        select(QASession).where(
            QASession.user_id == user.id,
            QASession.title == prefix,
            QASession.is_archived == False,  # noqa: E712
        )
    )
    row = result.scalar_one_or_none()
    return row.id if row else None


# ======================== 路由 ========================
@router.post("/ask", response_model=SuccessResponse[AskResponse])
async def ask(
    payload: AskRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """同步问答: 调用 AnswerGenerator.generate 返回完整答案.

    无 session_id 时自动创建新会话. 触发限流.
    """
    await rate_limit_dependency(request, endpoint="qa.ask")

    context_manager = DialogContextManager()
    session_id = await _ensure_session(context_manager, current_user, payload.session_id)

    generator = get_generator()
    try:
        result = await generator.generate(
            query=payload.query,
            session_id=session_id,
            user_id=current_user.id,
            department_id=current_user.department_id,
            top_k=payload.top_k,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("问答生成失败 session={}: {}", session_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"问答生成失败: {exc}",
        )

    return SuccessResponse[AskResponse](
        data=AskResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []) or [],
            rewritten_query=result.get("rewritten_query"),
            retrieved_chunks=result.get("retrieved_chunks", []) or [],
            latency_ms=float(result.get("latency_ms", 0.0) or 0.0),
            cache_hit=bool(result.get("cache_hit", False)),
            answer_source=result.get("answer_source", "llm"),
            session_id=session_id,
            turn_count=int(result.get("turn_count", 0) or 0),
        )
    )


@router.post("/stream")
async def ask_stream(
    payload: StreamRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """SSE 流式问答: 每个 token / 元数据帧作为独立 SSE event 推送.

    前端约定:
    - event: token  -> data: 文本片段;
    - event: meta   -> data: JSON 元数据 (含 citations / latency / session_id);
    - event: error  -> data: 错误描述 (异常时).
    """
    await rate_limit_dependency(request, endpoint="qa.stream")

    context_manager = DialogContextManager()
    session_id = await _ensure_session(context_manager, current_user, payload.session_id)
    generator = get_generator()

    async def event_generator():
        try:
            async for chunk in generator.generate_stream(
                query=payload.query,
                session_id=session_id,
                user_id=current_user.id,
                department_id=current_user.department_id,
                top_k=payload.top_k,
            ):
                # 元数据帧 (JSON 字符串, 带 __meta__ 标记)
                if chunk.lstrip().startswith("{") and '"__meta__"' in chunk:
                    yield {"event": "meta", "data": chunk}
                else:
                    yield {"event": "token", "data": chunk}
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式问答异常 session={}: {}", session_id, str(exc))
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": f"流式生成失败: {exc}", "session_id": session_id},
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())


@router.get("/history/{session_id}", response_model=PaginatedResponse[MessageOut])
async def get_history(
    session_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取会话历史消息 (从 QAMessage 表读取)."""
    pg_session_id = await _resolve_pg_session_id(db, session_id, current_user)
    if pg_session_id is None:
        return PaginatedResponse[MessageOut](
            data=[],
            total=0,
            page=page,
            page_size=page_size,
            total_pages=0,
        )

    # 权限: 会话必须属于当前用户
    sess_result = await db.execute(select(QASession).where(QASession.id == pg_session_id))
    sess = sess_result.scalar_one_or_none()
    if sess is None or sess.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权访问该会话",
        )

    count_stmt = (
        select(func.count()).select_from(QAMessage).where(QAMessage.session_id == pg_session_id)
    )
    total = (await db.execute(count_stmt)).scalar_one()

    offset = (page - 1) * page_size
    stmt = (
        select(QAMessage)
        .where(QAMessage.session_id == pg_session_id)
        .order_by(QAMessage.id.asc())
        .offset(offset)
        .limit(page_size)
    )
    msgs = (await db.execute(stmt)).scalars().all()

    return PaginatedResponse[MessageOut](
        data=[
            MessageOut(
                id=m.id,
                session_id=m.session_id,
                role=m.role,
                user_query=m.user_query,
                rewritten_query=m.rewritten_query,
                answer=m.answer,
                answer_source=m.answer_source,
                latency_ms=m.latency_ms,
                cache_hit=m.cache_hit,
                feedback=m.feedback,
                created_at=m.created_at,
            )
            for m in msgs
        ],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if page_size else 0,
    )


@router.get("/sessions", response_model=PaginatedResponse[SessionOut])
async def list_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """当前用户会话列表 (分页, 仅未归档)."""
    count_stmt = (
        select(func.count())
        .select_from(QASession)
        .where(
            QASession.user_id == current_user.id,
            QASession.is_archived == False,  # noqa: E712
        )
    )
    total = (await db.execute(count_stmt)).scalar_one()

    offset = (page - 1) * page_size
    stmt = (
        select(QASession)
        .where(
            QASession.user_id == current_user.id,
            QASession.is_archived == False,  # noqa: E712
        )
        .order_by(QASession.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    sessions = (await db.execute(stmt)).scalars().all()

    return PaginatedResponse[SessionOut](
        data=[
            SessionOut(
                id=s.id,
                user_id=s.user_id,
                title=s.title,
                department_id=s.department_id,
                turn_count=s.turn_count,
                is_archived=s.is_archived,
                created_at=s.created_at,
            )
            for s in sessions
        ],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if page_size else 0,
    )


@router.post("/feedback/{message_id}", response_model=SuccessResponse[dict])
async def feedback(
    message_id: int,
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """用户反馈 (1=正确 / -1=错误 / 0=取消)."""
    msg_result = await db.execute(select(QAMessage).where(QAMessage.id == message_id))
    msg = msg_result.scalar_one_or_none()
    if msg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"消息不存在: {message_id}",
        )
    if msg.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权对该消息反馈",
        )

    msg.feedback = payload.feedback
    await db.flush()

    logger.info(
        "用户反馈: msg_id={} feedback={} by={}",
        message_id,
        payload.feedback,
        current_user.id,
    )

    return SuccessResponse[dict](
        data={
            "message_id": message_id,
            "feedback": payload.feedback,
        }
    )


@router.delete("/sessions/{session_id}", response_model=SuccessResponse[dict])
async def archive_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """归档会话 (清理 Redis 热数据, PostgreSQL 标记 is_archived=True)."""
    pg_session_id = await _resolve_pg_session_id(db, session_id, current_user)
    if pg_session_id is not None:
        # 权限校验
        sess_result = await db.execute(select(QASession).where(QASession.id == pg_session_id))
        sess = sess_result.scalar_one_or_none()
        if sess is not None and sess.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="无权归档该会话",
            )

    context_manager = DialogContextManager()
    await context_manager.archive_session(session_id)

    return SuccessResponse[dict](
        data={
            "session_id": session_id,
            "archived": True,
        }
    )
