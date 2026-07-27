"""
问答会话与消息 ORM

设计:
1. QASession: 多轮对话会话 (绑定用户, TTL 由 Redis 管理)
2. QAMessage: 单轮消息 (含 user_query / answer / citations / retrieved_chunks)
3. Citation: 答案溯源 (answer_id + chunk_id + 引用片段)
"""

from sqlalchemy import JSON, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IDMixin, TimestampMixin


class QASession(Base, IDMixin, TimestampMixin):
    """问答会话表"""

    __tablename__ = "qa_sessions"
    __table_args__ = (
        Index("ix_qa_session_user", "user_id"),
        {"comment": "问答会话表"},
    )

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    title: Mapped[str | None] = mapped_column(String(256), nullable=True, comment="会话标题")
    department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id"), nullable=True, comment="会话归属部门 (权限快照)"
    )
    turn_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_archived: Mapped[bool] = mapped_column(default=False, nullable=False)

    messages: Mapped[list["QAMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="QAMessage.id"
    )


class QAMessage(Base, IDMixin, TimestampMixin):
    """问答消息表 (单轮对话记录)"""

    __tablename__ = "qa_messages"
    __table_args__ = (
        Index("ix_qa_msg_session", "session_id"),
        Index("ix_qa_msg_user", "user_id"),
        {"comment": "问答消息表"},
    )

    session_id: Mapped[int] = mapped_column(ForeignKey("qa_sessions.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    role: Mapped[str] = mapped_column(
        String(16), nullable=False, comment="user / assistant / system"
    )
    user_query: Mapped[str | None] = mapped_column(Text, nullable=True, comment="用户原始问题")
    rewritten_query: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="改写后的查询 (融合历史上下文)"
    )
    answer: Mapped[str | None] = mapped_column(Text, nullable=True, comment="模型回答")
    answer_source: Mapped[str] = mapped_column(
        String(32),
        default="llm",
        nullable=False,
        comment="回答来源: llm / cache / fallback",
    )
    retrieved_chunks: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="检索召回的 chunk 列表 (含得分)"
    )
    citations: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="答案溯源引用 [{chunk_id, doc_id, doc_title, snippet}]"
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="响应耗时")
    cache_hit: Mapped[bool] = mapped_column(default=False, nullable=False)
    feedback: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="用户反馈: 1 正确 / -1 错误 / NULL 未反馈"
    )

    session: Mapped[QASession] = relationship(back_populates="messages")
