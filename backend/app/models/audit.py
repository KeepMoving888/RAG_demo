"""
审计日志 ORM

记录关键操作 (文档上传/删除/权限变更/图谱抽取), 满足企业级合规审计需求.
"""

from sqlalchemy import String, Integer, ForeignKey, Index, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IDMixin, TimestampMixin


class AuditLog(Base, IDMixin, TimestampMixin):
    """审计日志表"""

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_user", "user_id"),
        Index("ix_audit_action", "action"),
        Index("ix_audit_resource", "resource_type", "resource_id"),
        {"comment": "审计日志表"},
    )

    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(
        String(64), nullable=False,
        comment="动作: doc.upload / doc.delete / auth.login / graph.rebuild",
    )
    resource_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="详情")
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="success", nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
