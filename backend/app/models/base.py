"""
Enterprise RAG Knowledge Base - SQLAlchemy ORM 基类与公共 Mixin

设计:
1. Base: 声明式基类, 全局统一 metadata
2. TimestampMixin: created_at / updated_at 自动维护
3. SoftDeleteMixin: is_deleted 软删除标记
4. 所有时区使用 UTC 存储, 业务层按需转换
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import BigInteger, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# 兼容 Python 3.10 (datetime.UTC 在 3.11+ 才有)
UTC = timezone.utc


class Base(DeclarativeBase):
    """声明式基类"""

    metadata: Any  # 全局共享 metadata, 供 Alembic 迁移引用


class TimestampMixin:
    """时间戳 Mixin: 自动维护创建/更新时间 (UTC)"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
        comment="创建时间 (UTC)",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
        comment="更新时间 (UTC)",
    )


class SoftDeleteMixin:
    """软删除 Mixin"""

    is_deleted: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        index=True,
        comment="软删除标记",
    )


class IDMixin:
    """BigInt 主键 Mixin (兼容大规模 ID)"""

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="主键 ID",
    )


__all__ = ["Base", "TimestampMixin", "SoftDeleteMixin", "IDMixin"]
