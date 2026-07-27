"""
文档与分块 ORM

设计要点:
1. Document: 文档元数据 (含部门权限标签 department_id, 公开文档为 NULL)
2. DocumentChunk: 文档分块 (含 parent_chunk_id 父子层级, 供语义分块保留上下文)
3. ParseTask: 异步解析任务状态 (Celery task_id 关联)
4. 可见性规则:
   - department_id IS NULL → 全公司可见
   - department_id = X → 仅部门 X 可见
"""

from datetime import datetime
from typing import Any

from sqlalchemy import String, Integer, Float, ForeignKey, Index, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IDMixin, TimestampMixin, SoftDeleteMixin


class Document(Base, IDMixin, TimestampMixin, SoftDeleteMixin):
    """文档表"""

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_doc_department", "department_id"),
        Index("ix_doc_status", "status"),
        {"comment": "文档表"},
    )

    title: Mapped[str] = mapped_column(String(256), nullable=False, comment="文档标题")
    file_name: Mapped[str] = mapped_column(String(256), nullable=False, comment="原始文件名")
    file_path: Mapped[str] = mapped_column(String(512), nullable=False, comment="存储路径")
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, comment="字节")
    file_format: Mapped[str] = mapped_column(
        String(16), nullable=False, comment="格式: pdf / docx / image / txt / md"
    )
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    md5_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True, comment="文件 MD5, 去重用")
    department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id"), nullable=True,
        comment="归属部门 (NULL = 全公司可见)",
    )
    category: Mapped[str] = mapped_column(
        String(32), default="other", nullable=False,
        comment="文档分类: product_manual / policy / faq / other",
    )
    status: Mapped[str] = mapped_column(
        String(16), default="pending", nullable=False,
        comment="解析状态: pending / parsing / chunking / embedding / ready / failed",
    )
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="页数 (PDF)")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    parsed_at: Mapped[datetime | None] = mapped_column(nullable=True, comment="解析完成时间")

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base, IDMixin, TimestampMixin):
    """文档分块表 (PostgreSQL pgvector 暂不启用, 真正向量存储于 Milvus)"""

    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_chunk_doc", "document_id"),
        Index("ix_chunk_parent", "parent_chunk_id"),
        Index("ix_chunk_dept", "department_id"),
        {"comment": "文档分块表"},
    )

    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), nullable=False)
    department_id: Mapped[int | None] = mapped_column(
        nullable=True, comment="冗余字段, 检索时直接过滤部门权限"
    )
    parent_chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_chunks.id"), nullable=True,
        comment="父分块 ID (语义层级保留)",
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="文档内序号")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="分块文本")
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    heading_path: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="标题层级路径, 如 '第3章/3.2 安装/3.2.1 步骤'",
    )
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="页码")
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSON, nullable=True, comment="扩展元数据 (bbox, font_size 等)"
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class ParseTask(Base, IDMixin, TimestampMixin):
    """异步解析任务表 (关联 Celery task_id)"""

    __tablename__ = "parse_tasks"
    __table_args__ = (
        Index("ix_parse_task_doc", "document_id"),
        {"comment": "异步解析任务表"},
    )

    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), nullable=False)
    celery_task_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    stage: Mapped[str] = mapped_column(
        String(32), default="queued", nullable=False,
        comment="阶段: queued / parsing / chunking / embedding / done / failed",
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False, comment="0.0-1.0")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
