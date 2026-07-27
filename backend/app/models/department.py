"""
部门与用户 ORM

权限设计:
- 部门 (Department): 一级业务单元, 拥有可见文档范围
- 用户 (User): 归属部门, 拥有 role (admin / staff)
- 文档可见性: doc.department_id IN (用户部门 + 公开文档 NULL)
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IDMixin, SoftDeleteMixin, TimestampMixin


class Department(Base, IDMixin, TimestampMixin, SoftDeleteMixin):
    """部门表"""

    __tablename__ = "departments"
    __table_args__ = (
        UniqueConstraint("code", name="uq_department_code"),
        {"comment": "部门表"},
    )

    code: Mapped[str] = mapped_column(String(64), nullable=False, comment="部门代码")
    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="部门名称")
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id"), nullable=True, comment="上级部门 ID"
    )
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)

    users: Mapped[list["User"]] = relationship(back_populates="department")


class User(Base, IDMixin, TimestampMixin, SoftDeleteMixin):
    """用户表"""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("email", name="uq_user_email"),
        Index("ix_user_department", "department_id"),
        {"comment": "用户表"},
    )

    email: Mapped[str] = mapped_column(String(128), nullable=False, comment="邮箱 (登录名)")
    hashed_password: Mapped[str] = mapped_column(
        String(256), nullable=False, comment="bcrypt 哈希密码"
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False, comment="姓名")
    department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id"), nullable=True, comment="所属部门 (NULL 表示跨部门可见)"
    )
    role: Mapped[str] = mapped_column(
        String(16), default="staff", nullable=False, comment="角色: admin / staff"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime, default=None, nullable=True, comment="最近登录时间"
    )

    department: Mapped[Department | None] = relationship(back_populates="users")
