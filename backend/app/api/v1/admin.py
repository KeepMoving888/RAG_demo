"""
管理后台 API 路由

提供用户 / 部门 / 系统统计 / 审计日志接口, 全部依赖 ``require_role("admin")``.

设计要点:
1. 用户管理: 列表 / 创建 / 更新 (含禁用 / 启用 / 改部门 / 改角色);
2. 部门管理: 列表 / 创建;
3. 系统统计: 用户数 / 文档数 / 会话数 / 检索次数 (审计日志 action=retrieval.search 计数) /
   缓存命中率 (检索缓存命中数 / 检索总数) / 图谱节点数;
4. 审计日志: 分页 + action 过滤, 供合规追溯.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user, hash_password, require_role
from app.database import get_db
from app.graphrag.neo4j_store import neo4j_store
from app.models import AuditLog, Department, Document, QASession, User
from app.schemas.common import PaginatedResponse, SuccessResponse
from app.utils.logger import logger

router = APIRouter()


# ======================== Schemas ========================
class UserOut(BaseModel):
    """用户对外结构"""
    id: int
    email: str
    name: str
    department_id: Optional[int] = None
    role: str
    is_active: bool
    last_login_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class CreateUserRequest(BaseModel):
    """创建用户请求"""
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    name: str = Field(min_length=1, max_length=64)
    department_id: Optional[int] = None
    role: str = Field(default="staff", pattern="^(admin|staff)$")


class UpdateUserRequest(BaseModel):
    """更新用户请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=64)
    department_id: Optional[int] = None
    role: Optional[str] = Field(None, pattern="^(admin|staff)$")
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=6, max_length=128)


class DepartmentOut(BaseModel):
    """部门对外结构"""
    id: int
    code: str
    name: str
    parent_id: Optional[int] = None
    description: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class CreateDepartmentRequest(BaseModel):
    """创建部门请求"""
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    parent_id: Optional[int] = None
    description: Optional[str] = Field(None, max_length=512)


class SystemStats(BaseModel):
    """系统统计"""
    users: int = 0
    documents: int = 0
    qa_sessions: int = 0
    retrieval_count: int = 0
    cache_hit_rate: float = 0.0
    graph_nodes: int = 0


class AuditLogOut(BaseModel):
    """审计日志对外结构"""
    id: int
    user_id: Optional[int] = None
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    detail: Optional[dict[str, Any]] = None
    ip_address: Optional[str] = None
    status: str = "success"
    error: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ======================== 辅助 ========================
def _user_to_out(user: User) -> UserOut:
    return UserOut(
        id=user.id, email=user.email, name=user.name,
        department_id=user.department_id, role=user.role,
        is_active=user.is_active, last_login_at=user.last_login_at,
        created_at=user.created_at,
    )


def _dept_to_out(dept: Department) -> DepartmentOut:
    return DepartmentOut(
        id=dept.id, code=dept.code, name=dept.name,
        parent_id=dept.parent_id, description=dept.description,
        created_at=dept.created_at,
    )


async def _write_audit(
    db: AsyncSession,
    action: str,
    admin_id: int,
    status_: str = "success",
    error: Optional[str] = None,
    resource_id: Optional[str] = None,
    detail: Optional[dict] = None,
) -> None:
    """写入审计日志 (失败不阻断主流程)."""
    try:
        log = AuditLog(
            user_id=admin_id, action=action,
            resource_type="admin", resource_id=resource_id,
            detail=detail or {}, status=status_, error=error,
        )
        db.add(log)
        await db.flush()
    except Exception as exc:  # noqa: BLE001
        logger.warning("审计日志写入失败 action={}: {}", action, str(exc))


# ======================== 用户管理 ========================
@router.get("/users", response_model=PaginatedResponse[UserOut])
async def list_users(
    department_id: Optional[int] = Query(None, description="按部门过滤"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """用户列表 (分页 + 部门过滤)."""
    base = select(User).where(User.is_deleted == False)  # noqa: E712
    if department_id is not None:
        base = base.where(User.department_id == department_id)

    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    offset = (page - 1) * page_size
    stmt = base.order_by(User.created_at.desc()).offset(offset).limit(page_size)
    users = (await db.execute(stmt)).scalars().all()

    return PaginatedResponse[UserOut](
        data=[_user_to_out(u) for u in users],
        total=total, page=page, page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if page_size else 0,
    )


@router.post("/users", response_model=SuccessResponse[UserOut], status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: CreateUserRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """创建用户 (admin)."""
    # 校验部门
    if payload.department_id is not None:
        dept_result = await db.execute(
            select(Department).where(
                Department.id == payload.department_id,
                Department.is_deleted == False,  # noqa: E712
            )
        )
        if dept_result.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"部门不存在: {payload.department_id}",
            )

    # email 唯一
    exist = await db.execute(select(User).where(User.email == payload.email))
    if exist.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"邮箱已注册: {payload.email}",
        )

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        name=payload.name,
        department_id=payload.department_id,
        role=payload.role,
        is_active=True,
    )
    db.add(user)
    await db.flush()

    await _write_audit(
        db, "admin.user.create", admin.id,
        resource_id=str(user.id),
        detail={"email": user.email, "role": user.role},
    )

    return SuccessResponse[UserOut](data=_user_to_out(user))


@router.put("/users/{user_id}", response_model=SuccessResponse[UserOut])
async def update_user(
    user_id: int,
    payload: UpdateUserRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """更新用户 (含禁用 / 启用 / 改部门 / 改角色 / 重置密码)."""
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_deleted == False)  # noqa: E712
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户不存在: {user_id}",
        )

    changes: dict[str, Any] = {}
    if payload.name is not None:
        user.name = payload.name
        changes["name"] = payload.name
    if payload.department_id is not None:
        # 校验部门
        dept_result = await db.execute(
            select(Department).where(
                Department.id == payload.department_id,
                Department.is_deleted == False,  # noqa: E712
            )
        )
        if dept_result.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"部门不存在: {payload.department_id}",
            )
        user.department_id = payload.department_id
        changes["department_id"] = payload.department_id
    if payload.role is not None:
        user.role = payload.role
        changes["role"] = payload.role
    if payload.is_active is not None:
        user.is_active = payload.is_active
        changes["is_active"] = payload.is_active
    if payload.password is not None:
        user.hashed_password = hash_password(payload.password)
        changes["password_reset"] = True

    await db.flush()

    await _write_audit(
        db, "admin.user.update", admin.id,
        resource_id=str(user_id),
        detail=changes,
    )

    return SuccessResponse[UserOut](data=_user_to_out(user))


# ======================== 部门管理 ========================
@router.get("/departments", response_model=SuccessResponse[list[DepartmentOut]])
async def list_departments(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """部门列表 (全量, 不分页)."""
    result = await db.execute(
        select(Department).where(Department.is_deleted == False)  # noqa: E712
        .order_by(Department.id.asc())
    )
    depts = result.scalars().all()
    return SuccessResponse[list[DepartmentOut]](data=[_dept_to_out(d) for d in depts])


@router.post(
    "/departments",
    response_model=SuccessResponse[DepartmentOut],
    status_code=status.HTTP_201_CREATED,
)
async def create_department(
    payload: CreateDepartmentRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """创建部门."""
    # code 唯一
    exist = await db.execute(
        select(Department).where(Department.code == payload.code)
    )
    if exist.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"部门代码已存在: {payload.code}",
        )

    # 校验 parent_id
    if payload.parent_id is not None:
        parent_result = await db.execute(
            select(Department).where(
                Department.id == payload.parent_id,
                Department.is_deleted == False,  # noqa: E712
            )
        )
        if parent_result.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"上级部门不存在: {payload.parent_id}",
            )

    dept = Department(
        code=payload.code, name=payload.name,
        parent_id=payload.parent_id, description=payload.description,
    )
    db.add(dept)
    await db.flush()

    await _write_audit(
        db, "admin.department.create", admin.id,
        resource_id=str(dept.id),
        detail={"code": dept.code, "name": dept.name},
    )

    return SuccessResponse[DepartmentOut](data=_dept_to_out(dept))


# ======================== 系统统计 ========================
@router.get("/stats", response_model=SuccessResponse[SystemStats])
async def system_stats(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """系统统计: 用户数 / 文档数 / 会话数 / 检索次数 / 缓存命中率 / 图谱节点数."""
    # 用户数
    users_count = (await db.execute(
        select(func.count(User.id)).where(User.is_deleted == False)  # noqa: E712
    )).scalar_one()

    # 文档数
    docs_count = (await db.execute(
        select(func.count(Document.id)).where(Document.is_deleted == False)  # noqa: E712
    )).scalar_one()

    # 会话数
    sessions_count = (await db.execute(
        select(func.count(QASession.id)).where(QASession.is_archived == False)  # noqa: E712
    )).scalar_one()

    # 检索次数 (审计日志 action=retrieval.search)
    retrieval_count = (await db.execute(
        select(func.count(AuditLog.id)).where(AuditLog.action == "retrieval.search")
    )).scalar_one()

    # 缓存命中率 (从 QAMessage 表统计 cache_hit 占比)
    try:
        from app.models import QAMessage
        total_msgs = (await db.execute(
            select(func.count(QAMessage.id))
        )).scalar_one()
        hit_msgs = (await db.execute(
            select(func.count(QAMessage.id)).where(QAMessage.cache_hit == True)  # noqa: E712
        )).scalar_one()
        cache_hit_rate = (hit_msgs / total_msgs) if total_msgs > 0 else 0.0
    except Exception:  # noqa: BLE001
        cache_hit_rate = 0.0

    # 图谱节点数
    graph_nodes = 0
    try:
        stats = await neo4j_store.stats()
        graph_nodes = int(stats.get("nodes", 0) or 0)
    except Exception:  # noqa: BLE001
        pass

    return SuccessResponse[SystemStats](data=SystemStats(
        users=int(users_count or 0),
        documents=int(docs_count or 0),
        qa_sessions=int(sessions_count or 0),
        retrieval_count=int(retrieval_count or 0),
        cache_hit_rate=round(float(cache_hit_rate or 0.0), 4),
        graph_nodes=int(graph_nodes or 0),
    ))


# ======================== 审计日志 ========================
@router.get("/audit-logs", response_model=PaginatedResponse[AuditLogOut])
async def list_audit_logs(
    action: Optional[str] = Query(None, description="按 action 过滤"),
    user_id: Optional[int] = Query(None, description="按用户过滤"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """审计日志列表 (分页 + action 过滤)."""
    base = select(AuditLog)
    if action:
        base = base.where(AuditLog.action == action)
    if user_id is not None:
        base = base.where(AuditLog.user_id == user_id)

    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    offset = (page - 1) * page_size
    stmt = base.order_by(AuditLog.created_at.desc()).offset(offset).limit(page_size)
    logs = (await db.execute(stmt)).scalars().all()

    return PaginatedResponse[AuditLogOut](
        data=[
            AuditLogOut(
                id=l.id, user_id=l.user_id, action=l.action,
                resource_type=l.resource_type, resource_id=l.resource_id,
                detail=l.detail, ip_address=l.ip_address,
                status=l.status, error=l.error, created_at=l.created_at,
            )
            for l in logs
        ],
        total=total, page=page, page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if page_size else 0,
    )
