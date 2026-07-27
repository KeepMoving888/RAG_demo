"""
认证与用户管理 API 路由

提供登录 / 当前用户信息 / 注册三类接口, 全部使用 JWT Bearer 鉴权.
所有响应通过 ``SuccessResponse[T]`` 包装, 写入 ``AuditLog`` 满足企业级合规要求.

设计要点:
1. 登录: 邮箱 + 密码校验通过后颁发 JWT, payload 内嵌 ``sub`` / ``email`` /
   ``dept`` / ``role`` 四项声明, 供下游中间件与权限依赖直接消费, 避免重复查库.
2. 注册: 仅 ``admin`` 可调用 (依赖 ``require_role("admin")``), 防止任意人开账号.
3. 审计: 登录成功 / 失败 / 注册均写 ``AuditLog``, 含 IP / UA / 状态, 便于追溯.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.context import get_current_user_id, request_context
from app.core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    require_role,
    verify_password,
)
from app.database import get_db
from app.models import AuditLog, Department, User
from app.schemas.common import SuccessResponse
from app.utils.logger import logger

router = APIRouter()


# ======================== Schemas ========================
class LoginRequest(BaseModel):
    """登录请求"""
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class RegisterRequest(BaseModel):
    """注册请求 (admin 调用)"""
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    name: str = Field(min_length=1, max_length=64)
    department_id: Optional[int] = None
    role: str = Field(default="staff", pattern="^(admin|staff)$")


class UserOut(BaseModel):
    """用户对外结构"""
    id: int
    email: EmailStr
    name: str
    department_id: Optional[int] = None
    role: str
    is_active: bool
    last_login_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    """登录响应"""
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ======================== 辅助 ========================
async def _write_audit(
    db: AsyncSession,
    action: str,
    user_id: Optional[int],
    status_: str = "success",
    error: Optional[str] = None,
    resource_id: Optional[str] = None,
    detail: Optional[dict] = None,
) -> None:
    """写入审计日志 (失败不阻断主流程)."""
    try:
        ctx = request_context.get()
        log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type="user",
            resource_id=resource_id,
            detail=detail or {},
            ip_address=ctx.ip_address,
            user_agent=None,
            status=status_,
            error=error,
        )
        db.add(log)
        await db.flush()
    except Exception as exc:  # noqa: BLE001
        logger.warning("审计日志写入失败 action={}: {}", action, str(exc))


def _user_to_out(user: User) -> UserOut:
    """ORM -> UserOut"""
    return UserOut(
        id=user.id,
        email=user.email,
        name=user.name,
        department_id=user.department_id,
        role=user.role,
        is_active=user.is_active,
        last_login_at=user.last_login_at,
    )


# ======================== 路由 ========================
@router.post("/login", response_model=SuccessResponse[LoginResponse])
async def login(
    payload: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """邮箱密码登录, 返回 JWT 与用户信息.

    校验流程:
    1. 按 email 查找用户 (含 is_deleted=False 过滤);
    2. ``verify_password`` 校验 bcrypt 哈希;
    3. 校验通过颁发 JWT, 并更新 ``last_login_at``;
    4. 写入 ``auth.login`` 审计日志.
    """
    result = await db.execute(
        select(User).where(User.email == payload.email, User.is_deleted == False)  # noqa: E712
    )
    user = result.scalar_one_or_none()

    if user is None or not verify_password(payload.password, user.hashed_password):
        await _write_audit(
            db, "auth.login", None, status_="failed",
            error="邮箱或密码错误",
            resource_id=payload.email,
            detail={"email": payload.email},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        await _write_audit(
            db, "auth.login", user.id, status_="failed",
            error="账号已禁用",
            resource_id=str(user.id),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已禁用, 请联系管理员",
        )

    # 颁发 JWT
    token = create_access_token({
        "sub": str(user.id),
        "email": user.email,
        "dept": user.department_id,
        "role": user.role,
    })

    # 更新最近登录时间 (last_login_at 列为 TIMESTAMP WITHOUT TIME ZONE, 需 naive datetime)
    user.last_login_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.flush()

    await _write_audit(
        db, "auth.login", user.id, status_="success",
        resource_id=str(user.id),
        detail={"email": user.email, "role": user.role},
    )

    logger.info("用户登录成功: id={} email={}", user.id, user.email)

    return SuccessResponse[LoginResponse](data=LoginResponse(
        access_token=token,
        token_type="bearer",
        user=_user_to_out(user),
    ))


@router.get("/me", response_model=SuccessResponse[UserOut])
async def me(current_user: User = Depends(get_current_user)):
    """获取当前登录用户信息."""
    return SuccessResponse[UserOut](data=_user_to_out(current_user))


@router.post(
    "/register",
    response_model=SuccessResponse[UserOut],
    dependencies=[Depends(require_role("admin"))],
)
async def register(
    payload: RegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """注册新用户 (仅 admin 可调用).

    幂等性: email 唯一约束保证重复注册返回 409, 不会创建重复账号.
    """
    # 校验部门存在
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

    # 校验 email 唯一
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
        db, "auth.register", admin.id, status_="success",
        resource_id=str(user.id),
        detail={
            "new_user_id": user.id, "new_user_email": user.email,
            "role": user.role, "department_id": user.department_id,
        },
    )

    logger.info(
        "管理员 {} 注册新用户: id={} email={} role={}",
        admin.id, user.id, user.email, user.role,
    )

    return SuccessResponse[UserOut](data=_user_to_out(user))
