"""
安全模块: JWT + 密码哈希 + 权限控制

权限模型:
1. JWT 携带 {user_id, email, department_id, role}
2. 角色权限: admin (全公司) / staff (本部门 + 公开)
3. 部门权限:
   - admin 可访问所有部门文档
   - staff 仅可访问 department_id == self.department_id 或 department_id IS NULL 的文档
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import User

# ======================== 密码哈希 ========================
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ======================== OAuth2 ========================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def hash_password(password: str) -> str:
    """bcrypt 哈希密码"""
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """校验密码"""
    return _pwd_context.verify(plain, hashed)


# ======================== JWT ========================
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """生成 JWT"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    """解码 JWT"""
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"无效的认证凭据: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ======================== 依赖注入 ========================
async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """获取当前登录用户 (强制鉴权)"""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="凭据缺少用户标识"
        )

    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()

    if not user or not user.is_active or user.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在或已禁用"
        )
    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """获取当前用户 (可选, 未登录返回 None)"""
    if not token:
        return None
    try:
        return await get_current_user(token, db)
    except HTTPException:
        return None


def require_role(*roles: str):
    """角色权限依赖工厂"""

    async def _check(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"权限不足, 需要 {roles} 之一",
            )
        return user

    return _check


async def require_department_access(
    doc_department_id: Optional[int],
    user: User = Depends(get_current_user),
) -> bool:
    """
    部门权限检查:
    - admin 角色全公司可见
    - staff 仅可见本部门 + 公开 (NULL)
    """
    if user.role == "admin":
        return True
    if doc_department_id is None:
        return True  # 公开文档
    if doc_department_id == user.department_id:
        return True
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="无权访问该部门文档",
    )
