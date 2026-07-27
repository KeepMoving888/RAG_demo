"""
数据库初始化脚本

功能:
1. 创建所有表 (Base.metadata.create_all);
2. 创建默认部门 (研发部 / 销售部 / HR部 / 财务部 / 售后部 / 行政部);
3. 创建默认 admin 用户 (email=admin@semitech.cn, password=admin123);
4. 幂等执行: 重复运行不会报错, 已存在的记录跳过.

用法:
    python -m scripts.init_db
    或: python scripts/init_db.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

# 确保 backend 在 sys.path 中 (脚本可独立运行)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from sqlalchemy import select  # noqa: E402

from app.core.security import hash_password  # noqa: E402
from app.database import AsyncSessionLocal, engine  # noqa: E402
from app.models import Base, Department, User  # noqa: E402
from app.utils.logger import logger  # noqa: E402


# ======================== 默认数据 ========================
_DEFAULT_DEPARTMENTS = [
    {"code": "RD", "name": "研发部", "description": "产品研发与技术"},
    {"code": "SALES", "name": "销售部", "description": "销售与渠道"},
    {"code": "HR", "name": "HR部", "description": "人力资源"},
    {"code": "FIN", "name": "财务部", "description": "财务与结算"},
    {"code": "AFTER_SALES", "name": "售后部", "description": "售后技术支持"},
    {"code": "ADMIN", "name": "行政部", "description": "行政与运营"},
]

_DEFAULT_ADMIN = {
    "email": "admin@semitech.cn",
    "password": "admin123",
    "name": "系统管理员",
    "role": "admin",
}


# ======================== 初始化逻辑 ========================
async def _create_tables() -> None:
    """创建所有表 (幂等, IF NOT EXISTS 语义)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据库表已就绪 (create_all)")


async def _seed_departments() -> int:
    """创建默认部门, 返回新增数."""
    added = 0
    async with AsyncSessionLocal() as session:
        for dept in _DEFAULT_DEPARTMENTS:
            result = await session.execute(
                select(Department).where(Department.code == dept["code"])
            )
            if result.scalar_one_or_none() is not None:
                continue
            session.add(Department(**dept))
            added += 1
        await session.commit()
    logger.info("默认部门已就绪: 新增 {} 个", added)
    return added


async def _seed_admin() -> bool:
    """创建默认 admin 用户, 返回是否新增."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.email == _DEFAULT_ADMIN["email"])
        )
        if result.scalar_one_or_none() is not None:
            logger.info("默认 admin 用户已存在, 跳过")
            return False

        # 查找 ADMIN 部门作为默认归属
        dept_result = await session.execute(
            select(Department).where(Department.code == "ADMIN")
        )
        admin_dept: Optional[Department] = dept_result.scalar_one_or_none()

        user = User(
            email=_DEFAULT_ADMIN["email"],
            hashed_password=hash_password(_DEFAULT_ADMIN["password"]),
            name=_DEFAULT_ADMIN["name"],
            department_id=admin_dept.id if admin_dept else None,
            role=_DEFAULT_ADMIN["role"],
            is_active=True,
        )
        session.add(user)
        await session.commit()
    logger.info(
        "默认 admin 用户已创建: email={} (请尽快修改默认密码!)",
        _DEFAULT_ADMIN["email"],
    )
    return True


async def main() -> None:
    """初始化主流程."""
    logger.info("=" * 60)
    logger.info("开始数据库初始化...")
    logger.info("=" * 60)

    await _create_tables()
    dept_added = await _seed_departments()
    admin_added = await _seed_admin()

    logger.info("=" * 60)
    logger.info(
        "初始化完成: 新增部门 {} 个, admin {}",
        dept_added, "已创建" if admin_added else "已存在",
    )
    logger.info("=" * 60)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
