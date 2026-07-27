"""
pytest 公共 fixtures

设计要点:
1. ``mock_settings``: 临时设置 LLM_PROVIDER=offline 与 MILVUS_HOST=invalid,
   强制全链路走降级路径, 使测试在无 GPU / 无 Milvus / 无 LLM API 环境可运行;
2. ``db_session``: 内存 SQLite 异步会话, 测试隔离, 互不污染;
3. ``client``: FastAPI TestClient, 依赖注入覆盖 ``get_db``.

降级链验证:
- LLM=offline: 离线模式 LLM (规则/模板应答);
- MILVUS_HOST=invalid: Milvus 连接失败, 检索降级为 BM25 only;
- Redis 不可用: 缓存 / 限流 / 对话上下文均降级, 业务可用.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# 确保 backend 在 sys.path 中
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# 收集阶段忽略 Locust 压测脚本 (它依赖 locust 运行时, 不是单元测试)
collect_ignore = ["load_test.py"]


# ======================== 环境变量 (最早设置) ========================
# 必须在导入 app.* 之前设置, 强制全链路降级
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("MILVUS_HOST", "invalid")
os.environ.setdefault("MILVUS_PORT", "19530")
os.environ.setdefault("NEO4J_HOST", "invalid")
os.environ.setdefault("REDIS_HOST", "invalid")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DEBUG", "true")
# 使用 SQLite 内存库做测试
os.environ.setdefault("POSTGRES_HOST", "localhost")


# ======================== mock_settings ========================
@pytest.fixture(scope="session")
def mock_settings():
    """返回覆盖了离线 / 降级配置的 Settings 实例."""
    from app.config import Settings

    settings = Settings(
        llm_provider="offline",
        milvus_host="invalid",
        neo4j_host="invalid",
        redis_host="invalid",
        app_env="test",
        debug=True,
    )
    return settings


# ======================== 内存 SQLite 引擎 ========================
@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """每测试函数独立的内存 SQLite 异步会话.

    每个测试函数获得全新数据库, 避免数据污染.
    使用 aiosqlite 驱动; 若环境未装 aiosqlite, 降级为静态断言可见.
    """
    try:
        from app.models import Base
    except ImportError as exc:
        pytest.skip(f"模型加载失败, 跳过: {exc}")

    # 内存 SQLite (每连接独立, 需 StaticPool 共享)
    try:
        from sqlalchemy.pool import StaticPool

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            echo=False,
        )
    except ImportError:
        pytest.skip("aiosqlite 未安装, 跳过 DB 相关测试")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    SessionLocal = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with SessionLocal() as session:
        try:
            yield session
            await session.rollback()
        finally:
            await session.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ======================== FastAPI TestClient ========================
@pytest.fixture(scope="function")
def client(mock_settings):
    """FastAPI TestClient, 依赖注入覆盖 get_db 为内存 SQLite.

    若 FastAPI / TestClient 不可用或应用导入失败, 跳过.
    """
    try:
        from fastapi.testclient import TestClient
        from sqlalchemy.pool import StaticPool

        from app.database import get_db
        from app.main import app
        from app.models import Base
    except ImportError as exc:
        pytest.skip(f"应用加载失败, 跳过: {exc}")

    import asyncio

    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        echo=False,
    )

    # 同步建表 (在事件循环内)
    async def _init():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    try:
        asyncio.get_event_loop().run_until_complete(_init())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_init())

    TestSessionLocal = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with TestSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = _override_get_db

    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()

        # 清理测试引擎
        async def _dispose():
            await test_engine.dispose()

        try:
            asyncio.get_event_loop().run_until_complete(_dispose())
        except Exception:
            pass


# ======================== 事件循环 ========================
@pytest.fixture(scope="function")
def event_loop():
    """每测试函数独立事件循环."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
