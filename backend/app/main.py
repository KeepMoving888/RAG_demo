"""
Enterprise RAG Knowledge Base - FastAPI 应用入口

启动流程:
1. 加载配置 (config.settings)
2. 初始化日志 (utils.logger)
3. 注册中间件 (CORS / 请求上下文 / 限流 / Prometheus)
4. 注册路由 (api/v1/*)
5. 生命周期 (lifespan): 启动时建表, 关闭时清理连接
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.utils.logger import logger
from app.core.context import set_request_context, RequestContext


# ======================== 生命周期 ========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("Enterprise RAG Knowledge Base 启动中...")
    logger.info("环境: {} | 调试: {} | LLM Provider: {}",
                settings.app_env, settings.debug, settings.llm_provider)
    logger.info("=" * 60)

    # 启动时建表 (开发环境)
    if settings.app_env == "development":
        try:
            from app.models import Base
            from app.database import engine
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("开发模式: 已自动建表")
        except Exception as e:
            logger.warning("自动建表失败 (可改用 alembic 迁移): {}", str(e))

    # 初始化 Milvus Collection
    try:
        from app.rag.milvus_store import milvus_store
        await milvus_store.init_collection()
        logger.info("Milvus Collection 初始化完成")
    except Exception as e:
        logger.warning("Milvus 初始化失败 (RAG 将降级为 BM25 only): {}", str(e))

    # 初始化 Neo4j Schema
    try:
        from app.graphrag.neo4j_store import neo4j_store
        await neo4j_store.init_schema()
        logger.info("Neo4j Schema 初始化完成")
    except Exception as e:
        logger.warning("Neo4j 初始化失败 (GraphRAG 将降级): {}", str(e))

    logger.info("应用启动完成")
    yield

    # 关闭清理
    from app.database import engine
    await engine.dispose()
    logger.info("应用已关闭")


# ======================== 应用实例 ========================
app = FastAPI(
    title=settings.app_name,
    description="面向企业内部员工自助答疑场景的智能知识库问答系统",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ======================== 中间件 ========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """请求上下文中间件: 注入 user_id / IP / request_id"""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex

    # 解析 JWT 获取 user_id (轻量解析, 不查 DB)
    user_id = None
    department_id = None
    role = None
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            from app.core.security import decode_access_token
            payload = decode_access_token(auth[7:])
            user_id = int(payload.get("sub", 0)) or None
            department_id = payload.get("dept")
            role = payload.get("role")
        except Exception:
            pass

    set_request_context(RequestContext(
        user_id=user_id,
        department_id=department_id,
        role=role,
        ip_address=request.client.host if request.client else None,
        request_id=request_id,
    ))

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ======================== Prometheus 指标 ========================
if settings.enable_metrics:
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=False,
            excluded_handlers=["/metrics", "/health"],
        ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
        logger.info("Prometheus 指标已挂载 /metrics")
    except ImportError:
        logger.warning("prometheus-fastapi-instrumentator 未安装, 跳过指标挂载")


# ======================== 路由注册 ========================
try:
    from app.api.v1 import api_router
    app.include_router(api_router)
    logger.info("API v1 路由注册完成")
except ImportError as e:
    logger.warning("API 路由未完全就绪 (开发中): {}", str(e))


# ======================== 健康检查 ========================
@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查"""
    from app.database import check_db_connection
    db_ok = await check_db_connection()
    return {
        "status": "healthy" if db_ok else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {
            "database": "ok" if db_ok else "down",
        },
    }


@app.get("/", tags=["系统"])
async def root():
    """根路径"""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ======================== 全局异常处理 ========================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常兜底"""
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    logger.exception("未捕获异常 [{}]: {}", request_id, str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "code": -1,
            "message": f"内部错误: {exc.__class__.__name__}",
            "data": None,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
