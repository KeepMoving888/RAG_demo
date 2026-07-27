"""
知识图谱 API 路由

提供自然语言查询 / 实体检索 / 关系检索 / 最短路径 / 重建图谱 / 统计接口.

设计要点:
1. /query: 调用 ``GraphCypherQAChain.query`` 将自然语言转 Cypher 只读查询并执行,
   返回 cypher / result_text / records (含图谱节点与关系);
2. /entities: 通过 ``neo4j_store.get_entity`` 按 name + 可选 type 检索单个实体;
3. /relations: 通过 ``neo4j_store.get_neighbors`` 获取 N 跳邻居;
4. /paths: 通过 ``neo4j_store.find_paths`` 查找最短路径;
5. /rebuild: 重建图谱 (仅 admin), 触发 Celery 任务对全部 ready 文档批量抽取;
6. /stats: 图谱统计 (节点 / 关系数量, 按标签分组).
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user, require_role
from app.database import get_db
from app.graphrag.cypher_chain import GraphCypherQAChain
from app.graphrag.neo4j_store import neo4j_store
from app.models import Document, User
from app.schemas.common import SuccessResponse
from app.utils.logger import logger

router = APIRouter()


# ======================== Schemas ========================
class GraphQueryRequest(BaseModel):
    """图谱查询请求"""
    natural_language: str = Field(min_length=1, max_length=2048)
    department_id: Optional[int] = None


class GraphQueryResponse(BaseModel):
    """图谱查询响应"""
    cypher: str
    result_text: str
    records: list[dict[str, Any]] = []
    latency_ms: float = 0.0


class EntityOut(BaseModel):
    """实体对外结构"""
    element_id: Optional[str] = None
    labels: list[str] = []
    name: Optional[str] = None
    type: Optional[str] = None
    source_chunks: list[int] = []
    properties: dict[str, Any] = {}


class RelationOut(BaseModel):
    """关系 (邻居) 对外结构"""
    neighbor: dict[str, Any] = {}
    labels: list[str] = []
    relations: list[str] = []


class PathOut(BaseModel):
    """路径对外结构"""
    nodes: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []


class GraphStatsOut(BaseModel):
    """图谱统计"""
    available: bool = False
    nodes: int = 0
    relationships: int = 0
    by_label: dict[str, int] = {}


class RebuildResponse(BaseModel):
    """重建图谱响应"""
    triggered: bool = True
    document_count: int = 0
    task_ids: list[str] = []


# ======================== 路由 ========================
@router.post("/query", response_model=SuccessResponse[GraphQueryResponse])
async def graph_query(
    payload: GraphQueryRequest,
    current_user: User = Depends(get_current_user),
):
    """自然语言图谱查询: NL -> Cypher -> 执行 -> 格式化.

    Cypher 生成后经严格白名单校验 (拒绝写操作 / 多语句 / 注释),
    保证图谱不被破坏.
    """
    chain = GraphCypherQAChain()
    try:
        result = await chain.query(
            natural_language=payload.natural_language,
            department_id=payload.department_id or current_user.department_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("图谱查询失败: {}", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图谱查询失败: {exc}",
        )

    return SuccessResponse[GraphQueryResponse](data=GraphQueryResponse(
        cypher=result.get("cypher", ""),
        result_text=result.get("result_text", ""),
        records=result.get("records", []) or [],
        latency_ms=float(result.get("latency_ms", 0.0) or 0.0),
    ))


@router.get("/entities", response_model=SuccessResponse[EntityOut])
async def get_entity(
    name: str = Query(..., min_length=1, max_length=256, description="实体名称"),
    entity_type: Optional[str] = Query(None, description="实体类型, 如 Product"),
    current_user: User = Depends(get_current_user),
):
    """按名称 (可选类型) 检索单个实体."""
    try:
        entity = await neo4j_store.get_entity(name, entity_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("实体检索失败 name={}: {}", name, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"实体检索失败: {exc}",
        )

    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"实体不存在: {name}",
        )

    return SuccessResponse[EntityOut](data=EntityOut(
        element_id=entity.get("element_id"),
        labels=entity.get("labels", []) or [],
        name=entity.get("name"),
        type=entity.get("type"),
        source_chunks=entity.get("source_chunks", []) or [],
        properties=entity.get("properties", {}) or {},
    ))


@router.get("/relations", response_model=SuccessResponse[list[RelationOut]])
async def get_relations(
    entity_name: str = Query(..., min_length=1, description="中心实体名称"),
    hops: int = Query(1, ge=1, le=5, description="跳数 (1-5)"),
    limit: int = Query(50, ge=1, le=200, description="返回上限"),
    current_user: User = Depends(get_current_user),
):
    """获取 N 跳邻居 (关系检索)."""
    try:
        records = await neo4j_store.get_neighbors(entity_name, hops=hops)
    except Exception as exc:  # noqa: BLE001
        logger.exception("关系检索失败 entity={}: {}", entity_name, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"关系检索失败: {exc}",
        )

    relations = [
        RelationOut(
            neighbor=rec.get("neighbor", {}) or {},
            labels=rec.get("labels", []) or [],
            relations=rec.get("relations", []) or [],
        )
        for rec in (records or [])[:limit]
    ]

    return SuccessResponse[list[RelationOut]](data=relations)


@router.get("/paths", response_model=SuccessResponse[list[PathOut]])
async def get_paths(
    source: str = Query(..., min_length=1, description="起始实体名称"),
    target: str = Query(..., min_length=1, description="目标实体名称"),
    max_hops: int = Query(3, ge=1, le=6, description="最大跳数 (1-6)"),
    current_user: User = Depends(get_current_user),
):
    """最短路径查找 (实体关系链路可解释性)."""
    try:
        records = await neo4j_store.find_paths(
            source_name=source, target_name=target, max_hops=max_hops
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("路径查找失败 {} -> {}: {}", source, target, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"路径查找失败: {exc}",
        )

    paths = [
        PathOut(
            nodes=rec.get("p", {}).get("nodes", []) or [] if isinstance(rec.get("p"), dict) else [],
            relationships=rec.get("p", {}).get("relationships", []) or [] if isinstance(rec.get("p"), dict) else [],
        )
        for rec in (records or [])
    ]

    return SuccessResponse[list[PathOut]](data=paths)


@router.post(
    "/rebuild",
    response_model=SuccessResponse[RebuildResponse],
    dependencies=[Depends(require_role("admin"))],
)
async def rebuild_graph(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """重建图谱 (仅 admin): 遍历全部 status=ready 文档, 触发批量抽取任务."""
    from app.graphrag.extractor_tasks import extract_entities_task

    result = await db.execute(
        select(Document).where(
            Document.status == "ready",
            Document.is_deleted == False,  # noqa: E712
        )
    )
    docs = result.scalars().all()

    task_ids: list[str] = []
    for doc in docs:
        try:
            async_result = extract_entities_task.delay(doc.id)
            tid = async_result.id if hasattr(async_result, "id") else str(async_result)
            task_ids.append(tid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("投递抽取任务失败 doc_id={}: {}", doc.id, str(exc))

    logger.info(
        "图谱重建已触发: docs={} tasks={} by admin={}",
        len(docs), len(task_ids), admin.id,
    )

    return SuccessResponse[RebuildResponse](data=RebuildResponse(
        triggered=True,
        document_count=len(docs),
        task_ids=task_ids,
    ))


@router.get("/stats", response_model=SuccessResponse[GraphStatsOut])
async def graph_stats(
    current_user: User = Depends(get_current_user),
):
    """图谱统计 (节点 / 关系数量, 按标签分组)."""
    try:
        stats = await neo4j_store.stats()
    except Exception as exc:  # noqa: BLE001
        logger.exception("图谱统计失败: {}", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图谱统计失败: {exc}",
        )

    return SuccessResponse[GraphStatsOut](data=GraphStatsOut(
        available=bool(stats.get("available", False)),
        nodes=int(stats.get("nodes", 0) or 0),
        relationships=int(stats.get("relationships", 0) or 0),
        by_label={str(k): int(v) for k, v in (stats.get("by_label", {}) or {}).items()},
    ))
