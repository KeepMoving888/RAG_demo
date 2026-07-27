"""
检索 API 路由

提供三路混合检索 / 检索得分解释 / 缓存失效接口.

设计要点:
1. /search: 调用 ``HybridRetriever.retrieve`` 执行向量 + BM25 + RRF + 精排,
   返回 chunks + scores + retrieval_detail (可解释性);
2. /explain: 对单个 chunk 调用 ``explain_retrieval``, 输出 BM25 tf/idf 分项 + 术语命中;
3. /cache/invalidate: 失效全部检索缓存 (仅 admin), 用于文档更新后强制重算.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user, require_role
from app.database import get_db
from app.models import User
from app.rag.retriever import HybridRetriever
from app.schemas.common import SuccessResponse
from app.utils.logger import logger

router = APIRouter()


# ======================== Schemas ========================
class ChunkResult(BaseModel):
    """检索结果 chunk"""
    id: str = ""
    content: str = ""
    score: float = 0.0
    source: str = "unknown"
    heading_path: str = ""
    page_number: int = 0
    document_id: int = 0
    department_id: int = 0
    rank: int = 0


class RetrievalDetail(BaseModel):
    """检索可解释性详情"""
    vector_count: int = 0
    bm25_count: int = 0
    rrf_fused_count: int = 0
    reranked_count: int = 0
    stages_latency: dict[str, float] = {}
    reranker_available: bool = False
    rerank_method: str = "none"
    term_hits: list[str] = []
    expanded_query: Optional[str] = None


class SearchResponse(BaseModel):
    """检索响应"""
    chunks: list[ChunkResult] = []
    scores: list[float] = []
    latency_ms: float = 0.0
    cache_hit: bool = False
    retrieval_detail: RetrievalDetail = RetrievalDetail()


class ExplainResponse(BaseModel):
    """检索解释响应"""
    query: str
    expanded_query: str
    chunk_id: str
    term_hits: list[str] = []
    bm25_explanation: dict[str, Any] = {}


class CacheInvalidateResponse(BaseModel):
    """缓存失效响应"""
    invalidated: int = 0


# ======================== 辅助 ========================
def _get_retriever() -> HybridRetriever:
    """获取 HybridRetriever 实例.

    优先调用 ``get_retriever`` 单例工厂 (与 generator.py 一致),
    若工厂不可用则降级为直接实例化.
    """
    try:
        from app.rag.retriever import get_retriever  # type: ignore
        return get_retriever()
    except ImportError:
        return HybridRetriever()


def _to_chunk_result(chunk: dict) -> ChunkResult:
    """dict -> ChunkResult"""
    return ChunkResult(
        id=str(chunk.get("id", "")),
        content=str(chunk.get("content", "")),
        score=float(chunk.get("score", 0.0) or 0.0),
        source=str(chunk.get("source", "unknown")),
        heading_path=str(chunk.get("heading_path", "") or ""),
        page_number=int(chunk.get("page_number", 0) or 0),
        document_id=int(chunk.get("document_id", 0) or 0),
        department_id=int(chunk.get("department_id", 0) or 0),
        rank=int(chunk.get("rank", 0) or 0),
    )


# ======================== 路由 ========================
@router.get("/search", response_model=SuccessResponse[SearchResponse])
async def search(
    query: str = Query(..., min_length=1, max_length=2048, description="检索查询"),
    top_k: int = Query(5, ge=1, le=50, description="最终返回 chunk 数"),
    recall_k: int = Query(50, ge=1, le=200, description="召回阶段每路返回数"),
    enable_rerank: bool = Query(True, description="是否启用 Cross-Encoder 精排"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """执行三路混合检索 (向量 + BM25 + RRF + 精排).

    返回 chunks + scores + retrieval_detail (可解释性, 含各阶段延迟).
    权限隔离: 由 retriever 内部按 department_id 过滤 (Milvus partition + BM25 应用层).
    """
    retriever = _get_retriever()

    try:
        result = await retriever.retrieve(
            query=query,
            department_id=current_user.department_id,
            top_k=top_k,
            recall_k=recall_k,
            enable_rerank=enable_rerank,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("检索失败 query={!r}: {}", query, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检索失败: {exc}",
        )

    chunks = [_to_chunk_result(c) for c in result.get("chunks", [])]
    detail_dict = result.get("retrieval_detail", {}) or {}
    detail = RetrievalDetail(
        vector_count=int(detail_dict.get("vector_count", 0) or 0),
        bm25_count=int(detail_dict.get("bm25_count", 0) or 0),
        rrf_fused_count=int(detail_dict.get("rrf_fused_count", 0) or 0),
        reranked_count=int(detail_dict.get("reranked_count", 0) or 0),
        stages_latency=detail_dict.get("stages_latency", {}) or {},
        reranker_available=bool(detail_dict.get("reranker_available", False)),
        rerank_method=str(detail_dict.get("rerank_method", "none")),
        term_hits=list(detail_dict.get("term_hits", []) or []),
        expanded_query=detail_dict.get("expanded_query"),
    )

    return SuccessResponse[SearchResponse](data=SearchResponse(
        chunks=chunks,
        scores=[float(s) for s in result.get("scores", []) or []],
        latency_ms=float(result.get("latency_ms", 0.0) or 0.0),
        cache_hit=bool(result.get("cache_hit", False)),
        retrieval_detail=detail,
    ))


@router.get("/explain", response_model=SuccessResponse[ExplainResponse])
async def explain(
    query: str = Query(..., min_length=1, max_length=2048, description="检索查询"),
    chunk_id: str = Query(..., description="待解释的 chunk ID"),
    current_user: User = Depends(get_current_user),
):
    """检索得分构成解释 (query + chunk_id).

    返回 BM25 tf/idf 分项 + 术语命中情况, 用于排查「为何该 chunk 被召回 / 未被召回」.
    """
    retriever = _get_retriever()

    try:
        result = retriever.explain_retrieval(query=query, chunk_id=chunk_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("检索解释失败 query={!r} chunk={}: {}", query, chunk_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检索解释失败: {exc}",
        )

    return SuccessResponse[ExplainResponse](data=ExplainResponse(
        query=result.get("query", query),
        expanded_query=result.get("expanded_query", query),
        chunk_id=result.get("chunk_id", chunk_id),
        term_hits=list(result.get("term_hits", []) or []),
        bm25_explanation=result.get("bm25_explanation", {}) or {},
    ))


@router.post(
    "/cache/invalidate",
    response_model=SuccessResponse[CacheInvalidateResponse],
    dependencies=[Depends(require_role("admin"))],
)
async def invalidate_cache(
    admin: User = Depends(require_role("admin")),
):
    """失效全部检索缓存 (仅 admin).

    用于文档更新 / 重建索引后强制重算, 避免返回过期结果.
    """
    retriever = _get_retriever()
    try:
        count = await retriever.invalidate_cache()
    except Exception as exc:  # noqa: BLE001
        logger.exception("缓存失效失败: {}", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"缓存失效失败: {exc}",
        )

    logger.info("检索缓存已失效: count={} by admin={}", count, admin.id)

    return SuccessResponse[CacheInvalidateResponse](data=CacheInvalidateResponse(
        invalidated=int(count or 0),
    ))
