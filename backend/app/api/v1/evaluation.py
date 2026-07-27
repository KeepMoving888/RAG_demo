"""
检索评估 API 路由

提供消融实验对比 / 单策略评估 / 评估数据集查看接口.

设计要点:
1. /ablation: 调用 ``RetrievalEvaluator.ablation_study`` 批量跑全部策略,
   返回各策略的 Recall@5 / MRR / NDCG@5 / Precision@5 + 对比表格 + 自动分析;
2. /strategy: 单策略评估, 入参 strategy (默认 full);
3. /dataset: 查看评估数据集 (data/seed/rag_eval_dataset.json), 供前端展示标注样本.

评估在离线模式下用 BM25 近似复现向量召回, 保证全链路在无 GPU 环境可运行.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.core.security import get_current_user
from app.models import User
from app.rag.evaluator import ABLATION_STRATEGIES, RetrievalEvaluator
from app.schemas.common import SuccessResponse
from app.utils.logger import logger

router = APIRouter()


# ======================== Schemas ========================
class StrategyMetrics(BaseModel):
    """单策略评估指标"""

    strategy: str
    sample_count: int = 0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    precision_at_5: float = 0.0
    avg_latency_ms: float = 0.0
    error: str | None = None


class AblationResponse(BaseModel):
    """消融实验响应"""

    strategies: list[StrategyMetrics] = []
    comparison_table: str = ""
    best_strategy: str = "none"
    analysis: str = ""


class DatasetItem(BaseModel):
    """评估数据集单条样本"""

    query: str
    relevant_chunk_ids: list[str] = []
    relevance_level: dict[str, int] = {}


class DatasetResponse(BaseModel):
    """评估数据集响应"""

    path: str
    size: int = 0
    items: list[DatasetItem] = []


# ======================== 辅助 ========================
def _get_evaluator() -> RetrievalEvaluator:
    """获取 RetrievalEvaluator 实例."""
    return RetrievalEvaluator()


def _to_metrics(raw: dict) -> StrategyMetrics:
    """原始指标 dict -> StrategyMetrics"""
    return StrategyMetrics(
        strategy=raw.get("strategy", ""),
        sample_count=int(raw.get("sample_count", 0) or 0),
        recall_at_5=float(raw.get("recall@5", 0.0) or 0.0),
        mrr=float(raw.get("mrr", 0.0) or 0.0),
        ndcg_at_5=float(raw.get("ndcg@5", 0.0) or 0.0),
        precision_at_5=float(raw.get("precision@5", 0.0) or 0.0),
        avg_latency_ms=float(raw.get("avg_latency_ms", 0.0) or 0.0),
        error=raw.get("error"),
    )


# ======================== 路由 ========================
@router.get("/ablation", response_model=SuccessResponse[AblationResponse])
async def ablation_study(
    current_user: User = Depends(get_current_user),
):
    """消融实验对比: 批量跑全部策略, 返回各策略指标 + 对比表格 + 自动分析.

    策略列表: vector_only / bm25_only / rrf / full / full_with_terminology.
    """
    evaluator = _get_evaluator()
    try:
        result = evaluator.ablation_study()
    except Exception as exc:  # noqa: BLE001
        logger.exception("消融实验失败: {}", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"消融实验失败: {exc}",
        )

    strategies = [_to_metrics(s) for s in result.get("strategies", []) or []]

    return SuccessResponse[AblationResponse](
        data=AblationResponse(
            strategies=strategies,
            comparison_table=result.get("comparison_table", "") or "",
            best_strategy=result.get("best_strategy", "none") or "none",
            analysis=result.get("analysis", "") or "",
        )
    )


@router.get("/strategy", response_model=SuccessResponse[StrategyMetrics])
async def evaluate_strategy(
    strategy: str = Query("full", description=f"策略, 可选: {ABLATION_STRATEGIES}"),
    current_user: User = Depends(get_current_user),
):
    """单策略评估: 返回 Recall@5 / MRR / NDCG@5 / Precision@5 / 平均延迟."""
    if strategy not in ABLATION_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的策略: {strategy}, 允许: {ABLATION_STRATEGIES}",
        )

    evaluator = _get_evaluator()
    try:
        raw = evaluator.evaluate(strategy=strategy)
    except Exception as exc:  # noqa: BLE001
        logger.exception("策略评估失败 strategy={}: {}", strategy, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"策略评估失败: {exc}",
        )

    return SuccessResponse[StrategyMetrics](data=_to_metrics(raw))


@router.get("/dataset", response_model=SuccessResponse[DatasetResponse])
async def get_dataset(
    current_user: User = Depends(get_current_user),
):
    """查看评估数据集 (data/seed/rag_eval_dataset.json)."""
    evaluator = _get_evaluator()
    items = [
        DatasetItem(
            query=str(s.get("query", "")),
            relevant_chunk_ids=[str(c) for c in s.get("relevant_chunk_ids", []) or []],
            relevance_level={
                str(k): int(v) for k, v in (s.get("relevance_level", {}) or {}).items()
            },
        )
        for s in evaluator._dataset  # noqa: SLF001
    ]

    return SuccessResponse[DatasetResponse](
        data=DatasetResponse(
            path=evaluator._eval_path,  # noqa: SLF001
            size=evaluator.dataset_size,
            items=items,
        )
    )
