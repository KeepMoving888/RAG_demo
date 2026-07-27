"""app.rag.evaluator —— 检索质量消融评估

设计要点
--------
本模块用于离线评估不同检索策略的质量，通过消融实验（Ablation Study）量化
各组件（向量召回 / BM25 / RRF 融合 / 术语扩展 / 精排）的贡献，指导参数调优
与架构决策。

评估指标
~~~~~~~~
- **Recall@K**：前 K 个结果中命中相关文档的比例。衡量召回能力。
- **MRR（Mean Reciprocal Rank）**：第一个相关文档排名倒数的均值。衡量
  排序质量。
- **NDCG@K**：归一化折损累积增益，考虑相关性等级（多级）。衡量排序质量
  的标准指标。
- **Precision@K**：前 K 个结果中相关文档的比例。衡量精确率。

消融策略
~~~~~~~~
- ``vector_only``：仅向量召回
- ``bm25_only``：仅 BM25 召回
- ``rrf``：向量 + BM25 + RRF 融合（无术语扩展 / 无精排）
- ``full``：RRF + 精排（无术语扩展）
- ``full_with_terminology``：RRF + 精排 + 术语扩展（完整链路）

通过对比各策略指标，可量化每个组件的边际贡献。例如：
- ``rrf`` vs ``vector_only`` / ``bm25_only`` → 融合的增益
- ``full`` vs ``rrf`` → 精排的增益
- ``full_with_terminology`` vs ``full`` → 术语扩展的增益

数据来源
~~~~~~~~
``data/seed/rag_eval_dataset.json``（种子数据，约 30 条标注查询），每条含：
::

    {
        "query": "车规 eMMC 的 RoHS 合规要求",
        "relevant_chunk_ids": ["chunk_001", "chunk_042"],
        "relevance_level": {"chunk_001": 3, "chunk_042": 2}
    }

每次评估上报 ``RETRIEVAL_RECALL.labels(strategy=...).set(recall)``，供
Prometheus 监控可视化。
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from app.utils.logger import logger

try:
    from app.metrics import RETRIEVAL_RECALL
except ImportError:  # pragma: no cover
    # metrics 不可用时的 no-op 降级
    class _DummyMetric:
        def labels(self, **kwargs):
            return self

        def set(self, value):
            pass

    RETRIEVAL_RECALL = _DummyMetric()  # type: ignore


# 默认评估数据集路径
_DEFAULT_EVAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "data",
    "seed",
    "rag_eval_dataset.json",
)

# 消融策略列表
ABLATION_STRATEGIES = [
    "vector_only",
    "bm25_only",
    "rrf",
    "full",
    "full_with_terminology",
]


class RetrievalEvaluator:
    """检索质量消融评估器。

    Parameters
    ----------
    eval_path : str | None
        评估数据集 JSON 路径，默认 ``data/seed/rag_eval_dataset.json``。
    retriever : HybridRetriever | None
        待评估的检索器实例，None 则内部创建。
    """

    def __init__(
        self,
        eval_path: str | None = None,
        retriever: Any | None = None,
    ) -> None:
        self._eval_path: str = eval_path or _DEFAULT_EVAL_PATH
        self._dataset: list[dict[str, Any]] = []
        self._retriever = retriever
        self._load_dataset()

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def _load_dataset(self) -> None:
        """加载评估数据集。"""
        if os.path.exists(self._eval_path):
            try:
                with open(self._eval_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    self._dataset = data
                    logger.info(
                        "评估数据集加载成功: %s, 条目数=%d",
                        self._eval_path,
                        len(data),
                    )
                    return
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("评估数据集加载失败: %s", exc)

        # 兜底：空数据集
        self._dataset = []
        logger.warning("评估数据集为空，使用兜底空集: %s", self._eval_path)

    def _get_retriever(self):
        """懒获取检索器实例。"""
        if self._retriever is None:
            from app.rag.retriever import HybridRetriever

            self._retriever = HybridRetriever()
        return self._retriever

    # ------------------------------------------------------------------
    # 评估指标
    # ------------------------------------------------------------------
    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int = 5,
    ) -> float:
        """Recall@K：前 K 个结果中命中相关文档的比例。

        .. math::
            \\text{Recall@K} = \\frac{|\\text{retrieved}_k \\cap \\text{relevant}|}{|\\text{relevant}|}
        """
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = len(set(top_k) & set(relevant_ids))
        return hits / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
        """MRR：第一个相关文档排名倒数。

        .. math::
            \\text{MRR} = \\frac{1}{\\text{rank of first relevant doc}}
        """
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: list[str],
        relevance_map: dict[str, int],
        k: int = 5,
    ) -> float:
        """NDCG@K：归一化折损累积增益。

        考虑相关性等级（relevance_map 中的值，如 3=强相关, 2=相关, 1=弱相关）。

        .. math::
            \\text{DCG@K} = \\sum_{i=1}^{K} \\frac{2^{rel_i} - 1}{\\log_2(i + 1)}

            \\text{NDCG@K} = \\frac{\\text{DCG@K}}{\\text{IDCG@K}}
        """
        if not relevance_map:
            return 0.0

        # DCG
        dcg = 0.0
        for i, rid in enumerate(retrieved_ids[:k], start=1):
            rel = relevance_map.get(rid, 0)
            if rel > 0:
                dcg += (2**rel - 1) / math.log2(i + 1)

        # IDCG（理想排序）
        ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
        idcg = sum(
            (2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1) if rel > 0
        )

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def precision_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int = 5,
    ) -> float:
        """Precision@K：前 K 个结果中相关文档的比例。

        .. math::
            \\text{Precision@K} = \\frac{|\\text{retrieved}_k \\cap \\text{relevant}|}{K}
        """
        if k == 0:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = len(set(top_k) & set(relevant_ids))
        return hits / k

    # ------------------------------------------------------------------
    # 单策略评估
    # ------------------------------------------------------------------
    def evaluate(self, strategy: str = "full") -> dict[str, Any]:
        """评估指定策略。

        Parameters
        ----------
        strategy : str
            检索策略，见 ``ABLATION_STRATEGIES``。

        Returns
        -------
        dict
            ``{"strategy", "sample_count", "recall@5", "mrr", "ndcg@5",
            "precision@5", "avg_latency_ms"}``。
        """
        import asyncio

        if not self._dataset:
            logger.warning("评估数据集为空，无法评估")
            return {
                "strategy": strategy,
                "sample_count": 0,
                "recall@5": 0.0,
                "mrr": 0.0,
                "ndcg@5": 0.0,
                "precision@5": 0.0,
                "avg_latency_ms": 0.0,
            }

        retriever = self._get_retriever()
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        precisions: list[float] = []
        latencies: list[float] = []

        # 尝试获取事件循环，无则新建
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("loop closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        for sample in self._dataset:
            query = sample.get("query", "")
            relevant_ids = sample.get("relevant_chunk_ids", [])
            relevance_map = sample.get("relevance_level", {})

            if not query or not relevant_ids:
                continue

            # 调用同步版检索
            result = loop.run_until_complete(
                retriever.retrieve_sync(query, top_k=5, strategy=strategy)
            )

            retrieved_ids = [c["id"] for c in result.get("chunks", [])]
            latencies.append(result.get("latency_ms", 0.0))

            recalls.append(self.recall_at_k(retrieved_ids, relevant_ids, k=5))
            mrrs.append(self.mrr(retrieved_ids, relevant_ids))
            ndcgs.append(self.ndcg_at_k(retrieved_ids, relevance_map, k=5))
            precisions.append(self.precision_at_k(retrieved_ids, relevant_ids, k=5))

        n = len(recalls)
        metrics = {
            "strategy": strategy,
            "sample_count": n,
            "recall@5": round(sum(recalls) / n, 4) if n else 0.0,
            "mrr": round(sum(mrrs) / n, 4) if n else 0.0,
            "ndcg@5": round(sum(ndcgs) / n, 4) if n else 0.0,
            "precision@5": round(sum(precisions) / n, 4) if n else 0.0,
            "avg_latency_ms": round(sum(latencies) / n, 2) if n else 0.0,
        }

        # 上报指标
        RETRIEVAL_RECALL.labels(strategy=strategy).set(metrics["recall@5"])
        logger.info(
            "策略评估完成: %s | recall@5=%.4f, mrr=%.4f, ndcg@5=%.4f, "
            "precision@5=%.4f, avg_latency=%.1fms, samples=%d",
            strategy,
            metrics["recall@5"],
            metrics["mrr"],
            metrics["ndcg@5"],
            metrics["precision@5"],
            metrics["avg_latency_ms"],
            n,
        )
        return metrics

    # ------------------------------------------------------------------
    # 消融实验
    # ------------------------------------------------------------------
    def ablation_study(self) -> dict[str, Any]:
        """批量跑所有策略，返回对比表格。

        Returns
        -------
        dict
            ``{"strategies": [各策略指标], "comparison_table": str,
            "best_strategy": str, "analysis": str}``。
        """
        results: list[dict[str, Any]] = []
        for strategy in ABLATION_STRATEGIES:
            try:
                metrics = self.evaluate(strategy)
                results.append(metrics)
            except Exception as exc:  # noqa: BLE001
                logger.error("策略 %s 评估失败: %s", strategy, exc)
                results.append(
                    {
                        "strategy": strategy,
                        "sample_count": 0,
                        "recall@5": 0.0,
                        "mrr": 0.0,
                        "ndcg@5": 0.0,
                        "precision@5": 0.0,
                        "avg_latency_ms": 0.0,
                        "error": str(exc),
                    }
                )

        # 生成对比表格
        table = self._format_comparison_table(results)

        # 找最优策略（以 NDCG@5 为主指标）
        valid = [r for r in results if "error" not in r and r["sample_count"] > 0]
        best = max(valid, key=lambda x: x["ndcg@5"]) if valid else None
        best_strategy = best["strategy"] if best else "none"

        # 自动分析
        analysis = self._generate_analysis(results)

        logger.info(
            "消融实验完成: 策略数=%d, 最优=%s (ndcg@5=%.4f)",
            len(results),
            best_strategy,
            best["ndcg@5"] if best else 0.0,
        )

        return {
            "strategies": results,
            "comparison_table": table,
            "best_strategy": best_strategy,
            "analysis": analysis,
        }

    def _format_comparison_table(self, results: list[dict[str, Any]]) -> str:
        """格式化对比表格为字符串。"""
        header = (
            f"{'Strategy':<28} {'Recall@5':<12} {'MRR':<10} {'NDCG@5':<12} "
            f"{'P@5':<10} {'Latency(ms)':<14} {'Samples':<10}"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for r in results:
            lines.append(
                f"{r['strategy']:<28} {r['recall@5']:<12.4f} {r['mrr']:<10.4f} "
                f"{r['ndcg@5']:<12.4f} {r['precision@5']:<10.4f} "
                f"{r['avg_latency_ms']:<14.1f} {r['sample_count']:<10}"
            )
        return "\n".join(lines)

    def _generate_analysis(self, results: list[dict[str, Any]]) -> str:
        """根据消融结果生成自动分析。"""
        if not results:
            return "无可用结果。"

        def get(name: str) -> dict[str, Any]:
            for r in results:
                if r["strategy"] == name:
                    return r
            return {"ndcg@5": 0.0, "recall@5": 0.0}

        parts: list[str] = []

        # 融合增益
        rrf = get("rrf")
        vec = get("vector_only")
        bm25 = get("bm25_only")
        if rrf["ndcg@5"] > 0:
            best_single = max(vec["ndcg@5"], bm25["ndcg@5"])
            if best_single > 0:
                gain = (rrf["ndcg@5"] - best_single) / best_single * 100
                parts.append(
                    f"RRF 融合相比最佳单路召回 NDCG@5 提升 {gain:.1f}% "
                    f"({best_single:.4f} → {rrf['ndcg@5']:.4f})"
                )

        # 精排增益
        full = get("full")
        if full["ndcg@5"] > 0 and rrf["ndcg@5"] > 0:
            gain = (full["ndcg@5"] - rrf["ndcg@5"]) / rrf["ndcg@5"] * 100
            parts.append(
                f"Cross-Encoder 精排 NDCG@5 提升 {gain:.1f}% "
                f"({rrf['ndcg@5']:.4f} → {full['ndcg@5']:.4f})"
            )

        # 术语扩展增益
        full_term = get("full_with_terminology")
        if full_term["ndcg@5"] > 0 and full["ndcg@5"] > 0:
            gain = (full_term["ndcg@5"] - full["ndcg@5"]) / full["ndcg@5"] * 100
            parts.append(
                f"术语扩展 NDCG@5 提升 {gain:.1f}% "
                f"({full['ndcg@5']:.4f} → {full_term['ndcg@5']:.4f})"
            )

        if not parts:
            parts.append("各组件增益不显著，建议检查数据集标注质量。")

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def dataset_size(self) -> int:
        """评估数据集条目数。"""
        return len(self._dataset)
