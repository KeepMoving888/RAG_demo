"""app.rag.retriever —— HybridRetriever 混合检索统一入口

设计要点
--------
本模块是整个 RAG 检索链路的编排者，组合以下组件：

- **MilvusStore**：稠密向量召回（BGE-M3 encode + 分区隔离 search）
- **BM25Retriever**：稀疏词项召回（术语加权 BM25）
- **CrossEncoderReranker**：Cross-Encoder 精排（三级降级）
- **TerminologyExpander**：术语查询扩展 + 词项加权
- **RetrievalCache**：检索结果缓存

检索流程（七步）
~~~~~~~~~~~~~~~~
1. **查缓存**：命中则直接返回（标记 cache_hit=True），跳过全部计算。
2. **术语扩展**：``expand_query`` 注入同义词，``boost_term_weight`` 生成
   术语词项权重。
3. **Stage 1 召回**（双路并行）：
   - Path A（向量）：BGE-M3 encode query → Milvus 分区 search，recall_k=50
   - Path B（BM25）：术语加权 search，recall_k=50
   - 两路均按 department_id 过滤（_public + 本部门），权限隔离落地。
4. **RRF 融合**：两路结果 RRF 融合 → top-50 候选。
5. **Stage 2 精排**：CrossEncoderReranker 精排 → top-5。
6. **写缓存**：将结果写入 RetrievalCache。
7. **返回结构化结果**：含 chunks、scores、latency、retrieval_detail。

每步通过 ``record_retrieval_stage(stage, latency_ms)`` 上报延迟指标，
便于性能监控与瓶颈定位。

权限隔离要点
~~~~~~~~~~~~
``department_id`` 贯穿整个链路：向量检索在 Milvus partition 侧隔离，
BM25 检索在应用层按 ``department_id`` 字段过滤（BM25 无 partition 概念）。
两路合并前确保候选均属于 ``[_public, <dept>]``。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from app.config import settings
from app.utils.logger import logger

try:
    from app.metrics import record_retrieval_stage
except ImportError:  # pragma: no cover

    def record_retrieval_stage(stage: str, latency_ms: float) -> None:
        """metrics 不可用时的 no-op 降级。"""
        pass


from app.rag.bm25_retriever import BM25Retriever
from app.rag.cache import RetrievalCache, get_retrieval_cache
from app.rag.embedder import BGEM3Embedder, get_embedder
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.milvus_store import MilvusStore, milvus_store
from app.rag.reranker import CrossEncoderReranker, get_reranker
from app.rag.terminology import TerminologyExpander, get_terminology
from app.rag.tokenizer import get_tokenizer


class HybridRetriever:
    """混合检索统一入口。

    组合向量召回 + BM25 召回 + RRF 融合 + Cross-Encoder 精排 + 术语扩展 +
    结果缓存，提供企业级 RAG 检索能力。

    Parameters
    ----------
    vector_store : MilvusStore
        向量存储实例（默认用单例 ``milvus_store``）。
    bm25 : BM25Retriever
        BM25 检索器实例。
    reranker : CrossEncoderReranker
        精排器实例。
    terminology : TerminologyExpander
        术语扩展器实例。
    embedder : BGEM3Embedder
        向量化器实例。
    cache : RetrievalCache
        检索缓存实例。
    """

    def __init__(
        self,
        vector_store: MilvusStore | None = None,
        bm25: BM25Retriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        terminology: TerminologyExpander | None = None,
        embedder: BGEM3Embedder | None = None,
        cache: RetrievalCache | None = None,
    ) -> None:
        self._vector_store: MilvusStore = vector_store or milvus_store
        self._bm25: BM25Retriever = bm25 or BM25Retriever()
        self._reranker: CrossEncoderReranker = reranker or get_reranker()
        self._terminology: TerminologyExpander = terminology or get_terminology()
        self._embedder: BGEM3Embedder = embedder or get_embedder()
        self._cache: RetrievalCache = cache or get_retrieval_cache()
        self._tokenizer = get_tokenizer()

        # 默认参数
        self._rrf_k: int = getattr(settings, "rrf_k", 60)
        self._default_recall_k: int = getattr(settings, "retrieval_recall_k", 50)
        self._default_top_k: int = getattr(settings, "retrieval_top_k", 5)

    # ------------------------------------------------------------------
    # 主检索方法
    # ------------------------------------------------------------------
    async def retrieve(
        self,
        query: str,
        department_id: int | None,
        top_k: int = 5,
        recall_k: int = 50,
        enable_rerank: bool = True,
        enable_graph: bool = False,
    ) -> dict[str, Any]:
        """混合检索主方法。

        Parameters
        ----------
        query : str
            用户查询。
        department_id : int | None
            用户部门 ID（权限隔离用）。None 表示只检索公开数据。
        top_k : int
            最终返回 chunk 数，默认取 settings.retrieval_top_k。
        recall_k : int
            召回阶段每路返回数，默认取 settings.retrieval_recall_k。
        enable_rerank : bool
            是否启用 Cross-Encoder 精排。False 则直接返回 RRF 融合后的 top-k。
        enable_graph : bool
            是否启用图谱增强（预留接口，当前未实现）。

        Returns
        -------
        dict
            检索结果，结构见模块 docstring。
        """
        total_start = time.monotonic()
        top_k = top_k or self._default_top_k
        recall_k = recall_k or self._default_recall_k

        # ---- Step 1: 查缓存 ----
        cached = await self._cache.get(query, department_id, top_k)
        if cached is not None:
            cached["cache_hit"] = True
            cached["latency_ms"] = round((time.monotonic() - total_start) * 1000, 2)
            logger.debug("检索缓存命中，直接返回: query='%s'", query)
            return cached

        # ---- Step 2: 术语扩展 ----
        expanded_query, term_hits = self._terminology.expand_query(query)
        # 生成术语词项权重
        query_tokens = self._tokenizer.tokenize(expanded_query)
        weighted_tokens = self._terminology.boost_term_weight(query_tokens, term_hits)
        term_weights = {tok: w for tok, w in weighted_tokens if w > 1.0}

        stages_latency: dict[str, float] = {}

        # ---- Step 3: Stage 1 双路召回（并行）----
        vector_task = self._vector_recall(expanded_query, department_id, recall_k)
        bm25_task = asyncio.to_thread(
            self._bm25_recall,
            expanded_query,
            department_id,
            recall_k,
            term_weights,
        )

        t0 = time.monotonic()
        vector_results, bm25_results = await asyncio.gather(
            vector_task, bm25_task, return_exceptions=True
        )
        # 异常容错：任一路失败返回空列表
        if isinstance(vector_results, Exception):
            logger.error("向量召回异常: %s", vector_results)
            vector_results = []
        if isinstance(bm25_results, Exception):
            logger.error("BM25 召回异常: %s", bm25_results)
            bm25_results = []
        stages_latency["vector_ms"] = round((time.monotonic() - t0) * 1000, 2)
        record_retrieval_stage("vector_recall", stages_latency["vector_ms"])

        # BM25 延迟单独记录（与向量并行，取完成时间）
        stages_latency["bm25_ms"] = stages_latency["vector_ms"]
        record_retrieval_stage("bm25_recall", stages_latency["bm25_ms"])

        # ---- Step 4: RRF 融合 ----
        t0 = time.monotonic()
        ranked_lists = [vector_results, bm25_results]
        fused = reciprocal_rank_fusion(ranked_lists, k=self._rrf_k, key_field="content")
        stages_latency["rrf_ms"] = round((time.monotonic() - t0) * 1000, 2)
        record_retrieval_stage("rrf_fusion", stages_latency["rrf_ms"])

        # ---- Step 5: Stage 2 精排 ----
        rerank_method = "none"
        t0 = time.monotonic()
        if enable_rerank and fused:
            ranked = await self._reranker.rerank(query, fused, top_k=top_k)
            rerank_method = ranked[0].get("rerank_method", "none") if ranked else "none"
        else:
            # 不启用精排，直接取 RRF top-k
            ranked = fused[:top_k]
            for i, doc in enumerate(ranked, 1):
                doc["rank"] = i
                doc["rerank_method"] = "none"
        stages_latency["rerank_ms"] = round((time.monotonic() - t0) * 1000, 2)
        record_retrieval_stage("rerank", stages_latency["rerank_ms"])

        total_ms = round((time.monotonic() - total_start) * 1000, 2)

        # ---- Step 6: 组装结果 ----
        chunks = self._format_chunks(ranked)
        scores = [c["score"] for c in chunks]

        result: dict[str, Any] = {
            "chunks": chunks,
            "scores": scores,
            "latency_ms": total_ms,
            "cache_hit": False,
            "retrieval_detail": {
                "vector_count": len(vector_results),
                "bm25_count": len(bm25_results),
                "rrf_fused_count": len(fused),
                "reranked_count": len(ranked),
                "stages_latency": stages_latency,
                "reranker_available": self._reranker.is_loaded,
                "rerank_method": rerank_method,
                "term_hits": term_hits,
                "expanded_query": expanded_query if term_hits else query,
            },
        }

        # ---- Step 7: 写缓存 ----
        await self._cache.set(query, department_id, top_k, result)

        logger.info(
            "混合检索完成: query='%s', dept=%s, vector=%d, bm25=%d, fused=%d, "
            "reranked=%d, latency=%.1fms, rerank=%s, terms=%s",
            query,
            department_id,
            len(vector_results),
            len(bm25_results),
            len(fused),
            len(ranked),
            total_ms,
            rerank_method,
            term_hits,
        )
        return result

    # ------------------------------------------------------------------
    # 召回子方法
    # ------------------------------------------------------------------
    async def _vector_recall(
        self,
        query: str,
        department_id: int | None,
        recall_k: int,
    ) -> list[dict[str, Any]]:
        """向量召回：BGE-M3 encode + Milvus 分区 search。"""
        if not self._vector_store.is_available:
            logger.debug("Milvus 不可用，向量召回返回空")
            return []

        # 编码 query
        query_embedding = await self._embedder.embed_one(query)
        # 分区检索
        results = await self._vector_store.search(
            query_embedding=query_embedding,
            department_id=department_id,
            top_k=recall_k,
            recall_k=recall_k,
        )
        return results

    def _bm25_recall(
        self,
        query: str,
        department_id: int | None,
        recall_k: int,
        term_weights: dict[str, float],
    ) -> list[dict[str, Any]]:
        """BM25 召回（术语加权 + 部门过滤）。

        BM25 无 partition 概念，权限隔离在应用层按 department_id 过滤。
        """
        if not self._bm25.is_ready:
            return []

        results = self._bm25.search(
            query=query,
            top_k=recall_k,
            score_threshold=0.0,
            term_weights=term_weights if term_weights else None,
        )
        # 应用层权限过滤：只保留 _public(0) 或本部门的 chunk
        if department_id is not None:
            results = [
                r
                for r in results
                if r.get("department_id") == 0 or r.get("department_id") == department_id
            ]
        return results

    # ------------------------------------------------------------------
    # 结果格式化
    # ------------------------------------------------------------------
    def _format_chunks(self, ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """格式化为统一 chunk 输出结构。

        确保每个 chunk 含契约要求的字段：id, content, score, source,
        heading_path, page_number, document_id, department_id。
        """
        chunks: list[dict[str, Any]] = []
        for doc in ranked:
            chunk = {
                "id": doc.get("chunk_id") or doc.get("id", ""),
                "content": doc.get("content", ""),
                "score": round(float(doc.get("score", 0.0)), 6),
                "source": doc.get("source", "unknown"),
                "heading_path": doc.get("heading_path", ""),
                "page_number": doc.get("page_number", 0),
                "document_id": doc.get("document_id", 0),
                "department_id": doc.get("department_id", 0),
                "rank": doc.get("rank", 0),
            }
            chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------------
    # 同步版（供评估脚本用）
    # ------------------------------------------------------------------
    async def retrieve_sync(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "full",
    ) -> dict[str, Any]:
        """同步版检索（供评估脚本用，不依赖 Milvus / CrossEncoder）。

        评估脚本在离线环境运行，无 Milvus 与 GPU，本方法只用 BM25 + RRF
        近似复现混合检索，用于消融对比。

        Parameters
        ----------
        query : str
            查询。
        top_k : int
            返回数。
        strategy : str
            策略：``vector_only`` / ``bm25_only`` / ``rrf`` / ``full`` /
            ``full_with_terminology``。在离线模式下，vector_only 用 BM25
            近似复现，full 等同 rrf + rerank（rerank 降级为 rrf_fallback）。

        Returns
        -------
        dict
            与 ``retrieve`` 相同结构。
        """
        total_start = time.monotonic()
        stages_latency: dict[str, float] = {
            "vector_ms": 0.0,
            "bm25_ms": 0.0,
            "rrf_ms": 0.0,
            "rerank_ms": 0.0,
        }

        # 术语扩展（仅 full_with_terminology 启用）
        term_hits: list[str] = []
        expanded_query = query
        term_weights: dict[str, float] = {}
        if strategy in ("full", "full_with_terminology"):
            expanded_query, term_hits = self._terminology.expand_query(query)
            query_tokens = self._tokenizer.tokenize(expanded_query)
            weighted_tokens = self._terminology.boost_term_weight(query_tokens, term_hits)
            term_weights = {tok: w for tok, w in weighted_tokens if w > 1.0}

        t0 = time.monotonic()
        # 离线模式下 vector 用 BM25 近似复现
        vector_results = (
            self._bm25.search(
                query=expanded_query,
                top_k=50,
                term_weights=term_weights if term_weights else None,
            )
            if strategy != "bm25_only"
            else []
        )
        stages_latency["vector_ms"] = round((time.monotonic() - t0) * 1000, 2)

        t0 = time.monotonic()
        bm25_results = (
            self._bm25.search(
                query=expanded_query, top_k=50, term_weights=term_weights if term_weights else None
            )
            if strategy in ("bm25_only", "rrf", "full", "full_with_terminology")
            else []
        )
        stages_latency["bm25_ms"] = round((time.monotonic() - t0) * 1000, 2)

        # 融合
        t0 = time.monotonic()
        if strategy in ("rrf", "full", "full_with_terminology"):
            fused = reciprocal_rank_fusion([vector_results, bm25_results], k=self._rrf_k)
        elif strategy == "vector_only":
            fused = vector_results
        elif strategy == "bm25_only":
            fused = bm25_results
        else:
            fused = bm25_results
        stages_latency["rrf_ms"] = round((time.monotonic() - t0) * 1000, 2)

        # 精排（离线降级为 rrf_fallback）
        ranked = fused[:top_k]
        for i, doc in enumerate(ranked, 1):
            doc["rank"] = i
            doc["rerank_method"] = "rrf_fallback"
        stages_latency["rerank_ms"] = 0.0

        total_ms = round((time.monotonic() - total_start) * 1000, 2)
        chunks = self._format_chunks(ranked)

        return {
            "chunks": chunks,
            "scores": [c["score"] for c in chunks],
            "latency_ms": total_ms,
            "cache_hit": False,
            "retrieval_detail": {
                "vector_count": len(vector_results),
                "bm25_count": len(bm25_results),
                "rrf_fused_count": len(fused),
                "reranked_count": len(ranked),
                "stages_latency": stages_latency,
                "reranker_available": False,
                "rerank_method": "rrf_fallback",
                "term_hits": term_hits,
                "strategy": strategy,
            },
        }

    # ------------------------------------------------------------------
    # 可解释性
    # ------------------------------------------------------------------
    def explain_retrieval(self, query: str, chunk_id: str) -> dict[str, Any]:
        """解释某 chunk 为何被召回 / 排名如何。

        Parameters
        ----------
        query : str
            查询。
        chunk_id : str
            目标 chunk ID。

        Returns
        -------
        dict
            含 BM25 得分分项、术语命中情况、向量分数（若有）。
        """
        # 术语扩展信息
        expanded_query, term_hits = self._terminology.expand_query(query)

        # BM25 解释：找到 chunk 在语料中的索引
        bm25_explain: dict[str, Any] = {"available": False}
        if self._bm25.is_ready:
            for idx, doc in enumerate(self._bm25._corpus):
                if str(doc.get("chunk_id", "")) == str(chunk_id):
                    bm25_explain = {
                        "available": True,
                        **self._bm25.explain_score(expanded_query, idx),
                    }
                    break

        return {
            "query": query,
            "expanded_query": expanded_query,
            "chunk_id": chunk_id,
            "term_hits": term_hits,
            "bm25_explanation": bm25_explain,
        }

    # ------------------------------------------------------------------
    # 索引管理（供 ingestion 调用）
    # ------------------------------------------------------------------
    def build_bm25_index(self, documents: list[dict[str, Any]]) -> int:
        """构建 BM25 索引（供 ingestion 流程调用）。"""
        return self._bm25.build_index(documents)

    async def invalidate_cache(self) -> int:
        """失效全部检索缓存（文档更新后调用）。"""
        return await self._cache.invalidate_all()
