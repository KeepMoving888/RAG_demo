"""
检索质量消融评估脚本

功能:
1. 基于种子文档与评估数据集, 执行真实 BM25 检索评估
2. 对比 bm25_only / vector_only / vector_only_milvus(真实) / hybrid_rrf /
   hybrid_milvus_rrf(真实) 多种策略
3. 计算 Recall@5 / MRR / NDCG@5 / Precision@5
4. 输出对比表格, 写入 ablation_results.json

策略矩阵:
- bm25_only:           真实 BM25 检索
- vector_only:         近似向量排序 (公开基准近似, 无 Milvus 环境降级)
- vector_only_milvus:  真实 Milvus + BGE-M3 向量检索 (Milvus 在线时启用)
- hybrid_rrf:          真实 BM25 + 近似向量 + RRF
- hybrid_milvus_rrf:   真实 BM25 + 真实 Milvus 向量 + RRF (推荐生产)

用法:
    python -m scripts.run_ablation
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windows 事件循环修复: asyncpg + pymilvus 在 ProactorEventLoop 下不稳定
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 确保 backend 在 sys.path 中
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.rag.bm25_retriever import BM25Retriever  # noqa: E402
from app.rag.fusion import reciprocal_rank_fusion  # noqa: E402
from app.utils.logger import logger  # noqa: E402

# 真实 Milvus + BGE-M3 (懒加载, 失败则降级为模拟)
try:
    from app.rag.embedder import get_embedder  # noqa: E402
    from app.rag.milvus_store import milvus_store  # noqa: E402
    _REAL_VECTOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REAL_VECTOR_AVAILABLE = False
    milvus_store = None  # type: ignore[assignment]


# ======================== 路径常量 ========================
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_SEED_DIR = _BACKEND_DIR / "data" / "seed"
_EVAL_FILE = _SEED_DIR / "rag_eval_dataset.json"
_OUTPUT_FILE = _SEED_DIR / "ablation_results.json"

TOP_K = 5
RRF_K = 60  # RRF 平滑常数, SIGIR 2009 经验最优


# ======================== 指标计算 (与 app.rag.evaluator 保持一致) ========================

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Recall@K: 前 K 个结果中命中相关文档的比例."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """MRR: 第一个相关文档排名倒数."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevance_map: Dict[str, int], k: int = 5) -> float:
    """NDCG@K: 归一化折损累积增益 (支持多级相关性)."""
    if not relevance_map:
        return 0.0
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_map.get(rid, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(
        (2**rel - 1) / math.log2(i + 1)
        for i, rel in enumerate(ideal_rels, start=1)
        if rel > 0
    )
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Precision@K: 前 K 个结果中相关文档的比例."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / k


def _aggregate(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    """聚合 per-query 指标为整体均值."""
    n = len(per_query)
    if n == 0:
        return {
            "recall@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0,
            "precision@5": 0.0, "sample_count": 0,
        }
    return {
        "recall@5": round(sum(r["recall@5"] for r in per_query) / n, 4),
        "mrr": round(sum(r["mrr"] for r in per_query) / n, 4),
        "ndcg@5": round(sum(r["ndcg@5"] for r in per_query) / n, 4),
        "precision@5": round(sum(r["precision@5"] for r in per_query) / n, 4),
        "sample_count": n,
    }


# ======================== 数据加载 ========================

def _load_corpus() -> List[Dict[str, Any]]:
    """加载种子文档作为语料.

    种子文档定义在 scripts.seed_docs._SEED_DOCUMENTS, 评估数据集中的
    relevant_chunk_ids ("1".."18") 与种子文档列表的 1-indexed 位置对齐,
    故此处为每个文档赋予字符串 id.
    """
    from scripts.seed_docs import _SEED_DOCUMENTS

    corpus: List[Dict[str, Any]] = []
    for idx, doc in enumerate(_SEED_DOCUMENTS, start=1):
        corpus.append({
            "id": str(idx),
            "content": doc["content"],
            "title": doc.get("title", ""),
            "category": doc.get("category", ""),
        })
    return corpus


def _stable_hash(text: str) -> int:
    """稳定哈希 (跨进程可复现, 不受 PYTHONHASHSEED 影响)."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)


# ======================== 策略: bm25_only (真实) ========================

def eval_bm25_only(
    bm25: BM25Retriever,
    dataset: List[Dict[str, Any]],
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """真实 BM25 检索评估.

    直接调用 BM25Retriever.search 执行真实的 rank_bm25 检索,
    指标完全基于真实检索结果计算.
    """
    per_query: List[Dict[str, Any]] = []
    for sample in dataset:
        query = sample.get("query", "")
        relevant_ids = sample.get("relevant_chunk_ids", [])
        relevance_map = sample.get("relevance_level", {})
        if not query or not relevant_ids:
            continue

        results = bm25.search(query, top_k=top_k)
        retrieved_ids = [r["id"] for r in results]

        per_query.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "recall@5": recall_at_k(retrieved_ids, relevant_ids, k=top_k),
            "mrr": mrr(retrieved_ids, relevant_ids),
            "ndcg@5": ndcg_at_k(retrieved_ids, relevance_map, k=top_k),
            "precision@5": precision_at_k(retrieved_ids, relevant_ids, k=top_k),
        })
    return per_query


# ======================== 策略: vector_only (近似参考值) ========================

def _simulate_vector_ranking(
    bm25: BM25Retriever,
    query: str,
    all_docs: List[Dict[str, Any]],
    top_n: int = 20,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """近似向量检索排序 (无 Milvus 环境下的近似).

    公开基准参考 (bge-m3 on Chinese enterprise KB, MTEB zh / BEIR):
    - 对含明确型号/标准号的查询, 向量检索字面匹配弱于 BM25
    - 对语义/概念查询, 向量检索召回有优势
    - 整体在中文技术文档场景, 纯向量检索 P@5 / NDCG@5 通常比 BM25 低 5-15%

    近似策略:
    1. 取 BM25 全量分数作为相关性先验 (向量与 BM25 在 top 段有较大重叠)
    2. 对 BM25 命中文档: 近似分数 = BM25 分数 * 衰减系数 + 确定性噪声
       (衰减近似向量对字面匹配的不敏感)
    3. 对 BM25 未命中文档: 小概率被向量语义召回 (近似语义召回能力)
    4. 按近似分数排序, 取 top_n
    """
    rng = random.Random(_stable_hash(query) + seed)

    # BM25 全量分数 (top_k 设为语料大小, 取全部)
    bm25_results = bm25.search(query, top_k=len(all_docs))
    bm25_scores = {r["id"]: r.get("score", 0.0) for r in bm25_results}

    scored: List[tuple] = []
    for doc in all_docs:
        doc_id = doc["id"]
        bm25_score = bm25_scores.get(doc_id, 0.0)

        if bm25_score > 0:
            # BM25 命中: 向量分数 = BM25 * 0.7 + 噪声 (字面匹配衰减)
            sim_score = bm25_score * 0.7 + rng.uniform(-0.5, 0.5)
        else:
            # BM25 未命中: 10% 概率被语义召回
            sim_score = rng.uniform(0.1, 0.8) if rng.random() < 0.1 else 0.0

        if sim_score > 0:
            scored.append((sim_score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


def eval_vector_only(
    bm25: BM25Retriever,
    dataset: List[Dict[str, Any]],
    corpus: List[Dict[str, Any]],
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """近似向量检索评估.

    NOTE: 真实向量检索需要 Milvus + embedding 模型 (BAAI/bge-m3),
    离线环境无法运行. 此处基于公开基准的行为特征近似向量排序,
    指标仅供横向参考, 非真实检索结果.

    公开参考区间 (bge-m3, Chinese enterprise KB):
        Recall@5:   0.55 - 0.70
        MRR:        0.45 - 0.60
        NDCG@5:     0.50 - 0.65
        Precision@5: 0.35 - 0.50
    """
    per_query: List[Dict[str, Any]] = []
    for sample in dataset:
        query = sample.get("query", "")
        relevant_ids = sample.get("relevant_chunk_ids", [])
        relevance_map = sample.get("relevance_level", {})
        if not query or not relevant_ids:
            continue

        simulated = _simulate_vector_ranking(bm25, query, corpus, top_n=top_k)
        retrieved_ids = [r["id"] for r in simulated]

        per_query.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "recall@5": recall_at_k(retrieved_ids, relevant_ids, k=top_k),
            "mrr": mrr(retrieved_ids, relevant_ids),
            "ndcg@5": ndcg_at_k(retrieved_ids, relevance_map, k=top_k),
            "precision@5": precision_at_k(retrieved_ids, relevant_ids, k=top_k),
        })
    return per_query


# ======================== 策略: vector_only_milvus (真实 Milvus + BGE-M3) ========================

async def _ensure_milvus_ready() -> bool:
    """检查并初始化 Milvus Collection.

    Returns:
        True 表示 Milvus 可用并已初始化.
    """
    if not _REAL_VECTOR_AVAILABLE or milvus_store is None:
        logger.warning("Milvus 或 embedder 模块未安装, 跳过真实向量评估")
        return False
    if milvus_store.is_available:
        return True
    ok = await milvus_store.init_collection()
    if not ok:
        logger.warning("Milvus 初始化失败, 跳过真实向量评估")
        return False
    # 加载 embedder 模型 (避免首次检索延迟计入指标)
    embedder = get_embedder()
    await embedder.embed(["warmup"])  # 预热
    return True


async def eval_vector_only_milvus(
    dataset: List[Dict[str, Any]],
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """真实 Milvus + BGE-M3 向量检索评估.

    对每个 query 调用真实 BGE-M3 编码 -> Milvus 分区检索.
    评估场景: search_all_partitions=True 跨全部分区搜索.

    注: 评估数据集 relevant_chunk_ids 与种子文档 1-indexed 位置对齐,
    故此处用 Milvus 返回的 document_id 做匹配 (DB document_id == 种子位置).
    """
    embedder = get_embedder()
    per_query: List[Dict[str, Any]] = []
    for sample in dataset:
        query = sample.get("query", "")
        relevant_ids = sample.get("relevant_chunk_ids", [])
        relevance_map = sample.get("relevance_level", {})
        if not query or not relevant_ids:
            continue

        # 真实向量编码
        q_vec = await embedder.embed_one(query)
        # 真实 Milvus 检索 (评估场景: 搜全部分区)
        results = await milvus_store.search(
            query_embedding=q_vec,
            department_id=None,
            top_k=top_k,
            recall_k=top_k * 2,
            search_all_partitions=True,
        )
        # 用 document_id 做匹配 (与种子位置对齐), 多个 chunk 同属一个文档时去重
        seen_docs = set()
        retrieved_ids: List[str] = []
        for r in results:
            doc_id = str(r.get("document_id", 0))
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                retrieved_ids.append(doc_id)

        per_query.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "recall@5": recall_at_k(retrieved_ids, relevant_ids, k=top_k),
            "mrr": mrr(retrieved_ids, relevant_ids),
            "ndcg@5": ndcg_at_k(retrieved_ids, relevance_map, k=top_k),
            "precision@5": precision_at_k(retrieved_ids, relevant_ids, k=top_k),
        })
    return per_query


# ======================== 策略: hybrid_milvus_rrf (真实 BM25 + 真实 Milvus + RRF) ========================

async def eval_hybrid_milvus_rrf(
    bm25: BM25Retriever,
    dataset: List[Dict[str, Any]],
    top_k: int = TOP_K,
    rrf_k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """真实 BM25 + 真实 Milvus + RRF 融合评估 (推荐生产配置).

    流程:
    1. 真实 BM25 检索 top-20
    2. 真实 Milvus + BGE-M3 检索 top-40 (评估场景: 搜全部分区), 按文档去重取 top-20
    3. RRF 融合 (k=60)
    4. 取融合后 top-5 计算指标

    注: Milvus 可能返回同文档多个 chunk, 需先按 document_id 去重 (保留最高分 chunk).
    """
    embedder = get_embedder()
    per_query: List[Dict[str, Any]] = []
    for sample in dataset:
        query = sample.get("query", "")
        relevant_ids = sample.get("relevant_chunk_ids", [])
        relevance_map = sample.get("relevance_level", {})
        if not query or not relevant_ids:
            continue

        # 真实 BM25 top-20 (id 字段 = 种子位置字符串)
        bm25_results = bm25.search(query, top_k=20)
        bm25_normalized = [
            {"doc_id": str(r.get("id", "")), **r} for r in bm25_results
        ]

        # 真实 Milvus top-40 (评估场景: 搜全部分区)
        q_vec = await embedder.embed_one(query)
        vector_results = await milvus_store.search(
            query_embedding=q_vec,
            department_id=None,
            top_k=40,
            recall_k=80,
            search_all_partitions=True,
        )
        # 按文档去重 (保留最高分 chunk, Milvus 默认按 score 降序)
        seen_docs: set = set()
        vector_normalized: List[Dict[str, Any]] = []
        for r in vector_results:
            doc_id = str(r.get("document_id", 0))
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                vector_normalized.append({"doc_id": doc_id, **r})
            if len(vector_normalized) >= 20:
                break

        # RRF 融合 (按 doc_id 去重)
        fused = reciprocal_rank_fusion(
            [bm25_normalized, vector_normalized],
            k=rrf_k,
            key_field="doc_id",
        )
        retrieved_ids = [str(f.get("doc_id", "")) for f in fused[:top_k]]

        per_query.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "recall@5": recall_at_k(retrieved_ids, relevant_ids, k=top_k),
            "mrr": mrr(retrieved_ids, relevant_ids),
            "ndcg@5": ndcg_at_k(retrieved_ids, relevance_map, k=top_k),
            "precision@5": precision_at_k(retrieved_ids, relevant_ids, k=top_k),
        })
    return per_query


# ======================== 策略: hybrid_rrf (BM25真实 + 向量近似 + RRF) ========================

def eval_hybrid_rrf(
    bm25: BM25Retriever,
    dataset: List[Dict[str, Any]],
    corpus: List[Dict[str, Any]],
    top_k: int = TOP_K,
    rrf_k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """hybrid_rrf 评估: 真实 BM25 + 近似向量 + RRF 融合.

    流程:
    1. 真实 BM25 检索 top-20
    2. 近似向量检索 top-20 (基于 BM25 先验 + 噪声)
    3. 对两路结果执行 RRF 融合 (k=60, SIGIR 2009 经验最优)
    4. 取融合后 top-5 计算指标

    RRF 公式: score(d) = Σ_i 1 / (k + rank_i(d)), rank 从 1 开始
    """
    per_query: List[Dict[str, Any]] = []
    for sample in dataset:
        query = sample.get("query", "")
        relevant_ids = sample.get("relevant_chunk_ids", [])
        relevance_map = sample.get("relevance_level", {})
        if not query or not relevant_ids:
            continue

        # 真实 BM25 top-20
        bm25_results = bm25.search(query, top_k=20)
        # 近似向量 top-20
        vector_results = _simulate_vector_ranking(
            bm25, query, corpus, top_n=20
        )

        # RRF 融合 (按 id 去重)
        fused = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            k=rrf_k,
            key_field="id",
        )
        retrieved_ids = [f["id"] for f in fused[:top_k]]

        per_query.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "recall@5": recall_at_k(retrieved_ids, relevant_ids, k=top_k),
            "mrr": mrr(retrieved_ids, relevant_ids),
            "ndcg@5": ndcg_at_k(retrieved_ids, relevance_map, k=top_k),
            "precision@5": precision_at_k(retrieved_ids, relevant_ids, k=top_k),
        })
    return per_query


# ======================== 输出格式化 ========================

def _format_table(strategies: List[tuple]) -> str:
    """格式化对比表格为字符串."""
    header = (
        f"{'Strategy':<16} {'Recall@5':<12} {'MRR':<10} {'NDCG@5':<12} "
        f"{'P@5':<10} {'Samples':<10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for name, metrics in strategies:
        lines.append(
            f"{name:<16} {metrics['recall@5']:<12.4f} {metrics['mrr']:<10.4f} "
            f"{metrics['ndcg@5']:<12.4f} {metrics['precision@5']:<10.4f} "
            f"{metrics['sample_count']:<10}"
        )
    return "\n".join(lines)


def _generate_analysis(
    bm25_m: Dict[str, Any],
    vector_m: Dict[str, Any],
    hybrid_m: Dict[str, Any],
) -> str:
    """根据消融结果生成自动分析."""
    parts: List[str] = []

    # BM25 vs vector_only
    if bm25_m["ndcg@5"] > 0 and vector_m["ndcg@5"] > 0:
        diff = (bm25_m["ndcg@5"] - vector_m["ndcg@5"]) / vector_m["ndcg@5"] * 100
        parts.append(
            f"bm25_only 相比 vector_only NDCG@5 {'高' if diff > 0 else '低'} {abs(diff):.1f}% "
            f"({bm25_m['ndcg@5']:.4f} vs {vector_m['ndcg@5']:.4f})"
        )

    # hybrid vs bm25
    if bm25_m["ndcg@5"] > 0 and hybrid_m["ndcg@5"] > 0:
        gain = (hybrid_m["ndcg@5"] - bm25_m["ndcg@5"]) / bm25_m["ndcg@5"] * 100
        parts.append(
            f"hybrid_rrf 相比 bm25_only NDCG@5 {'提升' if gain > 0 else '下降'} {abs(gain):.1f}% "
            f"({bm25_m['ndcg@5']:.4f} -> {hybrid_m['ndcg@5']:.4f})"
        )

    if not parts:
        parts.append("各策略指标差异不显著, 建议检查数据集标注质量.")

    return " | ".join(parts)


# ======================== 主流程 ========================

def main() -> int:
    """主流程: 加载数据 -> 构建索引 -> 多策略评估 -> 输出结果.

    策略矩阵:
    - bm25_only:           真实 BM25
    - vector_only:         近似向量 (无 Milvus 环境的降级参考)
    - vector_only_milvus:  真实 Milvus + BGE-M3 (Milvus 在线时启用)
    - hybrid_rrf:          真实 BM25 + 近似向量 + RRF
    - hybrid_milvus_rrf:   真实 BM25 + 真实 Milvus + RRF (推荐生产)
    """
    # 1. 加载语料与评估集
    logger.info("加载种子文档与评估数据集...")
    corpus = _load_corpus()
    logger.info("种子文档数: {}", len(corpus))

    with open(_EVAL_FILE, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)
    logger.info("评估样本数: {}", len(dataset))

    # 2. 构建 BM25 索引
    logger.info("构建 BM25 索引 (rank_bm25 + jieba 分词)...")
    bm25 = BM25Retriever()
    n_indexed = bm25.build_index(corpus, content_field="content")
    logger.info("BM25 索引就绪, 文档数={}, 平均文档长度={:.1f} tokens",
                bm25.corpus_size, bm25.avg_doc_length)

    # 3. 评估 bm25_only (真实)
    logger.info("评估 bm25_only (真实检索)...")
    bm25_pq = eval_bm25_only(bm25, dataset)
    bm25_metrics = _aggregate(bm25_pq)
    logger.info("bm25_only 完成: ndcg@5={:.4f}, recall@5={:.4f}",
                bm25_metrics["ndcg@5"], bm25_metrics["recall@5"])

    # 4. 评估 vector_only (近似参考值)
    logger.info("评估 vector_only (近似参考值)...")
    vector_pq = eval_vector_only(bm25, dataset, corpus)
    vector_metrics = _aggregate(vector_pq)
    logger.info("vector_only 完成: ndcg@5={:.4f}, recall@5={:.4f}",
                vector_metrics["ndcg@5"], vector_metrics["recall@5"])

    # 5. 评估 hybrid_rrf (BM25真实 + 向量近似)
    logger.info("评估 hybrid_rrf (BM25真实 + 向量近似 + RRF)...")
    hybrid_pq = eval_hybrid_rrf(bm25, dataset, corpus)
    hybrid_metrics = _aggregate(hybrid_pq)
    logger.info("hybrid_rrf 完成: ndcg@5={:.4f}, recall@5={:.4f}",
                hybrid_metrics["ndcg@5"], hybrid_metrics["recall@5"])

    # 6. 评估真实 Milvus 策略 (若 Milvus 在线)
    vector_milvus_pq: List[Dict[str, Any]] = []
    vector_milvus_metrics: Optional[Dict[str, Any]] = None
    hybrid_milvus_pq: List[Dict[str, Any]] = []
    hybrid_milvus_metrics: Optional[Dict[str, Any]] = None
    milvus_online = False

    try:
        milvus_online = asyncio.run(_ensure_milvus_ready())
    except Exception as exc:
        logger.warning("Milvus 初始化异常, 跳过真实向量评估: {}", exc)
        milvus_online = False

    if milvus_online:
        logger.info("评估 vector_only_milvus (真实 Milvus + BGE-M3)...")
        try:
            vector_milvus_pq = asyncio.run(eval_vector_only_milvus(dataset))
            vector_milvus_metrics = _aggregate(vector_milvus_pq)
            logger.info("vector_only_milvus 完成: ndcg@5={:.4f}, recall@5={:.4f}",
                        vector_milvus_metrics["ndcg@5"],
                        vector_milvus_metrics["recall@5"])
        except Exception as exc:
            logger.error("vector_only_milvus 评估失败: {}", exc)
            vector_milvus_metrics = None

        logger.info("评估 hybrid_milvus_rrf (真实 BM25 + 真实 Milvus + RRF)...")
        try:
            hybrid_milvus_pq = asyncio.run(eval_hybrid_milvus_rrf(bm25, dataset))
            hybrid_milvus_metrics = _aggregate(hybrid_milvus_pq)
            logger.info("hybrid_milvus_rrf 完成: ndcg@5={:.4f}, recall@5={:.4f}",
                        hybrid_milvus_metrics["ndcg@5"],
                        hybrid_milvus_metrics["recall@5"])
        except Exception as exc:
            logger.error("hybrid_milvus_rrf 评估失败: {}", exc)
            hybrid_milvus_metrics = None
    else:
        logger.warning("Milvus 未就绪, 跳过 vector_only_milvus 与 hybrid_milvus_rrf")

    # 7. 输出对比表格
    strategies: List[tuple] = [
        ("bm25_only", bm25_metrics),
        ("vector_only", vector_metrics),
        ("hybrid_rrf", hybrid_metrics),
    ]
    if vector_milvus_metrics is not None:
        strategies.append(("vector_only_milvus", vector_milvus_metrics))
    if hybrid_milvus_metrics is not None:
        strategies.append(("hybrid_milvus_rrf", hybrid_milvus_metrics))

    table = _format_table(strategies)

    print("\n" + "=" * 90)
    print("检索质量消融评估结果")
    print("=" * 90)
    print()
    print(table)
    print()

    # 最优策略
    best = max(strategies, key=lambda x: x[1]["ndcg@5"])
    print(f"最优策略 (按 NDCG@5): {best[0]} (ndcg@5={best[1]['ndcg@5']:.4f})")
    print()

    # 自动分析 (含真实 Milvus 数据对比)
    analysis_parts: List[str] = []
    base_analysis = _generate_analysis(bm25_metrics, vector_metrics, hybrid_metrics)
    analysis_parts.append(base_analysis)
    if vector_milvus_metrics is not None:
        analysis_parts.append(
            f"vector_only_milvus (真实 Milvus) 相比 vector_only (近似) "
            f"NDCG@5 {'高' if vector_milvus_metrics['ndcg@5'] > vector_metrics['ndcg@5'] else '低'} "
            f"{abs(vector_milvus_metrics['ndcg@5'] - vector_metrics['ndcg@5']) * 100 / max(vector_metrics['ndcg@5'], 1e-9):.1f}% "
            f"({vector_metrics['ndcg@5']:.4f} -> {vector_milvus_metrics['ndcg@5']:.4f})"
        )
    if hybrid_milvus_metrics is not None:
        analysis_parts.append(
            f"hybrid_milvus_rrf (真实双路) 相比 hybrid_rrf (近似) "
            f"NDCG@5 {'高' if hybrid_milvus_metrics['ndcg@5'] > hybrid_metrics['ndcg@5'] else '低'} "
            f"{abs(hybrid_milvus_metrics['ndcg@5'] - hybrid_metrics['ndcg@5']) * 100 / max(hybrid_metrics['ndcg@5'], 1e-9):.1f}% "
            f"({hybrid_metrics['ndcg@5']:.4f} -> {hybrid_milvus_metrics['ndcg@5']:.4f})"
        )
    analysis = " | ".join(analysis_parts)
    print("自动分析:")
    print(f"  {analysis}")
    print()
    print("=" * 90)

    # 8. 写入结果文件
    strategy_payloads = [
        ("bm25_only", bm25_metrics, bm25_pq),
        ("vector_only", vector_metrics, vector_pq),
        ("hybrid_rrf", hybrid_metrics, hybrid_pq),
    ]
    if vector_milvus_metrics is not None:
        strategy_payloads.append(("vector_only_milvus", vector_milvus_metrics, vector_milvus_pq))
    if hybrid_milvus_metrics is not None:
        strategy_payloads.append(("hybrid_milvus_rrf", hybrid_milvus_metrics, hybrid_milvus_pq))

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "top_k": TOP_K,
        "rrf_k": RRF_K,
        "corpus_size": len(corpus),
        "dataset_size": len(dataset),
        "milvus_online": milvus_online,
        "strategies": [
            {"strategy": name, **metrics, "per_query": per_query}
            for name, metrics, per_query in strategy_payloads
        ],
        "comparison_table": table,
        "best_strategy": best[0],
        "analysis": analysis,
        "notes": {
            "bm25_only": "真实 BM25 检索 (rank_bm25.BM25Okapi + jieba 中英双语分词)",
            "vector_only": (
                "近似参考值, 真实运行需 Milvus + embedding 模型 (BAAI/bge-m3). "
                "基于公开基准 (MTEB zh / BEIR) 行为特征近似: "
                "对含型号/标准号的查询, 向量检索字面匹配弱于 BM25."
            ),
            "vector_only_milvus": (
                "真实 Milvus + BGE-M3 向量检索. "
                "query 经 BGE-M3 编码后入 Milvus IVF_FLAT 索引检索."
            ),
            "hybrid_rrf": (
                "真实 BM25 top-20 + 近似向量 top-20 + RRF(k=60) 融合. "
                "BM25 部分为真实检索, 向量部分为近似, RRF 融合为真实计算."
            ),
            "hybrid_milvus_rrf": (
                "真实 BM25 top-20 + 真实 Milvus top-20 + RRF(k=60) 融合. "
                "推荐生产配置."
            ),
        },
    }

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入: {_OUTPUT_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
