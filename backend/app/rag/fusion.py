"""app.rag.fusion —— 多路召回结果融合（RRF / 加权融合）

设计要点：为何选择 RRF 而非加权融合
------------------------------------
混合检索需要把向量检索与 BM25 检索两路结果合并为一个排序。两种主流方案：

1. **加权融合（Weighted Fusion）**
   - 做法：对各路分数归一化后按权重相加，如 ``0.5 * norm(vector_score) +
     0.5 * norm(bm25_score)``。
   - 缺点：
     * 需要对每路分数做归一化（min-max / z-score），归一化策略本身引入
       超参数，且对 outlier 敏感；
     * 不同检索器的分数尺度差异大（向量余弦 ∈ [0,1]，BM25 可达数十），
       归一化后仍可能因分布偏斜导致一路主导；
     * 权重需调参，且不同查询最优权重不同，泛化性差。

2. **RRF（Reciprocal Rank Fusion，倒数排名融合）**
   - 做法：``score(d) = Σ_i 1 / (k + rank_i(d))``，``rank`` 从 1 开始，
     ``k`` 为平滑常数（经验值 60）。
   - 优点：
     * **尺度无关**：只用排名不用原始分数，天然消除不同检索器分数尺度差异；
     * **对 outlier 鲁棒**：排名第 1 与第 2 的贡献差为
       ``1/61 - 1/62 ≈ 0.000266``，而排名第 100 与第 101 差仅
       ``1/160 - 1/161 ≈ 0.0000389``，高排名文档差异被放大、低排名被平滑，
       与人工判断的注意力分配一致；
     * **无需调参**：k=60 由 Cormack et al. (2009) 在大规模实验中验证为
       鲁棒最优，跨数据集泛化好；
     * 实现简单、计算开销低。

结论：生产环境首选 RRF，本模块同时保留 ``weighted_fusion`` 供对比实验，
但 ``HybridRetriever`` 默认使用 RRF。

参考文献
~~~~~~~~
Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
*Reciprocal rank fusion outperforms condorcet and individual rank learning
methods*. SIGIR 2009.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.utils.logger import logger


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    key_field: str = "content",
) -> list[dict]:
    """倒数排名融合（RRF）。

    将多路检索的排序列表融合为一个综合排序列表。

    Parameters
    ----------
    ranked_lists : list[list[dict]]
        多路检索结果，每个子列表已按相关性降序排列。每个 dict 至少包含
        ``key_field`` 指定的字段用于去重标识。
    k : int, default 60
        平滑常数。k 越大，排名差异对分数的影响越平缓。k=60 是 SIGIR 2009
        经验最优值。
    key_field : str, default "content"
        用于判定文档唯一性的字段。同一文档在不同路中可能 score 不同，但
        ``content``（或 chunk_id）应一致。

    Returns
    -------
    list[dict]
        融合后按 RRF 分数降序排列的文档列表，每个 dict 新增 ``rrf_score``
        字段，并保留各路原始分数于 ``source_scores`` 字段。

    Notes
    -----
    公式：``score(d) = Σ_i 1 / (k + rank_i(d))``，``rank`` 从 1 开始。
    若文档在第 i 路未出现，则该路贡献为 0。
    """
    if not ranked_lists:
        return []

    # 融合分数累加器：key -> {"rrf_score": float, "doc": dict, "sources": [...]}
    fused: dict[str, dict] = {}

    for list_idx, ranked in enumerate(ranked_lists):
        for rank, doc in enumerate(ranked, start=1):
            # 取 key 用于去重；缺省回退到 id 再到对象 id 字符串
            doc_key = doc.get(key_field) or doc.get("id") or str(id(doc))
            if not isinstance(doc_key, str):
                doc_key = str(doc_key)

            contribution = 1.0 / (k + rank)

            if doc_key not in fused:
                fused[doc_key] = {
                    "doc": dict(doc),  # 浅拷贝避免污染原始数据
                    "rrf_score": 0.0,
                    "sources": [],
                }
            fused[doc_key]["rrf_score"] += contribution
            # 记录该文档在各路的来源与分数，便于可解释性
            fused[doc_key]["sources"].append(
                {
                    "list_index": list_idx,
                    "rank": rank,
                    "original_score": doc.get("score", 0.0),
                }
            )

    # 按融合分数降序排列
    result: list[dict] = []
    for item in sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True):
        doc = item["doc"]
        doc["rrf_score"] = round(item["rrf_score"], 6)
        doc["source_scores"] = item["sources"]
        # 融合后的统一 score 字段设为 rrf_score，供下游统一处理
        doc["score"] = doc["rrf_score"]
        result.append(doc)

    logger.debug(
        "RRF 融合完成: 输入路数=%d，融合后文档数=%d，k=%d",
        len(ranked_lists),
        len(result),
        k,
    )
    return result


def weighted_fusion(
    ranked_lists: Sequence[list[dict]],
    weights: Sequence[float],
    key_field: str = "content",
) -> list[dict]:
    """加权融合（对比基线，默认不使用）。

    对各路分数做 min-max 归一化后按权重相加。保留此方法用于消融实验对比
    RRF 的效果，生产环境使用 RRF。

    Parameters
    ----------
    ranked_lists : sequence[list[dict]]
        多路检索结果。
    weights : sequence[float]
        各路权重，长度须与 ``ranked_lists`` 一致，权重和无需归一。
    key_field : str, default "content"
        文档唯一性字段。

    Returns
    -------
    list[dict]
        融合后按加权分数降序排列的文档列表。

    Warns
    -----
    此方法对 outlier 敏感，且需调权重；推荐使用 ``reciprocal_rank_fusion``。
    """
    if not ranked_lists:
        return []
    if len(ranked_lists) != len(weights):
        raise ValueError(
            f"ranked_lists 长度({len(ranked_lists)})与 weights 长度({len(weights)})不一致"
        )

    # 权重归一化
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights] if total_w > 0 else list(weights)

    # 各路分数 min-max 归一化
    normalized_lists: list[list[dict]] = []
    for ranked in ranked_lists:
        if not ranked:
            normalized_lists.append([])
            continue
        scores = [d.get("score", 0.0) for d in ranked]
        min_s, max_s = min(scores), max(scores)
        denom = (max_s - min_s) if (max_s - min_s) > 1e-12 else 1.0
        norm_list = [{**d, "_norm_score": (d.get("score", 0.0) - min_s) / denom} for d in ranked]
        normalized_lists.append(norm_list)

    # 累加加权分数
    fused: dict[str, dict] = {}
    for list_idx, (norm_list, w) in enumerate(zip(normalized_lists, norm_weights)):
        for doc in norm_list:
            doc_key = doc.get(key_field) or doc.get("id") or str(id(doc))
            if not isinstance(doc_key, str):
                doc_key = str(doc_key)
            contribution = doc["_norm_score"] * w
            if doc_key not in fused:
                fused[doc_key] = {"doc": dict(doc), "weighted_score": 0.0}
            fused[doc_key]["weighted_score"] += contribution

    result: list[dict] = []
    for item in sorted(fused.values(), key=lambda x: x["weighted_score"], reverse=True):
        doc = item["doc"]
        doc.pop("_norm_score", None)
        doc["weighted_score"] = round(item["weighted_score"], 6)
        doc["score"] = doc["weighted_score"]
        result.append(doc)

    logger.debug("加权融合完成: 融合后文档数=%d，权重=%s", len(result), norm_weights)
    return result
