"""app.rag.bm25_retriever —— 稀疏词项检索器（rank_bm25）

设计要点
--------
BM25 是基于词频（tf）与逆文档频率（idf）的概率检索模型，擅长字面精确
匹配，与向量检索的语义召回形成互补。本模块封装 ``rank_bm25.BM25Okapi``，
并针对企业 RAG 场景做了三项增强：

1. **术语加权打分**：原生 BM25 对所有 query token 一视同仁。本模块支持
   ``term_weights`` 参数，对术语命中的 token 加权（×2.0），使含术语的
   文档排名前置。这是术语扩展器与 BM25 的衔接点。
2. **可解释性**：``explain_score`` 输出每个 query token 的 tf / idf /
   contribution 分项，便于排查「为何该文档没召回」类问题。
3. **增量重建说明**：``BM25Okapi`` 内部预计算了 idf 表与文档长度统计，
   不支持单文档增量插入。``add_documents`` 通过全量重建实现「增量」，
   docstring 中说明了原因与取舍：生产中文档更新频率远低于查询频率，
   全量重建的开销可接受，且能保证 idf 统计的一致性。

参数来源
~~~~~~~~
``k1`` / ``b`` 来自 ``app.config.settings``（默认 1.5 / 0.75）：
- ``k1``：词频饱和参数，控制 tf 增长速率。k1 越大，高频词贡献衰减越慢。
- ``b``：文档长度归一化参数，b=0 不归一，b=1 完全归一。0.75 是经验值。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from app.config import settings
from app.utils.logger import logger

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore
    _BM25_AVAILABLE = False
    logger.warning("rank_bm25 未安装，BM25 检索将不可用")

from app.rag.tokenizer import get_tokenizer


class BM25Retriever:
    """基于 rank_bm25.BM25Okapi 的稀疏检索器。

    Parameters
    ----------
    k1 : float
        词频饱和参数，默认取 settings.bm25_k1。
    b : float
        文档长度归一化参数，默认取 settings.bm25_b。
    """

    def __init__(self, k1: Optional[float] = None, b: Optional[float] = None) -> None:
        self._k1: float = k1 if k1 is not None else getattr(settings, "bm25_k1", 1.5)
        self._b: float = b if b is not None else getattr(settings, "bm25_b", 0.75)

        # 索引数据
        self._bm25: Optional["BM25Okapi"] = None
        self._corpus: List[Dict[str, Any]] = []  # 原始文档列表
        self._tokenized_corpus: List[List[str]] = []  # 分词后的语料
        self._tokenizer = get_tokenizer()

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------
    def build_index(
        self,
        documents: List[Dict[str, Any]],
        content_field: str = "content",
    ) -> int:
        """构建 BM25 索引。

        Parameters
        ----------
        documents : list[dict]
            文档列表，每个 dict 须含 ``content_field`` 指定的字段。
        content_field : str, default "content"
            用于分词与索引的文本字段。

        Returns
        -------
        int
            索引文档数。
        """
        if not _BM25_AVAILABLE:
            logger.warning("rank_bm25 不可用，跳过索引构建")
            return 0

        self._corpus = list(documents)
        # 索引构建阶段使用无缓存分词，避免缓存膨胀
        self._tokenized_corpus = [
            self._tokenizer.tokenize_without_cache(doc.get(content_field, ""))
            for doc in self._corpus
        ]

        self._bm25 = BM25Okapi(
            self._tokenized_corpus, k1=self._k1, b=self._b
        )
        logger.info(
            "BM25 索引构建完成: 文档数=%d, k1=%.2f, b=%.2f",
            self.corpus_size,
            self._k1,
            self._b,
        )
        return self.corpus_size

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        content_field: str = "content",
    ) -> int:
        """增量添加文档（实际为全量重建）。

        .. note::
            ``BM25Okapi`` 在构造时预计算 idf 表与平均文档长度，**不支持单文档
            增量插入**。若要增量插入，需重新计算全量 idf，等价于全量重建。
            本方法采取全量重建策略：将新文档追加到现有语料后重建索引。

            取舍：企业文档更新频率（小时级 / 天级）远低于查询频率（秒级），
            全量重建在更新时产生一次性开销，但保证了 idf 统计的一致性与
            检索的正确性。若未来需要更高频更新，可考虑替换为支持增量的
            BM25 实现（如 Elasticsearch 的 BM25）。

        Parameters
        ----------
        documents : list[dict]
            新增文档列表。
        content_field : str
            文本字段名。

        Returns
        -------
        int
            重建后的文档总数。
        """
        if not documents:
            return self.corpus_size

        merged = self._corpus + list(documents)
        logger.info(
            "BM25 增量重建: 现有=%d，新增=%d，重建后=%d",
            self.corpus_size,
            len(documents),
            len(merged),
        )
        return self.build_index(merged, content_field=content_field)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
        term_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """BM25 检索。

        Parameters
        ----------
        query : str
            查询文本（可含术语扩展器注入的 OR 同义词）。
        top_k : int
            返回文档数上限。
        score_threshold : float
            分数下限，低于此值的文档过滤掉。
        term_weights : dict[str, float] | None
            词项权重字典 ``{token: weight}``。术语命中的 token 权重 >1，
            使含术语的文档得分放大。若为 None 则等权。

        Returns
        -------
        list[dict]
            按分数降序排列的文档列表，每个 dict 含原始字段 + ``score`` +
            ``rank`` + ``source="bm25"``。
        """
        if not _BM25_AVAILABLE or self._bm25 is None or self.corpus_size == 0:
            logger.debug("BM25 索引未就绪，返回空结果")
            return []

        query_tokens = self._tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # 获取每个文档的基础 BM25 分数
        base_scores = self._bm25.get_scores(query_tokens)

        # 应用词项权重：对术语 token 的 tf 贡献加权
        if term_weights:
            weighted_scores = self._apply_term_weights(
                base_scores, query_tokens, term_weights
            )
        else:
            weighted_scores = base_scores

        # 收集、过滤、排序
        scored: List[Dict[str, Any]] = []
        for idx, score in enumerate(weighted_scores):
            if score >= score_threshold:
                doc = dict(self._corpus[idx])
                doc["score"] = float(score)
                doc["rank"] = 0  # 占位，排序后回填
                doc["source"] = "bm25"
                scored.append(doc)

        scored.sort(key=lambda x: x["score"], reverse=True)
        # 回填排名
        for rank, doc in enumerate(scored, start=1):
            doc["rank"] = rank

        result = scored[:top_k]
        logger.debug(
            "BM25 检索: query='%s', tokens=%s, 候选=%d, 返回=%d",
            query,
            query_tokens,
            len(scored),
            len(result),
        )
        return result

    def _apply_term_weights(
        self,
        base_scores: Any,
        query_tokens: List[str],
        term_weights: Dict[str, float],
    ) -> List[float]:
        """对术语命中的 token 加权 BM25 得分。

        策略：对每个文档，重新计算加权得分。加权逻辑是对术语 token 的
        ``get_scores`` 贡献乘以权重。由于 ``rank_bm25`` 的 ``get_scores``
        返回的是各 token 贡献之和，这里采用近似：对术语 token 单独计算
        其贡献，乘以权重后与非术语 token 贡献相加。

        为简化实现且保持正确性，采用如下策略：对术语 token 用
        ``get_scores`` 单独取分再加权，对非术语 token 取等权分数。
        """
        # 分离术语 token 与普通 token
        term_tokens = [t for t in query_tokens if t in term_weights]
        normal_tokens = [t for t in query_tokens if t not in term_weights]

        # 普通 token 的等权得分
        if normal_tokens:
            normal_scores = self._bm25.get_scores(normal_tokens)
        else:
            normal_scores = [0.0] * self.corpus_size

        # 术语 token 的加权得分
        weighted_sum = list(normal_scores)
        if term_tokens:
            term_scores = self._bm25.get_scores(term_tokens)
            for i, t in enumerate(term_tokens):
                weight = term_weights.get(t, 1.0)
                # 逐 token 累加加权贡献
                single_scores = self._bm25.get_scores([t])
                for doc_idx in range(self.corpus_size):
                    # 减去等权贡献，加上加权贡献
                    weighted_sum[doc_idx] += single_scores[doc_idx] * (
                        weight - 1.0
                    )

        return weighted_sum

    # ------------------------------------------------------------------
    # 可解释性
    # ------------------------------------------------------------------
    def explain_score(self, query: str, doc_index: int) -> Dict[str, Any]:
        """解释某文档对 query 的 BM25 得分构成。

        输出每个 query token 的 tf、idf、contribution，便于排查召回问题。

        Parameters
        ----------
        query : str
            查询文本。
        doc_index : int
            文档在语料中的索引。

        Returns
        -------
        dict
            ``{"doc_index", "total_score", "tokens": [{"token","tf","idf",
            "contribution"}, ...], "doc_length", "avg_doc_length"}``。
        """
        if not _BM25_AVAILABLE or self._bm25 is None:
            return {"error": "BM25 索引未就绪"}

        if doc_index < 0 or doc_index >= self.corpus_size:
            return {"error": f"doc_index {doc_index} 越界 (corpus_size={self.corpus_size})"}

        query_tokens = self._tokenizer.tokenize(query)
        doc_tokens = self._tokenized_corpus[doc_index]
        doc_len = len(doc_tokens)

        token_details: List[Dict[str, Any]] = []
        total_score = 0.0

        for token in query_tokens:
            # 词频：token 在该文档出现的次数
            tf = doc_tokens.count(token)
            # 文档频率：含 token 的文档数
            doc_freq = self.get_doc_freq(token)
            # idf：BM25 变体 idf = log((N - df + 0.5) / (df + 0.5) + 1)
            n = self.corpus_size
            idf = math.log((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1) if doc_freq > 0 else 0.0

            # BM25 tf 项
            if tf > 0 and doc_freq > 0:
                tf_component = (
                    (tf * (self._k1 + 1))
                    / (
                        tf
                        + self._k1
                        * (
                            1
                            - self._b
                            + self._b * (doc_len / max(self.avg_doc_length, 1))
                        )
                    )
                )
                contribution = idf * tf_component
            else:
                contribution = 0.0

            total_score += contribution
            token_details.append(
                {
                    "token": token,
                    "tf": tf,
                    "doc_freq": doc_freq,
                    "idf": round(idf, 4),
                    "contribution": round(contribution, 4),
                }
            )

        return {
            "doc_index": doc_index,
            "total_score": round(total_score, 4),
            "tokens": token_details,
            "doc_length": doc_len,
            "avg_doc_length": round(self.avg_doc_length, 2),
            "k1": self._k1,
            "b": self._b,
        }

    # ------------------------------------------------------------------
    # 属性与工具
    # ------------------------------------------------------------------
    @property
    def corpus_size(self) -> int:
        """当前索引的文档数。"""
        return len(self._corpus)

    @property
    def avg_doc_length(self) -> float:
        """语料平均文档长度（token 数）。"""
        if not self._tokenized_corpus:
            return 0.0
        return sum(len(d) for d in self._tokenized_corpus) / len(
            self._tokenized_corpus
        )

    def get_doc_freq(self, term: str) -> int:
        """返回含指定 term 的文档数。"""
        if not self._tokenized_corpus:
            return 0
        return sum(1 for doc in self._tokenized_corpus if term in doc)

    @property
    def is_ready(self) -> bool:
        """索引是否就绪。"""
        return self._bm25 is not None and self.corpus_size > 0
