"""app.rag —— 企业级 RAG 混合检索与权限隔离模块

本模块是 Enterprise RAG Knowledge Base 的检索核心，提供以下能力：

1. **混合检索（Hybrid Retrieval）**：融合稠密向量（Milvus + BGE-M3）与稀疏词项
   （BM25）两路召回，通过 RRF 融合后交由 Cross-Encoder 精排，兼顾语义召回与
   字面精确匹配。
2. **权限隔离（Permission Isolation）**：在 Milvus 侧采用 partition 级别隔离，
   每个部门一个独立 partition，外加全局 ``_public`` 公开 partition；检索时只搜
   本部门 + 公开两个分区，从向量库层面实现行级权限控制，避免在应用层过滤的
   性能与安全风险。
3. **术语扩展（Terminology Expansion）**：维护企业内部术语词典（产品代号、
   部门简称、认证标准等），对查询进行同义词扩展与词项加权，显著提升术语类
   问题的召回率。
4. **可观测与可解释**：每个检索阶段记录延迟指标，BM25 提供 tf/idf 分项解释，
   RRF 与 rerank 方法均上报，便于离线评估与在线排查。
5. **生产级降级链**：Milvus / CrossEncoder / Redis 任一依赖不可用时自动降级，
   保证核心检索链路不中断。

设计要点详见各子模块 docstring。
"""

from app.rag.tokenizer import BilingualTokenizer, get_tokenizer
from app.rag.terminology import TerminologyExpander, get_terminology
from app.rag.fusion import reciprocal_rank_fusion, weighted_fusion
from app.rag.bm25_retriever import BM25Retriever
from app.rag.embedder import BGEM3Embedder, get_embedder
from app.rag.cache import RetrievalCache, get_retrieval_cache
from app.rag.milvus_store import MilvusStore, milvus_store
from app.rag.reranker import CrossEncoderReranker, get_reranker
from app.rag.retriever import HybridRetriever
from app.rag.evaluator import RetrievalEvaluator

__all__ = [
    "BilingualTokenizer",
    "get_tokenizer",
    "TerminologyExpander",
    "get_terminology",
    "reciprocal_rank_fusion",
    "weighted_fusion",
    "BM25Retriever",
    "BGEM3Embedder",
    "get_embedder",
    "RetrievalCache",
    "get_retrieval_cache",
    "MilvusStore",
    "milvus_store",
    "CrossEncoderReranker",
    "get_reranker",
    "HybridRetriever",
    "RetrievalEvaluator",
]
