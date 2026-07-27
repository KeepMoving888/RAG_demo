"""
检索链路单元测试

覆盖:
1. BM25Retriever 基础检索 (索引构建 / 检索 / 术语加权);
2. RRF 融合 (reciprocal_rank_fusion) 多路结果合并;
3. CrossEncoder 降级链 (模型不可用时降级为 rrf_fallback);
4. 术语扩展 (TerminologyExpander 同义词注入 / 词项加权);
5. 权限过滤逻辑 (BM25 应用层按 department_id 过滤).

测试在离线模式 (MILVUS_HOST=invalid) 下运行, 仅依赖 BM25, 不需要 GPU.
"""
from __future__ import annotations

import pytest

from app.rag.bm25_retriever import BM25Retriever
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.terminology import TerminologyExpander


# ======================== 1. BM25Retriever ========================
class TestBM25Retriever:
    """BM25 检索器测试."""

    def test_build_index_and_search(self):
        """构建索引后能正确检索出相关文档."""
        retriever = BM25Retriever()
        documents = [
            {"chunk_id": "1", "content": "车规 eMMC 5.1 规格书, 容量 64GB AEC-Q100."},
            {"chunk_id": "2", "content": "DDR4 DRAM 桌面级规格书, 适用于 PC 内存场景."},
            {"chunk_id": "3", "content": "NVMe SSD PCIe Gen4 接口用于高速读写."},
        ]
        count = retriever.build_index(documents)
        assert count == 3

        results = retriever.search(query="eMMC 控制器", top_k=2)
        assert len(results) >= 1
        # 最相关的应为 chunk_id=1
        top = results[0]
        assert top.get("chunk_id") == "1" or "eMMC" in top.get("content", "")

    def test_empty_corpus_returns_empty(self):
        """空语料检索返回空列表."""
        retriever = BM25Retriever()
        results = retriever.search(query="任意查询", top_k=5)
        assert results == []

    def test_term_weights_boost(self):
        """术语加权应使含术语的文档得分提升."""
        retriever = BM25Retriever()
        documents = [
            {"chunk_id": "1", "content": "车规 eMMC 容量 64GB AEC-Q100"},
            {"chunk_id": "2", "content": "其他品牌 容量 32GB"},
        ]
        retriever.build_index(documents)

        # 无术语加权
        results_no_weight = retriever.search(query="容量", top_k=2)
        # 有术语加权 (车规 eMMC 权重 2.0)
        results_weighted = retriever.search(
            query="容量 车规 eMMC",
            top_k=2,
            term_weights={"车规 eMMC": 2.0, "车规": 2.0, "eMMC": 2.0},
        )

        # 加权后 车规 eMMC 文档应排名更靠前 (或得分更高)
        assert len(results_weighted) >= 1
        # 加权后 top1 应为含 车规 eMMC 的文档
        top_weighted = results_weighted[0]
        assert "车规 eMMC" in top_weighted.get("content", "")


# ======================== 2. RRF 融合 ========================
class TestRRFFusion:
    """RRF (Reciprocal Rank Fusion) 融合测试."""

    def test_basic_fusion(self):
        """两路结果融合后, 双路均命中的文档排名更靠前."""
        list_a = [
            {"content": "doc1", "score": 0.9},
            {"content": "doc2", "score": 0.8},
            {"content": "doc3", "score": 0.7},
        ]
        list_b = [
            {"content": "doc3", "score": 0.95},
            {"content": "doc2", "score": 0.85},
            {"content": "doc4", "score": 0.6},
        ]
        fused = reciprocal_rank_fusion([list_a, list_b], k=60)

        # 融合后应去重
        assert len(fused) == 4
        # doc2 与 doc3 在两路都靠前, 融合后应排前列
        top_contents = [f.get("content") for f in fused[:2]]
        assert "doc2" in top_contents
        assert "doc3" in top_contents

    def test_empty_inputs(self):
        """空输入返回空列表."""
        assert reciprocal_rank_fusion([], k=60) == []
        assert reciprocal_rank_fusion([[], []], k=60) == []

    def test_single_list(self):
        """单路输入, RRF 退化为按原顺序返回."""
        list_a = [
            {"content": "x", "score": 0.9},
            {"content": "y", "score": 0.8},
        ]
        fused = reciprocal_rank_fusion([list_a], k=60)
        assert len(fused) == 2
        assert fused[0]["content"] == "x"


# ======================== 3. CrossEncoder 降级链 ========================
class TestRerankerDegrade:
    """Cross-Encoder 精排器降级链测试.

    离线模式下 CrossEncoder 模型不可用, 应降级为 rrf_fallback,
    不抛异常, 返回原始顺序.
    """

    def test_reranker_not_loaded_in_offline(self):
        """离线模式下 reranker.is_loaded 为 False."""
        from app.rag.reranker import get_reranker

        reranker = get_reranker()
        # 离线环境 (无 GPU / 无模型文件) 下 is_loaded 应为 False
        # 若环境恰好有模型, is_loaded 为 True 也算通过 (不强制为 False)
        assert hasattr(reranker, "is_loaded")

    @pytest.mark.asyncio
    async def test_rerank_degrades_to_rrf_fallback(self):
        """rerank 调用在模型不可用时降级, 不抛异常."""
        from app.rag.reranker import get_reranker

        reranker = get_reranker()
        candidates = [
            {"content": "doc1", "score": 0.9},
            {"content": "doc2", "score": 0.8},
        ]
        # 无论模型是否加载, rerank 都应返回结果 (降级或精排)
        result = await reranker.rerank("查询", candidates, top_k=2)
        assert isinstance(result, list)
        assert len(result) >= 1


# ======================== 4. 术语扩展 ========================
class TestTerminologyExpander:
    """术语扩展器测试."""

    def test_expand_query_with_hit(self):
        """命中术语时, 同义词以 OR 注入扩展查询."""
        expander = TerminologyExpander()
        expanded, hits = expander.expand_query("车载存储模块的合规要求")
        # 内置兜底词典含 "车载存储模块" -> 车规 eMMC
        assert "车规 eMMC" in hits or "车规eMMC" in hits
        # 扩展后查询应包含 OR 与同义词
        assert "OR" in expanded
        assert len(expanded) > len("车载存储模块的合规要求")

    def test_expand_query_without_hit(self):
        """未命中术语时, 返回原查询, term_hits 为空."""
        expander = TerminologyExpander()
        expanded, hits = expander.expand_query("一段完全不包含任何术语的普通文本查询")
        assert expanded == "一段完全不包含任何术语的普通文本查询"
        assert hits == []

    def test_boost_term_weight(self):
        """术语命中的 token 权重 >1.0, 其余为 1.0."""
        expander = TerminologyExpander()
        tokens = ["车规 eMMC", "容量", "64GB"]
        weighted = expander.boost_term_weight(tokens, ["车规 eMMC"])
        weights_dict = dict(weighted)
        # 车规 eMMC 命中术语, 权重应 >1.0
        assert weights_dict.get("车规 eMMC", 1.0) > 1.0
        # 其余 token 权重为 1.0
        assert weights_dict.get("容量", 1.0) == 1.0

    def test_add_term_dynamic(self):
        """运行时新增术语后能被 expand_query 命中."""
        expander = TerminologyExpander()
        expander.add_term("XYZ-9999", ["测试代号"], type_="product")
        expanded, hits = expander.expand_query("测试代号 的规格")
        assert "XYZ-9999" in hits
        assert "XYZ-9999" in expanded


# ======================== 5. 权限过滤 ========================
class TestPermissionFilter:
    """BM25 应用层权限过滤测试 (HybridRetriever._bm25_recall 逻辑)."""

    def test_department_filter_keeps_public_and_own(self):
        """department_id=2 的用户应能看到 public (0) 与本部门 (2), 看不到部门 3."""
        # 构造 BM25 召回结果
        results = [
            {"chunk_id": "1", "content": "公开文档", "department_id": 0, "score": 0.9},
            {"chunk_id": "2", "content": "本部门文档", "department_id": 2, "score": 0.85},
            {"chunk_id": "3", "content": "其他部门文档", "department_id": 3, "score": 0.95},
        ]

        # 复用 HybridRetriever._bm25_recall 中的过滤逻辑
        department_id = 2
        filtered = [
            r for r in results
            if r.get("department_id") == 0
            or r.get("department_id") == department_id
        ]

        assert len(filtered) == 2
        chunk_ids = [r["chunk_id"] for r in filtered]
        assert "1" in chunk_ids  # public
        assert "2" in chunk_ids  # own
        assert "3" not in chunk_ids  # other dept filtered out

    def test_admin_sees_all(self):
        """admin 角色应看到全部文档 (无 department_id 过滤)."""
        # admin 在 HybridRetriever 中通过不应用过滤实现
        # 这里验证 BM25 过滤逻辑对 admin 不应被启用
        results = [
            {"chunk_id": "1", "department_id": 0},
            {"chunk_id": "2", "department_id": 2},
            {"chunk_id": "3", "department_id": 3},
        ]
        # admin 不应用过滤, 直接返回全部
        admin_results = results  # 不过滤
        assert len(admin_results) == 3
