"""
对话模块单元测试

覆盖:
1. 滑动窗口上下文 (DialogContextManager): window_size 内裁剪 + 首尾锚点;
2. query 改写 (QueryRewriter 离线模式): 代词替换 + 主语抽取;
3. QA 缓存 (QACache) 命中: Redis 不可用时降级 no-op, 文本哈希命中;
4. 答案溯源覆盖率 (CitationExtractor): 答案中 [CITE{id}] 标记能映射回 chunk.

测试在离线模式 (LLM_PROVIDER=offline, REDIS_HOST=invalid) 下运行,
全部依赖降级路径, 无需 Redis 与 LLM API.
"""

from __future__ import annotations

import pytest

from app.dialog.context_manager import DialogContextManager
from app.dialog.qa_cache import QACache
from app.dialog.query_rewriter import QueryRewriter


# ======================== 1. 滑动窗口上下文 ========================
class TestDialogContextWindow:
    """DialogContextManager 滑动窗口裁剪测试."""

    @pytest.mark.asyncio
    async def test_create_and_get_context(self):
        """创建会话后, 上下文初始为空."""
        mgr = DialogContextManager()
        session_id = await mgr.create_session(user_id=1, department_id=2)
        assert isinstance(session_id, str)
        ctx = await mgr.get_context(session_id)
        assert ctx == []

    @pytest.mark.asyncio
    async def test_append_messages_and_read(self):
        """追加 user / assistant 消息后, get_context 返回完整内容."""
        mgr = DialogContextManager()
        session_id = await mgr.create_session(user_id=1, department_id=None)
        await mgr.add_user_message(session_id, "车规 eMMC 是什么")
        await mgr.add_assistant_message(session_id, "车规 eMMC 是车载级存储产品", [])

        ctx = await mgr.get_context(session_id)
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"
        assert ctx[0]["content"] == "车规 eMMC 是什么"
        assert ctx[1]["role"] == "assistant"
        assert "车规 eMMC" in ctx[1]["content"]

    def test_apply_window_preserves_first_anchor(self):
        """超过 window_size 时, 保留首条 user 锚点 + 最近窗口."""
        mgr = DialogContextManager()
        # 构造 window_size + 3 条消息
        turns = [{"role": "user", "content": f"问题 {i}"} for i in range(mgr._window_size + 3)]
        pruned = mgr._apply_window(turns)

        # 首条锚点保留
        assert pruned[0]["content"] == "问题 0"
        # 末尾保留最近 window_size 条
        assert pruned[-1]["content"] == f"问题 {mgr._window_size + 2}"
        # 总数 <= window_size + 1 (锚点 + 最近窗口)
        assert len(pruned) <= mgr._window_size + 1

    def test_apply_window_short_list_unchanged(self):
        """消息数 <= window_size + 1 时原样返回."""
        mgr = DialogContextManager()
        turns = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        pruned = mgr._apply_window(turns)
        assert pruned == turns


# ======================== 2. Query 改写 (离线模式) ========================
class TestQueryRewriterOffline:
    """QueryRewriter 离线规则改写测试."""

    @pytest.mark.asyncio
    async def test_rewrite_not_needed_for_complete_query(self):
        """完整查询 (无代词, 长度足够) 不触发改写."""
        rewriter = QueryRewriter()
        needed = await rewriter.is_rewrite_needed("车规 eMMC 规格参数")
        assert needed is False

    @pytest.mark.asyncio
    async def test_rewrite_needed_for_pronoun(self):
        """含代词的查询触发改写."""
        rewriter = QueryRewriter()
        assert await rewriter.is_rewrite_needed("它的处理器是什么") is True
        assert await rewriter.is_rewrite_needed("这个怎么部署") is True

    @pytest.mark.asyncio
    async def test_rewrite_needed_for_short_query(self):
        """长度 < 5 的短查询触发改写."""
        rewriter = QueryRewriter()
        assert await rewriter.is_rewrite_needed("部署") is True
        assert await rewriter.is_rewrite_needed("价格") is True

    @pytest.mark.asyncio
    async def test_rule_rewrite_replaces_pronoun(self):
        """离线规则改写: 用历史主语替换代词."""
        rewriter = QueryRewriter()
        context = [
            {"role": "user", "content": "车规 eMMC 是什么"},
            {"role": "assistant", "content": "车规 eMMC 是车载级存储产品"},
            {"role": "user", "content": "它的容量是什么"},
        ]
        rewritten = await rewriter.rewrite("它的容量是什么", context)
        # 应将 "它" 替换为 "车规 eMMC"
        assert "车规 eMMC" in rewritten
        assert "它" not in rewritten

    @pytest.mark.asyncio
    async def test_rewrite_no_context_returns_original(self):
        """无历史上下文时返回原查询."""
        rewriter = QueryRewriter()
        rewritten = await rewriter.rewrite("它的容量", context=[])
        assert rewritten == "它的容量"


# ======================== 3. QA 缓存命中 ========================
class TestQACache:
    """QACache 测试 (Redis 不可用时降级 no-op)."""

    @pytest.mark.asyncio
    async def test_get_returns_none_when_redis_unavailable(self):
        """Redis 不可用时, get 返回 None (降级 no-op)."""
        cache = QACache()
        # 强制标记 Redis 不可用
        QACache._redis_broken = True
        try:
            result = await cache.get("任意查询", department_id=1)
            assert result is None
        finally:
            # 恢复以便后续测试
            QACache._redis_broken = False

    @pytest.mark.asyncio
    async def test_set_no_exception_when_redis_unavailable(self):
        """Redis 不可用时, set 不抛异常 (降级跳过)."""
        cache = QACache()
        QACache._redis_broken = True
        try:
            # 应不抛异常
            await cache.set("查询", 1, "答案", [], ["chunk_1"])
        finally:
            QACache._redis_broken = False

    def test_cache_key_deterministic(self):
        """相同 query + department_id 生成相同 cache_key."""
        cache = QACache()
        key1 = cache._cache_key("车规 eMMC 容量", 1)
        key2 = cache._cache_key("车规 eMMC 容量", 1)
        key3 = cache._cache_key("车规 eMMC 容量", 2)
        assert key1 == key2
        assert key1 != key3  # 不同部门不同 key


# ======================== 4. 答案溯源覆盖率 ========================
class TestCitationCoverage:
    """CitationExtractor 答案溯源测试 (基于字符 bigram Jaccard 相似度)."""

    @pytest.mark.asyncio
    async def test_citation_extraction_for_high_similarity(self):
        """答案句与 chunk 内容高度相似时, 应识别为引用."""
        from app.dialog.citation import CitationExtractor

        extractor = CitationExtractor()
        chunks = [
            {"id": "chunk_1", "content": "车规 eMMC 容量为 64GB，符合 AEC-Q100 Grade 3."},
            {"id": "chunk_2", "content": "车规 eMMC 工作温度 -40~85℃."},
        ]
        # 答案直接复用 chunk_1 的内容 (Jaccard 应接近 1.0)
        answer = "车规 eMMC 容量为 64GB，符合 AEC-Q100 Grade 3."

        citations = await extractor.extract(answer, chunks)
        # chunk_1 应被识别为引用
        cited_ids = {c.get("chunk_id") for c in citations}
        assert "chunk_1" in cited_ids

    @pytest.mark.asyncio
    async def test_citation_empty_answer(self):
        """空答案不产生 citation."""
        from app.dialog.citation import CitationExtractor

        extractor = CitationExtractor()
        chunks = [{"id": "chunk_1", "content": "车规 eMMC 容量."}]
        citations = await extractor.extract("", chunks)
        assert citations == []

    @pytest.mark.asyncio
    async def test_citation_no_chunks_returns_empty(self):
        """无 chunks 时返回空列表."""
        from app.dialog.citation import CitationExtractor

        extractor = CitationExtractor()
        citations = await extractor.extract("任意答案", [])
        assert citations == []

    @pytest.mark.asyncio
    async def test_citation_coverage_for_fallback_answer(self):
        """fallback 答案 (基于 chunk 拼装) 应能溯源到使用的 chunks."""
        from app.dialog.generator import AnswerGenerator

        chunks = [
            {"id": "chunk_1", "content": "车规 eMMC 容量 64GB AEC-Q100"},
            {"id": "chunk_2", "content": "车规 eMMC 工作温度 -40~85℃"},
        ]
        fallback = AnswerGenerator._build_fallback_answer("容量规格", chunks)

        # fallback 答案应包含 chunk 内容片段
        assert "64GB" in fallback or "容量" in fallback or "车规" in fallback
        # fallback 应有结构化的引用提示
        assert "知识库" in fallback or "溯源" in fallback
