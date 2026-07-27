"""
答案生成编排器 - 组合多轮对话 / 改写 / 缓存 / 检索 / 溯源

设计要点
========

1. 编排职责: AnswerGenerator 是多轮对话模块的「门面」, 串联:
   DialogContextManager (上下文) -> QueryRewriter (改写) -> QACache (缓存)
   -> HybridRetriever (检索) -> BaseLLM (生成) -> CitationExtractor (溯源),
   并完成持久化与指标上报.

2. 缓存优先: 改写后先查缓存, 命中则跳过检索与 LLM, 直接返回 (answer_source=cache),
   将 P99 延迟从秒级降至毫秒级. 仅在未命中时走完整 RAG 链路.

3. 失败兜底: LLM 调用失败时, 基于检索 chunk 拼装 fallback 答案 (answer_source=fallback),
   保证用户始终拿到有据可依的回复, 而非裸异常.

4. 流式生成 (generate_stream): 先同步完成「改写 + 缓存判断 + 检索」, 再流式输出
   LLM token; 流末尾追加一个 JSON 元数据帧 (含 citations / metadata), 供前端在
   流结束后渲染溯源面板.

5. Prompt 约定: 检索片段以 [RETRIEVED_CONTEXT] 块注入, 每段前缀 [CITE{chunk_id}],
   供离线 LLM 解析与 CitationExtractor 对齐.
"""

import json
import time
from collections.abc import AsyncIterator

from app.config import settings
from app.dialog.citation import CitationExtractor
from app.dialog.context_manager import DialogContextManager
from app.dialog.qa_cache import QACache
from app.dialog.query_rewriter import QueryRewriter
from app.metrics import DIALOG_TURN_COUNT, QA_RESPONSE_LATENCY
from app.utils.logger import logger


class AnswerGenerator:
    """
    答案生成编排器.

    组合 DialogContextManager + QueryRewriter + QACache + CitationExtractor,
    并通过 app.rag.retriever.get_retriever() 获取 HybridRetriever 单例.
    """

    # RAG 问答系统 prompt
    _SYSTEM_PROMPT = (
        "你是企业知识库问答助手。基于以下检索到的知识片段回答用户问题。\n"
        "要求:\n"
        "1. 答案必须基于检索内容, 不得编造;\n"
        "2. 关键事实后可自然标注来源;\n"
        "3. 若检索内容不足以回答, 如实说明并建议补充相关资料。\n\n"
        "[RETRIEVED_CONTEXT]\n{context_block}\n[/RETRIEVED_CONTEXT]"
    )

    def __init__(
        self,
        context_manager: DialogContextManager | None = None,
        rewriter: QueryRewriter | None = None,
        cache: QACache | None = None,
        citation_extractor: CitationExtractor | None = None,
    ) -> None:
        self._context_manager = context_manager or DialogContextManager()
        self._rewriter = rewriter or QueryRewriter()
        self._cache = cache or QACache()
        self._citation = citation_extractor or CitationExtractor()

    # ======================== 主流程: 非流式 ========================
    async def generate(
        self,
        query: str,
        session_id: str,
        user_id: int,
        department_id: int | None,
        top_k: int = 5,
    ) -> dict:
        """
        完整问答生成流程.

        Returns:
            {
              "answer", "citations", "rewritten_query", "retrieved_chunks",
              "latency_ms", "cache_hit", "answer_source", "session_id", "turn_count"
            }
        """
        started = time.time()

        # 1. 加载历史上下文
        context = await self._context_manager.get_context(session_id)

        # 2. 查询改写
        rewritten_query = await self._rewriter.rewrite(query, context)

        # 3. 缓存查询
        cached = await self._cache.get(rewritten_query, department_id)
        if cached is not None:
            latency_ms = (time.time() - started) * 1000
            result = {
                "answer": cached["answer"],
                "citations": cached["citations"],
                "rewritten_query": rewritten_query,
                "retrieved_chunks": [{"id": cid} for cid in cached.get("retrieved_chunk_ids", [])],
                "latency_ms": round(latency_ms, 2),
                "cache_hit": True,
                "answer_source": "cache",
                "session_id": session_id,
                "turn_count": 0,
            }
            # 缓存命中也需写入会话上下文与持久化
            await self._finalize(
                session_id, query, rewritten_query, result, user_id, from_cache=True
            )
            result["turn_count"] = await self._context_manager.get_turn_count(session_id)
            self._report_metrics(latency_ms, result["turn_count"])
            logger.info(
                "QA 缓存命中: session={} latency={:.0f}ms",
                session_id,
                latency_ms,
            )
            return result

        # 4. 检索
        chunks = await self._retrieve(rewritten_query, department_id, top_k)

        # 5. 构建 prompt
        messages = self._build_messages(query, rewritten_query, context, chunks)

        # 6. LLM 生成 (含失败兜底)
        answer, answer_source = await self._generate_with_fallback(messages, query, chunks)

        # 7. 答案溯源
        citations = await self._citation.extract(answer, chunks)

        # 8. 写入缓存
        chunk_ids = [str(c.get("id", "")) for c in chunks if c.get("id")]
        await self._cache.set(rewritten_query, department_id, answer, citations, chunk_ids)

        latency_ms = (time.time() - started) * 1000
        result = {
            "answer": answer,
            "citations": citations,
            "rewritten_query": rewritten_query,
            "retrieved_chunks": chunks,
            "latency_ms": round(latency_ms, 2),
            "cache_hit": False,
            "answer_source": answer_source,
            "session_id": session_id,
            "turn_count": 0,
        }

        # 9 & 10. 写入上下文 + 持久化
        await self._finalize(session_id, query, rewritten_query, result, user_id, from_cache=False)
        result["turn_count"] = await self._context_manager.get_turn_count(session_id)

        # 11. 指标上报
        self._report_metrics(latency_ms, result["turn_count"])

        logger.info(
            "QA 生成完成: session={} source={} latency={:.0f}ms citations={}",
            session_id,
            answer_source,
            latency_ms,
            len(citations),
        )
        return result

    # ======================== 主流程: 流式 ========================
    async def generate_stream(
        self,
        query: str,
        session_id: str,
        user_id: int,
        department_id: int | None,
        top_k: int = 5,
    ) -> AsyncIterator[str]:
        """
        流式问答生成.

        产出契约:
        - 先 yield 若干文本片段 (LLM token 流);
        - 最后 yield 一个 JSON 字符串 (含 __meta__ 标记 + citations + metadata),
          供前端在流结束后解析溯源信息.
        """
        started = time.time()

        # 1. 同步阶段: 上下文 + 改写 + 缓存 + 检索
        context = await self._context_manager.get_context(session_id)
        rewritten_query = await self._rewriter.rewrite(query, context)
        cached = await self._cache.get(rewritten_query, department_id)

        if cached is not None:
            # 缓存命中: 按句切分作分片流式输出
            answer = cached["answer"]
            citations = cached["citations"]
            for chunk_text in self._split_for_stream(answer):
                yield chunk_text
            latency_ms = (time.time() - started) * 1000
            result = {
                "answer": answer,
                "citations": citations,
                "rewritten_query": rewritten_query,
                "retrieved_chunks": [{"id": cid} for cid in cached.get("retrieved_chunk_ids", [])],
                "latency_ms": round(latency_ms, 2),
                "cache_hit": True,
                "answer_source": "cache",
                "session_id": session_id,
                "turn_count": 0,
            }
            await self._finalize(
                session_id, query, rewritten_query, result, user_id, from_cache=True
            )
            result["turn_count"] = await self._context_manager.get_turn_count(session_id)
            self._report_metrics(latency_ms, result["turn_count"])
            yield self._meta_frame(result)
            return

        # 检索
        chunks = await self._retrieve(rewritten_query, department_id, top_k)
        messages = self._build_messages(query, rewritten_query, context, chunks)

        # 2. 流式生成
        answer_parts: list[str] = []
        answer_source = "llm"
        try:
            from app.llm import get_llm

            llm = get_llm()
            async for token in llm.agenerate_stream(messages, temperature=settings.llm_temperature):
                if token:
                    answer_parts.append(token)
                    yield token
        except Exception as e:
            logger.warning("流式 LLM 生成失败, 回退拼装答案: {}", str(e))
            fallback_answer = self._build_fallback_answer(query, chunks)
            answer_parts = [fallback_answer]
            answer_source = "fallback"
            yield fallback_answer

        answer = "".join(answer_parts)

        # 3. 溯源 + 缓存 + 持久化 + 元数据帧
        citations = await self._citation.extract(answer, chunks)
        chunk_ids = [str(c.get("id", "")) for c in chunks if c.get("id")]
        await self._cache.set(rewritten_query, department_id, answer, citations, chunk_ids)

        latency_ms = (time.time() - started) * 1000
        result = {
            "answer": answer,
            "citations": citations,
            "rewritten_query": rewritten_query,
            "retrieved_chunks": chunks,
            "latency_ms": round(latency_ms, 2),
            "cache_hit": False,
            "answer_source": answer_source,
            "session_id": session_id,
            "turn_count": 0,
        }
        await self._finalize(session_id, query, rewritten_query, result, user_id, from_cache=False)
        result["turn_count"] = await self._context_manager.get_turn_count(session_id)
        self._report_metrics(latency_ms, result["turn_count"])

        yield self._meta_frame(result)

    # ======================== 检索 ========================
    async def _retrieve(
        self,
        rewritten_query: str,
        department_id: int | None,
        top_k: int,
    ) -> list[dict]:
        """调用 HybridRetriever 获取召回 chunks (方法内 import 避免循环依赖)."""
        try:
            from app.rag.retriever import get_retriever

            retriever = get_retriever()
            result = await retriever.retrieve(
                rewritten_query,
                department_id,
                top_k=top_k,
                recall_k=settings.retrieval_recall_k,
                enable_rerank=True,
            )
            chunks = result.get("chunks", []) if isinstance(result, dict) else []
            logger.debug(
                "检索完成: query={!r} chunks={} latency={:.0f}ms",
                rewritten_query,
                len(chunks),
                result.get("latency_ms", 0) if isinstance(result, dict) else 0,
            )
            return chunks
        except Exception as e:
            logger.error("检索失败, 返回空结果: {}", str(e))
            return []

    # ======================== Prompt 构建 ========================
    def _build_messages(
        self,
        query: str,
        rewritten_query: str,
        context: list[dict],
        chunks: list[dict],
    ) -> list[dict]:
        """
        构建 LLM messages:
        - system: RAG 指令 + [RETRIEVED_CONTEXT] 块 (每段前缀 [CITE{chunk_id}]);
        - 历史: 上下文消息透传 (user/assistant);
        - user: 改写后的检索查询.
        """
        context_lines: list[str] = []
        for chunk in chunks:
            chunk_id = str(chunk.get("id", ""))
            content = chunk.get("content", "") or ""
            heading = chunk.get("heading_path", "") or ""
            header = f"[{heading}] " if heading else ""
            context_lines.append(f"[CITE{chunk_id}] {header}{content}")
        context_block = "\n".join(context_lines) if context_lines else "(无相关检索内容)"

        system_content = self._SYSTEM_PROMPT.format(context_block=context_block)

        messages: list[dict] = [{"role": "system", "content": system_content}]
        # 透传历史上下文 (已由滑动窗口裁剪)
        for turn in context:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        # 当前问题
        messages.append({"role": "user", "content": rewritten_query})
        return messages

    # ======================== 生成 + 兜底 ========================
    async def _generate_with_fallback(
        self,
        messages: list[dict],
        query: str,
        chunks: list[dict],
    ) -> tuple[str, str]:
        """
        LLM 生成, 失败时回退到基于 chunk 拼装的 fallback 答案.

        Returns:
            (answer, answer_source)  source ∈ {"llm", "fallback"}
        """
        try:
            from app.llm import get_llm

            llm = get_llm()
            resp = await llm.agenerate(messages, temperature=settings.llm_temperature)
            if resp.text and resp.text.strip():
                return resp.text, "llm"
            # 空回复视为失败
            logger.warning("LLM 返回空回复, 回退 fallback")
        except Exception as e:
            logger.warning("LLM 生成异常, 回退 fallback: {}", str(e))

        return self._build_fallback_answer(query, chunks), "fallback"

    @staticmethod
    def _build_fallback_answer(query: str, chunks: list[dict]) -> str:
        """基于检索 chunk 拼装 fallback 答案 (LLM 不可用时兜底)."""
        if not chunks:
            return (
                f"关于「{query}」, 当前知识库中未找到高度相关内容. "
                "建议补充相关文档或联系对应职能部门."
            )
        parts = [f"根据知识库检索, 关于「{query}」的相关内容如下:"]
        for i, chunk in enumerate(chunks[:5], 1):
            content = (chunk.get("content", "") or "").strip()
            if content:
                parts.append(f"{i}. {content[:200]}")
        parts.append("\n以上内容引用自知识库文档, 可在「答案溯源」面板查看原始出处.")
        return "\n".join(parts)

    # ======================== 收尾: 上下文 + 持久化 ========================
    async def _finalize(
        self,
        session_id: str,
        query: str,
        rewritten_query: str,
        result: dict,
        user_id: int,
        from_cache: bool,
    ) -> None:
        """写入会话上下文并持久化 QAMessage."""
        answer = result["answer"]
        citations = result["citations"]

        # 9. 写入上下文 (用户问题 + 助手回答)
        await self._context_manager.add_user_message(session_id, query)
        await self._context_manager.add_assistant_message(session_id, answer, citations)

        # 10. 持久化到 QAMessage
        await self._persist_message(
            session_id=session_id,
            user_id=user_id,
            query=query,
            rewritten_query=rewritten_query,
            result=result,
        )

    async def _persist_message(
        self,
        session_id: str,
        user_id: int,
        query: str,
        rewritten_query: str,
        result: dict,
    ) -> None:
        """持久化单轮 QAMessage (惰性确保 QASession 存在)."""
        try:
            # 获取 / 惰性创建 PostgreSQL QASession 主键
            pg_session_id = await self._context_manager._ensure_pg_session(session_id)
            if pg_session_id is None:
                logger.debug("pg_session_id 不可用, 跳过 QAMessage 持久化")
                return

            from app.database import db_session
            from app.models import QAMessage

            async with db_session() as db:
                msg = QAMessage(
                    session_id=int(pg_session_id),
                    user_id=user_id,
                    role="assistant",
                    user_query=query,
                    rewritten_query=rewritten_query,
                    answer=result["answer"],
                    answer_source=result["answer_source"],
                    retrieved_chunks=result.get("retrieved_chunks"),
                    citations=result.get("citations"),
                    latency_ms=int(result.get("latency_ms", 0) or 0),
                    cache_hit=bool(result.get("cache_hit")),
                )
                db.add(msg)
            logger.debug("QAMessage 已持久化: session={}", session_id)
        except Exception as e:
            # 持久化失败不影响在线问答
            logger.warning("QAMessage 持久化失败: {}", str(e))

    # ======================== 指标与工具 ========================
    def _report_metrics(self, latency_ms: float, turn_count: int) -> None:
        """上报响应延迟与对话轮数指标."""
        try:
            QA_RESPONSE_LATENCY.observe(latency_ms)
            DIALOG_TURN_COUNT.observe(turn_count)
        except Exception:
            pass

    @staticmethod
    def _meta_frame(result: dict) -> str:
        """构造流末尾的元数据 JSON 帧 (带 __meta__ 标记便于前端识别)."""
        meta = {"__meta__": True, **result}
        return json.dumps(meta, ensure_ascii=False)

    @staticmethod
    def _split_for_stream(text: str) -> list[str]:
        """将完整文本按句切分, 作分片流式输出 (缓存命中场景)."""
        if not text:
            return []
        import re

        parts = re.split(r"(?<=[。!?\!？；;\n])", text)
        return [p for p in parts if p]


# ======================== 单例工厂 ========================
_generator: AnswerGenerator | None = None


def get_generator() -> AnswerGenerator:
    """获取 AnswerGenerator 单例."""
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator
