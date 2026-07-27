"""
查询改写器 - 解决多轮对话中的代词指代与意图偏移

设计要点
========

1. 为何需要查询改写?
   - 多轮对话中用户常使用代词 (它/这个/那个), 或省略主语. 若直接拿原 query
     去检索, 召回会因缺少实体而偏移, 导致 RAG 答非所问.
   - 改写器融合历史上下文, 把「它怎么部署?」还原为「车规 eMMC 怎么部署?」,
     使检索查询具备独立完整性.

2. 三段式策略 (兼顾质量与可用性):
   - 在线模式: 调用 LLM 改写 (质量优先);
   - 离线模式: 基于规则改写 (识别代词, 用上一轮主语替换, 零依赖);
   - 失败兜底: 返回原 query, 不阻断主流程.

3. 改写触发判定 (is_rewrite_needed):
   - 含代词或过短 (<5 字) 才改写, 避免对已是完整查询的请求多花一次 LLM 调用.
"""

import re

from app.config import settings
from app.utils.logger import logger


# 代词词典 (触发改写的信号词)
_PRONOUNS = ("它", "他", "她", "这个", "那个", "这", "那", "其", "该", "上述", "前面提到")

# 主语候选正则 (用于离线规则改写时从历史中抽取实体)
_PRODUCT_RE = re.compile(r"\b([A-Z]{2,}-?[A-Z0-9]{2,})\b")
_DEPT_RE = re.compile(r"([\u4e00-\u9fa5]{2,8}(?:部|处|中心|科|组))")
_PERSON_RE = re.compile(r"([\u4e00-\u9fa5]{2,4})(?:工程师|经理|老师|主任|总监)")
_POLICY_RE = re.compile(r"《([^》]{2,30})》")


class QueryRewriter:
    """
    查询改写器: 在线走 LLM, 离线走规则, 失败兜底原 query.

    改写 Prompt 模板 (写入 LLM):

        你是查询改写助手。基于历史对话, 将用户最新问题改写为独立完整的检索查询。
        规则:
        1. 解析代词指代 (他/它/这个/那个)
        2. 融合必要的历史实体 (产品名/部门名/人名)
        3. 保留用户最新问题的核心意图
        4. 改写后查询不超过 100 字

        历史对话:
        {context}

        用户最新问题: {query}

        改写后查询 (只输出查询本身, 不要解释):
    """

    REWRITE_SYSTEM_PROMPT = (
        "你是查询改写助手。基于历史对话, 将用户最新问题改写为独立完整的检索查询。\n"
        "规则:\n"
        "1. 解析代词指代 (他/它/这个/那个)\n"
        "2. 融合必要的历史实体 (产品名/部门名/人名)\n"
        "3. 保留用户最新问题的核心意图\n"
        "4. 改写后查询不超过 100 字\n\n"
        "只输出改写后的查询本身, 不要解释、不要加引号。"
    )

    def __init__(self) -> None:
        self._offline = settings.is_offline_mode

    async def is_rewrite_needed(self, query: str) -> bool:
        """
        判断是否需要改写:
        - 含代词 -> 需要 (指代待消解);
        - 长度 < 5 字 -> 需要 (信息过少, 可能省略主语);
        - 否则不需要 (已是完整查询, 跳过节省 LLM 调用).
        """
        if not query:
            return False
        if len(query.strip()) < 5:
            return True
        return any(p in query for p in _PRONOUNS)

    async def rewrite(self, query: str, context: list[dict]) -> str:
        """
        改写查询.

        Args:
            query: 用户最新问题.
            context: 历史对话消息列表 (来自 DialogContextManager.get_context).

        Returns:
            改写后的独立完整查询; 无需改写或失败时返回原 query.
        """
        # 无历史上下文: 无需改写
        if not context:
            return query

        # 判定是否需要改写
        if not await self.is_rewrite_needed(query):
            return query

        # 离线模式: 规则改写
        if self._offline:
            rewritten = self._rule_rewrite(query, context)
            if rewritten and rewritten != query:
                logger.debug("规则改写: {!r} -> {!r}", query, rewritten)
                return rewritten
            return query

        # 在线模式: LLM 改写
        try:
            rewritten = await self._llm_rewrite(query, context)
            if rewritten:
                rewritten = rewritten.strip().strip('"').strip("'").strip("。")
                if rewritten and rewritten != query:
                    logger.debug("LLM 改写: {!r} -> {!r}", query, rewritten)
                    return rewritten
        except Exception as e:
            logger.warning("LLM 查询改写失败, 回退规则改写: {}", str(e))
            rewritten = self._rule_rewrite(query, context)
            if rewritten and rewritten != query:
                return rewritten

        return query

    # ======================== LLM 改写 ========================
    async def _llm_rewrite(self, query: str, context: list[dict]) -> str:
        """调用 LLM 完成改写."""
        from app.llm import get_llm

        context_text = self._format_context(context)
        user_prompt = (
            f"历史对话:\n{context_text}\n\n"
            f"用户最新问题: {query}\n\n"
            f"改写后查询 (只输出查询本身, 不要解释):"
        )
        messages = [
            {"role": "system", "content": self.REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        llm = get_llm()
        resp = await llm.agenerate(messages, temperature=0.0, max_tokens=128)
        return resp.text

    # ======================== 规则改写 (离线 / 兜底) ========================
    def _rule_rewrite(self, query: str, context: list[dict]) -> str:
        """
        基于规则的离线改写:
        1. 从历史 user 消息中抽取最近一个主语实体 (产品/部门/人名/制度名);
        2. 若 query 含代词, 用主语替换代词或前缀拼接;
        3. 找不到主语则返回原 query.
        """
        subject = self._extract_last_subject(context)
        if not subject:
            return query

        # 简单代词直接替换
        replaced = query
        for pron in ("它", "他", "她", "这个", "那个", "该", "其"):
            if pron in replaced:
                replaced = replaced.replace(pron, subject)
        if replaced != query:
            return replaced[:100]

        # 无显式代词但过短: 前缀拼接主语
        if len(query.strip()) < 5:
            return f"关于{subject}: {query}"[:100]

        return query

    def _extract_last_subject(self, context: list[dict]) -> str:
        """从历史 user 消息中抽取最近出现的主语实体."""
        # 倒序遍历 user 消息
        for msg in reversed(context):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not content:
                continue
            # 优先级: 制度名 > 产品名 > 部门名 > 人名
            for pattern in (_POLICY_RE, _PRODUCT_RE, _DEPT_RE, _PERSON_RE):
                m = pattern.search(content)
                if m:
                    return m.group(1)
        return ""

    @staticmethod
    def _format_context(context: list[dict]) -> str:
        """将上下文消息格式化为 prompt 可读文本."""
        if not context:
            return "(无)"
        lines: list[str] = []
        for t in context[-6:]:  # 最多取最近 6 轮, 控制 prompt 长度
            role = t.get("role", "user")
            content = t.get("content", "")
            label = "用户" if role == "user" else "助手"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)
