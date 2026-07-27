"""
离线模式 LLM

设计目的:
- 无需任何 API Key 即可完整体验 RAG 全链路 (开源即跑)
- 基于规则 + 模板生成可读回复, 重点保证检索/解析/图谱链路可观测
- 生产环境通过 LLM_PROVIDER=openai/deepseek 切换真实模型

实现策略:
1. 问答生成: 基于检索 chunk 拼装带引用的答案 (模仿 RAG 标准格式)
2. JSON 抽取: 基于正则 + 关键词匹配抽取实体关系
3. Cypher 生成: 基于查询模式映射预定义 Cypher 模板
"""

import json
import re
import time
from collections.abc import AsyncIterator

from app.llm.base import BaseLLM, LLMResponse
from app.utils.logger import logger


class OfflineLLM(BaseLLM):
    """离线模式 LLM"""

    provider = "offline"

    def __init__(self, model: str = "offline-rule-based"):
        super().__init__(model=model, temperature=0.0, max_tokens=1024)
        logger.info("OfflineLLM 已就绪 (无需 API Key, 全链路种子数据驱动)")

    async def agenerate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        start = time.time()

        # 提取最后一条用户消息
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        # JSON 模式: 尝试从 prompt 中识别抽取任务
        if response_format and response_format.get("type") == "json_object":
            text = self._json_response(user_msg)
        else:
            text = self._chat_response(user_msg, messages)

        return LLMResponse(
            text=text,
            model=self.model,
            usage={"prompt_tokens": len(user_msg), "completion_tokens": len(text)},
            latency_ms=(time.time() - start) * 1000,
        )

    async def agenerate_stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        resp = await self.agenerate(messages, **kwargs)
        # 仿流式: 按句号切分
        for chunk in re.split(r"(?<=[。\.!\?])\s+", resp.text):
            if chunk:
                yield chunk + " "

    # ======================== 内部策略 ========================
    def _chat_response(self, query: str, messages: list[dict]) -> str:
        """
        问答生成: 基于检索上下文拼装带引用的答案

        Prompt 约定: system 消息中包含 [RETRIEVED_CONTEXT] 块
        """
        # 提取检索上下文
        context_chunks: list[str] = []
        for m in messages:
            content = m.get("content", "")
            if "[RETRIEVED_CONTEXT]" in content:
                # 解析上下文块
                ctx_match = re.search(
                    r"\[RETRIEVED_CONTEXT\](.*?)\[/RETRIEVED_CONTEXT\]",
                    content,
                    re.DOTALL,
                )
                if ctx_match:
                    for line in ctx_match.group(1).strip().split("\n"):
                        line = line.strip()
                        if line.startswith("[CITE"):
                            context_chunks.append(line)

        if not context_chunks:
            return (
                f"关于「{query}」, 当前知识库中未找到高度相关内容. "
                "建议补充相关文档或联系对应职能部门."
            )

        # 拼装带引用的答案
        parts = [f"根据知识库检索, 关于「{query}」的解答如下:\n"]
        for i, chunk in enumerate(context_chunks[:5], 1):
            # 提取引用 ID
            cite_match = re.match(r"\[CITE[^\]]*\](.*)", chunk, re.DOTALL)
            content = cite_match.group(1).strip() if cite_match else chunk
            parts.append(f"{i}. {content}")
        parts.append("\n以上内容引用自知识库文档, 可在「答案溯源」面板查看原始出处.")
        return "\n".join(parts)

    def _json_response(self, prompt: str) -> str:
        """JSON 抽取: 识别实体关系 / Cypher 等任务"""
        # 实体关系抽取
        if "entities" in prompt.lower() or "实体" in prompt:
            return self._extract_entities(prompt)

        # Cypher 生成
        if "cypher" in prompt.lower() or "MATCH" in prompt:
            return self._generate_cypher(prompt)

        # 默认空 JSON
        return json.dumps({"result": "ok"}, ensure_ascii=False)

    def _extract_entities(self, text: str) -> str:
        """基于正则的实体关系抽取 (离线降级)"""
        entities: list[dict] = []
        relations: list[dict] = []

        # 产品名模式: 大写字母 + 数字 (e.g. eMMC, P300)
        for m in re.finditer(r"\b([A-Z]{2,}-?[A-Z0-9]{2,})\b", text):
            entities.append(
                {
                    "name": m.group(1),
                    "type": "Product",
                    "properties": {"matched_text": m.group(0)},
                }
            )

        # 部门名 (中文 + 部/处/中心)
        for m in re.finditer(r"([\u4e00-\u9fa5]{2,8}(?:部|处|中心|科|组))", text):
            name = m.group(1)
            if not any(e["name"] == name for e in entities):
                entities.append({"name": name, "type": "Department"})

        # 人名 (X 工程师 / X 经理 / X 老师)
        for m in re.finditer(r"([\u4e00-\u9fa5]{2,4})(?:工程师|经理|老师|主任|总监)", text):
            name = m.group(1)
            if not any(e["name"] == name for e in entities):
                entities.append({"name": name, "type": "Person"})

        # 制度名 (《xxx》)
        for m in re.finditer(r"《([^》]{2,30})》", text):
            name = m.group(1)
            if not any(e["name"] == name for e in entities):
                entities.append({"name": name, "type": "Policy"})

        return json.dumps(
            {"entities": entities, "relations": relations},
            ensure_ascii=False,
        )

    def _generate_cypher(self, prompt: str) -> str:
        """基于查询模式映射 Cypher 模板"""
        # 提取用户问题中的实体名
        products = re.findall(r"\b([A-Z]{2,}-?[A-Z0-9]{2,})\b", prompt)
        departments = re.findall(r"([\u4e00-\u9fa5]{2,8}(?:部|处|中心))", prompt)

        if products:
            return json.dumps(
                {
                    "cypher": (
                        f"MATCH (p:Product {{name: '{products[0]}'}})-[r]-(n) "
                        f"RETURN p, type(r) AS relation, n LIMIT 20"
                    ),
                },
                ensure_ascii=False,
            )

        if departments:
            return json.dumps(
                {
                    "cypher": (
                        f"MATCH (d:Department {{name: '{departments[0]}'}})-[r]-(n) "
                        f"RETURN d, type(r) AS relation, n LIMIT 20"
                    ),
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "cypher": "MATCH (n) RETURN n LIMIT 10",
                "note": "未识别明确实体, 返回默认查询",
            },
            ensure_ascii=False,
        )
