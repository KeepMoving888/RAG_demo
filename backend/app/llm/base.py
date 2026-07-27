"""
LLM 抽象基类

设计:
1. 统一接口: agenerate / agenerate_stream
2. 4 类任务: chat / extract / cypher / rewrite (不同 prompt 模板)
3. 降级链: openai/deepseek → offline (保证全链路可运行)
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """LLM 响应"""

    text: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    raw: dict | None = None
    latency_ms: float = 0.0


class BaseLLM(ABC):
    """LLM 抽象基类"""

    provider: str = "base"

    def __init__(self, model: str, temperature: float = 0.2, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def agenerate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """同步生成"""
        ...

    async def agenerate_stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """流式生成 (默认降级为一次性返回)"""
        resp = await self.agenerate(messages, temperature, max_tokens)
        yield resp.text

    async def aextract_json(self, prompt: str) -> dict:
        """抽取 JSON (用于实体关系抽取 / Cypher 生成)"""
        resp = await self.agenerate(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        import json

        try:
            return json.loads(resp.text)
        except json.JSONDecodeError:
            # 兼容模型返回带 markdown 代码块的情况
            import re

            match = re.search(r"\{[\s\S]*\}", resp.text)
            if match:
                return json.loads(match.group(0))
            raise
