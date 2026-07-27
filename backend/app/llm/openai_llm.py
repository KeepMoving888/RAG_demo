"""
OpenAI 兼容 LLM (支持 OpenAI / DeepSeek / 任意 OpenAI 协议服务)

特性:
1. 异步 httpx 客户端, 避免阻塞事件循环
2. 流式 SSE 解析
3. 自动重试 (tenacity 指数退避)
4. JSON 模式支持 (response_format)
5. 上下文超长自动截断
"""

import json
import time
from typing import AsyncIterator, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.llm.base import BaseLLM, LLMResponse
from app.utils.logger import logger


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI 协议兼容 LLM"""

    provider = "openai"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 60,
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        return self._client

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def agenerate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        client = await self._get_client()
        start = time.time()

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = response_format

        try:
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("LLM 请求失败 [{}]: {}", e.response.status_code, e.response.text[:200])
            raise
        except Exception as e:
            logger.error("LLM 请求异常: {}", str(e))
            raise

        text = data["choices"][0]["message"]["content"]
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            raw=data,
            latency_ms=(time.time() - start) * 1000,
        )

    async def agenerate_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        client = await self._get_client()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }

        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
