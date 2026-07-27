"""LLM 工厂: 按 settings.llm_provider 选择实现"""

from functools import lru_cache
from typing import Optional

from app.config import settings
from app.llm.base import BaseLLM


@lru_cache
def get_llm(provider: Optional[str] = None) -> BaseLLM:
    """获取 LLM 单例"""
    provider = provider or settings.llm_provider

    if provider == "offline":
        from app.llm.offline_llm import OfflineLLM
        return OfflineLLM(model="offline-rule-based")

    if provider == "openai":
        from app.llm.openai_llm import OpenAICompatibleLLM
        return OpenAICompatibleLLM(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )

    if provider == "deepseek":
        from app.llm.openai_llm import OpenAICompatibleLLM
        return OpenAICompatibleLLM(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_api_base,
        )

    raise ValueError(f"不支持的 LLM provider: {provider}")
