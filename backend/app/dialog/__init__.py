"""
多轮对话模块 - 上下文管理 / 查询改写 / QA 缓存 / 答案溯源 / 生成编排

模块组成
========

- DialogContextManager: 基于 Redis Hash 的滑动窗口多轮对话上下文
  (窗口 + 长程锚点 + 超限归档 + 进程内降级).
- QueryRewriter: 查询改写器 (LLM 在线 / 规则离线 / 失败兜底), 消解代词指代.
- QACache: 高频问答缓存 (文本哈希命中 + LRU 淘汰 + chunk 反向失效).
- CitationExtractor: 答案溯源 (字符 bigram Jaccard 匹配 + 覆盖率上报).
- AnswerGenerator: 门面编排器, 串联上述组件 + HybridRetriever + BaseLLM,
  提供 generate / generate_stream 两种入口.

外部推荐通过 get_generator() 获取单例使用.
"""

from app.dialog.citation import CitationExtractor
from app.dialog.context_manager import DialogContextManager
from app.dialog.generator import AnswerGenerator, get_generator
from app.dialog.qa_cache import QACache
from app.dialog.query_rewriter import QueryRewriter

__all__ = [
    "DialogContextManager",
    "QueryRewriter",
    "QACache",
    "CitationExtractor",
    "AnswerGenerator",
    "get_generator",
]
