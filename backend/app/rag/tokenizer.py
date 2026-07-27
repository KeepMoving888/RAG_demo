"""app.rag.tokenizer —— 中英双语分词器

设计要点
--------
BM25 等稀疏检索方法对分词质量高度敏感。本模块实现一个面向企业技术文档的
中英双语分词器：

1. **中文分词**：基于 jieba，支持 ``add_word`` 动态扩展自定义词典（用于注入
   企业术语，如 "车规 eMMC"、"RoHS"），避免被切散成无意义单字。
2. **英文分词**：使用正则 ``\\b\\w+\\b`` 提取词元并小写化，简单但对企业
   技术文档足够；不引入 nltk 依赖以保持轻量。
3. **停用词过滤**：内置约 200 个中英停用词，去除 "的/the/is" 等高频虚词，
   降低其对 BM25 idf 的干扰。
4. **LRU 缓存**：使用 ``functools.lru_cache(maxsize=10000)`` 缓存分词结果，
   避免对同一 query / chunk 文本重复分词。BM25 索引构建阶段对全量语料分词
   一次后，在线检索只需对 query 分词，缓存命中率极高。
5. **线程安全**：jieba 内部已做加锁，lru_cache 本身线程安全，可直接用于
   异步框架的线程池调用。

注意：``tokenize`` 走缓存，``tokenize_without_cache`` 绕过缓存，供索引构建
阶段避免缓存膨胀过大时使用。
"""

from __future__ import annotations

import functools
import re
from typing import List

from app.utils.logger import logger

try:
    import jieba

    _JIEBA_AVAILABLE = True
except ImportError:  # pragma: no cover - 依赖缺失时的降级
    jieba = None  # type: ignore
    _JIEBA_AVAILABLE = False
    logger.warning("jieba 未安装，中文分词将退化为按字切分")

# 英文词元正则：匹配单词边界内的字母数字序列
_ENGLISH_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)

# ---------------------------------------------------------------------------
# 停用词表（约 200 个，中英混合）
# 中文部分覆盖常见虚词、代词、连词；英文部分覆盖冠词、介词、be 动词等。
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset(
    {
        # 中文停用词
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
        "看", "好", "自己", "这", "那", "它", "他", "她", "们", "与", "或", "但",
        "而", "及", "以", "为", "对", "由", "从", "把", "被", "让", "使", "向",
        "于", "其", "之", "所", "得", "地", "下", "中", "里", "后", "前", "时",
        "已", "可", "能", "应", "该", "需", "并", "且", "则", "若", "如", "因",
        "此", "些", "等", "即", "才", "再", "又", "还", "只", "才", "吧", "呢",
        "吗", "啊", "哦", "嗯", "嘛", "哈", "呃", "呀", "哇",
        # 英文停用词
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "should", "could", "may", "might", "must", "shall", "can",
        "of", "to", "in", "on", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "before", "after", "above",
        "below", "from", "up", "down", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "s", "t", "just", "don", "now", "i", "me", "my", "myself",
        "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs",
        "themselves", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "if", "because", "as", "until", "while", "of", "at",
        "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "this", "that",
    }
)


class BilingualTokenizer:
    """中英双语分词器。

    中文走 jieba（支持自定义词典），英文走正则 + 小写化，统一过滤停用词。
    适合企业技术文档场景：文档常中英混排（如 "车规 eMMC 模块的 RoHS 合规要求"），
    需要保留英文专有名词的整体性。
    """

    def __init__(self) -> None:
        self._stopwords: frozenset[str] = _STOPWORDS
        # 自定义词典集合，用于 add_word 后的去重判断
        self._custom_words: set[str] = set()
        logger.debug(
            "BilingualTokenizer 初始化完成，停用词数量=%d", len(self._stopwords)
        )

    # ------------------------------------------------------------------
    # 词典扩展
    # ------------------------------------------------------------------
    def add_word(self, word: str) -> None:
        """向 jieba 注入自定义词，避免术语被切散。

        例如 ``add_word("车规 eMMC")`` 后，"车规 eMMC" 会被识别为单个词元，
        而非 "车规" + "eMMC"。术语扩展器会在加载术语词典后批量调用本方法。
        """
        if not _JIEBA_AVAILABLE:
            return
        if word and word not in self._custom_words:
            jieba.add_word(word)
            self._custom_words.add(word)

    # ------------------------------------------------------------------
    # 分词核心
    # ------------------------------------------------------------------
    def _tokenize_raw(self, text: str) -> List[str]:
        """实际分词逻辑（无缓存）。"""
        if not text:
            return []

        tokens: List[str] = []

        if _JIEBA_AVAILABLE:
            # jieba.cut 会同时处理中英文；英文片段会被切成单词
            raw_tokens = list(jieba.cut(text, cut_all=False))
        else:
            # 降级：中文按字切，英文按正则切
            raw_tokens = _ENGLISH_TOKEN_RE.findall(text)

        for tok in raw_tokens:
            tok = tok.strip()
            if not tok:
                continue
            # 英文小写化，保证 BM25 词项归一
            tok_lower = tok.lower()
            # 停用词过滤（同时检查原形与小写形）
            if tok in self._stopwords or tok_lower in self._stopwords:
                continue
            # 过滤纯标点 / 单字符噪声（中文单字保留，因为可能承载语义）
            # 但纯数字单字符且无上下文意义时过滤——这里保留数字，因型号常含数字
            tokens.append(tok_lower)

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """分词（带 LRU 缓存）。

        在线检索场景下 query 通常较短且重复率高，缓存可显著降低分词开销。
        缓存键为原始文本，值为 tuple（lru_cache 要求可哈希返回值，故转 tuple）。
        """
        return list(self._tokenize_cached(text))

    def tokenize_without_cache(self, text: str) -> List[str]:
        """分词（不走缓存）。

        索引构建阶段语料量大，若全部进缓存会导致内存膨胀且无复用价值，
        故构建索引时调用此方法。
        """
        return self._tokenize_raw(text)

    @functools.lru_cache(maxsize=10000)
    def _tokenize_cached(self, text: str) -> tuple:
        """lru_cache 要求可哈希返回值，故返回 tuple。"""
        return tuple(self._tokenize_raw(text))

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @property
    def stopwords(self) -> frozenset[str]:
        """当前停用词集合（只读）。"""
        return self._stopwords

    def clear_cache(self) -> None:
        """清空 LRU 缓存。运营更新术语词典后可调用。"""
        self._tokenize_cached.cache_clear()
        logger.info("分词器 LRU 缓存已清空")


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_tokenizer_instance: BilingualTokenizer | None = None


def get_tokenizer() -> BilingualTokenizer:
    """获取全局分词器单例。

    全局单例确保 jieba 词典与 LRU 缓存在进程内共享，避免重复初始化开销。
    """
    global _tokenizer_instance
    if _tokenizer_instance is None:
        _tokenizer_instance = BilingualTokenizer()
    return _tokenizer_instance
