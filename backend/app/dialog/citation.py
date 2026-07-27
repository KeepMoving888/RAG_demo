"""
答案溯源 - 基于字符级 Jaccard 相似度的引用抽取

设计要点
========

1. 为何需要答案溯源?
   - 企业级 RAG 必须可审计: 每条答案需标注来源文档与页码, 否则无法满足合规与
     可信要求. 前端据此渲染「答案溯源」面板, 用户可一键跳转原文.

2. 为何用 Jaccard 相似度而非 BM25 / 向量相似度?
   - BM25 依赖倒排索引与语料统计, 不适合「答案句 vs 单个 chunk」的细粒度比对;
   - 向量相似度需额外 embedding 调用, 引入延迟与模型依赖;
   - 字符级 Jaccard (这里用字符二元组 bigram) 是零依赖的轻量算法, 在「短句 vs
     短段落」场景下区分度足够, 适合作为引用判定的近似匹配.

3. 匹配算法:
   a. 将 answer 按句号/问号/叹号切分为句子;
   b. 每个句子与每个 retrieved_chunk 的 content 计算字符 bigram Jaccard 相似度;
   c. 相似度 > 0.3 视为该句引用了该 chunk;
   d. 每个 chunk 至少匹配 1 句才进入 citations;
   e. snippet 取匹配句在 chunk content 中的位置 + 前后各 50 字上下文.

4. 覆盖率 (compute_coverage): 引用句数 / 总句数, 上报 CITATION_COVERAGE 指标,
   用于监控答案的「有据可依」比例.
"""

import re

from app.metrics import CITATION_COVERAGE
from app.utils.logger import logger


# 句子切分: 按中英文句号/问号/叹号/分号切分, 保留分隔符
_SENT_SPLIT_RE = re.compile(r"(?<=[。!?\!？；;])\s*")

# 相似度阈值: 超过则视为引用
_SIM_THRESHOLD = 0.3

# snippet 上下文半径 (字符)
_SNIPPET_RADIUS = 50


class CitationExtractor:
    """答案溯源抽取器: 从答案文本中识别引用的 chunk 并生成溯源信息."""

    def __init__(self, threshold: float = _SIM_THRESHOLD) -> None:
        self._threshold = threshold

    async def extract(
        self, answer: str, retrieved_chunks: list[dict]
    ) -> list[dict]:
        """
        抽取答案中引用的 chunk.

        Args:
            answer: LLM 生成的答案文本.
            retrieved_chunks: 检索召回的 chunk 列表, 每项需含
                {id, content, document_id, page_number, heading_path, ...}.

        Returns:
            citations 列表, 每项:
            {chunk_id, document_id, doc_title, snippet, page_number, heading_path, score}
        """
        if not answer or not retrieved_chunks:
            return []

        sentences = self._split_sentences(answer)
        if not sentences:
            return []

        citations: list[dict] = []
        # 每个 chunk 取其匹配句的最大相似度作为 score
        for chunk in retrieved_chunks:
            chunk_id = str(chunk.get("id", ""))
            content = chunk.get("content", "") or ""
            if not chunk_id or not content:
                continue

            best_score = 0.0
            best_sentence = ""
            for sent in sentences:
                score = self._jaccard_similarity(sent, content)
                if score > best_score:
                    best_score = score
                    best_sentence = sent

            if best_score >= self._threshold:
                snippet = self._make_snippet(best_sentence, content)
                citations.append({
                    "chunk_id": chunk_id,
                    "document_id": chunk.get("document_id"),
                    "doc_title": chunk.get("doc_title") or chunk.get("title") or "",
                    "snippet": snippet,
                    "page_number": chunk.get("page_number"),
                    "heading_path": chunk.get("heading_path") or "",
                    "score": round(best_score, 4),
                })

        # 按相似度降序
        citations.sort(key=lambda c: c["score"], reverse=True)

        # 上报覆盖率
        coverage = self.compute_coverage(answer, citations, sentences)
        try:
            CITATION_COVERAGE.set(coverage)
        except Exception:
            pass

        logger.debug(
            "答案溯源: sentences={} citations={} coverage={:.2f}",
            len(sentences), len(citations), coverage,
        )
        return citations

    # ======================== 覆盖率 ========================
    def compute_coverage(
        self,
        answer: str,
        citations: list[dict],
        sentences: list[str] | None = None,
    ) -> float:
        """
        计算答案覆盖率 = 被引用的句子数 / 总句数.

        若已传入 sentences 则复用, 否则重新切分.
        """
        if sentences is None:
            sentences = self._split_sentences(answer)
        if not sentences:
            return 0.0

        cited_count = 0
        for sent in sentences:
            for c in citations:
                snippet = c.get("snippet", "") or ""
                # 若该句出现在某 citation 的 snippet 上下文中, 视为被引用
                if sent and sent[:20] in snippet:
                    cited_count += 1
                    break

        return round(cited_count / len(sentences), 4)

    # ======================== 句子切分 ========================
    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """按中英文标点切分句子, 过滤空白与过短片段."""
        if not text:
            return []
        parts = _SENT_SPLIT_RE.split(text)
        sentences: list[str] = []
        for p in parts:
            p = p.strip()
            # 过滤纯标点或过短 (少于 2 字) 的片段
            if len(p) >= 2 and not re.fullmatch(r"[\s。!?\!？；;.,，、·]+", p):
                sentences.append(p)
        return sentences

    # ======================== Jaccard 相似度 ========================
    @staticmethod
    def _char_bigrams(text: str) -> set[str]:
        """提取字符二元组 (bigram) 集合, 兼顾区分度与零依赖."""
        if not text:
            return set()
        # 去除空白与标点, 降低噪声
        cleaned = re.sub(r"[\s\n\r\t。!?\!？；;.,，、·\"'()（）\[\]【】]", "", text)
        if len(cleaned) < 2:
            return {cleaned} if cleaned else set()
        return {cleaned[i: i + 2] for i in range(len(cleaned) - 1)}

    def _jaccard_similarity(self, sentence: str, chunk_content: str) -> float:
        """
        字符 bigram Jaccard 相似度:
            |A ∩ B| / |A ∪ B|
        A = 句子 bigram 集合, B = chunk 内容 bigram 集合.
        """
        set_a = self._char_bigrams(sentence)
        set_b = self._char_bigrams(chunk_content)
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    # ======================== snippet 生成 ========================
    @staticmethod
    def _make_snippet(sentence: str, chunk_content: str) -> str:
        """
        生成引用片段: 在 chunk content 中定位匹配句, 取前后各 50 字上下文.
        若精确匹配失败, 回退为句子本身.
        """
        if not sentence:
            return ""
        # 尝试在 chunk 中定位 (取句子前缀避免标点差异导致失配)
        anchor = sentence[:20] if len(sentence) > 20 else sentence
        pos = chunk_content.find(anchor)
        if pos < 0:
            # 回退: 用句子首个非空白片段再试
            for seg in re.split(r"[\s,，。;；]+", sentence):
                if seg and chunk_content.find(seg) >= 0:
                    pos = chunk_content.find(seg)
                    break
        if pos < 0:
            return sentence[:120]

        start = max(0, pos - _SNIPPET_RADIUS)
        end = min(len(chunk_content), pos + len(sentence) + _SNIPPET_RADIUS)
        snippet = chunk_content[start:end].strip()
        # 边界省略号提示
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(chunk_content) else ""
        return f"{prefix}{snippet}{suffix}"
