"""语义分块器：沿标题层级切分，输出带层级路径的检索分块。

本模块是文档摄入流水线的核心难点。

分块策略（与固定窗口分块对比）
================================
固定窗口分块（如每 512 token 硬切）的问题：
- 会从句子/段落中间切断，破坏语义完整性，导致检索召回的相关片段缺失上下文；
- 无视文档结构，标题与其正文可能落入不同分块，检索时无法判断归属章节。

本实现的语义分块策略：
1. **沿标题层级切分**：根据解析器推断的 heading_level（H1→H2→H3）建立标题栈，
   每个叶子节点的正文作为一个 candidate chunk，保证分块边界与语义边界对齐。
2. **超大块二次切分**：candidate chunk 超过 ``chunk_size``（512 token）时按段落
   二次切分，并保留 ``chunk_overlap``（64 token）滑动窗口，避免边界信息丢失。
3. **过小块合并**：candidate chunk 小于 ``chunk_min_size``（64 token）时与相邻
   同级块合并，避免产生大量低信息量的碎片分块，浪费向量库存储。
4. **表格独立分块**：表格作为独立 chunk 保留，metadata 标记 ``is_table=True``，
   便于检索后特殊渲染。
5. **父子关系保留**：为每个标题创建概要 chunk，子块通过 ``parent_chunk_id``
   指向父标题的概要块，支持检索时向上回溯父级上下文，提升答案完整性。

说明：``parent_chunk_id`` 在分块阶段为返回列表中的本地索引（从 0 起），
持久化时由任务层映射为数据库 DocumentChunk.id。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.config import settings
from app.ingestion.parsers.base import ParsedDocument, TableBlock
from app.utils.logger import logger

# CJK 字符正则：用于 token 估算与中文断行修复
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class Chunk:
    """语义分块结果。

    Attributes:
        content: 分块文本内容。
        heading_path: 标题层级路径，如 ``"第3章/3.2 安装/3.2.1 步骤"``。
        page_number: 来源页码。
        parent_chunk_id: 父标题概要块的本地索引（持久化时映射为 DB id）。
        chunk_index: 分块在文档中的顺序索引（持久化前由任务层重排）。
        token_count: 估算 token 数。
        char_count: 字符数。
        metadata: 扩展元数据（is_table / is_summary 等）。
    """

    content: str
    heading_path: str = ""
    page_number: int = 0
    parent_chunk_id: Optional[int] = None
    chunk_index: int = 0
    token_count: int = 0
    char_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticChunker:
    """语义分块器。

    将 ``ParsedDocument`` 转换为带层级路径与父子关系的 ``Chunk`` 列表。
    """

    def __init__(self) -> None:
        self.chunk_size: int = getattr(settings, "chunk_size", 512)
        self.chunk_overlap: int = getattr(settings, "chunk_overlap", 64)
        self.chunk_min_size: int = getattr(settings, "chunk_min_size", 64)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def chunk(self, parsed_doc: ParsedDocument) -> list[Chunk]:
        """对清洗后的文档执行语义分块。

        Args:
            parsed_doc: 已清洗的 ``ParsedDocument``。

        Returns:
            ``list[Chunk]``，按文档顺序排列，``chunk_index`` 已重排。
        """
        # 第一步：沿标题层级切分，产出含父子关系的原始候选块
        raw_chunks = self._split_by_headings(parsed_doc)
        logger.info(f"标题层级切分产出 {len(raw_chunks)} 个候选块")

        # 第二步：超大块按段落二次切分（含 overlap）
        split_chunks: list[Chunk] = []
        for chunk in raw_chunks:
            if chunk.metadata.get("is_summary"):
                split_chunks.append(chunk)
                continue
            if chunk.token_count > self.chunk_size:
                split_chunks.extend(self._split_large_chunk(chunk, self.chunk_size))
            else:
                split_chunks.append(chunk)

        # 第三步：合并过小块
        merged = self._merge_small_chunks(split_chunks)

        # 第四步：重排 chunk_index 并补全统计
        final = self._reindex(merged)
        logger.info(f"语义分块完成: 最终 {len(final)} 个分块")
        return final

    # ------------------------------------------------------------------
    # 第一步：沿标题层级切分
    # ------------------------------------------------------------------

    def _split_by_headings(self, parsed_doc: ParsedDocument) -> list[Chunk]:
        """遍历页面，按标题栈切分并建立父子关系。

        维护标题栈 ``[(level, title, summary_local_index), ...]``：
        - 遇到 heading：弹出栈顶 level >= 当前 level 的节点，为新标题创建概要块，
          其 parent 指向新栈顶（即外层标题）；正文块归属此标题。
        - 遇到 body：累积到当前标题缓冲，标题变更时落盘为候选块。
        - 遇到 table：作为独立块，parent 指向当前标题。
        """
        chunks: list[Chunk] = []
        # 标题栈：(level, title, summary_chunk_local_index)
        heading_stack: list[tuple[int, str, int]] = []

        body_buffer: list[str] = []
        body_page = 0
        # 当前生效标题的概要块本地索引
        current_summary_idx: Optional[int] = None

        def _flush_body() -> None:
            nonlocal body_buffer, body_page, current_summary_idx
            if not body_buffer:
                return
            content = "\n".join(body_buffer).strip()
            if not content:
                body_buffer = []
                return
            path = self._build_heading_path([(lv, t) for lv, t, _ in heading_stack])
            token = self._estimate_tokens(content)
            chunk = Chunk(
                content=content,
                heading_path=path,
                page_number=body_page,
                parent_chunk_id=current_summary_idx,
                token_count=token,
                char_count=len(content),
                metadata={"is_body": True},
            )
            chunks.append(chunk)
            body_buffer = []

        for page in parsed_doc.pages:
            page_num = page.page_num
            for block in page.blocks:
                text = block.text.strip()
                if not text:
                    continue
                if block.heading_level > 0:
                    _flush_body()
                    # 弹出同级及更深标题，保持层级单调递增
                    while heading_stack and heading_stack[-1][0] >= block.heading_level:
                        heading_stack.pop()
                    parent_idx = heading_stack[-1][2] if heading_stack else None
                    new_path = self._build_heading_path(
                        [(lv, t) for lv, t, _ in heading_stack] + [(block.heading_level, text)]
                    )
                    summary = Chunk(
                        content=new_path,
                        heading_path=new_path,
                        page_number=page_num,
                        parent_chunk_id=parent_idx,
                        token_count=self._estimate_tokens(new_path),
                        char_count=len(new_path),
                        metadata={"is_summary": True, "heading_level": block.heading_level},
                    )
                    chunks.append(summary)
                    current_summary_idx = len(chunks) - 1
                    heading_stack.append((block.heading_level, text, current_summary_idx))
                else:
                    body_buffer.append(text)
                    body_page = page_num

            # 表格作为独立块
            for table in page.tables:
                _flush_body()
                path = self._build_heading_path([(lv, t) for lv, t, _ in heading_stack])
                markdown = table.markdown.strip()
                if not markdown:
                    continue
                chunk = Chunk(
                    content=markdown,
                    heading_path=path,
                    page_number=page_num,
                    parent_chunk_id=current_summary_idx,
                    token_count=self._estimate_tokens(markdown),
                    char_count=len(markdown),
                    metadata={"is_table": True},
                )
                chunks.append(chunk)

        _flush_body()
        return chunks

    # ------------------------------------------------------------------
    # 第二步：超大块二次切分（按段落，含 overlap）
    # ------------------------------------------------------------------

    def _split_large_chunk(self, chunk: Chunk, max_tokens: int) -> list[Chunk]:
        """按段落二次切分超大块，保留 overlap 滑动窗口。

        切分粒度优先为段落（``\\n`` 分隔）；若单段落仍显著超限，则按 token
        硬切。相邻子块之间保留约 ``chunk_overlap`` token 的尾部段落作为重叠，
        避免切分边界丢失上下文。
        """
        paragraphs = [p for p in chunk.content.split("\n") if p.strip()]
        if not paragraphs:
            return [chunk]

        sub_chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            # 当前已累积且加入本段会超限时，先落盘当前块
            if current and current_tokens + para_tokens > max_tokens:
                content = "\n".join(current)
                sub_chunks.append(self._make_sub_chunk(content, chunk))
                # 构建 overlap：从已累积段落尾部取约 overlap token
                overlap = self._take_overlap(current, self.chunk_overlap)
                current = overlap
                current_tokens = sum(self._estimate_tokens(p) for p in overlap)
            current.append(para)
            current_tokens += para_tokens

        if current:
            content = "\n".join(current)
            sub_chunks.append(self._make_sub_chunk(content, chunk))

        # 处理单段落仍过大的情况：按 token 硬切
        final: list[Chunk] = []
        for sc in sub_chunks:
            if sc.token_count > max_tokens * 1.5:
                final.extend(self._hard_split(sc, max_tokens, chunk))
            else:
                final.append(sc)
        return final

    def _take_overlap(self, paragraphs: list[str], overlap_tokens: int) -> list[str]:
        """从段落列表尾部取累计 token 约 overlap 的段落作为重叠。"""
        overlap: list[str] = []
        acc = 0
        for para in reversed(paragraphs):
            t = self._estimate_tokens(para)
            if overlap and acc + t > overlap_tokens * 1.5:
                break
            overlap.insert(0, para)
            acc += t
            if acc >= overlap_tokens:
                break
        return overlap

    def _hard_split(self, chunk: Chunk, max_tokens: int, parent: Chunk) -> list[Chunk]:
        """对无段落边界的超大文本按 token 硬切，含 overlap。"""
        # 按 token 长度切字符（粗略：1 token ≈ 1 CJK 字符 / 0.77 英文词）
        text = chunk.content
        # 用近似比例将 token 预算转为字符预算
        char_budget = max(1, int(max_tokens / self._token_char_ratio(text)))
        overlap_chars = max(1, int(self.chunk_overlap / self._token_char_ratio(text)))
        result: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + char_budget, len(text))
            piece = text[start:end]
            result.append(self._make_sub_chunk(piece, parent))
            if end >= len(text):
                break
            start = end - overlap_chars
        return result

    def _make_sub_chunk(self, content: str, parent: Chunk) -> Chunk:
        """构造继承父块属性的子分块。"""
        return Chunk(
            content=content,
            heading_path=parent.heading_path,
            page_number=parent.page_number,
            parent_chunk_id=parent.parent_chunk_id,
            token_count=self._estimate_tokens(content),
            char_count=len(content),
            metadata={**parent.metadata, "split_from": True},
        )

    # ------------------------------------------------------------------
    # 第三步：合并过小块
    # ------------------------------------------------------------------

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """合并过小块：将 token 不足 ``chunk_min_size`` 的正文块与相邻同级块合并。

        概要块（is_summary）与表格块（is_table）不参与合并，以保留结构锚点。
        合并仅发生在相邻正文块之间，避免跨标题拼接造成语义错乱。
        """
        result: list[Chunk] = []
        buffer: Optional[Chunk] = None

        for chunk in chunks:
            is_mergeable = not (chunk.metadata.get("is_summary") or chunk.metadata.get("is_table"))
            if not is_mergeable:
                if buffer is not None:
                    result.append(buffer)
                    buffer = None
                result.append(chunk)
                continue

            if buffer is None:
                buffer = chunk
            elif buffer.token_count < self.chunk_min_size:
                # 缓冲未达最小阈值，合并当前块
                buffer = self._combine(buffer, chunk)
            else:
                # 缓冲已达标，落盘并启用新缓冲
                result.append(buffer)
                buffer = chunk

        if buffer is not None:
            # 末尾残余：若过小则并入上一个结果块（若存在且可合并），否则单独保留
            if (
                buffer.token_count < self.chunk_min_size
                and result
                and not result[-1].metadata.get("is_summary")
                and not result[-1].metadata.get("is_table")
            ):
                result[-1] = self._combine(result[-1], buffer)
            else:
                result.append(buffer)
        return result

    @staticmethod
    def _combine(a: Chunk, b: Chunk) -> Chunk:
        """合并两个块，保留 a 的位置属性与父关系。"""
        content = a.content + "\n" + b.content
        return Chunk(
            content=content,
            heading_path=a.heading_path,
            page_number=a.page_number,
            parent_chunk_id=a.parent_chunk_id,
            token_count=SemanticChunker._estimate_tokens(content),
            char_count=len(content),
            metadata={**a.metadata, **{k: v for k, v in b.metadata.items() if k not in a.metadata}},
        )

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数。

        中文按字符数近似（BPE 对中文基本 1 字符 ≈ 1 token）；英文按词数 × 1.3
        （英文经 BPE 拆分后 token 数通常多于词数）。混合文本取两者之和。
        该估算用于分块决策，无需精确，重在快速与稳定。
        """
        if not text:
            return 0
        cjk = len(_CJK_PATTERN.findall(text))
        non_cjk = _CJK_PATTERN.sub(" ", text)
        words = len(non_cjk.split())
        return cjk + int(words * 1.3)

    @staticmethod
    def _token_char_ratio(text: str) -> float:
        """估算 token/字符比，用于硬切时将 token 预算换算为字符预算。"""
        if not text:
            return 1.0
        tokens = SemanticChunker._estimate_tokens(text)
        return tokens / len(text) if len(text) else 1.0

    @staticmethod
    def _build_heading_path(headings: list[tuple[int, str]]) -> str:
        """构建标题层级路径，如 ``"第3章/3.2 安装/3.2.1 步骤"``。

        Args:
            headings: ``(level, title)`` 列表，已按层级单调排列。
        """
        return "/".join(title.strip() for _, title in headings if title.strip())

    @staticmethod
    def _reindex(chunks: list[Chunk]) -> list[Chunk]:
        """重排 chunk_index 并补全 token/char 统计。"""
        for idx, chunk in enumerate(chunks):
            chunk.chunk_index = idx
            if not chunk.token_count:
                chunk.token_count = SemanticChunker._estimate_tokens(chunk.content)
            if not chunk.char_count:
                chunk.char_count = len(chunk.content)
        return chunks
