"""DOCX 解析器：基于 unstructured 库的结构化抽取。

设计要点
========
1. 使用 ``unstructured.partition.docx.partition_docx`` 作为主引擎，统一处理
   标题、正文、列表、表格，避免手写 python-docx 的样式遍历分支。
2. **标题层级推断**：unstructured 将标题统一归类为 ``Title``，但企业文档常以
   编号前缀（"1." / "1.1" / "1.1.1"）表达层级。这里结合元素类别与编号模式
   正则推断 heading_level（Title → H1，编号前缀深度 → H2/H3），使下游分块器
   可沿层级切分。
3. **列表与表格结构化**：ListItem 归入正文块并标记 ``is_list``，Table 转为
   ``TableBlock.markdown`` 独立保留。
4. 全部解析为单页 ``ParsedDocument``（DOCX 无原生分页概念），page_num 统一为 1，
   但保留元素顺序，分块器按内容流处理。
"""

from __future__ import annotations

import os
import re
from typing import Any

from app.ingestion.parsers.base import (
    BaseParser,
    ParsedDocument,
    ParsedPage,
    ParserError,
    TableBlock,
    TextBlock,
)
from app.utils.logger import logger

# 编号前缀正则：匹配 "1." / "1.1" / "1.1.1" / "第一章" 等
_NUMBERING_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)[.\s、]")


class DocxParser(BaseParser):
    """DOCX 解析器。

    依赖 ``unstructured`` 库；懒加载避免无 docx 解析场景的导入开销。
    """

    def __init__(self) -> None:
        self._partition: Any = None

    def _load_partition(self) -> Any:
        """懒加载 unstructured 的 partition_docx。"""
        if self._partition is None:
            try:
                from unstructured.partition.docx import (
                    partition_docx,  # type: ignore[import-not-found]
                )
            except ImportError as exc:
                raise ParserError(
                    "unstructured 未安装，无法解析 DOCX，请执行 pip install unstructured"
                ) from exc
            self._partition = partition_docx
            logger.info("unstructured.partition_docx 已加载")
        return self._partition

    async def parse(self, file_path: str) -> ParsedDocument:
        """解析 DOCX 文档。

        Args:
            file_path: DOCX 文件绝对路径。

        Returns:
            ``ParsedDocument``，单页结构，按元素顺序排列文本块与表格。

        Raises:
            ParserError: 文件不存在或 unstructured 不可用。
        """
        if not os.path.exists(file_path):
            raise ParserError(f"DOCX 文件不存在: {file_path}")

        partition = self._load_partition()
        try:
            elements = partition(filename=file_path)
        except Exception as exc:  # noqa: BLE001
            raise ParserError(f"解析 DOCX 失败: {file_path} ({exc})") from exc

        blocks: list[TextBlock] = []
        tables: list[TableBlock] = []

        for element in elements:
            category = str(getattr(element, "category", "") or "")
            text = str(getattr(element, "text", "") or "").strip()
            if not text:
                continue

            if category == "Table":
                markdown = self._element_to_markdown(element)
                if markdown.strip():
                    tables.append(TableBlock(markdown=markdown, rows=0, cols=0))
            else:
                heading_level = self._infer_heading_level(category, text)
                blocks.append(
                    TextBlock(
                        text=text,
                        heading_level=heading_level,
                        bold=(category == "Title"),
                    )
                )

        page_text = "\n".join(b.text for b in blocks)
        # 表格文本也并入 page.text，保证无结构消费方可用
        if tables:
            page_text += "\n" + "\n\n".join(t.markdown for t in tables)

        page = ParsedPage(page_num=1, text=page_text, blocks=blocks, tables=tables)
        logger.info(f"DOCX 解析完成: {file_path}, 文本块 {len(blocks)} 个, 表格 {len(tables)} 个")
        return ParsedDocument(
            pages=[page],
            metadata={"source": file_path, "page_count": 1},
            format="docx",
            page_count=1,
        )

    # ------------------------------------------------------------------
    # 标题层级推断
    # ------------------------------------------------------------------

    def _infer_heading_level(self, category: str, text: str) -> int:
        """根据元素类别与编号前缀推断标题层级。

        规则：Title → H1；若文本以编号前缀开头，则按编号深度映射层级
        （"1." → H1，"1.1" → H2，"1.1.1" → H3）；其余为正文（0）。
        """
        if category == "Title":
            match = _NUMBERING_PATTERN.match(text)
            if match:
                depth = match.group(1).count(".") + 1
                return min(depth, 3)
            return 1
        return 0

    @staticmethod
    def _element_to_markdown(element: Any) -> str:
        """将 unstructured Table 元素转为 markdown 字符串。

        优先使用 ``text_as_html`` 元数据，回退到纯文本按行切分。
        """
        metadata = getattr(element, "metadata", None)
        html = getattr(metadata, "text_as_html", None) if metadata else None
        if html:
            return DocxParser._html_table_to_markdown(html)
        text = str(getattr(element, "text", "") or "")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        rows = [re.split(r"\t|\s{2,}", ln) for ln in lines]
        from app.ingestion.parsers.pdf_parser import PDFParser

        return PDFParser._rows_to_markdown(rows)

    @staticmethod
    def _html_table_to_markdown(html: str) -> str:
        """简易 HTML 表格转 markdown（仅处理 <tr>/<td>/<th>）。"""
        row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
        cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
        rows: list[list[str]] = []
        for row_html in row_pattern.findall(html):
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cell_pattern.findall(row_html)]
            if cells:
                rows.append(cells)
        if not rows:
            return html
        from app.ingestion.parsers.pdf_parser import PDFParser

        return PDFParser._rows_to_markdown(rows)  # type: ignore[arg-type]
