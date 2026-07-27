"""文档解析器抽象基类与统一数据结构定义。

设计要点
========
1. 定义统一的 ``ParsedDocument`` 数据结构，承载解析后的页面、文本块、表格、
   图片，使不同格式的解析器（PDF / DOCX / Image）输出完全一致，下游清洗与
   分块模块无需关心文档来源格式。
2. ``BaseParser`` 为所有具体解析器提供统一接口 ``parse``，采用抽象基类而非
   duck typing，以便在静态检查阶段即可发现接口不匹配问题。
3. 保留 bbox / font_size / heading_level 等结构化信息，供下游
   ``SemanticChunker`` 沿标题层级切分，避免「固定窗口分块」割裂语义的问题。
4. 解析失败统一抛出 ``ParserError``，便于 Celery 任务层捕获并标记文档状态。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ParserError(Exception):
    """解析器统一异常。

    所有具体解析器在遇到无法恢复的错误（文件损坏、不支持的格式、依赖缺失等）
    时应抛出此异常。上层 Celery 任务统一捕获后将 Document 状态置为 ``failed``，
    保证单文档失败不会拖垮整条流水线。
    """


# ---------------------------------------------------------------------------
# 块级数据结构：保留样式与位置信息，支撑语义分块
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    """文本块：承载一段连续文本及其样式信息。

    保留 ``bbox`` 与 ``font_size`` 的目的：PDF 中标题通常字号更大、加粗，
    据此可推断 ``heading_level``，使分块器能沿标题层级切分，而非粗暴按字数切。
    """

    text: str
    # 边界框 (x0, y0, x1, y1)，坐标系原点为页面左上角，单位随解析器而定
    bbox: tuple[float, float, float, float] | None = None
    font_size: float | None = None
    bold: bool = False
    # 标题层级：0=正文, 1=H1, 2=H2, 3=H3；由解析器推断
    heading_level: int = 0


@dataclass
class TableBlock:
    """表格块：以 markdown 字符串形式存储，便于直接嵌入分块上下文。

    采用 markdown 而非二维列表的原因：markdown 表格对 LLM 友好，且可被
    嵌入模型直接读取，无需额外反序列化。
    """

    markdown: str
    bbox: tuple[float, float, float, float] | None = None
    rows: int = 0
    cols: int = 0


@dataclass
class ImageBlock:
    """图片块：记录图片位置与可选的 OCR 文本。

    当文档含扫描图片时，由 ``ImageParser`` / ``PDFParser`` 的 OCR 兜底分支
    填充 ``ocr_text``，使图片中的文字也能进入检索索引。
    """

    bbox: tuple[float, float, float, float] | None = None
    # 图片落盘路径（若解析器选择导出图片）
    image_path: str | None = None
    # OCR 识别结果（若有）
    ocr_text: str | None = None


# ---------------------------------------------------------------------------
# 页级与文档级数据结构
# ---------------------------------------------------------------------------


@dataclass
class ParsedPage:
    """单页解析结果：包含纯文本、结构化块、表格、图片。

    ``text`` 是所有文本块的拼接，供无需结构信息的清洗规则使用；
    ``blocks`` 保留结构信息，供分块器使用。
    """

    page_num: int
    text: str = ""
    blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)
    images: list[ImageBlock] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """完整文档解析结果：多页 + 元数据。

    ``metadata`` 用于承载文档级信息（如来源、作者、创建时间、总页数），
    会透传至最终 Chunk 的 metadata，便于检索时溯源。
    """

    pages: list[ParsedPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    format: str = ""
    page_count: int = 0

    def __post_init__(self) -> None:
        # 未显式指定页数时按实际页数补全
        if not self.page_count and self.pages:
            self.page_count = len(self.pages)


# ---------------------------------------------------------------------------
# 解析器抽象基类
# ---------------------------------------------------------------------------


class BaseParser(ABC):
    """解析器抽象基类。

    所有具体解析器（PDFParser / DocxParser / ImageParser）必须实现 ``parse``，
    返回统一的 ``ParsedDocument``。采用抽象基类约束接口，便于在 ``ParserRegistry``
    中统一路由，也便于单元测试以同一断言校验不同格式解析器的输出结构。
    """

    @abstractmethod
    async def parse(self, file_path: str) -> ParsedDocument:
        """解析文档，返回结构化结果。

        Args:
            file_path: 文档绝对路径。

        Returns:
            ``ParsedDocument``，包含所有页面与结构化块。

        Raises:
            ParserError: 文件不存在、格式不支持或解析过程出错时统一抛出。
        """
        raise NotImplementedError

    @staticmethod
    def _read_file_bytes(file_path: str) -> bytes:
        """读取文件字节，供子类复用。

        集中处理文件读取异常，统一转换为 ``ParserError``。
        """
        import os

        if not os.path.exists(file_path):
            raise ParserError(f"文件不存在: {file_path}")
        try:
            with open(file_path, "rb") as fh:
                return fh.read()
        except OSError as exc:
            raise ParserError(f"读取文件失败: {file_path} ({exc})") from exc
