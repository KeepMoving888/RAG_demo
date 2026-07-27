"""解析器子包：聚合导出各类解析器与统一数据结构。

通过此 ``__init__`` 集中导出，调用方只需 ``from app.ingestion.parsers import ...``
即可获取全部解析能力，无需感知内部模块划分。
"""
from app.ingestion.parsers.base import (
    BaseParser,
    ImageBlock,
    ParsedDocument,
    ParsedPage,
    ParserError,
    TableBlock,
    TextBlock,
)
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.image_parser import ImageParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.registry import ParserRegistry, parser_registry

__all__ = [
    "BaseParser",
    "DocxParser",
    "ImageBlock",
    "ImageParser",
    "PDFParser",
    "ParsedDocument",
    "ParsedPage",
    "ParserError",
    "ParserRegistry",
    "TableBlock",
    "TextBlock",
    "parser_registry",
]
