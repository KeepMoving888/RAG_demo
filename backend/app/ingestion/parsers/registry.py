"""解析器注册中心：按文件格式路由到对应解析器。

设计要点
========
1. 提供统一的 ``get_parser(file_format)`` 路由接口，调用方无需关心格式与
   解析器的映射细节，降低耦合。
2. 支持 ``register(format_, parser)`` 扩展接口，新增格式只需注册即可，无需
   修改既有调用代码，符合开闭原则。
3. 模块级 ``parser_registry`` 单例，避免重复构造解析器实例（解析器内部持有
   懒加载的引擎缓存，复用单例可命中缓存）。
4. 对未知格式抛出 ``ParserError``，统一异常出口。
"""
from __future__ import annotations

from typing import Optional

from app.ingestion.parsers.base import BaseParser, ParserError
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.image_parser import ImageParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.utils.logger import logger


class ParserRegistry:
    """解析器注册中心。

    内部维护 ``format -> BaseParser`` 映射。构造时按惯例注册内置格式，
    后续可通过 ``register`` 动态扩展。
    """

    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册内置格式解析器。"""
        # 同一解析器实例可服务多格式，复用单例以命中懒加载缓存
        pdf = PDFParser()
        docx = DocxParser()
        image = ImageParser()
        mapping = {
            "pdf": pdf,
            "docx": docx,
            "doc": docx,  # .doc 暂复用 docx 解析器，能力受限时降级
            "txt": _PlainTextParser(),
            "md": _PlainTextParser(),
            "png": image,
            "jpg": image,
            "jpeg": image,
            "tiff": image,
        }
        for fmt, parser in mapping.items():
            self._parsers[fmt] = parser
        logger.debug(f"ParserRegistry 已注册格式: {list(self._parsers.keys())}")

    def register(self, format_: str, parser: BaseParser) -> None:
        """注册或覆盖某格式的解析器。

        Args:
            format_: 文件格式小写扩展名（如 ``pdf``）。
            parser: ``BaseParser`` 实例。
        """
        self._parsers[format_.lower()] = parser
        logger.info(f"注册解析器: {format_} -> {parser.__class__.__name__}")

    def get_parser(self, file_format: str) -> BaseParser:
        """根据文件格式获取解析器。

        Args:
            file_format: 文件格式小写扩展名。

        Returns:
            对应的 ``BaseParser`` 实例。

        Raises:
            ParserError: 不支持的格式。
        """
        fmt = file_format.lower().lstrip(".")
        parser = self._parsers.get(fmt)
        if parser is None:
            supported = ", ".join(sorted(self._parsers.keys()))
            raise ParserError(f"不支持的文件格式: {file_format} (支持: {supported})")
        return parser

    def supports(self, file_format: str) -> bool:
        """判断是否支持某格式。"""
        return file_format.lower().lstrip(".") in self._parsers

    def supported_formats(self) -> list[str]:
        """返回所有已注册格式。"""
        return sorted(self._parsers.keys())


class _PlainTextParser(BaseParser):
    """纯文本解析器（txt / md）。

    将整篇文本作为单页处理，按空行切分段落以保留基本结构，
    供下游分块器按段落二次切分。
    """

    async def parse(self, file_path: str) -> "ParsedDocument":  # type: ignore[name-defined]
        import os

        from app.ingestion.parsers.base import (
            ParsedDocument,
            ParsedPage,
            ParserError,
            TextBlock,
        )

        if not os.path.exists(file_path):
            raise ParserError(f"文本文件不存在: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError as exc:
            raise ParserError(f"读取文本文件失败: {file_path} ({exc})") from exc

        blocks = [TextBlock(text=para) for para in content.split("\n\n") if para.strip()]
        page = ParsedPage(page_num=1, text=content, blocks=blocks)
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ParsedDocument(
            pages=[page],
            metadata={"source": file_path, "page_count": 1},
            format=ext or "txt",
            page_count=1,
        )


# 模块级单例：复用解析器内部懒加载引擎缓存
parser_registry = ParserRegistry()
