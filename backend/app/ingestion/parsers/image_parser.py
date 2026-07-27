"""图片解析器：基于 PaddleOCR 的单页文档抽取。

设计要点
========
1. PaddleOCR 懒加载，与 PDFParser 共享同一惰性策略：仅在真正处理图片时
   才加载模型，避免模块导入即触发数百 MB 模型加载。
2. 返回单页 ``ParsedDocument``（整张图视为一页），OCR 结果作为正文文本块。
3. OCR 不可用时降级返回空页并记录 warning，保证流水线不中断。
4. 支持 png / jpg / jpeg / tiff，tiff 多帧时仅取第一帧（企业扫描件常见）。
"""

from __future__ import annotations

import os
from typing import Any

from app.ingestion.parsers.base import (
    BaseParser,
    ParsedDocument,
    ParsedPage,
    ParserError,
    TextBlock,
)
from app.utils.logger import logger


class ImageParser(BaseParser):
    """图片解析器，使用 PaddleOCR 识别图片文本。"""

    def __init__(self) -> None:
        self._ocr_engine: Any = None

    def _load_ocr(self) -> Any | None:
        """懒加载 PaddleOCR 引擎，不可用时返回 None。"""
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR  # type: ignore[import-not-found]
            except ImportError:
                logger.warning("PaddleOCR 未安装，图片将无法 OCR，返回空文本")
                return None
            try:
                self._ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
                logger.info("PaddleOCR 引擎已加载（ImageParser）")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"PaddleOCR 初始化失败，OCR 降级关闭: {exc}")
                return None
        return self._ocr_engine

    async def parse(self, file_path: str) -> ParsedDocument:
        """解析图片，返回单页 ``ParsedDocument``。

        Args:
            file_path: 图片文件绝对路径（png/jpg/jpeg/tiff）。

        Returns:
            ``ParsedDocument``，单页，OCR 文本作为正文。

        Raises:
            ParserError: 文件不存在。
        """
        if not os.path.exists(file_path):
            raise ParserError(f"图片文件不存在: {file_path}")

        ocr = self._load_ocr()
        text = ""
        if ocr is not None:
            text = self._ocr_image(ocr, file_path)

        blocks = [TextBlock(text=text)] if text else []
        page = ParsedPage(page_num=1, text=text, blocks=blocks)
        logger.info(f"图片解析完成: {file_path}, 识别字符 {len(text)} 个")
        return ParsedDocument(
            pages=[page],
            metadata={"source": file_path, "page_count": 1, "ocr": bool(text)},
            format="image",
            page_count=1,
        )

    @staticmethod
    def _ocr_image(ocr: Any, file_path: str) -> str:
        """调用 PaddleOCR 识别单张图片，返回拼接文本。"""
        try:
            result = ocr.ocr(file_path, cls=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"OCR 识别失败 ({file_path}): {exc}")
            return ""

        lines: list[str] = []
        for page_result in result or []:
            for item in page_result or []:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    text_conf = item[1]
                    if isinstance(text_conf, (list, tuple)) and text_conf:
                        lines.append(str(text_conf[0]))
        return "\n".join(lines).strip()
