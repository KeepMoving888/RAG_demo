"""PDF 解析器：PyMuPDF 抽取文本/表格，PaddleOCR 兜底扫描件。

设计要点
========
1. **文本层优先**：文本层完整的 PDF 用 PyMuPDF ``page.get_text("dict")``
   抽取，保留字体大小、bbox，用于推断标题层级（font_size > 14 视为 H1，
   > 12 视为 H2，加粗且 > 10 视为 H3）。
2. **表格结构化抽取**：用 ``page.find_tables()`` 抽取并转为 markdown 格式，
   保留行列结构，便于下游独立分块。
3. **OCR 懒加载兜底**：扫描件或图片页（文本层为空 / 字符密度过低）触发
   PaddleOCR 兜底。懒加载的核心动机：绝大多数企业 PDF 都带文本层，
   预先 import PaddleOCR 会引入数百 MB 依赖并拖慢冷启动，仅在真正需要时
   才加载，可显著降低常规文档的处理开销。
4. **优雅降级**：PaddleOCR 不可用时记录 warning 并返回空文本而非崩溃，
   保证流水线对缺依赖环境仍可运行（仅丢失 OCR 能力）。
"""

from __future__ import annotations

import os
from typing import Any

from app.ingestion.parsers.base import (
    BaseParser,
    ImageBlock,
    ParsedDocument,
    ParsedPage,
    ParserError,
    TableBlock,
    TextBlock,
)
from app.utils.logger import logger

# 标题字号阈值：依据企业文档常见排版经验设定，可按需调整
_HEADING_FONT_H1 = 14.0
_HEADING_FONT_H2 = 12.0
_HEADING_FONT_H3 = 10.0
# 文本层字符密度阈值（字符/点²），低于此值判定为图片页需 OCR 兜底
_OCR_DENSITY_THRESHOLD = 0.01


class PDFParser(BaseParser):
    """PDF 解析器。

    解析流程：逐页抽取文本块与表格 → 推断标题层级 → 检测图片页触发 OCR 兜底 →
    组装 ``ParsedDocument``。所有依赖（PyMuPDF / PaddleOCR）懒加载，避免无该
    场景时的导入开销与潜在 ImportError。
    """

    def __init__(self) -> None:
        # 运行期才填充，避免构造期触发重依赖加载
        self._fitz_module: Any = None
        self._ocr_engine: Any = None

    # ------------------------------------------------------------------
    # 依赖懒加载
    # ------------------------------------------------------------------

    def _load_fitz(self) -> Any:
        """懒加载 PyMuPDF（fitz）。

        懒加载原因：PyMuPDF 为重依赖，部分容器镜像可能未预装；
        仅在真正解析 PDF 时加载，使模块本身可被安全 import。
        """
        if self._fitz_module is None:
            try:
                import fitz  # type: ignore[import-not-found]
            except ImportError as exc:  # 依赖缺失属可恢复的配置问题
                raise ParserError(
                    "PyMuPDF(fitz) 未安装，无法解析 PDF，请执行 pip install pymupdf"
                ) from exc
            self._fitz_module = fitz
            logger.info("PyMuPDF 已加载")
        return self._fitz_module

    def _load_ocr(self) -> Any | None:
        """懒加载 PaddleOCR 引擎。

        返回 None 表示 OCR 不可用。懒加载的核心收益：绝大多数 PDF 都有
        文本层，提前加载 PaddleOCR（含模型初始化）会带来秒级开销与数百 MB
        内存占用，仅在遇到扫描页时才加载可显著优化常规路径性能。
        """
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR  # type: ignore[import-not-found]
            except ImportError:
                logger.warning("PaddleOCR 未安装，扫描页/图片页将无法 OCR，返回空文本")
                return None
            try:
                self._ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
                logger.info("PaddleOCR 引擎已加载")
            except Exception as exc:  # noqa: BLE001 模型加载失败不应崩溃
                logger.warning(f"PaddleOCR 初始化失败，OCR 降级关闭: {exc}")
                return None
        return self._ocr_engine

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    async def parse(self, file_path: str) -> ParsedDocument:
        """解析 PDF 文档。

        Args:
            file_path: PDF 文件绝对路径。

        Returns:
            ``ParsedDocument``，每页含文本块、表格与（必要时）OCR 文本。

        Raises:
            ParserError: 文件不存在或 PyMuPDF 不可用。
        """
        if not os.path.exists(file_path):
            raise ParserError(f"PDF 文件不存在: {file_path}")

        fitz = self._load_fitz()
        try:
            doc = fitz.open(file_path)
        except Exception as exc:  # noqa: BLE001 PyMuPDF 内部错误统一转 ParserError
            raise ParserError(f"打开 PDF 失败: {file_path} ({exc})") from exc

        pages: list[ParsedPage] = []
        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                pages.append(self._parse_page(page, page_index + 1))
        finally:
            doc.close()

        logger.info(f"PDF 解析完成: {file_path}, 共 {len(pages)} 页")
        return ParsedDocument(
            pages=pages,
            metadata={"source": file_path, "page_count": len(pages)},
            format="pdf",
            page_count=len(pages),
        )

    # ------------------------------------------------------------------
    # 单页解析
    # ------------------------------------------------------------------

    def _parse_page(self, page: Any, page_num: int) -> ParsedPage:
        """解析单页：抽取文本块与表格，必要时 OCR 兜底。"""
        text_blocks = self._extract_text_blocks(page)
        tables = self._extract_tables(page)

        # 计算文本层字符密度，决定是否 OCR 兜底
        page_text = " ".join(b.text for b in text_blocks)
        char_count = len(page_text.strip())
        page_area = self._page_area(page)
        density = char_count / page_area if page_area > 0 else 0.0

        images: list[ImageBlock] = []
        if char_count == 0 or density < _OCR_DENSITY_THRESHOLD:
            ocr_text = self._ocr_page(page)
            if ocr_text:
                images.append(ImageBlock(ocr_text=ocr_text))
                if char_count == 0:
                    # 无文本层时用 OCR 文本填充 page.text，保证下游可用
                    page_text = ocr_text
                    if not text_blocks:
                        text_blocks.append(TextBlock(text=ocr_text))

        return ParsedPage(
            page_num=page_num,
            text=page_text,
            blocks=text_blocks,
            tables=tables,
            images=images,
        )

    def _extract_text_blocks(self, page: Any) -> list[TextBlock]:
        """用 ``page.get_text("dict")`` 抽取文本块，保留字体与 bbox。

        PyMuPDF 的 dict 结构为 block -> line -> span，span 才携带 font_size；
        这里将同一行的 spans 合并为一个 TextBlock，保留行内最大字号与 bbox 并集。
        """
        text_blocks: list[TextBlock] = []
        try:
            page_dict = page.get_text("dict")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"抽取文本块失败 (page={page}): {exc}")
            return text_blocks

        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:  # 0=文本块, 1=图片块
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = "".join(s.get("text", "") for s in spans)
                if not line_text.strip():
                    continue
                font_sizes = [s.get("size", 0.0) for s in spans]
                font_size = max(font_sizes) if font_sizes else 0.0
                bold = any("bold" in str(s.get("font", "")).lower() for s in spans)
                bbox = tuple(line.get("bbox", (0.0, 0.0, 0.0, 0.0)))  # type: ignore[arg-type]
                heading_level = self._detect_heading_level(font_size, bold)
                text_blocks.append(
                    TextBlock(
                        text=line_text,
                        bbox=bbox,
                        font_size=font_size,
                        bold=bold,
                        heading_level=heading_level,
                    )
                )
        return text_blocks

    def _extract_tables(self, page: Any) -> list[TableBlock]:
        """抽取页面表格并转为 markdown。

        使用 ``page.find_tables()`` 识别表格区域，``table.extract()`` 取出
        二维数据后转为 markdown 表格字符串，保留表头分隔行。
        """
        tables_out: list[TableBlock] = []
        try:
            finder = page.find_tables()
        except Exception as exc:  # noqa: BLE001 部分页面 find_tables 可能异常
            logger.debug(f"表格抽取跳过 (page={page}): {exc}")
            return tables_out

        for table in getattr(finder, "tables", []) or []:
            try:
                rows = table.extract()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"表格提取失败: {exc}")
                continue
            if not rows:
                continue
            tables_out.append(
                TableBlock(
                    markdown=self._rows_to_markdown(rows),
                    bbox=tuple(getattr(table, "bbox", (0.0, 0.0, 0.0, 0.0))),  # type: ignore[arg-type]
                    rows=len(rows),
                    cols=max((len(r) for r in rows), default=0),
                )
            )
        return tables_out

    # ------------------------------------------------------------------
    # OCR 兜底
    # ------------------------------------------------------------------

    def _ocr_page(self, page: Any) -> str:
        """对页面渲染后做 OCR，返回识别文本。

        若 OCR 引擎不可用则返回空串。渲染为图片再识别，保证扫描页与
        混合页均可处理。
        """
        ocr = self._load_ocr()
        if ocr is None:
            return ""
        try:
            fitz = self._fitz_module
            pix = page.get_pixmap(dpi=200)
            import io

            from PIL import Image  # type: ignore[import-not-found]

            img = Image.open(io.BytesIO(pix.tobytes("png")))
            result = ocr.ocr(img, cls=True)
        except Exception as exc:  # noqa: BLE001 OCR 失败不应中断整页
            logger.warning(f"OCR 处理失败 (page={page}): {exc}")
            return ""

        lines: list[str] = []
        for page_result in result or []:
            for item in page_result or []:
                # item 形如 [box, (text, confidence)]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    text_conf = item[1]
                    if isinstance(text_conf, (list, tuple)) and text_conf:
                        lines.append(str(text_conf[0]))
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _detect_heading_level(self, font_size: float, bold: bool) -> int:
        """根据字号与加粗推断标题层级。

        规则：font_size > 14 → H1；> 12 → H2；加粗且 > 10 → H3；其余为正文。
        字号阈值依据企业文档常见排版设定，对于扫描 OCR 文本（无字号信息）
        会返回 0，交由分块器按段落处理。
        """
        if font_size > _HEADING_FONT_H1:
            return 1
        if font_size > _HEADING_FONT_H2:
            return 2
        if bold and font_size > _HEADING_FONT_H3:
            return 3
        return 0

    @staticmethod
    def _rows_to_markdown(rows: list[list[Any]]) -> str:
        """将二维数据转为 markdown 表格字符串。"""
        if not rows:
            return ""
        normalized = [
            [
                ("" if cell is None else str(cell)).replace("|", "\\|").replace("\n", " ")
                for cell in row
            ]
            or [""] * max((len(r) for r in rows), default=1)
            for row in rows
        ]
        col_count = max(len(r) for r in normalized)
        for row in normalized:
            while len(row) < col_count:
                row.append("")
        header = normalized[0]
        separator = ["---"] * col_count
        body = normalized[1:] if len(normalized) > 1 else []
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    @staticmethod
    def _page_area(page: Any) -> float:
        """计算页面面积（点²），用于字符密度判定。"""
        try:
            rect = page.rect
            return float(rect.width * rect.height)
        except Exception:  # noqa: BLE001
            return 0.0
