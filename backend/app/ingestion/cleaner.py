"""文档清洗流水线。

设计要点
========
1. 每个清洗规则为独立方法，可单独单元测试，便于回归与审计。
2. 规则间保持幂等与顺序无关性（normalize 在结构清洗之后执行，避免误删有效空格）。
3. 每个规则记录被清洗/移除的字符数到日志，供运维审计清洗效果，定位异常文档。
4. 清洗作用于 ``ParsedDocument`` 的页内 ``text`` 与 ``blocks`` 两处，保证
   下游无论消费哪个字段都能拿到干净文本。

清洗规则
========
- ``remove_headers_footers``：基于重复模式检测页眉页脚
- ``remove_watermark``：基于文本频率与短文本特征识别水印
- ``normalize_whitespace``：合并多余空白、修复断行
- ``filter_garbled_chars``：过滤控制字符、替换 Wingdings 等乱码
- ``normalize_punctuation``：全角/半角统一、引号统一
- ``deduplicate``：跨页去重
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from app.ingestion.parsers.base import ParsedDocument, ParsedPage, TextBlock
from app.utils.logger import logger

# 页眉页脚判定阈值：同一短文本在 > 此比例的页面顶部/底部出现即视为页眉页脚
_HEADER_FOOTER_RATIO = 0.5
# 页眉页脚候选最大行数：只考察每页顶部/底部 N 行
_HEADER_FOOTER_SCAN_LINES = 3
# 水印候选最大长度：水印通常为短句
_WATERMARK_MAX_LEN = 30
# 水印出现页数阈值：在多数页面重复出现的短文本疑似水印
_WATERMARK_MIN_PAGES_RATIO = 0.5

# 控制字符（保留换行与制表符）
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 常见 Wingdings / 私有区乱码映射
_GARBLED_MAP = {
    "\uf0b2": "•",  # Wingdings 项目符号
    "\uf0a7": "▪",
    "\uf0d8": "→",
    "\uf0fc": "•",
    "\uf0a8": "○",
    "\uf0e3": "—",
}
_GARBLED_PATTERN = re.compile(
    "[" + re.escape("".join(_GARBLED_MAP.keys())) + "]"
)
# 私有区字符（U+E000–U+F8FF）整体视为可疑乱码
_PRIVATE_USE_PATTERN = re.compile(r"[\ue000-\uf8ff]")


class DocumentCleaner:
    """文档清洗器。

    通过组合多个原子清洗规则，去除页眉页脚、水印、乱码等噪声，提升后续
    检索质量。每个规则独立可测，``clean`` 为统一编排入口。
    """

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def clean(self, parsed_doc: ParsedDocument) -> ParsedDocument:
        """清洗主入口，按固定顺序串联所有规则。

        顺序设计：先做结构级清洗（页眉页脚、水印、跨页去重），再做字符级
        清洗（空白、乱码、标点），避免结构清洗误删被字符清洗折叠后的内容。
        """
        total_pages = len(parsed_doc.pages)
        logger.info(f"开始清洗文档: format={parsed_doc.format}, 页数={total_pages}")

        cleaned_pages = self.remove_watermark(parsed_doc.pages)
        cleaned_pages = self.remove_headers_footers(cleaned_pages)
        cleaned_pages = self.deduplicate(cleaned_pages)

        for page in cleaned_pages:
            page.text = self.normalize_whitespace(page.text)
            page.text = self.filter_garbled_chars(page.text)
            page.text = self.normalize_punctuation(page.text)
            for block in page.blocks:
                block.text = self.normalize_whitespace(block.text)
                block.text = self.filter_garbled_chars(block.text)
                block.text = self.normalize_punctuation(block.text)
            for table in page.tables:
                table.markdown = self.filter_garbled_chars(table.markdown)

        parsed_doc.pages = cleaned_pages
        parsed_doc.page_count = len(cleaned_pages)
        logger.info(f"文档清洗完成: 剩余页数={len(cleaned_pages)}")
        return parsed_doc

    # ------------------------------------------------------------------
    # 结构级清洗
    # ------------------------------------------------------------------

    def remove_headers_footers(self, pages: list[ParsedPage]) -> list[ParsedPage]:
        """检测并移除页眉页脚。

        策略：统计每页顶部 N 行与底部 N 行的短文本出现频次，若某文本在超过
        ``_HEADER_FOOTER_RATIO`` 比例的页面顶部/底部出现，则视为页眉/页脚，
        从所有页面中剔除。阈值依据企业文档经验设定，可按需调整。
        """
        if not pages:
            return pages
        threshold = max(2, int(len(pages) * _HEADER_FOOTER_RATIO))

        top_counter: Counter[str] = Counter()
        bottom_counter: Counter[str] = Counter()
        for page in pages:
            top_lines = self._head_lines(page.text, _HEADER_FOOTER_SCAN_LINES)
            bottom_lines = self._tail_lines(page.text, _HEADER_FOOTER_SCAN_LINES)
            for ln in top_lines:
                key = ln.strip()
                if key and len(key) <= _WATERMARK_MAX_LEN:
                    top_counter[key] += 1
            for ln in bottom_lines:
                key = ln.strip()
                if key and len(key) <= _WATERMARK_MAX_LEN:
                    bottom_counter[key] += 1

        headers = {t for t, c in top_counter.items() if c >= threshold}
        footers = {t for t, c in bottom_counter.items() if c >= threshold}
        removed = 0
        for page in pages:
            removed += self._strip_lines(page, headers, footers)
        if headers or footers:
            logger.info(
                f"移除页眉页脚: 页眉 {len(headers)} 条, 页脚 {len(footers)} 条, "
                f"共剔除 {removed} 处"
            )
        return pages

    def remove_watermark(self, pages: list[ParsedPage]) -> list[ParsedPage]:
        """识别并移除水印文本。

        策略：水印通常为在多数页面重复出现的短句，且往往独立成行。统计所有
        短行（<= _WATERMARK_MAX_LEN）的出现页数，超过阈值的视为水印并移除。
        相比基于透明度特征的图像水印检测，文本水印用频次统计更直接可靠。
        """
        if not pages:
            return pages
        threshold = max(2, int(len(pages) * _WATERMARK_MIN_PAGES_RATIO))
        line_page_count: Counter[str] = Counter()
        for page in pages:
            seen = set()
            for ln in page.text.splitlines():
                key = ln.strip()
                if key and len(key) <= _WATERMARK_MAX_LEN and key not in seen:
                    seen.add(key)
                    line_page_count[key] += 1

        watermarks = {t for t, c in line_page_count.items() if c >= threshold}
        removed = 0
        for page in pages:
            before = len(page.text)
            new_lines = [
                ln for ln in page.text.splitlines()
                if ln.strip() not in watermarks
            ]
            page.text = "\n".join(new_lines).strip()
            page.blocks = [
                b for b in page.blocks
                if b.text.strip() not in watermarks
            ]
            removed += before - len(page.text)
        if watermarks:
            logger.info(f"移除水印: {len(watermarks)} 条, 共剔除 {removed} 字符")
        return pages

    def deduplicate(self, pages: list[ParsedPage]) -> list[ParsedPage]:
        """跨页去重：同一段落在多页重复时只保留首次出现。

        场景：企业文档存在跨页重复的免责声明、版权信息等。以归一化后的段落
        为去重键，保留首次出现，后续命中即剔除，并记录去重条数。
        """
        seen_paragraphs: set[str] = set()
        removed_count = 0
        for page in pages:
            new_blocks: list[TextBlock] = []
            for block in page.blocks:
                key = re.sub(r"\s+", "", block.text).strip().lower()
                if not key:
                    continue
                if key in seen_paragraphs and len(key) > 20:
                    removed_count += 1
                    continue
                seen_paragraphs.add(key)
                new_blocks.append(block)
            page.blocks = new_blocks
            page.text = "\n".join(b.text for b in page.blocks)
        if removed_count:
            logger.info(f"跨页去重: 移除重复段落 {removed_count} 处")
        return pages

    # ------------------------------------------------------------------
    # 字符级清洗
    # ------------------------------------------------------------------

    def normalize_whitespace(self, text: str) -> str:
        """合并多余空白、修复断行。

        - 连续空格合并为单个
        - 连续空行合并为单个
        - 修复中文段落内被错误换行符打断的情况（行末非标点且下一行首为中文字符）
        """
        if not text:
            return text
        original_len = len(text)
        # 合并水平空白
        text = re.sub(r"[ \t]+", " ", text)
        # 修复中文断行：行末非标点、次行首为 CJK 时拼接
        text = re.sub(
            r"(?<=[^\n。，；：、！？\)\）」』])\n(?=[\u4e00-\u9fff])",
            "",
            text,
        )
        # 合并连续空行
        text = re.sub(r"\n{3,}", "\n\n", text)
        cleaned = original_len - len(text)
        if cleaned:
            logger.debug(f"空白规范化: 净调整 {cleaned} 字符")
        return text.strip()

    def filter_garbled_chars(self, text: str) -> str:
        """过滤控制字符、替换常见乱码（Wingdings 等）。

        私有区字符（U+E000–U+F8FF）若无映射则删除，避免污染向量空间。
        """
        if not text:
            return text
        original_len = len(text)
        # 替换已知乱码
        text = _GARBLED_PATTERN.sub(lambda m: _GARBLED_MAP[m.group(0)], text)
        # 删除控制字符
        text = _CONTROL_CHARS.sub("", text)
        # 删除未映射的私有区字符
        text = _PRIVATE_USE_PATTERN.sub("", text)
        # 用 NFKC 规范化兼容等价字符
        text = unicodedata.normalize("NFKC", text)
        cleaned = original_len - len(text)
        if cleaned:
            logger.debug(f"乱码过滤: 移除 {cleaned} 字符")
        return text

    def normalize_punctuation(self, text: str) -> str:
        """全角/半角统一、引号统一。

        将常见全角标点统一为半角（逗号、句号除外，中文场景保留全角更自然），
        引号统一为直引号，避免同一文档出现多种引号风格干扰检索匹配。
        """
        if not text:
            return text
        table = {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "「": '"',
            "」": '"',
            "『": '"',
            "』": '"',
            "（": "(",
            "）": ")",
            "【": "[",
            "】": "]",
            "；": ";",
            "：": ":",
            "！": "!",
            "？": "?",
            "，": ",",  # 中文逗号统一半角，便于跨语言检索
            "。": ".",  # 句号统一半角
        }
        result = text.translate(str.maketrans(table))
        # 合并重复标点（如 。。。。）为单个
        result = re.sub(r"([.!?,;:])\1{2,}", r"\1", result)
        cleaned = len(text) - len(result)
        if cleaned:
            logger.debug(f"标点规范化: 净调整 {cleaned} 字符")
        return result

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _head_lines(text: str, n: int) -> list[str]:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return lines[:n]

    @staticmethod
    def _tail_lines(text: str, n: int) -> list[str]:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return lines[-n:] if n else []

    @staticmethod
    def _strip_lines(page: ParsedPage, headers: set[str], footers: set[str]) -> int:
        """从页面文本与块中剔除页眉页脚行，返回剔除条数。"""
        removed = 0
        new_lines: list[str] = []
        for ln in page.text.splitlines():
            key = ln.strip()
            if key in headers or key in footers:
                removed += 1
                continue
            new_lines.append(ln)
        page.text = "\n".join(new_lines).strip()
        page.blocks = [
            b for b in page.blocks if b.text.strip() not in headers and b.text.strip() not in footers
        ]
        return removed
