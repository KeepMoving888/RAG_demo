"""文档摄入（Ingestion）模块。

聚合导出文档摄入流水线的核心组件：
- ``DocumentPipeline``：API 同步入口（提交/查询/重试/撤销）
- ``SemanticChunker``：语义分块器（沿标题层级切分 + 父子关系）
- ``DocumentCleaner``：清洗流水线（页眉页脚/水印/乱码/去重）
- ``ParserRegistry``：解析器路由中心

调用方只需 ``from app.ingestion import DocumentPipeline`` 即可完成文档提交，
内部由 Celery 任务异步驱动 解析→清洗→分块→向量化→落库 全链路。
"""
from app.ingestion.chunker import Chunk, SemanticChunker
from app.ingestion.cleaner import DocumentCleaner
from app.ingestion.embedder import BGEM3Embedder, embedder
from app.ingestion.parsers import (
    BaseParser,
    DocxParser,
    ImageParser,
    PDFParser,
    ParsedDocument,
    ParsedPage,
    ParserError,
    ParserRegistry,
    parser_registry,
)
from app.ingestion.pipeline import DocumentPipeline

__all__ = [
    "BGEM3Embedder",
    "BaseParser",
    "Chunk",
    "DocxParser",
    "DocumentCleaner",
    "DocumentPipeline",
    "ImageParser",
    "PDFParser",
    "ParsedDocument",
    "ParsedPage",
    "ParserError",
    "ParserRegistry",
    "SemanticChunker",
    "embedder",
    "parser_registry",
]
