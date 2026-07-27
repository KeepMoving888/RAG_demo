"""
GraphRAG 知识图谱增强检索模块

用途:
    为 Enterprise RAG Knowledge Base 提供基于 Neo4j 知识图谱的增强检索能力,
    在向量语义检索之外补充实体关系链路证据, 提升多跳推理问题的召回质量.

子模块:
    - schemas:       图谱 Schema (实体/关系类型枚举 + 数据结构)
    - extractor:     LLM 实体关系抽取 (含离线正则降级 + 跨 chunk 去重)
    - neo4j_store:   Neo4j 异步存储 (Schema 初始化 / 幂等写入 / 多跳查询)
    - cypher_chain:  自然语言转 Cypher 查询链 (含严格只读安全校验)
    - extractor_tasks: Celery 异步抽取任务 (队列 graphrag)
    - fusion:        向量 + 图谱双路融合检索 (路由策略 + 去重合并)

核心契约:
    - neo4j_store 单例在 app.main lifespan 中被 await neo4j_store.init_schema();
    - GraphFusion.fuse() 返回的 chunks 与 vector_results 同结构, 额外可能
      携带 source 字段 ("graph" / "graph+vector");
    - GraphCypherQAChain.query() 返回 {cypher, result_text, records, latency_ms}.
"""

from app.graphrag.cypher_chain import GraphCypherQAChain
from app.graphrag.extractor import EntityExtractor
from app.graphrag.fusion import GraphFusion
from app.graphrag.neo4j_store import Neo4jStore, neo4j_store
from app.graphrag.schemas import (
    ENTITY_LABELS_ZH,
    RELATION_LABELS_ZH,
    Entity,
    EntityType,
    ExtractionResult,
    Relation,
    RelationType,
)

__all__ = [
    # Schema
    "EntityType",
    "RelationType",
    "ENTITY_LABELS_ZH",
    "RELATION_LABELS_ZH",
    "Entity",
    "Relation",
    "ExtractionResult",
    # 组件
    "EntityExtractor",
    "Neo4jStore",
    "neo4j_store",
    "GraphCypherQAChain",
    "GraphFusion",
]
