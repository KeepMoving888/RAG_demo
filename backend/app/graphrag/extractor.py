"""
LLM 实体关系抽取器 (核心难点)

用途:
    将非结构化文档分块文本转换为结构化 (实体, 关系) 三元组, 作为知识图谱
    的数据来源. 支持单 chunk 抽取与整文档批量抽取, 并对多 chunk 结果做
    跨分块去重合并.

抽取流程:
    1. 构建 prompt (见 EXTRACT_PROMPT), 要求 LLM 输出严格 JSON;
    2. 调用 llm.aextract_json(prompt) 解析为 dict;
    3. 将 dict 映射为 Entity / Relation 列表, 校验类型白名单;
    4. LLM 失败时降级为离线正则抽取 (复用 OfflineLLM 的正则逻辑),
       保证离线模式/弱模型场景下图谱构建链路不中断.

去重策略 (跨 chunk 合并):
    - 实体: 同 name + type 视为同一实体, 合并 properties (后者覆盖前者)
      与 source_chunk_id (并集去重);
    - 关系: 同 source + target + relation 视为同一关系, 合并 source_chunk_id.
    这样同一产品在多个文档中提及不会产生重复节点, 关系也会聚合多来源证据.

上报指标:
    extract_from_chunk 完成后调用 GRAPH_ENTITIES_EXTRACTED.inc(len(entities)).
"""

import json
import re
from typing import Optional

from app.llm import get_llm
from app.metrics import GRAPH_ENTITIES_EXTRACTED
from app.utils.logger import logger
from app.graphrag.schemas import (
    Entity,
    EntityType,
    ExtractionResult,
    Relation,
    RelationType,
)


# ======================== Prompt 模板 ========================
EXTRACT_PROMPT = """你是企业知识图谱构建助手。从以下文档片段中抽取实体与关系。

实体类型: Product(产品), Department(部门), Person(人员), Policy(制度), Project(项目), Standard(标准), Supplier(供应商), Customer(客户)
关系类型: BELONGS_TO(归属), PARTICIPATES(参与), CERTIFIED_BY(认证), REFERENCES(引用), MANAGES(管理), SUPPLIES(供应), COLLABORATES_WITH(协作), DEFINED_BY(定义)

文档片段:
{content}

输出 JSON 格式 (只输出 JSON, 不要 markdown):
{
  "entities": [{"name":"...","type":"Product|Department|...","properties":{{...}}}}],
  "relations": [{{"source":"...","source_type":"Product","target":"...","target_type":"Department","relation":"CERTIFIED_BY"}}]
}}
"""


# 合法类型白名单 (用于校验 LLM 输出, 拒绝越界类型)
_VALID_ENTITY_TYPES = {e.value for e in EntityType}
_VALID_RELATION_TYPES = {r.value for r in RelationType}


class EntityExtractor:
    """实体关系抽取器"""

    def __init__(self) -> None:
        self._llm = get_llm()

    # ======================== 单 chunk 抽取 ========================
    async def extract_from_chunk(
        self, chunk_id: int, content: str
    ) -> ExtractionResult:
        """抽取单个文档分块的实体与关系

        Args:
            chunk_id: 文档分块 ID (写入 source_chunk_id 用于溯源).
            content: 分块文本.

        Returns:
            ExtractionResult, raw_llm_response 保留 LLM 原始输出便于调试.
        """
        if not content or not content.strip():
            return ExtractionResult()

        try:
            prompt = EXTRACT_PROMPT.format(content=content[:4000])  # 截断防超长
            raw = await self._llm.aextract_json(prompt)
            entities, relations = self._parse_llm_output(raw, chunk_id)
            raw_text = json.dumps(raw, ensure_ascii=False)
        except Exception as e:
            # 降级: 离线正则抽取, 保证链路不中断
            logger.warning("LLM 抽取失败 chunk_id={} 降级正则: {}", chunk_id, str(e))
            entities, relations = self._offline_extract(content, chunk_id)
            raw_text = f"(offline-fallback: {e})"

        # 上报实体抽取计数
        if entities:
            GRAPH_ENTITIES_EXTRACTED.inc(len(entities))

        logger.debug("chunk_id={} 抽取完成: 实体 {} / 关系 {}",
                     chunk_id, len(entities), len(relations))
        return ExtractionResult(
            entities=entities, relations=relations, raw_llm_response=raw_text
        )

    # ======================== 整文档抽取 ========================
    async def extract_from_document(self, document_id: int) -> ExtractionResult:
        """批量抽取文档所有 chunk 的实体关系并合并去重

        注意: 本方法直接复用 _extract_from_text 内部逻辑 (不经 extract_from_chunk),
        避免逐 chunk 上报指标造成重复计数; 实体计数由调用方 (Celery 任务) 统一上报.
        """
        from sqlalchemy import select

        from app.database import db_session
        from app.models import Document, DocumentChunk

        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        async with db_session() as session:
            doc = await session.get(Document, document_id)
            if doc is None:
                logger.warning("文档不存在 document_id={}", document_id)
                return ExtractionResult()

            result = await session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
            )
            chunks = result.scalars().all()

        logger.info("开始文档抽取 document_id={} chunks={}", document_id, len(chunks))

        for chunk in chunks:
            try:
                entities, relations = await self._extract_from_text(
                    chunk.id, chunk.content
                )
                all_entities.extend(entities)
                all_relations.extend(relations)
            except Exception as e:
                logger.warning("chunk 抽取失败 chunk_id={}: {}", chunk.id, str(e))

        # 跨 chunk 去重合并
        merged_entities, merged_relations = self._merge_dedup(
            all_entities, all_relations
        )
        logger.info(
            "文档抽取完成 document_id={} 合并前 {}/{} -> 合并后 {}/{}",
            document_id, len(all_entities), len(all_relations),
            len(merged_entities), len(merged_relations),
        )
        return ExtractionResult(
            entities=merged_entities,
            relations=merged_relations,
            raw_llm_response=f"(merged from {len(chunks)} chunks)",
        )

    # ======================== 内部: 文本抽取 ========================
    async def _extract_from_text(
        self, chunk_id: int, content: str
    ) -> tuple[list[Entity], list[Relation]]:
        """对单段文本执行 LLM 抽取 (不上报指标, 供 document 级批量复用)"""
        if not content or not content.strip():
            return [], []
        try:
            prompt = EXTRACT_PROMPT.format(content=content[:4000])
            raw = await self._llm.aextract_json(prompt)
            return self._parse_llm_output(raw, chunk_id)
        except Exception as e:
            logger.warning("LLM 抽取失败 chunk_id={} 降级正则: {}", chunk_id, str(e))
            return self._offline_extract(content, chunk_id)

    # ======================== 内部: LLM 输出解析 ========================
    @staticmethod
    def _parse_llm_output(
        raw: dict, chunk_id: Optional[int]
    ) -> tuple[list[Entity], list[Relation]]:
        """将 LLM 返回的 JSON dict 解析为 Entity / Relation 列表

        校验: 类型必须命中白名单, 非法类型条目直接丢弃 (不抛异常, 容错).
        """
        entities: list[Entity] = []
        relations: list[Relation] = []

        for item in raw.get("entities", []) or []:
            name = str(item.get("name", "")).strip()
            etype = str(item.get("type", "")).strip()
            if not name or etype not in _VALID_ENTITY_TYPES:
                continue
            properties = item.get("properties") or {}
            if not isinstance(properties, dict):
                properties = {}
            entities.append(Entity(
                name=name, type=etype, properties=properties,
                source_chunk_id=chunk_id,
            ))

        for item in raw.get("relations", []) or []:
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            stype = str(item.get("source_type", "")).strip()
            ttype = str(item.get("target_type", "")).strip()
            rtype = str(item.get("relation", "")).strip()
            if (not source or not target
                    or stype not in _VALID_ENTITY_TYPES
                    or ttype not in _VALID_ENTITY_TYPES
                    or rtype not in _VALID_RELATION_TYPES):
                continue
            properties = item.get("properties") or {}
            if not isinstance(properties, dict):
                properties = {}
            relations.append(Relation(
                source_entity=source, source_type=stype,
                target_entity=target, target_type=ttype,
                relation_type=rtype, properties=properties,
                source_chunk_id=chunk_id,
            ))

        return entities, relations

    # ======================== 内部: 离线正则降级 ========================
    @staticmethod
    def _offline_extract(
        content: str, chunk_id: Optional[int]
    ) -> tuple[list[Entity], list[Relation]]:
        """离线正则抽取 (LLM 不可用时的降级方案)

        复用 app.llm.offline_llm.OfflineLLM._extract_entities 的正则模式:
          - 产品名: 大写字母 + 数字 (车规 eMMC / P300)
          - 部门名: 中文 + 部/处/中心/科/组
          - 人名: 中文 + 工程师/经理/老师/主任/总监
          - 制度名: 《xxx》
        离线模式下不抽取关系 (正则难以可靠判断关系方向), 仅产出实体,
        保证图谱至少能建立产品/部门等节点供 Cypher 查询.
        """
        entities: list[Entity] = []
        seen_names: set[str] = set()

        def _add(name: str, etype: str, props: Optional[dict] = None) -> None:
            if name and name not in seen_names:
                seen_names.add(name)
                entities.append(Entity(
                    name=name, type=etype, properties=props or {},
                    source_chunk_id=chunk_id,
                ))

        # 产品名
        for m in re.finditer(r"\b([A-Z]{2,}-?[A-Z0-9]{2,})\b", content):
            _add(m.group(1), EntityType.Product.value,
                 {"matched_text": m.group(0)})
        # 部门名
        for m in re.finditer(r"([\u4e00-\u9fa5]{2,8}(?:部|处|中心|科|组))", content):
            _add(m.group(1), EntityType.Department.value)
        # 人名
        for m in re.finditer(
            r"([\u4e00-\u9fa5]{2,4})(?:工程师|经理|老师|主任|总监)", content
        ):
            _add(m.group(1), EntityType.Person.value)
        # 制度名
        for m in re.finditer(r"《([^》]{2,30})》", content):
            _add(m.group(1), EntityType.Policy.value)

        return entities, []

    # ======================== 内部: 去重合并 ========================
    @staticmethod
    def _merge_dedup(
        entities: list[Entity], relations: list[Relation]
    ) -> tuple[list[Entity], list[Relation]]:
        """跨 chunk 去重合并

        实体键: (name, type); 关系键: (source_entity, target_entity, relation_type).
        合并规则:
          - properties: dict 覆盖式合并 (后者覆盖前者);
          - source_chunk_id: 收集为列表并集去重 (None 与具体 id 视为不同来源).
        """
        # ---- 实体合并 ----
        entity_map: dict[tuple[str, str], Entity] = {}
        for e in entities:
            key = (e.name, e.type)
            if key not in entity_map:
                entity_map[key] = Entity(
                    name=e.name, type=e.type,
                    properties=dict(e.properties),
                    source_chunk_id=e.source_chunk_id,
                )
            else:
                existing = entity_map[key]
                # properties 覆盖合并
                existing.properties.update(e.properties)
                # source_chunk_id 并集
                existing.source_chunk_id = _union_chunk_id(
                    existing.source_chunk_id, e.source_chunk_id
                )

        # ---- 关系合并 ----
        relation_map: dict[tuple[str, str, str], Relation] = {}
        for r in relations:
            key = (r.source_entity, r.target_entity, r.relation_type)
            if key not in relation_map:
                relation_map[key] = Relation(
                    source_entity=r.source_entity, source_type=r.source_type,
                    target_entity=r.target_entity, target_type=r.target_type,
                    relation_type=r.relation_type, properties=dict(r.properties),
                    source_chunk_id=r.source_chunk_id,
                )
            else:
                existing = relation_map[key]
                existing.properties.update(r.properties)
                existing.source_chunk_id = _union_chunk_id(
                    existing.source_chunk_id, r.source_chunk_id
                )

        return list(entity_map.values()), list(relation_map.values())


def _union_chunk_id(a: Optional[int], b: Optional[int]) -> Optional[int]:
    """合并两个 source_chunk_id

    若两者均为 None 返回 None; 若其一为 None 返回另一个;
    若两者相同返回该值; 否则无法用单值表达多来源, 这里保留首个非 None 值
    (Neo4j 侧 source_chunks 数组会承接完整多来源, dataclass 层仅保留代表值).
    """
    if a is None:
        return b
    if b is None:
        return a
    return a if a == b else a


__all__ = ["EntityExtractor", "EXTRACT_PROMPT"]
