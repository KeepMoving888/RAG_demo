"""
Neo4j 图谱存储层 (核心)

用途:
    封装企业知识图谱的持久化与查询, 提供实体/关系的幂等写入、多跳邻居、
    最短路径、按来源 chunk 清理等能力, 供 extractor / cypher_chain /
    fusion 子模块调用.

为何选择 Neo4j 而非 networkx:
    1. 企业级: Neo4j 提供 ACID 事务、唯一约束、细粒度索引与 Bolt 协议,
       支撑生产级并发写入; networkx 是纯内存图, 重启即失, 无法跨进程共享.
    2. 多跳查询性能: Neo4j 原生图遍历引擎对 N 跳邻居、最短路径做了
       索引加速 (变量长度关系模式 MATCH -[r*1..3]-), networkx 在大图上
       逐跳展开易超时.
    3. Cypher 声明式查询: 业务侧 (cypher_chain) 可用自然语言转 Cypher
       表达复杂拓扑, networkx 需手写 Python 遍历, 表达力受限.
    4. 可观测性: Neo4j 自带查询计划与统计, 便于排查慢查询.

Schema 设计:
    节点:
        label = EntityType (Product/Department/Person/Policy/Project/
                            Standard/Supplier/Customer)
        属性: name(唯一), type, source_chunks(list[int]), properties(map),
              created_at(datetime)
        约束: 每种 label 对 name 建唯一约束 (CREATE CONSTRAINT ... REQUIRE
              n.name IS UNIQUE). 唯一约束在 Neo4j 中会自动创建后备索引,
              因此无需再单独 CREATE INDEX ON (name), 重复建索引反而会冲突.
              额外在 type 属性上建索引以加速按类型过滤的统计查询.
    关系:
        type = RelationType (CERTIFIED_BY / PARTICIPATES / ...)
        属性: source_chunks(list[int]), created_at(datetime)
    溯源机制:
        source_chunks 数组记录实体/关系来自哪些文档分块, 既支撑答案溯源
        (前端高亮来源 chunk), 又支撑文档更新时按 chunk 精准清理过期图数据
        (delete_by_source_chunk).

降级策略:
    Neo4j 不可用时 (未安装驱动 / 连接失败), is_available 返回 False,
    所有写方法返回空计数、读方法返回空列表, 并记录 warning, 保证主链路
    (向量检索 + BM25) 不受图谱故障影响, 实现 GraphRAG 与主检索解耦.
"""

from typing import Any

from app.config import settings
from app.graphrag.schemas import (
    Entity,
    EntityType,
    ExtractionResult,
    Relation,
    RelationType,
)
from app.utils.logger import logger

# 延迟导入 neo4j 驱动: 若运行环境未安装 neo4j 包, 模块仍可加载,
# 图谱功能整体降级为不可用, 不影响其余子系统启动.
try:
    from neo4j import AsyncGraphDatabase
    from neo4j.graph import Node, Path, Relationship

    _NEO4J_DRIVER_AVAILABLE = True
except ImportError:  # pragma: no cover - 依赖缺失时的降级路径
    AsyncGraphDatabase = None  # type: ignore[assignment]
    Node = Relationship = Path = None  # type: ignore[assignment]
    _NEO4J_DRIVER_AVAILABLE = False


class Neo4jStore:
    """Neo4j 异步图谱存储

    所有方法均为协程, 通过 AsyncGraphDatabase 的 Bolt 异步驱动执行.
    单例实例 neo4j_store 在 app.main 的 lifespan 中被调用 init_schema().
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._driver = None
        # 连接态标记: 仅在 verify_connectivity 成功后置 True
        self._available = False

    # ======================== 连接管理 ========================
    @property
    def is_available(self) -> bool:
        """图谱是否可用 (同步属性, 供调用方快速判断是否触发降级)"""
        return self._available and self._driver is not None

    async def _ensure_driver(self) -> bool:
        """惰性创建并校验驱动, 返回是否可用"""
        if self._driver is not None:
            return self.is_available
        if not _NEO4J_DRIVER_AVAILABLE:
            logger.warning("neo4j 驱动未安装, GraphRAG 降级为离线模式")
            self._available = False
            return False
        try:
            self._driver = AsyncGraphDatabase.driver(self._uri, auth=(self._user, self._password))
            await self._driver.verify_connectivity()
            self._available = True
            logger.info("Neo4j 连接成功: {}", self._uri)
        except Exception as e:  # 连接失败不抛出, 走降级
            self._available = False
            self._driver = None
            logger.warning("Neo4j 不可用 (GraphRAG 将降级): {}", str(e))
        return self.is_available

    async def close(self) -> None:
        """关闭驱动连接池 (应用关停时调用)"""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            self._available = False

    # ======================== Schema 初始化 ========================
    async def init_schema(self) -> None:
        """创建约束与索引 (幂等)

        幂等性: 所有语句带 IF NOT EXISTS, 重复执行不报错.
        唯一约束自动建立 name 上的后备索引, 因此不再单独建 name 索引;
        另在 type 属性上建索引以加速 stats() 按类型聚合.
        """
        if not await self._ensure_driver():
            return

        # 每种实体类型的 name 唯一约束 (同时提供索引 + 去重保证)
        constraint_stmts = [
            f"CREATE CONSTRAINT {et.value.lower()}_name_unique IF NOT EXISTS "
            f"FOR (n:{et.value}) REQUIRE n.name IS UNIQUE"
            for et in EntityType
        ]
        # type 属性索引 (按类型过滤/统计加速)
        index_stmts = [
            f"CREATE INDEX {et.value.lower()}_type_idx IF NOT EXISTS FOR (n:{et.value}) ON (n.type)"
            for et in EntityType
        ]

        async with self._driver.session() as session:
            for stmt in constraint_stmts + index_stmts:
                try:
                    await session.run(stmt)
                except Exception as e:  # 单条失败不阻断其余 schema 创建
                    logger.warning("Schema 语句执行失败 [{}]: {}", stmt, str(e))
        logger.info(
            "Neo4j Schema 初始化完成 (约束 {} 条 / 索引 {} 条)",
            len(constraint_stmts),
            len(index_stmts),
        )

    # ======================== 写入 ========================
    @staticmethod
    def _validate_label(label: str, allowed: set[str]) -> str:
        """校验 label/rel-type 命中白名单, 防止 Cypher 注入"""
        if label not in allowed:
            raise ValueError(f"非法的图谱标签: {label}")
        return label

    async def upsert_entity(self, entity: Entity) -> str:
        """幂等写入实体 (MERGE), 返回节点 element_id

        MERGE 语义: name 已存在则更新, 否则创建.
        ON MATCH 合并 source_chunks (去重) 与 properties (覆盖式合并).
        """
        if not await self._ensure_driver():
            return ""
        label = self._validate_label(entity.type, {e.value for e in EntityType})
        source_chunks = [entity.source_chunk_id] if entity.source_chunk_id else []

        cypher = (
            f"MERGE (n:{label} {{name: $name}}) "
            "ON CREATE SET n.created_at = datetime(), n.type = $type, "
            "n.source_chunks = $source_chunks "
            "ON MATCH SET "
            # 列表去重合并: 先剔除已在列表中的, 再追加新来源
            "n.source_chunks = "
            "[x IN n.source_chunks WHERE NOT x IN $source_chunks] + $source_chunks "
            # properties 展开为节点顶级属性 (Neo4j 不允许嵌套 Map 作为属性值)
            "SET n += $properties "
            "RETURN elementId(n) AS id"
        )
        params = {
            "name": entity.name,
            "type": entity.type,
            "source_chunks": source_chunks,
            "properties": entity.properties or {},
        }
        async with self._driver.session() as session:
            result = await session.run(cypher, parameters=params)
            records = [r async for r in result]
        return records[0]["id"] if records else ""

    async def upsert_relation(self, relation: Relation) -> str:
        """幂等写入关系 (两端节点必须已存在), 返回关系 element_id"""
        if not await self._ensure_driver():
            return ""
        src_label = self._validate_label(relation.source_type, {e.value for e in EntityType})
        tgt_label = self._validate_label(relation.target_type, {e.value for e in EntityType})
        rel_type = self._validate_label(relation.relation_type, {r.value for r in RelationType})
        source_chunks = [relation.source_chunk_id] if relation.source_chunk_id else []

        cypher = (
            f"MATCH (a:{src_label} {{name: $source}}) "
            f"MATCH (b:{tgt_label} {{name: $target}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "ON CREATE SET r.created_at = datetime(), r.source_chunks = $source_chunks "
            "ON MATCH SET r.source_chunks = "
            "[x IN r.source_chunks WHERE NOT x IN $source_chunks] + $source_chunks "
            "RETURN elementId(r) AS id"
        )
        params = {
            "source": relation.source_entity,
            "target": relation.target_entity,
            "source_chunks": source_chunks,
        }
        async with self._driver.session() as session:
            result = await session.run(cypher, parameters=params)
            records = [r async for r in result]
        return records[0]["id"] if records else ""

    async def batch_upsert(self, extraction: ExtractionResult) -> dict:
        """批量入库抽取结果

        实体先于关系写入 (关系 MATCH 依赖两端节点存在).
        逐条 auto-commit: 允许部分成功, 已成功的不回滚, 适配 ETL 场景.

        Returns:
            {"entities_added": int, "relations_added": int}
        """
        if not await self._ensure_driver():
            logger.warning(
                "Neo4j 不可用, batch_upsert 跳过 ({} 实体 / {} 关系)",
                len(extraction.entities),
                len(extraction.relations),
            )
            return {"entities_added": 0, "relations_added": 0}

        entities_added = 0
        relations_added = 0
        # 单 session 复用连接, 逐条 auto-commit
        async with self._driver.session() as session:
            for entity in extraction.entities:
                try:
                    await self._upsert_entity_in_session(session, entity)
                    entities_added += 1
                except Exception as e:
                    logger.warning("实体入库失败 [{}]: {}", entity.name, str(e))
            for rel in extraction.relations:
                try:
                    await self._upsert_relation_in_session(session, rel)
                    relations_added += 1
                except Exception as e:
                    logger.warning(
                        "关系入库失败 [{}->{}]: {}", rel.source_entity, rel.target_entity, str(e)
                    )
        logger.info("batch_upsert 完成: 实体 {} / 关系 {}", entities_added, relations_added)
        return {"entities_added": entities_added, "relations_added": relations_added}

    async def _upsert_entity_in_session(self, session, entity: Entity) -> None:
        """在已有 session 内写入实体 (复用 batch_upsert 的连接)"""
        label = self._validate_label(entity.type, {e.value for e in EntityType})
        source_chunks = [entity.source_chunk_id] if entity.source_chunk_id else []
        cypher = (
            f"MERGE (n:{label} {{name: $name}}) "
            "ON CREATE SET n.created_at = datetime(), n.type = $type, "
            "n.source_chunks = $source_chunks "
            "ON MATCH SET n.source_chunks = "
            "[x IN n.source_chunks WHERE NOT x IN $source_chunks] + $source_chunks "
            "SET n += $properties"
        )
        await session.run(
            cypher,
            parameters={
                "name": entity.name,
                "type": entity.type,
                "source_chunks": source_chunks,
                "properties": entity.properties or {},
            },
        )

    async def _upsert_relation_in_session(self, session, relation: Relation) -> None:
        """在已有 session 内写入关系"""
        src_label = self._validate_label(relation.source_type, {e.value for e in EntityType})
        tgt_label = self._validate_label(relation.target_type, {e.value for e in EntityType})
        rel_type = self._validate_label(relation.relation_type, {r.value for r in RelationType})
        source_chunks = [relation.source_chunk_id] if relation.source_chunk_id else []
        cypher = (
            f"MATCH (a:{src_label} {{name: $source}}) "
            f"MATCH (b:{tgt_label} {{name: $target}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "ON CREATE SET r.created_at = datetime(), r.source_chunks = $source_chunks "
            "ON MATCH SET r.source_chunks = "
            "[x IN r.source_chunks WHERE NOT x IN $source_chunks] + $source_chunks"
        )
        await session.run(
            cypher,
            parameters={
                "source": relation.source_entity,
                "target": relation.target_entity,
                "source_chunks": source_chunks,
            },
        )

    # ======================== 读取 ========================
    @staticmethod
    def _serialize(value: Any) -> Any:
        """将 Neo4j 图对象 (Node/Relationship/Path) 序列化为纯 dict/list

        保证 run_cypher 返回值可直接 JSON 序列化, 前端与下游无需依赖 neo4j 类型.
        """
        if not _NEO4J_DRIVER_AVAILABLE or value is None:
            return value
        if isinstance(value, Node):
            return {
                "element_id": value.element_id,
                "labels": list(value.labels),
                "name": value.get("name"),
                "type": value.get("type"),
                "source_chunks": value.get("source_chunks", []),
                "properties": {k: v for k, v in value.items() if k not in ("name", "type")},
            }
        if isinstance(value, Relationship):
            return {
                "element_id": value.element_id,
                "type": value.type,
                "source_chunks": value.get("source_chunks", []),
                "properties": dict(value),
            }
        if isinstance(value, Path):
            return {
                "nodes": [Neo4jStore._serialize(n) for n in value.nodes],
                "relationships": [
                    {"type": r.type, "source_chunks": r.get("source_chunks", [])}
                    for r in value.relationships
                ],
            }
        return value

    async def run_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        """执行只读 Cypher, 返回序列化后的记录列表

        注意: 本方法不做安全校验 (校验职责在 cypher_chain.validate_cypher),
        调用方需确保 Cypher 来源可信.
        """
        if not await self._ensure_driver():
            return []
        try:
            async with self._driver.session() as session:
                result = await session.run(cypher, parameters=params or {})
                records = [
                    {k: self._serialize(v) for k, v in record.items()} async for record in result
                ]
            return records
        except Exception as e:
            logger.warning("Cypher 执行失败: {} | cypher={}", str(e), cypher[:200])
            return []

    async def get_entity(self, name: str, entity_type: str | None = None) -> dict | None:
        """按名称 (可选类型) 查询单个实体"""
        if not await self._ensure_driver():
            return None
        if entity_type:
            self._validate_label(entity_type, {e.value for e in EntityType})
            cypher = f"MATCH (n:{entity_type} {{name: $name}}) RETURN n LIMIT 1"
        else:
            cypher = "MATCH (n {name: $name}) RETURN n LIMIT 1"
        records = await self.run_cypher(cypher, {"name": name})
        return records[0].get("n") if records else None

    async def get_neighbors(self, name: str, hops: int = 1) -> list[dict]:
        """获取 N 跳邻居 (多跳推理用)

        hops 经整数校验并钳制到 [1, 5], 避免 LLM/调用方传入过大值导致
        全图扫描; hops 作为整数直接拼入变量长度模式 (Cypher 不支持参数化上界).
        """
        if not await self._ensure_driver():
            return []
        hops = max(1, min(int(hops), 5))
        cypher = (
            f"MATCH (n {{name: $name}})-[r*1..{hops}]-(m) "
            "WHERE m.name <> $name "
            "RETURN DISTINCT m AS neighbor, labels(m) AS labels, "
            "[rel IN r | type(rel)] AS relations LIMIT 50"
        )
        return await self.run_cypher(cypher, {"name": name})

    async def find_paths(self, source_name: str, target_name: str, max_hops: int = 3) -> list[dict]:
        """最短路径查找 (实体关系链路可解释性)

        返回 Path 序列化结果, 用于回答「A 与 B 之间是什么关系」.
        """
        if not await self._ensure_driver():
            return []
        max_hops = max(1, min(int(max_hops), 6))
        cypher = (
            f"MATCH p = shortestPath("
            f"(a {{name: $source}})-[*..{max_hops}]-(b {{name: $target}})) "
            "RETURN p LIMIT 5"
        )
        return await self.run_cypher(cypher, {"source": source_name, "target": target_name})

    async def delete_by_source_chunk(self, chunk_id: int) -> None:
        """按来源 chunk 清理图数据 (文档更新/删除时调用)

        策略:
            1. 从所有节点的 source_chunks 中移除该 chunk_id;
            2. source_chunks 清空的节点 (不再有任何来源支撑) 连同其关系 DETACH DELETE;
            3. 同理清理关系上的 source_chunks, 清空的孤立关系删除.
        这保证文档更新后图谱不会残留过期实体, 同时不误删多源共引实体.
        """
        if not await self._ensure_driver():
            return
        async with self._driver.session() as session:
            # 节点: 移除来源, 清空则删除
            try:
                await session.run(
                    "MATCH (n) WHERE $chunk_id IN n.source_chunks "
                    "SET n.source_chunks = "
                    "[x IN n.source_chunks WHERE x <> $chunk_id]",
                    parameters={"chunk_id": chunk_id},
                )
                await session.run(
                    "MATCH (n) WHERE n.source_chunks = [] DETACH DELETE n",
                    parameters={},
                )
            except Exception as e:
                logger.warning("按 chunk 清理节点失败 chunk_id={}: {}", chunk_id, str(e))
            # 关系: 移除来源, 清空则删除
            try:
                await session.run(
                    "MATCH ()-[r]->() WHERE $chunk_id IN r.source_chunks "
                    "SET r.source_chunks = "
                    "[x IN r.source_chunks WHERE x <> $chunk_id]",
                    parameters={"chunk_id": chunk_id},
                )
                await session.run(
                    "MATCH ()-[r]->() WHERE r.source_chunks = [] DELETE r",
                    parameters={},
                )
            except Exception as e:
                logger.warning("按 chunk 清理关系失败 chunk_id={}: {}", chunk_id, str(e))
        logger.info("已按来源 chunk {} 清理图谱数据", chunk_id)

    async def stats(self) -> dict:
        """图谱统计 (节点/关系总数及分类型计数)"""
        if not await self._ensure_driver():
            return {"available": False, "nodes": 0, "relationships": 0, "by_label": {}}
        node_count = await self.run_cypher("MATCH (n) RETURN count(n) AS c LIMIT 1")
        rel_count = await self.run_cypher("MATCH ()-[r]->() RETURN count(r) AS c LIMIT 1")
        by_label = await self.run_cypher(
            "MATCH (n) UNWIND labels(n) AS lbl RETURN lbl, count(*) AS c ORDER BY c DESC LIMIT 50"
        )
        return {
            "available": True,
            "nodes": node_count[0].get("c", 0) if node_count else 0,
            "relationships": rel_count[0].get("c", 0) if rel_count else 0,
            "by_label": {r.get("lbl"): r.get("c") for r in by_label},
        }


# ======================== 单例 ========================
# app.main lifespan 中通过 `from app.graphrag.neo4j_store import neo4j_store`
# 调用 await neo4j_store.init_schema()
neo4j_store = Neo4jStore()

__all__ = ["Neo4jStore", "neo4j_store"]
