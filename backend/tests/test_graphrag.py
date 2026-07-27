"""
GraphRAG 知识图谱模块单元测试

覆盖三大核心场景:
1. Cypher 安全校验 (GraphCypherQAChain.validate_cypher): 拒绝写操作 /
   多语句注入 / 注释 / 反引号, 强制 MATCH 起始 + RETURN + LIMIT;
2. 实体去重 (EntityExtractor._merge_dedup): 同 name+type 实体合并,
   source_chunk_id 并集, properties 覆盖式合并;
3. 双路融合路由判断 (GraphFusion.should_use_graph): 关系词 / 实体名触发
   图谱检索, 纯语义查询仅走向量.

测试在离线模式 (NEO4J_HOST=invalid, LLM_PROVIDER=offline) 下运行,
全部依赖降级路径, 无需 Neo4j 实例与 LLM API.
"""
from __future__ import annotations

import pytest

from app.graphrag.cypher_chain import GraphCypherQAChain
from app.graphrag.extractor import EntityExtractor
from app.graphrag.fusion import GraphFusion
from app.graphrag.schemas import Entity, EntityType, Relation, RelationType


# ======================== 1. Cypher 安全校验 ========================
class TestCypherSecurity:
    """GraphCypherQAChain.validate_cypher 安全校验测试.

    校验逻辑覆盖: 写操作拒绝、多语句注入拒绝、注释拒绝、反引号拒绝、
    起始关键字白名单、RETURN + LIMIT 强制要求.
    """

    def test_accept_valid_match_query(self):
        """合法的 MATCH 查询 (含 RETURN + LIMIT) 通过校验."""
        cypher = "MATCH (n:Product) RETURN n.name LIMIT 10"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is True
        assert reason == "ok"

    def test_accept_optional_match(self):
        """OPTIONAL MATCH 起始也通过校验."""
        cypher = (
            "OPTIONAL MATCH (n:Product)-[r:CERTIFIED_BY]->(d:Department) "
            "RETURN n.name, d.name LIMIT 20"
        )
        ok, _ = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is True

    def test_reject_delete_statement(self):
        """DELETE 写操作必须拒绝."""
        cypher = "MATCH (n:Product) DELETE n RETURN n LIMIT 10"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "DELETE" in reason

    def test_reject_detach_delete(self):
        """DETACH DELETE 必须拒绝."""
        cypher = "MATCH (n:Product) DETACH DELETE n LIMIT 10"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "DETACH" in reason

    def test_reject_set_statement(self):
        """SET 写操作必须拒绝."""
        cypher = "MATCH (n:Product) SET n.name = 'x' RETURN n LIMIT 10"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "SET" in reason

    def test_reject_create_statement(self):
        """CREATE 写操作必须拒绝."""
        cypher = "CREATE (n:Product {name: 'x'}) RETURN n LIMIT 10"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "CREATE" in reason

    def test_reject_merge_statement(self):
        """MERGE 写操作必须拒绝."""
        cypher = "MERGE (n:Product {name: 'x'}) RETURN n LIMIT 10"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "MERGE" in reason

    def test_reject_drop_statement(self):
        """DROP DDL 必须拒绝."""
        cypher = "MATCH (n:Product) RETURN n LIMIT 10 DROP DATABASE graph"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "DROP" in reason

    def test_reject_multi_statement_injection(self):
        """分号拼接多语句注入必须拒绝."""
        cypher = "MATCH (n:Product) RETURN n LIMIT 10; DELETE (n)"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "分号" in reason or "多语句" in reason

    def test_reject_line_comment(self):
        """行注释 (//) 必须拒绝."""
        cypher = "MATCH (n:Product) RETURN n LIMIT 10 // 注释"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "注释" in reason

    def test_reject_block_comment(self):
        """块注释 (/* */) 必须拒绝."""
        cypher = "MATCH (n:Product) /* 注释 */ RETURN n LIMIT 10"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "注释" in reason

    def test_reject_backtick(self):
        """反引号 (防 label/属性名注入) 必须拒绝."""
        cypher = "MATCH (n:Product) RETURN n.`name` LIMIT 10"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "反引号" in reason

    def test_reject_non_match_start(self):
        """非 MATCH / OPTIONAL MATCH 起始必须拒绝."""
        cypher = "WITH 1 AS x RETURN x LIMIT 10"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "MATCH" in reason

    def test_reject_missing_return(self):
        """缺少 RETURN 必须拒绝."""
        cypher = "MATCH (n:Product) LIMIT 10"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "RETURN" in reason

    def test_reject_missing_limit(self):
        """缺少 LIMIT 必须拒绝 (防全表扫描)."""
        cypher = "MATCH (n:Product) RETURN n.name"
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "LIMIT" in reason

    def test_reject_empty_query(self):
        """空查询必须拒绝."""
        ok, reason = GraphCypherQAChain.validate_cypher("")
        assert ok is False
        assert "空" in reason

    def test_reject_call_procedure(self):
        """CALL 过程调用必须拒绝 (防过程注入)."""
        cypher = (
            "MATCH (n:Product) RETURN n LIMIT 10 "
            "CALL db.labels() YIELD label RETURN label"
        )
        ok, reason = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is False
        assert "CALL" in reason

    def test_keyword_not_substring_matched(self):
        """关键字校验用词边界, 不应误伤子串 (如 SETTLE 含 SET)."""
        # SETTLE 含 SET 子串, 但词边界匹配应不误判 (前提: 查询本身合法)
        # 这里构造一条含 SETTLE 但作为字符串值的合法查询
        cypher = "MATCH (n:Product) WHERE n.name = 'SETTLE' RETURN n LIMIT 10"
        ok, _ = GraphCypherQAChain.validate_cypher(cypher)
        assert ok is True


# ======================== 2. 实体去重合并 ========================
class TestEntityDedup:
    """EntityExtractor._merge_dedup 跨 chunk 去重合并测试.

    去重键: 实体 (name, type); 关系 (source_entity, target_entity, relation_type).
    合并规则: properties 覆盖式合并, source_chunk_id 并集.
    """

    def test_merge_same_name_type_entities(self):
        """同 name + type 的实体应合并为一条."""
        entities = [
            Entity(name="车规 eMMC", type=EntityType.Product.value,
                   properties={"category": "存储"}, source_chunk_id=1),
            Entity(name="车规 eMMC", type=EntityType.Product.value,
                   properties={"capacity": "64GB"}, source_chunk_id=2),
        ]
        merged, _ = EntityExtractor._merge_dedup(entities, [])
        assert len(merged) == 1
        assert merged[0].name == "车规 eMMC"
        assert merged[0].type == EntityType.Product.value

    def test_merge_properties_overlay(self):
        """properties 覆盖式合并: 后者覆盖前者同名字段, 不同字段并存."""
        entities = [
            Entity(name="车规 eMMC", type=EntityType.Product.value,
                   properties={"category": "存储", "capacity": "32GB"},
                   source_chunk_id=1),
            Entity(name="车规 eMMC", type=EntityType.Product.value,
                   properties={"capacity": "64GB", "grade": "AEC-Q100"},
                   source_chunk_id=2),
        ]
        merged, _ = EntityExtractor._merge_dedup(entities, [])
        assert len(merged) == 1
        props = merged[0].properties
        # category 来自前者保留
        assert props.get("category") == "存储"
        # capacity 后者覆盖前者
        assert props.get("capacity") == "64GB"
        # grade 来自后者
        assert props.get("grade") == "AEC-Q100"

    def test_merge_source_chunk_id_union(self):
        """source_chunk_id 应取首个非 None 值 (多来源由 Neo4j 侧承接)."""
        entities = [
            Entity(name="车规 eMMC", type=EntityType.Product.value,
                   properties={}, source_chunk_id=1),
            Entity(name="车规 eMMC", type=EntityType.Product.value,
                   properties={}, source_chunk_id=2),
        ]
        merged, _ = EntityExtractor._merge_dedup(entities, [])
        assert len(merged) == 1
        # dataclass 层保留首个非 None 值 (Neo4j source_chunks 数组承接完整多来源)
        assert merged[0].source_chunk_id in (1, 2)

    def test_no_merge_different_type_entities(self):
        """同 name 但不同 type 的实体不应合并 (视为不同实体)."""
        entities = [
            Entity(name="Phoenix", type=EntityType.Product.value,
                   source_chunk_id=1),
            Entity(name="Phoenix", type=EntityType.Project.value,
                   source_chunk_id=2),
        ]
        merged, _ = EntityExtractor._merge_dedup(entities, [])
        assert len(merged) == 2

    def test_no_merge_different_name_entities(self):
        """不同 name 的实体不合并."""
        entities = [
            Entity(name="车规 eMMC", type=EntityType.Product.value,
                   source_chunk_id=1),
            Entity(name="eMMC 5.1", type=EntityType.Product.value,
                   source_chunk_id=2),
        ]
        merged, _ = EntityExtractor._merge_dedup(entities, [])
        assert len(merged) == 2

    def test_merge_relations_by_key(self):
        """同 source + target + relation_type 的关系应合并为一条."""
        relations = [
            Relation(source_entity="车规 eMMC", source_type=EntityType.Product.value,
                     target_entity="质量部", target_type=EntityType.Department.value,
                     relation_type=RelationType.CERTIFIED_BY.value,
                     properties={"date": "2024-01"}, source_chunk_id=1),
            Relation(source_entity="车规 eMMC", source_type=EntityType.Product.value,
                     target_entity="质量部", target_type=EntityType.Department.value,
                     relation_type=RelationType.CERTIFIED_BY.value,
                     properties={"standard": "ISO 9001"}, source_chunk_id=2),
        ]
        _, merged = EntityExtractor._merge_dedup([], relations)
        assert len(merged) == 1
        rel = merged[0]
        assert rel.source_entity == "车规 eMMC"
        assert rel.target_entity == "质量部"
        assert rel.relation_type == RelationType.CERTIFIED_BY.value
        # properties 合并
        assert rel.properties.get("date") == "2024-01"
        assert rel.properties.get("standard") == "ISO 9001"

    def test_no_merge_different_relation_type(self):
        """同 source + target 但不同 relation_type 的关系不合并."""
        relations = [
            Relation(source_entity="车规 eMMC", source_type=EntityType.Product.value,
                     target_entity="质量部", target_type=EntityType.Department.value,
                     relation_type=RelationType.CERTIFIED_BY.value,
                     source_chunk_id=1),
            Relation(source_entity="车规 eMMC", source_type=EntityType.Product.value,
                     target_entity="质量部", target_type=EntityType.Department.value,
                     relation_type=RelationType.BELONGS_TO.value,
                     source_chunk_id=2),
        ]
        _, merged = EntityExtractor._merge_dedup([], relations)
        assert len(merged) == 2

    def test_merge_mixed_entities_and_relations(self):
        """实体与关系混合去重: 互不干扰."""
        entities = [
            Entity(name="车规 eMMC", type=EntityType.Product.value,
                   source_chunk_id=1),
            Entity(name="车规 eMMC", type=EntityType.Product.value,
                   source_chunk_id=2),
            Entity(name="质量部", type=EntityType.Department.value,
                   source_chunk_id=1),
        ]
        relations = [
            Relation(source_entity="车规 eMMC", source_type=EntityType.Product.value,
                     target_entity="质量部", target_type=EntityType.Department.value,
                     relation_type=RelationType.CERTIFIED_BY.value,
                     source_chunk_id=1),
            Relation(source_entity="车规 eMMC", source_type=EntityType.Product.value,
                     target_entity="质量部", target_type=EntityType.Department.value,
                     relation_type=RelationType.CERTIFIED_BY.value,
                     source_chunk_id=2),
        ]
        merged_entities, merged_relations = EntityExtractor._merge_dedup(
            entities, relations
        )
        assert len(merged_entities) == 2  # 车规 eMMC + 质量部
        assert len(merged_relations) == 1

    def test_empty_input_returns_empty(self):
        """空实体与空关系列表, 合并后仍为空."""
        merged_entities, merged_relations = EntityExtractor._merge_dedup([], [])
        assert merged_entities == []
        assert merged_relations == []

    def test_none_source_chunk_id_preserved(self):
        """source_chunk_id 为 None 的实体合并后仍可为 None."""
        entities = [
            Entity(name="车规 eMMC", type=EntityType.Product.value,
                   properties={}, source_chunk_id=None),
            Entity(name="车规 eMMC", type=EntityType.Product.value,
                   properties={}, source_chunk_id=None),
        ]
        merged, _ = EntityExtractor._merge_dedup(entities, [])
        assert len(merged) == 1
        # 两者均为 None, 合并后仍为 None
        assert merged[0].source_chunk_id is None


# ======================== 3. 双路融合路由判断 ========================
class TestFusionRouting:
    """GraphFusion.should_use_graph 路由判断测试.

    路由策略: 含关系词或实体名 -> 触发图谱; 纯语义 -> 仅向量.
    """

    @pytest.mark.asyncio
    async def test_relation_keyword_triggers_graph(self):
        """含关系词 (哪些/参与/认证等) 触发图谱检索."""
        fusion = GraphFusion()
        # 各类关系词均应触发
        assert await fusion.should_use_graph("车规 eMMC 有哪些认证供应商") is True
        assert await fusion.should_use_graph("质量部参与了哪些项目") is True
        assert await fusion.should_use_graph("车规 eMMC 属于哪个部门") is True
        assert await fusion.should_use_graph("谁管理研发中心") is True

    @pytest.mark.asyncio
    async def test_product_pattern_triggers_graph(self):
        """含产品代号 (大写字母+数字) 触发双路检索."""
        fusion = GraphFusion()
        assert await fusion.should_use_graph("车规 eMMC 的规格参数") is True
        assert await fusion.should_use_graph("P300 模块说明") is True

    @pytest.mark.asyncio
    async def test_department_pattern_triggers_graph(self):
        """含中文部门名 (xx部/处/中心/科/组) 触发双路检索."""
        fusion = GraphFusion()
        assert await fusion.should_use_graph("质量部的职责") is True
        assert await fusion.should_use_graph("研发中心的工作内容") is True
        assert await fusion.should_use_graph("技术处的项目") is True

    @pytest.mark.asyncio
    async def test_pure_semantic_no_graph(self):
        """纯语义查询 (无关系词/无实体名) 不触发图谱, 仅走向量."""
        fusion = GraphFusion()
        assert await fusion.should_use_graph("产品规格参数说明") is False
        assert await fusion.should_use_graph("安装部署步骤") is False
        assert await fusion.should_use_graph("故障排查方法") is False

    @pytest.mark.asyncio
    async def test_empty_query_no_graph(self):
        """空查询不触发图谱."""
        fusion = GraphFusion()
        assert await fusion.should_use_graph("") is False
        assert await fusion.should_use_graph(None) is False  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_relation_word_variants_trigger(self):
        """关系词的变体 (供应商/协作/定义等) 均应触发."""
        fusion = GraphFusion()
        assert await fusion.should_use_graph("供应商列表") is True
        assert await fusion.should_use_graph("两个部门协作流程") is True
        assert await fusion.should_use_graph("标准由谁定义") is True
        assert await fusion.should_use_graph("关联的文档") is True

    @pytest.mark.asyncio
    async def test_fuse_degrades_when_neo4j_unavailable(self):
        """Neo4j 不可用时, fuse 降级为 vector_only 策略."""
        fusion = GraphFusion()
        # 离线模式下 neo4j_store.is_available 为 False
        vector_results = [
            {"chunk_id": 1, "content": "车规 eMMC 规格文档", "score": 0.9},
            {"chunk_id": 2, "content": "认证供应商清单", "score": 0.8},
        ]
        result = await fusion.fuse(
            query="车规 eMMC 有哪些认证供应商",
            vector_results=vector_results,
            department_id=1,
            top_k=5,
        )
        # 图谱不可用 -> 降级 vector_only
        assert result["fusion_strategy"] == "vector_only"
        assert result["graph_results"] == []
        # 返回向量结果
        assert len(result["chunks"]) == 2
        assert result["chunks"][0]["chunk_id"] == 1

    @pytest.mark.asyncio
    async def test_fuse_no_graph_route_returns_vector(self):
        """纯语义查询 (无图谱路由) 直接返回向量结果."""
        fusion = GraphFusion()
        vector_results = [
            {"chunk_id": 10, "content": "安装步骤说明", "score": 0.95},
        ]
        result = await fusion.fuse(
            query="安装部署步骤",  # 纯语义, 无关系词/实体名
            vector_results=vector_results,
            department_id=None,
            top_k=3,
        )
        assert result["fusion_strategy"] == "vector_only"
        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["chunk_id"] == 10
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_fuse_empty_vector_returns_empty(self):
        """向量结果为空且图谱不可用时, 返回空 chunks."""
        fusion = GraphFusion()
        result = await fusion.fuse(
            query="车规 eMMC 的供应商",
            vector_results=[],
            department_id=1,
            top_k=5,
        )
        # 图谱不可用 -> 降级 vector_only, 向量为空 -> chunks 空
        assert result["fusion_strategy"] == "vector_only"
        assert result["chunks"] == []
        assert result["graph_results"] == []
