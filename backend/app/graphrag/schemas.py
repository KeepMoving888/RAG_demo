"""
GraphRAG 知识图谱 Schema 定义

用途:
    定义企业知识图谱的实体类型、关系类型及核心数据结构, 作为图谱构建
    (extractor)、存储 (neo4j_store)、查询 (cypher_chain)、融合 (fusion)
    四大子模块的共享契约层.

设计要点:
    1. 实体类型覆盖企业知识库八大核心对象 (产品/部门/人员/制度/项目/
       标准/供应商/客户), 与文档分块语义对齐.
    2. 关系类型覆盖组织-产品-供应链-合规四类拓扑, 支撑多跳推理
       (如「车规 eMMC 的认证供应商」需 Product-CERTIFIED_BY-Department
        再 SUPPLIES 反查 Supplier).
    3. 使用 dataclass 而非 Pydantic 模型: 图谱对象生命周期短、无需
       序列化校验, dataclass 更轻量且支持可变字段 (properties 合并).
    4. EntityType / RelationType 继承 (str, Enum): value 即 Cypher
       label/rel-type 字面量, 可直接安全拼接进 Cypher (经白名单校验).
"""

from dataclasses import dataclass, field
from enum import Enum


# ======================== 实体类型枚举 ========================
class EntityType(str, Enum):
    """企业知识图谱实体类型

    value 即 Neo4j 节点 label, 同时也是 Cypher 拼接白名单成员,
    任何来自 LLM 抽取的实体类型必须命中此枚举, 否则拒绝入库 (防注入).
    """

    Product = "Product"  # 产品 (如 车规 eMMC)
    Department = "Department"  # 部门 (如 质量部、研发中心)
    Person = "Person"  # 人员 (如 张三工程师)
    Policy = "Policy"  # 制度 (如 《供应商准入管理办法》)
    Project = "Project"  # 项目 (如 XX 产线改造项目)
    Standard = "Standard"  # 标准 (如 ISO 9001、GB/T 19001)
    Supplier = "Supplier"  # 供应商
    Customer = "Customer"  # 客户
    Certification = "Certification"  # 认证 (如 ISO 9001、CE)
    Patent = "Patent"  # 专利 (如 ZL202310458X)


# ======================== 关系类型枚举 ========================
class RelationType(str, Enum):
    """企业知识图谱关系类型

    value 即 Neo4j 关系 type, 同样作为 Cypher 拼接白名单.
    关系方向约定: source_entity -[relation_type]-> target_entity
    """

    BELONGS_TO = "BELONGS_TO"  # 归属 (Product->Department)
    PARTICIPATES = "PARTICIPATES"  # 参与 (Department->Project)
    CERTIFIED_BY = "CERTIFIED_BY"  # 认证 (Product->Department/Standard)
    REFERENCES = "REFERENCES"  # 引用 (Policy->Standard)
    MANAGES = "MANAGES"  # 管理 (Person->Department/Project)
    SUPPLIES = "SUPPLIES"  # 供应 (Supplier->Product)
    COLLABORATES_WITH = "COLLABORATES_WITH"  # 协作 (Department<->Department)
    DEFINED_BY = "DEFINED_BY"  # 定义 (Standard->Department)
    GOVERNED_BY = "GOVERNED_BY"  # 受约束 (Product->Policy)
    AUDITED_BY = "AUDITED_BY"  # 审核 (Department->Certification)
    PARTICIPATES_IN = "PARTICIPATES_IN"  # 参与 (Department->Certification)
    INVENTED_BY = "INVENTED_BY"  # 发明 (Product->Patent/Person)
    AUTHORED_BY = "AUTHORED_BY"  # 撰写 (Policy->Person/Department)
    MANUFACTURES = "MANUFACTURES"  # 生产 (Supplier->Product)


# ======================== 中文标签映射 ========================
# 供前端展示与结果格式化使用, 不参与图查询逻辑
ENTITY_LABELS_ZH: dict[str, str] = {
    EntityType.Product.value: "产品",
    EntityType.Department.value: "部门",
    EntityType.Person.value: "人员",
    EntityType.Policy.value: "制度",
    EntityType.Project.value: "项目",
    EntityType.Standard.value: "标准",
    EntityType.Supplier.value: "供应商",
    EntityType.Customer.value: "客户",
    EntityType.Certification.value: "认证",
    EntityType.Patent.value: "专利",
}

RELATION_LABELS_ZH: dict[str, str] = {
    RelationType.BELONGS_TO.value: "归属",
    RelationType.PARTICIPATES.value: "参与",
    RelationType.CERTIFIED_BY.value: "认证",
    RelationType.REFERENCES.value: "引用",
    RelationType.MANAGES.value: "管理",
    RelationType.SUPPLIES.value: "供应",
    RelationType.COLLABORATES_WITH.value: "协作",
    RelationType.DEFINED_BY.value: "定义",
    RelationType.GOVERNED_BY.value: "受约束",
    RelationType.AUDITED_BY.value: "审核",
    RelationType.PARTICIPATES_IN.value: "参与",
    RelationType.INVENTED_BY.value: "发明",
    RelationType.AUTHORED_BY.value: "撰写",
    RelationType.MANUFACTURES.value: "生产",
}


# ======================== 核心数据结构 ========================
@dataclass
class Entity:
    """图谱实体

    Attributes:
        name: 实体唯一名称 (作为 Neo4j 节点唯一约束键).
        type: 实体类型, 必须命中 EntityType 枚举.
        properties: 扩展属性 (如 category=传感器), 多次抽取会合并.
        source_chunk_id: 该实体的来源文档分块 ID, 用于溯源与文档更新时清理.
    """

    name: str
    type: str
    properties: dict = field(default_factory=dict)
    source_chunk_id: int | None = None


@dataclass
class Relation:
    """图谱关系

    Attributes:
        source_entity: 源实体名称.
        source_type: 源实体类型 (用于 MATCH 定位节点 label).
        target_entity: 目标实体名称.
        target_type: 目标实体类型.
        relation_type: 关系类型, 必须命中 RelationType 枚举.
        properties: 关系扩展属性.
        source_chunk_id: 来源文档分块 ID.
    """

    source_entity: str
    source_type: str
    target_entity: str
    target_type: str
    relation_type: str
    properties: dict = field(default_factory=dict)
    source_chunk_id: int | None = None


@dataclass
class ExtractionResult:
    """单次抽取 (chunk 或 document 级别) 的聚合结果

    Attributes:
        entities: 去重后的实体列表.
        relations: 去重后的关系列表.
        raw_llm_response: LLM 原始返回文本, 用于调试与可观测性.
    """

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    raw_llm_response: str = ""


__all__ = [
    "EntityType",
    "RelationType",
    "ENTITY_LABELS_ZH",
    "RELATION_LABELS_ZH",
    "Entity",
    "Relation",
    "ExtractionResult",
]
