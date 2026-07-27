# GraphRAG 知识图谱增强

本文档描述企业级 RAG 知识库的 GraphRAG 增强能力：通过 Neo4j 知识图谱捕获实体间关系链路，补足纯向量检索在多跳推理上的短板。内容覆盖设计动机、图谱 Schema、LLM 实体关系抽取、Cypher 查询编排与安全防护、双路融合策略与查询路由，以及实测效果对比。

## 1. 设计动机

纯向量检索的本质是"问题与文档片段的语义相似度匹配"，它隐含一个假设：答案所在的文档片段与问题在语义空间中相近。这一假设在以下场景失效：

- **跨部门多跳关系推理**：问题"负责认证 X 产品的部门，其参与制定的制度有哪些？"需要先找到产品 X → 认证部门 → 该部门参与的制度，是三跳关系链。答案文档（制度）与问题在语义上并不相近，向量检索难以召回。
- **聚合类问题**：问题"研发部参与了哪些产品的认证？"需要枚举关系，向量检索返回的是语义相近片段而非完整关系集合。
- **实体消歧**：问题中提到的"产品 A"在不同文档中可能指代不同实体，向量检索无法利用实体唯一标识进行精确匹配。

知识图谱以"实体-关系-实体"的三元组显式表达结构化关系，天然适合多跳推理。GraphRAG 将图谱检索作为向量检索之外的第三路召回，通过查询路由按问题类型分发，并在融合阶段去重精排，兼顾语义匹配与关系推理两类能力。

## 2. 图谱 Schema

Schema 在图谱初始化时定义，约束节点与关系的类型，保证 LLM 抽取结果的规范性。Schema 设计遵循"类型少而稳定、属性可扩展"原则。

### 2.1 实体类型

| 实体类型 | 标签 | 关键属性 | 说明 |
|----------|------|----------|------|
| 产品 | `Product` | `name`, `version`, `category`, `doc_ids[]` | 企业自研或采购的系统/产品 |
| 部门 | `Department` | `name`, `code`, `level` | 组织架构节点 |
| 人员 | `Person` | `name`, `title`, `department_id` | 文档中提及的相关人员 |
| 制度 | `Policy` | `name`, `policy_no`, `effective_date`, `status` | 规章制度、规范文件 |

### 2.2 关系类型

| 关系类型 | 方向 | 含义 | 示例 |
|----------|------|------|------|
| `BELONGS_TO` | Product → Department | 产品归属部门 | 产品 A 归属研发部 |
| `PARTICIPATES` | Department → Policy | 部门参与制定制度 | 研发部参与制定《编码规范》 |
| `CERTIFIES` | Department → Product | 部门认证产品 | 安全部认证产品 A |
| `REFERENCES` | Policy → Product | 制度引用产品 | 《准入标准》引用产品 A |
| `REFERENCES` | Policy → Policy | 制度引用制度 | 《细则》引用《总则》 |
| `AUTHORED_BY` | Policy → Person | 制度起草人 | 《编码规范》起草人张三 |

每个节点携带 `doc_ids[]` 属性，记录该实体来源的文档分块 ID，便于从图谱检索结果回溯到原文，参与后续融合与溯源。

```cypher
// Schema 约束（初始化时创建）
CREATE CONSTRAINT product_name IF NOT EXISTS
  FOR (n:Product) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT department_code IF NOT EXISTS
  FOR (n:Department) REQUIRE n.code IS UNIQUE;
CREATE CONSTRAINT policy_no IF NOT EXISTS
  FOR (n:Policy) REQUIRE n.policy_no IS UNIQUE;
```

## 3. LLM 实体关系抽取

图谱构建在文档入库流水线的第 7 阶段执行（见[架构文档](./architecture.md) 4.1 节）。Worker 对每个 chunk 调用 LLM 抽取实体与关系，经去重归一化后写入 Neo4j。

### 3.1 抽取流程

```
chunk ──► LLM(Prompt: 抽取实体+关系, 输出 JSON) ──► 实体归一化 ──► 去重 ──► 写入 Neo4j
                                                          │
                                                          ▼
                                                   更新节点 doc_ids[]
```

抽取 Prompt 约束 LLM 仅输出 Schema 内定义的实体与关系类型，并以结构化 JSON 返回，避免自由文本带来的解析负担。

```python
EXTRACT_PROMPT = """你是一个信息抽取助手。从以下文档片段中抽取实体与关系，
仅抽取以下类型：
实体: Product(产品), Department(部门), Person(人员), Policy(制度)
关系: BELONGS_TO(归属), PARTICIPATES(参与), CERTIFIES(认证), REFERENCES(引用), AUTHORED_BY(起草)

文档片段:
{chunk}

以 JSON 输出，格式:
{{
  "entities": [{{"type": "Product", "name": "...", "properties": {{...}}}}],
  "relations": [{{"head": "产品A", "head_type": "Product",
                  "relation": "BELONGS_TO",
                  "tail": "研发部", "tail_type": "Department"}}]
}}
仅输出 JSON，不要解释。"""
```

### 3.2 实体归一化与去重

LLM 抽取的实体名称存在异形（"研发部" / "研发部门" / "Research & Development"指同一实体）。归一化步骤：

1. **别名映射**：维护实体别名表（管理员可配置），将别名统一到规范名。
2. **模糊匹配**：对别名表中未命中的实体，与图中已有同名类型实体做编辑距离匹配（阈值 0.85），命中则合并。
3. **唯一性约束**：依赖 Neo4j 的唯一约束，重复写入触发 `MERGE` 而非 `CREATE`，避免产生重复节点。

`doc_ids[]` 在 `MERGE` 时追加当前 chunk_id，保证实体可回溯到所有来源片段。

```python
def upsert_entity(tx, entity: dict, chunk_id: str):
    """实体写入：MERGE 保证幂等，doc_ids 追加来源。"""
    cypher = f"""
    MERGE (n:{entity['type']} {{name: $name}})
    ON CREATE SET n.created_at = timestamp()
    SET n += $properties,
        n.doc_ids = coalesce(n.doc_ids, []) + $chunk_id
    """
    tx.run(cypher, name=entity["name"],
           properties=entity.get("properties", {}),
           chunk_id=chunk_id)
```

## 4. GraphCypherQAChain 查询编排

问答阶段，对判定为关系类的问题，系统通过 LangChain 的 `GraphCypherQAChain` 将自然语言翻译为 Cypher 查询并执行。Chain 的编排流程：

```
用户问题 ──► Cypher 生成 (LLM, 基于 Schema) ──► Cypher 校验 (白名单/只读) ──► 参数化执行 ──► 结果转自然语言 (LLM)
```

```python
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    return_intermediate_steps=True,   # 保留生成的 Cypher 用于审计
    cypher_prompt=CYPHER_GEN_PROMPT,  # 约束生成范围
    qa_prompt=QA_ANSWER_PROMPT,       # 结果转自然语言
    top_k=50,                          # 单查询返回上限
    use_function_calling=True,
)
```

`GraphCypherQAChain` 在生成 Cypher 前注入 Schema（节点/关系类型与属性），使 LLM 产出合法查询。生成的 Cypher 与执行结果作为 `intermediate_steps` 保留，写入审计日志，便于排查错误查询与定位幻觉。

## 5. Cypher 安全

自然语言转 Cypher 是 GraphRAG 最大的注入风险面。本系统采用纵深防御，任何单一层被绕过仍有下一层兜底。

### 5.1 白名单校验

生成的 Cypher 在执行前经过解析，仅允许只读子句：

```python
ALLOWED_CLAUSES = {"MATCH", "WHERE", "RETURN", "WITH", "ORDER BY",
                   "LIMIT", "SKIP", "OPTIONAL MATCH", "UNWIND", "DISTINCT"}
FORBIDDEN_CLAUSES = {"CREATE", "DELETE", "SET", "REMOVE", "MERGE",
                     "DROP", "CALL"}  # CALL 禁止, 防止调用过程写库

def validate_cypher(cypher: str) -> None:
    """校验 Cypher 仅含只读子句, 否则拒绝执行。"""
    ast = parse_cypher(cypher)
    for clause in ast.clauses:
        if clause.type not in ALLOWED_CLAUSES:
            raise CypherSecurityError(
                f"禁止的子句: {clause.type} (仅允许只读查询)")
```

### 5.2 参数化查询防注入

用户可控输入绝不字符串拼接到 Cypher 中，统一以 `$param` 形式传入。即使 LLM 生成包含用户输入的查询，参数化也保证其被当作字面值而非图模式。

```python
# 正确: 参数化
result = graph.query(
    "MATCH (p:Policy)-[:AUTHORED_BY]->(person:Person {name: $author}) "
    "RETURN p.name AS policy LIMIT $limit",
    params={"author": user_input, "limit": 20},
)

# 错误: 字符串拼接 (严禁)
# cypher = f"MATCH ... WHERE person.name = '{user_input}' ..."
```

### 5.3 只读用户与硬限制

- Neo4j 连接使用只读角色（仅 `READ` 权限），即使前两层防护被绕过也无法写入或删除。
- 单查询 `timeout=5s`，`LIMIT 200` 硬上限，防止恶意查询引发全图扫描拖垮数据库。
- 所有生成的 Cypher 与执行耗时写入 `audit_log`，便于事后审计与异常查询定位。

## 6. 双路融合策略

图谱检索与向量检索结果异构：图谱返回的是实体与关系（结构化），向量返回的是文本 chunk（非结构化）。融合策略需要将两者统一到可比较的形式。

```
用户问题
   │
   ├─► 向量检索 (语义匹配) ──► chunks[]
   │                              │
   └─► 图谱检索 (关系推理) ──► entities[] + relations[]
                                  │
                                  ▼
                          回溯到源 chunk (via node.doc_ids[])
                                  │
                                  ▼
                          合并去重 (by chunk_id)
                                  │
                                  ▼
                          bge-reranker-v2-m3 精排 ──► Top-5
```

图谱节点携带 `doc_ids[]`，可回溯到对应的文本 chunk。回溯后的 chunk 与向量检索召回的 chunk 合并去重（按 `chunk_id`），统一进入 Cross-Encoder 精排。这一设计让图谱检索结果能复用既有的精排与溯源链路，无需为图谱单独设计生成 Prompt。

图谱未能回溯到 chunk 的纯关系结果（如"产品 A 归属研发部"这类直接由图谱给出的关系事实），以结构化卡片形式作为补充上下文注入 Prompt，标注来源为图谱而非原文。

## 7. 查询路由

并非所有问题都需要走图谱检索——纯语义类问题走图谱只会增加延迟而无收益。查询路由器按问题类型分发：

| 问题类型 | 判断特征 | 路由目标 | 示例 |
|----------|----------|----------|------|
| 语义类 | 询问概念、定义、描述 | 仅向量检索 | "什么是 RTO 标准？" |
| 关系类 | 含关系动词（归属/参与/认证/引用）、枚举需求 | 仅图谱检索 | "研发部参与了哪些制度制定？" |
| 混合类 | 既需语义匹配又涉关系推理 | 双路融合 | "负责认证 X 产品的部门制定的制度有哪些？" |

路由判断由轻量级分类器完成（规则优先，规则未命中时调用小模型分类），避免每次问答都消耗一次大模型调用。

```python
class QueryRouter:
    """按问题类型路由到向量检索、图谱检索或双路融合。"""

    RELATION_KEYWORDS = {"归属", "参与", "认证", "引用", "起草",
                         "哪些", "列出", "属于", "负责"}

    def route(self, question: str) -> RetrieverStrategy:
        has_relation = any(kw in question for kw in self.RELATION_KEYWORDS)
        is_enum = any(kw in question for kw in ("哪些", "列出", "所有"))
        if has_relation and is_enum:
            return RetrieverStrategy.GRAPH_ONLY
        if has_relation:
            return RetrieverStrategy.HYBRID_FUSION
        return RetrieverStrategy.VECTOR_ONLY
```

## 8. 效果对比

在内部评测集上，按问题跳数分组对比三种检索策略。评测集按人工标注的关系跳数划分：单跳事实（320 题）、双跳关联（120 题）、三跳推理（60 题）。

| 问题类型 | 向量 only | 图谱 only | 双路融合 | 融合相对向量提升 |
|----------|-----------|-----------|----------|------------------|
| 单跳事实 | 0.88 | 0.82 | **0.91** | +3 pp |
| 双跳关联 | 0.52 | 0.89 | **0.93** | +41 pp |
| 三跳推理 | 0.23 | 0.76 | **0.84** | +61 pp |

关键结论：

1. **单跳事实**：向量检索（0.88）已较强，图谱（0.82）略低（因图谱覆盖不全时召回为空）。融合（0.91）小幅领先，代价是额外延迟。
2. **双跳关联**：向量检索断崖式下降至 0.52，图谱检索保持 0.89，融合达 0.93。这是 GraphRAG 价值最显著的区间。
3. **三跳推理**：向量检索几乎失效（0.23），图谱检索 0.76，融合 0.84。图谱在多跳场景不可替代。

权衡：双路融合相比单路增加约 30ms 延迟（图谱查询 P95 约 85ms，与向量检索并行执行，取较大值）。鉴于多跳场景下的巨大召回收益，查询路由对关系类与混合类问题默认启用双路融合。

## 9. 图谱维护

### 9.1 增量构建

文档重解析或删除时，图谱需同步更新：

- **重解析**：先删除该文档所有 chunk 关联的实体关系（按 `doc_ids` 反查），再重新抽取写入。对被多文档引用的实体，仅移除当前文档的 `doc_id`，不删除节点本身。
- **文档删除**：同上移除引用，节点若无任何 `doc_ids` 残留则删除（孤儿节点清理）。

### 9.2 图谱质量校验

定期运行图谱健康检查：

- 孤儿节点（无任何关系）数量。
- 重复实体（同类型同义不同名）检测。
- 关系孤岛（连通分量过小，疑似抽取遗漏）。

校验结果通过 `GET /api/v1/graph/schema` 的 `health` 字段暴露，供管理员评估是否需要重新全量构建图谱。

## 10. 相关文档

- [系统架构设计](./architecture.md)：GraphRAG 在整体架构中的位置与降级策略。
- [RAG 检索流水线设计](./rag-pipeline.md)：双路融合中向量检索侧的实现细节。
- [部署运维指南](./deployment.md)：Neo4j 部署配置与 `make graph-init` 初始化。
