"""
GraphCypherQAChain - 自然语言转 Cypher 查询链

用途:
    将用户的自然语言问题翻译为 Neo4j Cypher 只读查询并执行, 返回结构化
    结果与可读文本. 是 GraphRAG 中「图谱检索」的执行入口, 供 fusion
    模块调用以获取实体关系链路证据.

为何强制只读 Cypher 校验:
    LLM 生成的 Cypher 不可信, 必须在执行前做严格白名单校验, 防止:
      1. 写操作破坏图谱 (DELETE/SET/CREATE/MERGE/REMOVE/DROP);
      2. 多语句注入 (分号拼接第二条 DELETE);
      3. 注释绕过 (// 或 /* */ 隐藏恶意语句);
      4. 全表扫描拖垮实例 (强制 LIMIT).
    校验失败一律拒绝执行并返回原因, 不做「自动修复」, 避免语义漂移风险.

查询流程:
    1. LLM 将自然语言转为 {"cypher": "...", "explanation": "..."};
    2. 缺 LIMIT 时自动追加 LIMIT 50 (防全表扫描);
    3. validate_cypher 白名单校验;
    4. neo4j_store.run_cypher 执行;
    5. format_result 格式化为可读文本.
"""

import re
import time

from app.graphrag.neo4j_store import neo4j_store
from app.graphrag.schemas import ENTITY_LABELS_ZH, RELATION_LABELS_ZH
from app.llm import get_llm
from app.metrics import record_graph_query
from app.utils.logger import logger

# ======================== Cypher 生成 Prompt ========================
CYPHER_PROMPT = """你是 Cypher 查询生成助手。将用户自然语言转为 Neo4j Cypher 查询。

图谱 Schema（半导体存储行业知识图谱）:
- 节点 label:
  * Product(name)         存储产品, 如 eMMC、UFS、NAND Flash、DRAM、SSD、MCP
  * Department(name)      部门, 如 研发中心、质量保证部、合规认证部
  * Person(name)          人员, 如 封装工程师、认证工程师
  * Policy(name)          制度, 如 供应商准入管理办法、可靠性测试规范
  * Standard(name)        标准, 如 ISO 9001、IATF 16949
  * Supplier(name)        供应商, 如 晶圆代工厂、封测厂、基板厂
  * Customer(name)        客户, 如 车厂、消费电子品牌
  * Certification(name)   认证, 如 ISO 9001、BSCI、IATF 16949、CE、RoHS、FCC、UKCA
  * Patent(name)          专利, 如 ZL2024XXXXX 封装结构专利

- 关系 type:
  * CERTIFIED_BY          产品/供应商 → 认证 (如 eMMC -[:CERTIFIED_BY]-> CE)
  * BELONGS_TO            产品/人员 → 部门 (归属)
  * SUPPLIES              供应商 → 产品 (供货)
  * MANUFACTURES          供应商 → 产品 (代工生产)
  * GOVERNED_BY           产品 → 制度 (受约束)
  * AUDITED_BY            部门 → 认证 (审核维持)
  * PARTICIPATES_IN       部门 → 认证 (参与)
  * INVENTED_BY           产品/专利 → 人员 (发明)
  * AUTHORED_BY           制度 → 人员/部门 (撰写)
  * REFERENCES            产品/制度 → 标准 (引用)
  * DEFINED_BY            标准 → 部门 (定义)

用户问题: {natural_language}

输出 JSON: {{"cypher": "MATCH ...", "explanation": "..."}}
规则:
1. 只读查询, 必须 LIMIT, 禁止写操作;
2. 节点过滤优先用 name 属性 (如 WHERE p.name CONTAINS 'eMMC');
3. 中文实体名可直接作为参数 (如 WHERE p.name = 'eMMC 5.1');
4. 多跳查询明确关系方向 (如产品→认证、供应商→产品→认证).
"""


# 默认结果上限 (LLM 未给 LIMIT 时强制追加)
DEFAULT_LIMIT = 50

# Cypher 安全校验: 禁止的关键字 (写操作 / 危险过程 / 计划分析)
# DETACH 优先于 DELETE 检查, 以便 DETACH DELETE 报告更精确的拒绝原因
FORBIDDEN_KEYWORDS = [
    "DETACH",
    "DELETE",
    "SET",
    "CREATE",
    "MERGE",
    "REMOVE",
    "DROP",
    "CALL",
    "FOREACH",
    "LOAD",
    "EXPLAIN",
    "PROFILE",
    "GRANT",
    "REVOKE",
    "DENY",
    "ALTER",
    "RENAME",
    "CONSTRAINT",
    "INDEX",
    "DATABASE",
    "SHORTESTPATH",  # shortestPath 在变量长度中易触发全图遍历, 改用受控写法时单独放行
]

# 允许的起始关键字 (查询必须只读且以 MATCH / OPTIONAL MATCH 开头)
ALLOWED_START_KEYWORDS = ("MATCH", "OPTIONAL")


class GraphCypherQAChain:
    """自然语言 -> Cypher -> 执行 -> 格式化 查询链"""

    def __init__(self) -> None:
        self._llm = get_llm()

    # ======================== 主入口 ========================
    async def query(
        self,
        natural_language: str,
        department_id: int | None = None,
    ) -> dict:
        """自然语言图谱查询

        Args:
            natural_language: 用户自然语言问题.
            department_id: 部门 ID (预留 ACL 过滤, 当前由 fusion 层在反查
                chunk 时按部门裁剪, 图谱节点本身不直接携带部门).

        Returns:
            {"cypher": str, "result_text": str, "records": list[dict],
             "latency_ms": float}
        """
        start = time.time()

        # 1. LLM 生成 Cypher
        try:
            prompt = CYPHER_PROMPT.format(natural_language=natural_language)
            cypher_data = await self._llm.aextract_json(prompt)
            cypher = str(cypher_data.get("cypher", "")).strip()
            explanation = str(cypher_data.get("explanation", ""))
        except Exception as e:
            logger.warning("Cypher 生成失败: {}", str(e))
            return {
                "cypher": "",
                "result_text": f"查询生成失败: {e}",
                "records": [],
                "latency_ms": (time.time() - start) * 1000,
            }

        if not cypher:
            return {
                "cypher": "",
                "result_text": "未能生成有效查询",
                "records": [],
                "latency_ms": (time.time() - start) * 1000,
            }

        # 2. 强制 LIMIT (防全表扫描)
        if not re.search(r"\bLIMIT\b", cypher, re.IGNORECASE):
            cypher = cypher.rstrip().rstrip(";") + f" LIMIT {DEFAULT_LIMIT}"

        # 3. 安全校验
        ok, reason = self.validate_cypher(cypher)
        if not ok:
            logger.warning("Cypher 校验拒绝: {} | cypher={}", reason, cypher[:200])
            return {
                "cypher": cypher,
                "result_text": f"查询被拒绝: {reason}",
                "records": [],
                "latency_ms": (time.time() - start) * 1000,
            }

        # 4. 执行 (Neo4j 不可用时 run_cypher 返回空列表)
        records = await neo4j_store.run_cypher(cypher)
        latency_ms = (time.time() - start) * 1000

        status = "success" if records else "empty"
        record_graph_query(latency_ms, status=status)

        # 5. 格式化
        result_text = self.format_result(records)
        logger.info(
            "Cypher 查询完成 latency={:.0f}ms records={} | {}",
            latency_ms,
            len(records),
            explanation,
        )
        return {
            "cypher": cypher,
            "result_text": result_text,
            "records": records,
            "latency_ms": latency_ms,
        }

    # ======================== 安全校验 ========================
    @staticmethod
    def validate_cypher(cypher: str) -> tuple[bool, str]:
        """Cypher 安全校验

        校验项:
            1. 非空且不含分号 (防多语句注入);
            2. 不含行注释 (//) 与块注释 (/* */);
            3. 不含反引号 (防 label/属性名注入, 本系统 label 均为 ASCII 白名单);
            4. 不含禁止关键字 (写操作 / DDL / 过程调用);
            5. 以 MATCH / OPTIONAL MATCH 开头;
            6. 含 RETURN 与 LIMIT.

        Returns:
            (是否通过, 原因说明)
        """
        stripped = cypher.strip()
        if not stripped:
            return False, "空查询"

        upper = stripped.upper()

        # 多语句注入防护
        if ";" in stripped.rstrip().rstrip(";"):
            return False, "禁止多语句 (分号)"

        # 注释防护
        if "//" in stripped or "/*" in stripped or "*/" in stripped:
            return False, "禁止注释"

        # 反引号防护 (本系统无需反引号转义)
        if "`" in stripped:
            return False, "禁止反引号"

        # 禁止关键字 (词边界匹配, 避免误伤子串如 'SETTLE')
        for kw in FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{kw}\b", upper):
                return False, f"禁止的关键字: {kw}"

        # 起始关键字
        if not upper.startswith(ALLOWED_START_KEYWORDS):
            return False, "查询必须以 MATCH / OPTIONAL MATCH 开头"

        # 必须有 RETURN
        if not re.search(r"\bRETURN\b", upper):
            return False, "查询必须包含 RETURN"

        # 必须有 LIMIT
        if not re.search(r"\bLIMIT\b", upper):
            return False, "查询必须包含 LIMIT"

        return True, "ok"

    # ======================== 结果格式化 ========================
    @staticmethod
    def format_result(records: list[dict]) -> str:
        """将查询记录格式化为可读文本

        识别节点 (含 labels/name) 与关系 (含 type), 输出中文标签增强可读性;
        其余标量按 key=value 输出.
        """
        if not records:
            return "图谱中未找到匹配结果."

        lines = []
        for i, rec in enumerate(records, 1):
            parts: list[str] = []
            for key, value in rec.items():
                if isinstance(value, dict):
                    # 节点或关系对象
                    labels = value.get("labels")
                    if labels:  # 节点
                        label = labels[0] if labels else "Entity"
                        zh = ENTITY_LABELS_ZH.get(label, label)
                        name = value.get("name", "")
                        parts.append(f"{zh}「{name}」")
                    else:  # 关系或普通 map
                        rtype = value.get("type", "")
                        zh = RELATION_LABELS_ZH.get(rtype, rtype or key)
                        parts.append(f"[{zh}]")
                elif isinstance(value, list):
                    # 关系类型列表或路径
                    parts.append(f"{key}={value}")
                else:
                    parts.append(f"{key}={value}")
            lines.append(f"{i}. " + " | ".join(parts) if parts else f"{i}. (空记录)")
        return "\n".join(lines)


__all__ = ["GraphCypherQAChain", "CYPHER_PROMPT"]
