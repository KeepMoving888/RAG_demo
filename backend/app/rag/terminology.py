"""app.rag.terminology —— 企业术语词典与查询扩展

设计要点
--------
企业内部知识库存在大量「专有名词」——产品代号（车规 eMMC）、部门简称（QAC）、
认证标准（RoHS / ISO 9001）等。这些词在通用分词器下常被切散或与字面文档
不匹配，导致 BM25 召回率骤降。本模块通过维护一份术语词典，在检索前对查询
进行**同义词扩展**与**词项加权**，弥补字面匹配的语义鸿沟。

核心策略
~~~~~~~~
1. **查询扩展（Query Expansion）**：当 query 命中术语时，将该术语的同义词以
   OR 形式注入 BM25 检索文本。例如 query="车载存储模块的合规要求" 命中
   术语 车规 eMMC（synonyms=["车载存储模块","车规eMMC"]），扩展后文本变为
   "车载存储模块的合规要求 OR 车规 eMMC OR 车规eMMC"，使 BM25 能匹配到
   使用代号命名的文档。
2. **词项加权（Term Weighting）**：术语命中后，对应 token 在 BM25 打分时
   权重 ×2.0，使含术语的文档排名前置。详见 ``boost_term_weight``。
3. **动态扩展**：运营可通过 ``add_term`` 在运行时新增术语，无需重启服务。

词典结构
~~~~~~~~
::

    {
        "车规 eMMC": {"synonyms": ["车载存储模块", "车规eMMC"], "type": "product"},
        "RoHS": {"synonyms": ["有害物质限制指令"], "type": "standard"},
        "QAC": {"synonyms": ["质量保证中心"], "type": "department"}
    }

数据来源：``data/seed/terminology.json``（种子数据，约 50 条），生产环境
可对接术语管理后台动态加载。
"""

from __future__ import annotations

import json
import os

from app.utils.logger import logger

# 默认术语词典种子数据路径
_DEFAULT_TERM_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "data",
    "seed",
    "terminology.json",
)

# 内置兜底术语（当种子文件不存在时使用，保证离线模式可用）
# 注意: 同义词只作为主术语的别名出现, 不单独作为主术语条目, 避免命中时
#       返回错误的主术语 (如 "车载存储模块" 不应作为主术语, 它是 "车规 eMMC" 的别名)
_FALLBACK_TERMS: dict[str, dict] = {
    "车规 eMMC": {"synonyms": ["车载存储模块", "车规eMMC"], "type": "product"},
    "RoHS": {"synonyms": ["有害物质限制指令"], "type": "standard"},
    "有害物质限制指令": {"synonyms": ["RoHS"], "type": "standard"},
    "ISO 9001": {"synonyms": ["质量管理体系认证", "ISO9001"], "type": "standard"},
    "ISO9001": {"synonyms": ["质量管理体系认证", "ISO 9001"], "type": "standard"},
    "质量管理体系认证": {"synonyms": ["ISO 9001", "ISO9001"], "type": "standard"},
    "QAC": {"synonyms": ["质量保证中心"], "type": "department"},
    "质量保证中心": {"synonyms": ["QAC"], "type": "department"},
    "IEC 61010": {"synonyms": ["电气安全标准", "IEC61010"], "type": "standard"},
    "IEC61010": {"synonyms": ["电气安全标准", "IEC 61010"], "type": "standard"},
    "CE认证": {"synonyms": ["CE 标志", "欧盟合规认证"], "type": "standard"},
    "CE 标志": {"synonyms": ["CE认证", "欧盟合规认证"], "type": "standard"},
    "FCC认证": {"synonyms": ["FCC 认证", "联邦通信委员会认证"], "type": "standard"},
    "UL认证": {"synonyms": ["UL 认证", "美国保险商实验室认证"], "type": "standard"},
    "SaaS": {"synonyms": ["软件即服务"], "type": "product"},
    "PaaS": {"synonyms": ["平台即服务"], "type": "product"},
    "IaaS": {"synonyms": ["基础设施即服务"], "type": "product"},
    "OCR": {"synonyms": ["光学字符识别"], "type": "technology"},
    "NLP": {"synonyms": ["自然语言处理"], "type": "technology"},
    "RAG": {"synonyms": ["检索增强生成"], "type": "technology"},
    "检索增强生成": {"synonyms": ["RAG"], "type": "technology"},
    "SLA": {"synonyms": ["服务等级协议"], "type": "concept"},
    "MTTR": {"synonyms": ["平均修复时间"], "type": "metric"},
    "MTBF": {"synonyms": ["平均故障间隔时间"], "type": "metric"},
    "API网关": {"synonyms": ["API Gateway", "接口网关"], "type": "product"},
    "API Gateway": {"synonyms": ["API网关", "接口网关"], "type": "product"},
    "数据中台": {"synonyms": ["数据平台", "Data Middle Platform"], "type": "product"},
    "RBAC": {"synonyms": ["基于角色的访问控制"], "type": "concept"},
    "ABAC": {"synonyms": ["基于属性的访问控制"], "type": "concept"},
    "ETL": {"synonyms": ["数据抽取转换加载"], "type": "technology"},
    "CDC": {"synonyms": ["变更数据捕获"], "type": "technology"},
    "K8s": {"synonyms": ["Kubernetes", "容器编排平台"], "type": "product"},
    "Kubernetes": {"synonyms": ["K8s", "容器编排平台"], "type": "product"},
    "微服务": {"synonyms": ["Microservice"], "type": "architecture"},
    "Microservice": {"synonyms": ["微服务"], "type": "architecture"},
    "DevOps": {"synonyms": ["开发运维一体化"], "type": "concept"},
    "CI/CD": {"synonyms": ["持续集成持续部署"], "type": "concept"},
    "DDoS": {"synonyms": ["分布式拒绝服务攻击"], "type": "concept"},
    "WAF": {"synonyms": ["Web应用防火墙"], "type": "product"},
    "SIEM": {"synonyms": ["安全信息事件管理"], "type": "product"},
    "GDPR": {"synonyms": ["通用数据保护条例"], "type": "standard"},
    "通用数据保护条例": {"synonyms": ["GDPR"], "type": "standard"},
    "SOP": {"synonyms": ["标准作业程序"], "type": "concept"},
    "BOM": {"synonyms": ["物料清单"], "type": "concept"},
}


class TerminologyExpander:
    """企业术语扩展器。

    加载术语词典后，提供查询扩展与词项加速能力。术语命中后会将同义词以 OR
    形式注入检索文本，并对命中 token 在 BM25 打分时加权，显著提升术语类
    问题的召回率与排序质量。

    Parameters
    ----------
    term_path : str | None
        术语词典 JSON 路径，默认指向 ``data/seed/terminology.json``。
    """

    # 术语命中 token 的权重倍数
    TERM_WEIGHT_BOOST: float = 2.0

    def __init__(self, term_path: str | None = None) -> None:
        self._term_path: str = term_path or _DEFAULT_TERM_PATH
        # 词典：term -> {"synonyms": [...], "type": str}
        self._terms: dict[str, dict] = {}
        # 反向索引：同义词 -> 主术语（用于命中后查找完整同义词组）
        self._synonym_index: dict[str, str] = {}
        self._load()
        logger.info(
            "TerminologyExpander 已加载，术语数=%d，同义词索引数=%d",
            len(self._terms),
            len(self._synonym_index),
        )

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """从种子数据文件加载术语词典，失败时使用内置兜底词典。

        支持两种 JSON 格式:
        - dict: ``{"term": {"synonyms": [...], "type": "..."}}``
        - list: ``[{"term": "...", "synonyms": [...], "category": "..."}]``
        """
        loaded = False
        if os.path.exists(self._term_path):
            try:
                with open(self._term_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    self._terms = data
                    loaded = True
                elif isinstance(data, list):
                    # 列表格式: 每项含 term/synonyms/category 字段
                    self._terms = {}
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        term = item.get("term") or item.get("name")
                        if not term:
                            continue
                        synonyms = item.get("synonyms", [])
                        category = item.get("category") or item.get("type", "custom")
                        self._terms[term] = {"synonyms": synonyms, "type": category}
                    loaded = True
                if loaded:
                    logger.info(
                        "术语词典从种子文件加载成功: %s，条目数=%d",
                        self._term_path,
                        len(self._terms),
                    )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("术语词典加载失败(%s)，使用内置兜底词典: %s", self._term_path, exc)

        if not loaded:
            self._terms = dict(_FALLBACK_TERMS)
            logger.info("使用内置兜底术语词典，条目数=%d", len(self._terms))

        # 构建同义词反向索引
        self._synonym_index = {}
        for term, info in self._terms.items():
            self._synonym_index[term.lower()] = term
            for syn in info.get("synonyms", []):
                self._synonym_index[syn.lower()] = term

    # ------------------------------------------------------------------
    # 查询扩展
    # ------------------------------------------------------------------
    def expand_query(self, query: str) -> tuple[str, list[str]]:
        """扩展查询，注入术语同义词。

        扫描 query 中是否出现术语或其同义词；若命中，将同义词以 OR 形式追加
        到 query 末尾，提升 BM25 对术语类文档的召回。

        Parameters
        ----------
        query : str
            原始用户查询。

        Returns
        -------
        tuple[str, list[str]]
            ``(expanded_query, term_hits)``。
            - ``expanded_query``：扩展后的检索文本。若未命中任何术语，则与原
              query 相同。
            - ``term_hits``：命中的术语主名列表（去重），用于后续词项加权。

        示例
        -----
        >>> expander.expand_query("车载存储模块的合规要求")
        ("车载存储模块的合规要求 OR 车规 eMMC OR 车规eMMC", ["车规 eMMC"])
        """
        if not query:
            return query, []

        query_lower = query.lower()
        term_hits: list[str] = []
        seen_terms: set[str] = set()

        # 遍历同义词索引，检查 query 是否包含某术语/同义词
        # 注意：按长度降序匹配，优先匹配长术语，避免短词误命中
        sorted_keys = sorted(self._synonym_index.keys(), key=len, reverse=True)
        for syn_lower in sorted_keys:
            if syn_lower in query_lower:
                main_term = self._synonym_index[syn_lower]
                if main_term not in seen_terms:
                    seen_terms.add(main_term)
                    term_hits.append(main_term)

        if not term_hits:
            return query, []

        # 收集所有同义词（去重），以 OR 注入
        expansion_parts: list[str] = []
        expansion_seen: set[str] = set()
        for term in term_hits:
            info = self._terms.get(term, {})
            for syn in info.get("synonyms", []):
                if syn not in expansion_seen and syn.lower() not in query_lower:
                    expansion_seen.add(syn)
                    expansion_parts.append(syn)

        if not expansion_parts:
            # 命中术语但同义词已在 query 中，无需扩展
            return query, term_hits

        expanded_query = f"{query} OR " + " OR ".join(expansion_parts)
        logger.debug("查询扩展: '%s' -> '%s'，命中术语=%s", query, expanded_query, term_hits)
        return expanded_query, term_hits

    # ------------------------------------------------------------------
    # 词项加权
    # ------------------------------------------------------------------
    def boost_term_weight(self, tokens: list[str], term_hits: list[str]) -> list[tuple[str, float]]:
        """对术语命中的 token 施加权重提升。

        将分词后的 token 列表转为 ``(token, weight)`` 列表，术语相关的 token
        权重为 ``TERM_WEIGHT_BOOST``（默认 2.0），其余为 1.0。BM25 检索器在
        打分时会读取该权重，对高权 token 的 tf 贡献放大。

        Parameters
        ----------
        tokens : list[str]
            分词后的 token 列表。
        term_hits : list[str]
            ``expand_query`` 返回的命中术语主名列表。

        Returns
        -------
        list[tuple[str, float]]
            ``[(token, weight), ...]``。
        """
        if not term_hits:
            return [(tok, 1.0) for tok in tokens]

        # 构建术语全词集合（主名 + 同义词的小写形式）
        boost_words: set[str] = set()
        for term in term_hits:
            boost_words.add(term.lower())
            info = self._terms.get(term, {})
            for syn in info.get("synonyms", []):
                boost_words.add(syn.lower())

        result: list[tuple[str, float]] = []
        for tok in tokens:
            weight = self.TERM_WEIGHT_BOOST if tok.lower() in boost_words else 1.0
            result.append((tok, weight))
        return result

    # ------------------------------------------------------------------
    # 动态扩展
    # ------------------------------------------------------------------
    def add_term(self, term: str, synonyms: list[str], type_: str = "custom") -> None:
        """运行时新增术语，供运营后台动态扩展词典。

        新增后会同步更新同义词反向索引。注意：本方法不持久化到磁盘，重启后
        需重新加载；生产环境应配合术语管理后台落库。
        """
        if not term:
            return
        self._terms[term] = {"synonyms": list(synonyms), "type": type_}
        self._synonym_index[term.lower()] = term
        for syn in synonyms:
            self._synonym_index[syn.lower()] = term
        logger.info("术语动态新增: %s (synonyms=%s, type=%s)", term, synonyms, type_)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def terms(self) -> dict[str, dict]:
        """当前术语词典（只读视图）。"""
        return dict(self._terms)

    @property
    def term_count(self) -> int:
        """术语条目数。"""
        return len(self._terms)


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_terminology_instance: TerminologyExpander | None = None


def get_terminology() -> TerminologyExpander:
    """获取全局术语扩展器单例。"""
    global _terminology_instance
    if _terminology_instance is None:
        _terminology_instance = TerminologyExpander()
    return _terminology_instance
