"""
向量 + 图谱双路融合检索 (核心难点 - 对标生产级 GraphRAG)

用途:
    在向量语义检索之外, 叠加知识图谱关系链路检索, 弥补向量检索在
    「关系型问题」上的盲区, 提升企业知识库对多跳推理问题的召回质量.

双路互补性 (为何需要融合):
    1. 向量检索: 捕获语义相似文档. 擅长「车规 eMMC 的规格参数」这类
       事实型问题; 但对「车规 eMMC 的认证供应商是谁」这类需跨实体
       关系链路推理的问题, 相关文档可能分散在不同 chunk, 向量相似度
       不足以把多跳证据一并召回.
    2. 图谱检索: 捕获实体关系链路. 通过 Cypher 沿 Product->Department
       ->Supplier 等边遍历, 直接定位关系证据; 但图谱节点本身不含原文,
       需反查 source_chunks 回到文档.
    3. 融合: 向量保语义覆盖, 图谱补关系证据, 经去重 + 重排后送入生成.

路由策略 (基于查询特征选择策略, 避免无谓图谱开销):
    - 含关系词 (哪些/有什么/参与了/认证了/属于/管理/引用/协作/供应)
      -> 触发图谱检索 (关系型问题);
    - 含实体名 (产品代号 车规 eMMC / 部门名) -> 双路都触发 (实体+语义);
    - 纯语义查询 -> 仅向量检索 (省去图谱往返延迟).

融合策略标签:
    - vector_only: 仅向量 (纯语义查询 / 图谱不可用 / 图谱无结果);
    - graph_only: 仅图谱 (向量结果为空且图谱有结果);
    - hybrid: 双路均有结果, 基于 chunk_id 去重合并.

降级:
    Neo4j 不可用时返回空 graph_results, fusion_strategy 标记 vector_only,
    主检索链路不受影响.
"""

import re
import time

from app.graphrag.cypher_chain import GraphCypherQAChain
from app.graphrag.neo4j_store import neo4j_store
from app.utils.logger import logger

# 触发图谱检索的关系词 (覆盖组织/供应链/合规/协作四类)
_RELATION_KEYWORDS = [
    "哪些",
    "有什么",
    "有哪些",
    "参与了",
    "参与",
    "认证了",
    "认证",
    "属于",
    "归属",
    "管理",
    "管理了",
    "引用",
    "协作",
    "供应",
    "供应商",
    "定义",
    "负责",
    "关联",
    "关系",
]

# 实体名模式: 产品代号 (混合大小写如 eMMC/LPDDR4X/DDR4) 或中文部门名
# 产品代号: 首字母任意大小写, 后跟 2+ 大写字母或数字 (排除纯小写普通英文词)
_PRODUCT_PATTERN = re.compile(r"[A-Za-z][A-Z0-9]{2,}[A-Za-z0-9]*")
# 部门名: 排除 "部署/部分/处理" 等含 部/处 但非部门后缀的常用词
_DEPARTMENT_PATTERN = re.compile(r"[\u4e00-\u9fa5]{2,8}(?:部(?!署|分)|处(?!理)|中心|科|组)")


class GraphFusion:
    """向量 + 图谱双路融合检索器"""

    def __init__(self) -> None:
        self._cypher_chain = GraphCypherQAChain()

    # ======================== 路由判断 ========================
    async def should_use_graph(self, query: str) -> bool:
        """基于查询特征判断是否触发图谱检索

        - 含关系词 -> True (关系型问题);
        - 含实体名 -> True (双路触发);
        - 纯语义 -> False (仅向量, 省 Cypher 往返).
        """
        if not query:
            return False
        if any(kw in query for kw in _RELATION_KEYWORDS):
            return True
        if _PRODUCT_PATTERN.search(query) or _DEPARTMENT_PATTERN.search(query):
            return True
        return False

    # ======================== 图谱检索 ========================
    async def graph_retrieve(
        self,
        query: str,
        department_id: int | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """图谱检索: NL->Cypher->执行->反查 chunk

        流程:
            1. GraphCypherQAChain 将 query 转 Cypher 并执行, 得到图谱节点;
            2. 从节点 source_chunks 收集 chunk_id;
            3. 反查 DocumentChunk 表还原原文, 并按部门权限过滤;
            4. 转为与 vector_results 同结构的 chunk dict, 标记 source="graph".

        Args:
            query: 自然语言查询.
            department_id: 部门权限过滤 (NULL 视为全公司可见).
            top_k: 返回上限.

        Returns:
            chunk dict 列表, 字段: chunk_id, content, document_id,
            score, source="graph".
        """
        if not neo4j_store.is_available:
            return []

        result = await self._cypher_chain.query(query, department_id)
        records = result.get("records", [])
        if not records:
            return []

        # 收集所有 source_chunk_id (节点 + 关系上携带的来源)
        chunk_ids: set[int] = set()
        for rec in records:
            for value in rec.values():
                if isinstance(value, dict):
                    for cid in value.get("source_chunks", []) or []:
                        if isinstance(cid, int):
                            chunk_ids.add(cid)
                elif isinstance(value, list):
                    for cid in value:
                        if isinstance(cid, int):
                            chunk_ids.add(cid)

        if not chunk_ids:
            return []

        # 反查 DocumentChunk, 按部门权限过滤
        from sqlalchemy import select

        from app.database import db_session
        from app.models import DocumentChunk

        async with db_session() as session:
            res = await session.execute(
                select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
            )
            chunks = res.scalars().all()

        # 部门权限: NULL (全公司) 或匹配 department_id
        visible = []
        for c in chunks:
            if c.department_id is None or c.department_id == department_id:
                visible.append(c)

        # 保持图谱命中顺序 (chunk_ids 为 set 无序, 这里按 id 稳定排序)
        visible.sort(key=lambda c: c.id)
        results = [
            {
                "chunk_id": c.id,
                "content": c.content,
                "document_id": c.document_id,
                "score": 0.0,  # 图谱路径命中, 无向量相似分
                "source": "graph",
            }
            for c in visible[:top_k]
        ]
        logger.info("图谱检索召回 {} chunks (可见 {}/{})", len(results), len(visible), len(chunks))
        return results

    # ======================== 融合主入口 ========================
    async def fuse(
        self,
        query: str,
        vector_results: list[dict],
        department_id: int | None = None,
        top_k: int = 5,
    ) -> dict:
        """向量 + 图谱双路融合

        Args:
            query: 用户查询.
            vector_results: 上游向量检索 (含 BM25/RRF 融合) 结果,
                每条至少含 chunk_id / content / score.
            department_id: 部门权限.
            top_k: 最终返回数量.

        Returns:
            {
              "chunks": list[dict],          # 融合去重后的最终结果
              "graph_results": list[dict],   # 图谱路原始结果 (供溯源展示)
              "fusion_strategy": str,        # vector_only / graph_only / hybrid
              "latency_ms": float,
            }
        """
        start = time.time()

        # 1. 路由: 是否需要图谱
        use_graph = await self.should_use_graph(query)

        # 2. 降级: 图谱不可用或无需图谱 -> 仅向量
        if not use_graph or not neo4j_store.is_available:
            latency_ms = (time.time() - start) * 1000
            strategy = "vector_only"
            logger.debug(
                "融合策略={} (use_graph={} available={})",
                strategy,
                use_graph,
                neo4j_store.is_available,
            )
            return {
                "chunks": vector_results[:top_k],
                "graph_results": [],
                "fusion_strategy": strategy,
                "latency_ms": latency_ms,
            }

        # 3. 图谱检索
        graph_results = await self.graph_retrieve(query, department_id, top_k)

        # 图谱无结果 -> 仍仅向量
        if not graph_results:
            latency_ms = (time.time() - start) * 1000
            return {
                "chunks": vector_results[:top_k],
                "graph_results": [],
                "fusion_strategy": "vector_only",
                "latency_ms": latency_ms,
            }

        # 4. 策略判定
        if not vector_results:
            strategy = "graph_only"
        else:
            strategy = "hybrid"

        # 5. 基于 chunk_id 去重融合
        #    向量结果优先 (语义分高), 图谱结果补充 (关系证据),
        #    已存在的 chunk 标记 source="graph+vector" 表示双路命中.
        seen: set = set()
        fused: list[dict] = []
        for r in vector_results:
            cid = r.get("chunk_id")
            if cid is None or cid in seen:
                continue
            seen.add(cid)
            fused.append(dict(r))

        for r in graph_results:
            cid = r.get("chunk_id")
            if cid is None:
                continue
            if cid in seen:
                # 双路命中: 标记并集来源
                for f in fused:
                    if f.get("chunk_id") == cid:
                        f["source"] = "graph+vector"
                        break
            else:
                seen.add(cid)
                fused.append(dict(r))

        # 6. 精排: 此处置于 CrossEncoderReranker 上游
        #    当前按 score 稳定排序 (图谱 score=0 排后), 重排器可在调用方再接.
        #    保留向量原始分数排序, 图谱补充结果排末尾.
        fused.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        latency_ms = (time.time() - start) * 1000
        logger.info(
            "融合完成 strategy={} 向量={} 图谱={} 融合={} 耗时={:.0f}ms",
            strategy,
            len(vector_results),
            len(graph_results),
            len(fused),
            latency_ms,
        )
        return {
            "chunks": fused[:top_k],
            "graph_results": graph_results,
            "fusion_strategy": strategy,
            "latency_ms": latency_ms,
        }


__all__ = ["GraphFusion"]
