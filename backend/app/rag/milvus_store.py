"""app.rag.milvus_store —— Milvus 向量存储（分区级权限隔离）

设计要点：为何用 partition 实现权限隔离
---------------------------------------
企业知识库要求「不同部门只能检索本部门 + 公开文档」。实现行级权限隔离
有两种方案：

1. **应用层过滤**：检索后按 department_id 过滤结果。
   - 缺点：向量检索先在全库召回 top-k，再过滤可能导致结果不足（若 top-k
     中大量属于其他部门，过滤后所剩无几）；且全库扫描性能差、存在数据
     泄露风险（过滤前的中间结果可能被日志记录）。

2. **分区隔离（本方案）**：为每个 department_id 创建独立 partition，外加
   全局 ``_public`` 公开 partition。检索时只搜索 ``[_public, <dept>]`` 两
   个分区，从向量库层面保证只返回有权限的文档。

   - 优点：
     * **安全性**：未授权部门的向量根本不参与检索，无泄露风险；
     * **性能**：搜索空间从全库缩小到单部门分区，召回质量与速度双提升；
     * **简洁性**：权限逻辑下沉到存储层，应用层无需额外过滤代码。
   - 代价：partition 数量随部门数线性增长，但企业部门数通常 <1000，
     Milvus 单 collection 支持 4096 个 partition，完全够用。

为何选 IVF_FLAT 而非 HNSW
~~~~~~~~~~~~~~~~~~~~~~~~~~
- **IVF_FLAT**：聚类倒排索引，nlist=128 个聚类中心，查询时 nprobe=16
  个聚类。精度高（FLAT 无量化损失）、构建速度快、内存占用适中。企业文档
  量级（百万级 chunk）下查询延迟 <50ms，满足需求。
- **HNSW**：图索引，查询更快但内存占用高（需维护图结构）、构建慢、且
  对批量插入不友好。本场景文档更新以批量为主，IVF_FLAT 更合适。
- 若未来数据量达千万级且延迟要求更严，可平滑切换 HNSW，schema 不变。

Collection Schema
~~~~~~~~~~~~~~~~~
::

    Field            Type            说明
    ----             ----            ----
    id               INT64 (PK)      auto_id 主键
    chunk_id         VARCHAR(64)     业务 chunk 唯一标识
    document_id      INT64           所属文档 ID
    department_id    INT64           所属部门 ID（用于回溯）
    content          VARCHAR(4096)   chunk 文本（截断）
    heading_path     VARCHAR(512)    标题路径
    page_number      INT16           页码
    embedding        FLOAT_VECTOR    1024 维 BGE-M3 向量

降级策略
~~~~~~~~
Milvus 不可用时（未安装 / 连接失败），所有方法降级返回空列表并记录
warning，保证 BM25 通路仍可用，检索链路不中断。
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.config import settings
from app.utils.logger import logger

try:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    _MILVUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    Collection = None  # type: ignore
    CollectionSchema = None  # type: ignore
    DataType = None  # type: ignore
    FieldSchema = None  # type: ignore
    connections = None  # type: ignore
    utility = None  # type: ignore
    _MILVUS_AVAILABLE = False
    logger.warning("pymilvus 未安装，Milvus 向量检索将不可用")


class MilvusStore:
    """Milvus 向量存储（分区级权限隔离）。

    每个 department_id 对应一个 partition，外加 ``_public`` 公开 partition。
    检索时只搜 ``[_public, <dept>]``，实现行级权限隔离。

    Parameters
    ----------
    host : str
        Milvus 主机，默认取 settings.milvus_host。
    port : int
        Milvus 端口，默认取 settings.milvus_port。
    collection_name : str
        Collection 名，默认取 settings.milvus_collection。
    dimension : int
        向量维度，默认取 settings.dimension（1024）。
    """

    # 公开分区名
    PUBLIC_PARTITION = "_public"
    # 部门分区名前缀
    DEPT_PARTITION_PREFIX = "dept_"
    # 索引参数 (运行时从 settings 覆盖, 支持热调优)
    INDEX_TYPE = "IVF_FLAT"
    METRIC_TYPE = "COSINE"
    NLIST = 128
    NPROBE = 16

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        dimension: int | None = None,
    ) -> None:
        self._host: str = host or getattr(settings, "milvus_host", "localhost")
        self._port: int = port or getattr(settings, "milvus_port", 19530)
        self._collection_name: str = collection_name or getattr(
            settings, "milvus_collection", "rag_chunks"
        )
        self._dimension: int = dimension or getattr(settings, "dimension", 1024)

        self._collection: Collection | None = None
        self._connected: bool = False
        self._initialized: bool = False
        # 已创建的 partition 集合（避免重复 ensure）
        self._partitions: set[str] = set()
        # 索引参数: 优先用 settings 覆盖, 支持运行时调优
        self._nlist: int = int(getattr(settings, "milvus_nlist", self.NLIST))
        self._nprobe: int = int(getattr(settings, "milvus_nprobe", self.NPROBE))

    # ------------------------------------------------------------------
    # 连接与初始化
    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        """建立 Milvus 连接。"""
        if not _MILVUS_AVAILABLE:
            return False
        if self._connected:
            return True
        try:
            connections.connect(
                alias="default",
                host=self._host,
                port=str(self._port),
            )
            self._connected = True
            logger.info("Milvus 连接成功: %s:%s", self._host, self._port)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus 连接失败，向量检索降级: %s", exc)
            self._connected = False
            return False

    def _build_schema(self) -> CollectionSchema:
        """构造 Collection schema。"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="document_id", dtype=DataType.INT64),
            FieldSchema(name="department_id", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="heading_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page_number", dtype=DataType.INT16),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._dimension,
            ),
        ]
        return CollectionSchema(
            fields=fields,
            description="RAG chunk 向量索引（分区级权限隔离）",
            enable_dynamic_field=False,
        )

    async def init_collection(self) -> bool:
        """初始化 Collection：创建 collection + 默认 _public partition + 索引。

        在 ``app.main`` 的 lifespan 中调用，确保服务启动时 Collection 就绪。

        Returns
        -------
        bool
            是否初始化成功（Milvus 不可用时返回 False）。
        """
        if not self._connect():
            return False

        try:
            # 在线程池中执行同步的 pymilvus 调用
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_collection_sync)
            self._initialized = True
            logger.info("Milvus Collection 初始化完成: %s", self._collection_name)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Milvus Collection 初始化失败: %s", exc)
            return False

    def _init_collection_sync(self) -> None:
        """同步初始化 Collection（在线程池中执行）。"""
        if utility.has_collection(self._collection_name):
            self._collection = Collection(self._collection_name)
            logger.info("Milvus Collection 已存在，直接加载: %s", self._collection_name)
        else:
            schema = self._build_schema()
            self._collection = Collection(
                name=self._collection_name,
                schema=schema,
                using="default",
            )
            logger.info("Milvus Collection 已创建: %s", self._collection_name)

        # 创建 _public partition (容错: 已存在则跳过)
        existing_partitions = []
        try:
            # partitions 返回 Partition 对象列表, 取 name 做字符串比对
            existing_partitions = [getattr(p, "name", str(p)) for p in self._collection.partitions]
        except Exception as part_err:  # noqa: BLE001
            logger.debug("读取 partitions 列表失败: %s", part_err)

        if self.PUBLIC_PARTITION not in existing_partitions:
            try:
                self._collection.create_partition(self.PUBLIC_PARTITION)
                logger.debug("Milvus partition 已创建: %s", self.PUBLIC_PARTITION)
            except Exception as create_err:  # noqa: BLE001
                logger.debug("创建 partition 失败 (可能已存在): %s", create_err)
        self._partitions.add(self.PUBLIC_PARTITION)

        # 创建索引（若不存在）—— 必须在 load() 之前完成，否则 load 会因
        # "index not found" 报错 (MilvusException code=700)
        if not self._collection.has_index():
            index_params = {
                "index_type": self.INDEX_TYPE,
                "metric_type": self.METRIC_TYPE,
                "params": {"nlist": self._nlist},
            }
            try:
                self._collection.create_index(field_name="embedding", index_params=index_params)
                logger.info(
                    "Milvus 索引已创建: %s, metric=%s, nlist=%d",
                    self.INDEX_TYPE,
                    self.METRIC_TYPE,
                    self._nlist,
                )
            except Exception as idx_err:  # noqa: BLE001
                # 索引已存在或并发创建时, Milvus 报错; 忽略并继续 load
                logger.debug("创建索引失败 (可能已存在): %s", idx_err)

        # 加载到内存 (此时索引必然存在)
        try:
            self._collection.load()
        except Exception as load_err:  # noqa: BLE001
            logger.warning("Milvus load 失败 (尝试继续): %s", load_err)

    # ------------------------------------------------------------------
    # Partition 管理
    # ------------------------------------------------------------------
    def _dept_partition_name(self, department_id: int) -> str:
        """部门 ID -> partition 名。"""
        return f"{self.DEPT_PARTITION_PREFIX}{department_id}"

    async def ensure_partition(self, department_id: int) -> bool:
        """确保部门 partition 存在，不存在则创建。

        Parameters
        ----------
        department_id : int
            部门 ID。

        Returns
        -------
        bool
            是否成功。
        """
        if not self._connected or self._collection is None:
            return False

        partition_name = self._dept_partition_name(department_id)
        if partition_name in self._partitions:
            return True

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._ensure_partition_sync, partition_name)
            self._partitions.add(partition_name)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("创建 partition '%s' 失败: %s", partition_name, exc)
            return False

    def _ensure_partition_sync(self, partition_name: str) -> None:
        """同步创建 partition。"""
        existing_partitions = []
        try:
            existing_partitions = [
                getattr(p, "name", str(p))
                for p in self._collection.partitions  # type: ignore[union-attr]
            ]
        except Exception as part_err:  # noqa: BLE001
            logger.debug("读取 partitions 列表失败: %s", part_err)

        if partition_name not in existing_partitions:
            try:
                self._collection.create_partition(partition_name)  # type: ignore[union-attr]
                logger.debug("Milvus partition 已创建: %s", partition_name)
            except Exception as create_err:  # noqa: BLE001
                logger.debug(
                    "创建 partition '%s' 失败 (可能已存在): %s",
                    partition_name,
                    create_err,
                )

    # ------------------------------------------------------------------
    # 插入
    # ------------------------------------------------------------------
    async def insert_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        department_id: int | None,
    ) -> int:
        """批量插入 chunk 向量。

        被 ``app.ingestion.tasks`` 调用，将文档分块向量化后写入 Milvus。

        Parameters
        ----------
        chunks : list[dict]
            chunk 元数据列表，每个 dict 含 chunk_id, document_id, content,
            heading_path, page_number。
        embeddings : list[list[float]]
            与 chunks 一一对应的向量。
        department_id : int | None
            部门 ID。None 则入 ``_public`` partition。

        Returns
        -------
        int
            成功插入的条数。
        """
        if not self._connected or self._collection is None:
            logger.warning("Milvus 未就绪，跳过插入")
            return 0

        if len(chunks) != len(embeddings):
            logger.error("chunks 与 embeddings 数量不一致: %d vs %d", len(chunks), len(embeddings))
            return 0

        if not chunks:
            return 0

        # 确定目标 partition
        if department_id is None:
            partition_name = self.PUBLIC_PARTITION
        else:
            partition_name = self._dept_partition_name(department_id)
            await self.ensure_partition(department_id)

        # 构造插入数据
        data = [
            [c.get("chunk_id", "") for c in chunks],  # chunk_id
            [c.get("document_id", 0) for c in chunks],  # document_id
            [department_id if department_id is not None else 0 for _ in chunks],  # department_id
            [str(c.get("content", ""))[:4096] for c in chunks],  # content
            [str(c.get("heading_path", ""))[:512] for c in chunks],  # heading_path
            [int(c.get("page_number", 0)) for c in chunks],  # page_number
            embeddings,  # embedding
        ]

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._insert_sync,
                data,
                partition_name,
            )
            logger.info("Milvus 插入完成: %d 条, partition=%s", len(chunks), partition_name)
            return len(chunks)
        except Exception as exc:  # noqa: BLE001
            logger.error("Milvus 插入失败: %s", exc)
            return 0

    def _insert_sync(self, data: list[list[Any]], partition_name: str) -> None:
        """同步插入。"""
        self._collection.insert(data, partition_name=partition_name)  # type: ignore[union-attr]
        # 插入后 flush 使数据可见
        self._collection.flush()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    async def search(
        self,
        query_embedding: list[float],
        department_id: int | None,
        top_k: int,
        recall_k: int,
        search_all_partitions: bool = False,
    ) -> list[dict[str, Any]]:
        """向量检索（分区隔离）。

        只搜索 ``_public`` + 用户部门 partition，实现权限隔离。
        当 ``search_all_partitions=True`` 时跳过权限隔离，搜索全部分区
        (评估场景使用，生产环境慎用)。

        Parameters
        ----------
        query_embedding : list[float]
            查询向量。
        department_id : int | None
            用户部门 ID。None 则只搜 ``_public``。
        top_k : int
            最终返回数量。
        recall_k : int
            召回数量（通常 > top_k，供后续 rerank）。
        search_all_partitions : bool
            是否搜索全部分区（评估场景，绕过权限隔离）。默认 False。

        Returns
        -------
        list[dict]
            检索结果，每个 dict 含 chunk_id, document_id, department_id,
            content, heading_path, page_number, score, source="vector"。
        """
        if not self._connected or self._collection is None:
            logger.debug("Milvus 未就绪，向量检索返回空")
            return []

        # 确定搜索分区列表
        if search_all_partitions:
            # 评估场景: 搜索全部分区 (None 表示不指定 partition, Milvus 默认搜全库)
            partition_names = None
        else:
            partition_names = [self.PUBLIC_PARTITION]
            if department_id is not None:
                dept_part = self._dept_partition_name(department_id)
                if dept_part in self._partitions:
                    partition_names.append(dept_part)

        search_params = {
            "metric_type": self.METRIC_TYPE,
            "params": {"nprobe": self._nprobe},
        }
        # 输出字段
        output_fields = [
            "chunk_id",
            "document_id",
            "department_id",
            "content",
            "heading_path",
            "page_number",
        ]

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._search_sync,
                [query_embedding],
                partition_names,
                search_params,
                output_fields,
                recall_k,
            )
            logger.debug(
                "Milvus 检索完成: partitions=%s, recall_k=%d, 命中=%d",
                partition_names,
                recall_k,
                len(results),
            )
            return results[:top_k]
        except Exception as exc:  # noqa: BLE001
            logger.error("Milvus 检索失败: %s", exc)
            return []

    def _search_sync(
        self,
        query_vectors: list[list[float]],
        partition_names: list[str],
        search_params: dict[str, Any],
        output_fields: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """同步检索。"""
        search_results = self._collection.search(  # type: ignore[union-attr]
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=limit,
            partition_names=partition_names,
            output_fields=output_fields,
        )

        results: list[dict[str, Any]] = []
        for hits in search_results:
            for hit in hits:
                entity = hit.entity.to_dict() if hasattr(hit.entity, "to_dict") else {}
                # pymilvus 不同版本字段获取方式兼容
                row = {
                    "chunk_id": entity.get("chunk_id", "") or hit.entity.get("chunk_id", ""),
                    "document_id": entity.get("document_id", 0) or hit.entity.get("document_id", 0),
                    "department_id": entity.get("department_id", 0)
                    or hit.entity.get("department_id", 0),
                    "content": entity.get("content", "") or hit.entity.get("content", ""),
                    "heading_path": entity.get("heading_path", "")
                    or hit.entity.get("heading_path", ""),
                    "page_number": entity.get("page_number", 0) or hit.entity.get("page_number", 0),
                    "score": float(hit.score),
                    "source": "vector",
                }
                results.append(row)
        return results

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------
    async def delete_by_document(self, document_id: int) -> int:
        """按文档 ID 删除所有相关 chunk（重建索引用）。

        使用 Milvus 的 ``delete`` 表达式删除 ``document_id == <id>`` 的记录。

        Parameters
        ----------
        document_id : int
            文档 ID。

        Returns
        -------
        int
            删除的条数（Milvus 不一定返回精确数，可能返回 -1）。
        """
        if not self._connected or self._collection is None:
            return 0

        try:
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(None, self._delete_sync, document_id)
            logger.info("Milvus 按文档删除: document_id=%d, 删除条数=%s", document_id, count)
            return count if isinstance(count, int) else 0
        except Exception as exc:  # noqa: BLE001
            logger.error("Milvus 删除失败: %s", exc)
            return 0

    def _delete_sync(self, document_id: int) -> int:
        """同步删除。"""
        expr = f"document_id == {document_id}"
        result = self._collection.delete(expr)  # type: ignore[union-attr]
        self._collection.flush()  # type: ignore[union-attr]
        # pymilvus delete 返回 MutationResult，含 delete_count
        if hasattr(result, "delete_count"):
            return int(result.delete_count)
        return 0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        """Milvus 是否可用。"""
        return self._connected and self._collection is not None

    @property
    def collection_name(self) -> str:
        """Collection 名。"""
        return self._collection_name

    @property
    def partitions(self) -> set[str]:
        """已创建的 partition 集合。"""
        return set(self._partitions)


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
# 注意：app.main 通过 `from app.rag.milvus_store import milvus_store` 引用
milvus_store = MilvusStore()
