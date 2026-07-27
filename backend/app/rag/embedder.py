"""app.rag.embedder —— BGE-M3 向量化器

设计要点
--------
本模块负责将文本转换为 1024 维稠密向量，供 Milvus 向量检索使用。采用
``BAAI/bge-m3`` 多语言嵌入模型，对企业中英混排技术文档有良好效果。

关键设计决策：

1. **懒加载 + 单例**：sentence-transformers 模型加载耗时数秒且占内存，
   故在首次调用 ``embed`` 时才加载，避免服务启动阻塞。类级缓存确保整个
   进程只加载一次。
2. **Redis 缓存**：对热点 query 的向量做 Redis 缓存（key=``emb:{md5(text)}``，
   TTL=86400s）。企业 RAG 场景下 query 重复率高（FAQ、热门问题），
   缓存命中可省去模型推理开销。
3. **降级策略**：模型不可用时（离线模式 / GPU 不可用 / 模型文件缺失）
   返回固定 seed 的随机向量，保证检索链路不中断（向量检索退化为近似随机
   召回，但 BM25 通路仍正常）。固定 seed 确保同一文本的「随机向量」一致，
   避免缓存与索引不一致。

为何独立实现而非复用 ingestion 模块
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ingestion 模块已有向量化逻辑，但本模块独立实现以避免循环依赖：
``app.ingestion`` 依赖 ``app.rag.milvus_store``，若 ``app.rag.embedder``
反向依赖 ``app.ingestion`` 会形成环。两处逻辑相似但职责不同——ingestion
侧面向批量索引（吞吐优先），检索侧面向单 query 低延迟（缓存优先）。
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from typing import Any, Dict, List, Optional

from app.config import settings
from app.utils.logger import logger

try:
    import redis.asyncio as aioredis

    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore
    _REDIS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False


class BGEM3Embedder:
    """BGE-M3 向量化器（懒加载 + Redis 缓存 + 降级）。

    Parameters
    ----------
    model_name : str | None
        模型名，默认取 settings.embedding_model。
    dimension : int
        向量维度，默认取 settings.dimension（1024）。
    redis_url : str | None
        Redis 连接地址，None 则不启用缓存。
    """

    # 向量缓存 key 前缀
    _CACHE_PREFIX = "emb:"
    _CACHE_TTL = 86400  # 24 小时

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        redis_url: Optional[str] = None,
    ) -> None:
        self._model_name: str = (
            model_name or getattr(settings, "embedding_model", "BAAI/bge-m3")
        )
        self._dimension: int = dimension or getattr(settings, "dimension", 1024)
        self._redis_url: Optional[str] = redis_url or getattr(
            settings, "redis_url", None
        )

        # 懒加载的模型实例
        self._model: Optional["SentenceTransformer"] = None
        self._model_loaded: bool = False
        self._load_attempted: bool = False

        # Redis 客户端（懒连接）
        self._redis = None
        self._redis_connected: bool = False

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """懒加载 sentence-transformers 模型。"""
        if self._load_attempted:
            return
        self._load_attempted = True

        # 注意: 离线模式 (is_offline_mode) 仅用于跳过 LLM 远程 API 调用,
        # BGE-M3 是本地模型, 不依赖网络, 因此即使在离线模式下也应加载,
        # 以支持真实向量检索 (P0 优化). 仅当 sentence-transformers 未安装
        # 或模型文件缺失时才降级为随机向量.
        if not _ST_AVAILABLE:
            logger.warning(
                "sentence-transformers 未安装，向量化将降级为随机向量"
            )
            return

        # 从 settings 读取设备 (cpu / cuda / cuda:0 等)
        device = getattr(settings, "embedding_device", "cpu")
        # GPU 可用性校验: 若指定 cuda 但环境无 GPU, 自动降级到 cpu
        # 避免 CUDA 不可用时抛异常后降级为无意义的随机向量
        if device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning(
                        "配置 device=%s 但 CUDA 不可用, 自动降级为 cpu", device
                    )
                    device = "cpu"
                else:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(
                        "检测到 GPU: %s (%.1f GB), 将用于 BGE-M3 推理",
                        gpu_name, gpu_mem,
                    )
            except ImportError:
                logger.warning("torch 未安装, 无法使用 CUDA, 降级为 cpu")
                device = "cpu"

        try:
            self._model = SentenceTransformer(self._model_name, device=device)
            # FP16 半精度: GPU 下吞吐约 2x, CPU 下无效自动忽略
            if getattr(settings, "embedding_use_fp16", False) and device.startswith("cuda"):
                try:
                    import torch

                    self._model.model = self._model.model.half()
                    logger.info("BGE-M3 已切换 FP16 半精度推理")
                except Exception as fp16_err:  # noqa: BLE001
                    logger.warning("FP16 切换失败, 沿用 FP32: %s", fp16_err)
            self._model_loaded = True
            logger.info(
                "BGE-M3 模型加载成功: %s, device=%s, 维度=%d",
                self._model_name, device, self._dimension,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("BGE-M3 模型加载失败 (device=%s)，降级为随机向量: %s", device, exc)
            self._model_loaded = False

    # ------------------------------------------------------------------
    # Redis 缓存
    # ------------------------------------------------------------------
    async def _get_redis(self):
        """懒获取 Redis 连接。"""
        if not _REDIS_AVAILABLE or not self._redis_url:
            return None
        if not self._redis_connected:
            try:
                max_conn = int(getattr(settings, "redis_max_connections", 64))
                pool = aioredis.BlockingConnectionPool.from_url(
                    self._redis_url,
                    decode_responses=False,
                    max_connections=max_conn,
                )
                self._redis = aioredis.Redis(connection_pool=pool)
                await self._redis.ping()
                self._redis_connected = True
                logger.debug("Embedder Redis 缓存已连接 (pool_max=%d)", max_conn)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedder Redis 连接失败，缓存降级: %s", exc)
                self._redis = None
                self._redis_connected = False
        return self._redis

    def _cache_key(self, text: str) -> str:
        """生成缓存 key: emb:{md5(text)}。"""
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{self._CACHE_PREFIX}{text_hash}"

    async def _cache_get(self, text: str) -> Optional[List[float]]:
        """从 Redis 读取缓存的向量。"""
        redis = await self._get_redis()
        if redis is None:
            return None
        try:
            import json

            raw = await redis.get(self._cache_key(text))
            if raw is not None:
                return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Embedder 缓存读取失败: %s", exc)
        return None

    async def _cache_set(self, text: str, vector: List[float]) -> None:
        """写入向量到 Redis。"""
        redis = await self._get_redis()
        if redis is None:
            return
        try:
            import json

            await redis.set(
                self._cache_key(text),
                json.dumps(vector),
                ex=self._CACHE_TTL,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Embedder 缓存写入失败: %s", exc)

    # ------------------------------------------------------------------
    # 向量化
    # ------------------------------------------------------------------
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化。

        Parameters
        ----------
        texts : list[str]
            待向量化的文本列表。

        Returns
        -------
        list[list[float]]
            向量列表，每个向量维度为 ``self._dimension``。

        Notes
        -----
        优先查 Redis 缓存，命中则直接返回；未命中的文本调用模型推理后回写
        缓存。模型不可用时返回固定 seed 随机向量，保证维度一致。
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        miss_indices: List[int] = []
        miss_texts: List[str] = []

        # 1. 查缓存
        for i, text in enumerate(texts):
            cached = await self._cache_get(text)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        # 2. 全部命中则直接返回
        if not miss_texts:
            return results  # type: ignore[return-value]

        # 3. 未命中的走模型推理
        self._load_model()
        new_vectors: List[List[float]] = []

        if self._model_loaded and self._model is not None:
            try:
                # sentence-transformers 的 encode 是同步 CPU 密集操作，
                # 放到线程池避免阻塞事件循环
                loop = asyncio.get_event_loop()
                raw_vectors = await loop.run_in_executor(
                    None, self._model.encode, miss_texts
                )
                new_vectors = [list(v) for v in raw_vectors]
                logger.debug("BGE-M3 推理完成: %d 条文本", len(miss_texts))
            except Exception as exc:  # noqa: BLE001
                logger.error("BGE-M3 推理失败，降级为随机向量: %s", exc)
                new_vectors = [self._random_vector(t) for t in miss_texts]
        else:
            new_vectors = [self._random_vector(t) for t in miss_texts]

        # 4. 回写缓存
        for text, vector in zip(miss_texts, new_vectors):
            await self._cache_set(text, vector)

        # 5. 填充结果
        for idx, vector in zip(miss_indices, new_vectors):
            results[idx] = vector

        return results  # type: ignore[return-value]

    async def embed_one(self, text: str) -> List[float]:
        """单文本向量化（``embed`` 的便捷封装）。"""
        vectors = await self.embed([text])
        return vectors[0] if vectors else self._random_vector(text)

    # ------------------------------------------------------------------
    # 降级
    # ------------------------------------------------------------------
    def _random_vector(self, text: str) -> List[float]:
        """生成固定 seed 的随机向量（降级用）。

        使用文本的 md5 作为随机种子，确保同一文本始终映射到同一向量，
        避免缓存与索引的向量不一致。降级向量无语义意义，但保证维度与
        类型正确，检索链路不报错。
        """
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        vec = [rng.uniform(-1, 1) for _ in range(self._dimension)]
        # L2 归一化，与 BGE-M3 输出一致
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """模型是否已加载。"""
        return self._model_loaded

    @property
    def is_available(self) -> bool:
        """向量化是否可用（模型加载成功或可降级）。"""
        return True  # 总是可用（可降级为随机向量）

    @property
    def model_info(self) -> Dict[str, Any]:
        """模型信息。"""
        return {
            "model_name": self._model_name,
            "dimension": self._dimension,
            "loaded": self._model_loaded,
            "cache_enabled": self._redis_connected,
        }


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_embedder_instance: Optional[BGEM3Embedder] = None


def get_embedder() -> BGEM3Embedder:
    """获取全局向量化器单例。"""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = BGEM3Embedder()
    return _embedder_instance
