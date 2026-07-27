"""BGE-M3 向量化器：sentence-transformers 加载 + Redis 缓存 + 优雅降级。

设计要点
========
1. **懒加载 + 单例缓存**：模型加载耗时数秒、占用 GB 级内存，单例化避免重复
   加载；仅在首次 embed 调用时初始化，使模块导入零开销。
2. **Redis 缓存**：相同文本反复出现（如多文档共享模板段落）时命中缓存跳过
   推理，显著降低 GPU/CPU 占用。缓存 key = ``emb:{md5(text)}``，TTL 来自配置。
3. **优雅降级**：模型或 Redis 不可用时降级为随机向量（并记录 warning），
   保证离线环境与测试环境流水线可跑通，向量库 schema 不受影响。
4. **批量向量化**：按 ``settings.embedding_batch_size`` 分批 encode，兼顾吞吐
   与显存占用。
"""
from __future__ import annotations

import asyncio
import hashlib
import random
from typing import Any, Optional

from app.config import settings
from app.utils.logger import logger


class BGEM3Embedder:
    """BGE-M3 嵌入器（1024 维）。

    使用 ``sentence-transformers`` 加载 ``BAAI/bge-m3``。提供批量与单条
    向量化接口，内置 Redis 缓存与降级策略。
    """

    _instance: Optional["BGEM3Embedder"] = None

    def __new__(cls) -> "BGEM3Embedder":
        """单例：全局复用一个模型实例与 Redis 连接。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._model: Any = None
        self._redis: Any = None
        self._dimension: int = getattr(settings, "milvus_dimension", 1024)
        self._batch_size: int = getattr(settings, "embedding_batch_size", 32)
        self._cache_ttl: int = getattr(settings, "embedding_cache_ttl", 86400)
        self._model_name: str = getattr(settings, "embedding_model", "BAAI/bge-m3")
        self._degraded = False
        self._initialized = True

    # ------------------------------------------------------------------
    # 懒加载
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """懒加载 sentence-transformers 模型。

        失败时进入降级模式：后续 embed 返回随机向量，保证流水线在无 GPU/
        无模型权重的环境下仍可端到端运行（如离线测试环境）。
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            self._model = SentenceTransformer(self._model_name)
            # 校验维度一致性，避免入库维度不匹配
            real_dim = self._model.get_sentence_embedding_dimension()
            if real_dim and real_dim != self._dimension:
                logger.warning(
                    f"模型实际维度 {real_dim} 与配置维度 {self._dimension} 不一致，"
                    f"以模型实际维度为准"
                )
                self._dimension = real_dim
            logger.info(f"BGE-M3 模型已加载: {self._model_name}, dim={self._dimension}")
        except Exception as exc:  # noqa: BLE001 任意加载失败均降级
            logger.warning(
                f"加载嵌入模型失败，降级为随机向量（离线模式）: {exc}"
            )
            self._degraded = True
        return self._model

    def _load_redis(self) -> Any:
        """懒加载 Redis 异步客户端，不可用时返回 None。"""
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis  # type: ignore[import-not-found]

            redis_url = getattr(settings, "redis_url", "redis://localhost:6379/0")
            self._redis = aioredis.from_url(redis_url, decode_responses=False)
            logger.info("Redis 客户端已加载（嵌入缓存）")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Redis 不可用，嵌入缓存降级关闭: {exc}")
            self._redis = None
        return self._redis

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。

        先查 Redis 缓存命中部分，未命中部分按 ``embedding_batch_size`` 分批推理，
        结果回写缓存。降级模式下返回随机向量。
        """
        if not texts:
            return []
        # 降级模式直接返回随机向量
        if self._degraded:
            self._load_model()  # 触发降级标记（幂等）
            if self._degraded:
                return [self._random_vector() for _ in texts]

        redis = self._load_redis()
        cache_keys = [self._cache_key(t) for t in texts]

        # 批量查缓存
        cached = await self._mget_cached(redis, cache_keys)
        results: list[Optional[list[float]]] = [
            cached[i] if i < len(cached) else None for i in range(len(texts))
        ]
        # 收集未命中项
        miss_indices = [i for i, r in enumerate(results) if r is None]
        miss_texts = [texts[i] for i in miss_indices]

        if miss_texts:
            new_vectors = await self._encode_batch(miss_texts)
            # 回写缓存并填回结果
            await self._mset_cached(redis, [cache_keys[i] for i in miss_indices], new_vectors)
            for idx, vec in zip(miss_indices, new_vectors):
                results[idx] = vec

        return [r or self._random_vector() for r in results]

    async def embed_one(self, text: str) -> list[float]:
        """单条向量化，等价于 ``embed([text])[0]``。"""
        vectors = await self.embed([text])
        return vectors[0] if vectors else self._random_vector()

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    async def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """分批调用模型推理，CPU 密集故放入线程池避免阻塞事件循环。"""
        model = self._load_model()
        if self._degraded or model is None:
            return [self._random_vector() for _ in texts]

        loop = asyncio.get_event_loop()
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors = await loop.run_in_executor(
                None, lambda b=batch: self._run_encode(model, b)
            )
            all_vectors.extend(vectors)
        return all_vectors

    def _run_encode(self, model: Any, batch: list[str]) -> list[list[float]]:
        """实际调用 SentenceTransformer.encode，封装异常。"""
        try:
            import numpy as np  # type: ignore[import-not-found]

            embeddings = model.encode(
                batch,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [np.asarray(v, dtype=float).tolist() for v in embeddings]
        except Exception as exc:  # noqa: BLE001 推理失败降级为随机向量
            logger.warning(f"模型推理失败，本批降级为随机向量: {exc}")
            return [self._random_vector() for _ in batch]

    def _random_vector(self) -> list[float]:
        """生成降级用的随机向量（已归一化），保证可入库。"""
        vec = [random.gauss(0.0, 1.0) for _ in range(self._dimension)]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _cache_key(text: str) -> str:
        """生成缓存 key：``emb:{md5(text)}``。"""
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"emb:{digest}"

    async def _mget_cached(self, redis: Any, keys: list[str]) -> list[Optional[list[float]]]:
        """批量读缓存，Redis 不可用或解析失败返回全 None。"""
        if redis is None or not keys:
            return [None] * len(keys)
        try:
            raw = await redis.mget(keys)
            results: list[Optional[list[float]]] = []
            for item in raw:
                if item is None:
                    results.append(None)
                    continue
                results.append(self._deserialize(item))
            return results
        except Exception as exc:  # noqa: BLE001 缓存故障不应影响主流程
            logger.debug(f"缓存读取失败，跳过: {exc}")
            return [None] * len(keys)

    async def _mset_cached(self, redis: Any, keys: list[str], vectors: list[list[float]]) -> None:
        """批量写缓存，设置 TTL。"""
        if redis is None or not keys:
            return
        try:
            pipe = redis.pipeline()
            for key, vec in zip(keys, vectors):
                pipe.set(key, self._serialize(vec), ex=self._cache_ttl)
            await pipe.execute()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"缓存写入失败，跳过: {exc}")

    @staticmethod
    def _serialize(vec: list[float]) -> bytes:
        """将向量序列化为紧凑字节，减少 Redis 内存占用。"""
        import struct

        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _deserialize(data: Any) -> Optional[list[float]]:
        """反序列化缓存值为向量。"""
        import struct

        try:
            if isinstance(data, str):
                data = data.encode("latin-1")
            count = len(data) // 4
            return list(struct.unpack(f"{count}f", data))
        except Exception:  # noqa: BLE001
            return None


# 模块级单例：与 __new__ 单例一致，便于直接 import 使用
embedder = BGEM3Embedder()
