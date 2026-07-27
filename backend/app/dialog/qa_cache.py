"""
高频问答缓存 - Redis Hash + LRU 淘汰 + chunk 反向失效

设计要点
========

1. 为何用「文本哈希相等」而非语义相似度作为命中条件?
   - 语义相似度需要向量计算 (embedding), 这本身的开销与一次轻量检索相当,
     违背「缓存以极低成本换取重复请求」的初衷.
   - 文本完全一致 (md5 哈希) 才命中, 命中判定 O(1), 真正实现加速.
   - 语义近似的查询由下游检索层 (BM25 + 向量) 兜底, 缓存只负责「同一问题」.

2. LRU 淘汰: 维护 ZSET qa:cache:lru (member=cache_key, score=最后访问时间戳),
   超过 qa_cache_max_keys 时按 score 升序删除最久未访问的条目, 保证热点常驻.

3. chunk 反向失效: 维护 SET qa:chunkidx:{chunk_id} 记录「引用过该 chunk 的缓存 key」,
   文档更新/删除时按 chunk 反向失效相关缓存, 避免返回过期答案.

4. 命中统计: 每次命中/未命中上报 app.metrics.record_qa_cache, 便于命中率监控;
   并在缓存条目内维护 hit_count 用于热点分析.

5. 降级: Redis 不可用时降级为 no-op (get 返回 None, set 跳过), 不影响主流程.
"""

import hashlib
import json
import time
from typing import Any

from app.config import settings
from app.metrics import record_qa_cache
from app.utils.logger import logger


class QACache:
    """
    高频问答缓存.

    缓存条目 (Redis Hash, key = qa:{md5(rewritten_query|department_id)}):
        {
            "answer": str,
            "citations": list[dict],         # JSON 字符串
            "retrieved_chunk_ids": list[str], # JSON 字符串
            "created_at": float,
            "hit_count": int,
        }
    TTL = settings.qa_cache_ttl (秒)
    """

    KEY_PREFIX = "qa:"
    LRU_KEY = "qa:cache:lru"  # ZSET: member=cache_key, score=访问时间戳
    STATS_KEY = "qa:cache:stats"  # Hash: hits / misses 计数
    CHUNKIDX_PREFIX = "qa:chunkidx:"  # SET: 引用过某 chunk 的缓存 key 集合

    _redis: Any = None
    _redis_broken: bool = False

    def __init__(self) -> None:
        self._ttl: int = settings.qa_cache_ttl
        self._max_keys: int = settings.qa_cache_max_keys

    # ======================== Redis 连接 ========================
    @classmethod
    async def _get_redis(cls) -> Any:
        if cls._redis_broken:
            return None
        if cls._redis is None:
            try:
                import redis.asyncio as aioredis

                max_conn = int(getattr(settings, "redis_max_connections", 64))
                pool = aioredis.BlockingConnectionPool.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    max_connections=max_conn,
                )
                cls._redis = aioredis.Redis(connection_pool=pool)
            except Exception as e:  # pragma: no cover - 依赖环境
                cls._redis_broken = True
                logger.warning("Redis 连接失败, QA 缓存降级为 no-op: {}", str(e))
                return None
        return cls._redis

    def _cache_key(self, rewritten_query: str, department_id: int | None) -> str:
        """生成缓存 key: md5(rewritten_query|department_id)."""
        raw = f"{rewritten_query}|{department_id if department_id is not None else ''}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"{self.KEY_PREFIX}{digest}"

    # ======================== 读取 ========================
    async def get(self, rewritten_query: str, department_id: int | None) -> dict | None:
        """
        查询缓存.

        Returns:
            命中: {answer, citations, retrieved_chunk_ids, created_at, hit_count}
            未命中或 Redis 不可用: None
        """
        redis = await self._get_redis()
        if redis is None:
            record_qa_cache("miss")
            return None

        cache_key = self._cache_key(rewritten_query, department_id)
        try:
            raw = await redis.hgetall(cache_key)
            if not raw:
                record_qa_cache("miss")
                await self._incr_stat("misses")
                return None

            # 命中: 累加 hit_count, 刷新 LRU 时间戳
            pipe = redis.pipeline()
            pipe.hincrby(cache_key, "hit_count", 1)
            pipe.zadd(self.LRU_KEY, {cache_key: time.time()})
            pipe.expire(cache_key, self._ttl)
            await pipe.execute()

            record_qa_cache("hit")
            await self._incr_stat("hits")
            logger.debug("QA 缓存命中: key={}", cache_key[:16])
            return self._decode_entry(raw)
        except Exception as e:
            logger.warning("QA 缓存读取失败, 视为未命中: {}", str(e))
            record_qa_cache("miss")
            return None

    # ======================== 写入 ========================
    async def set(
        self,
        rewritten_query: str,
        department_id: int | None,
        answer: str,
        citations: list[dict],
        retrieved_chunk_ids: list[str],
    ) -> None:
        """写入缓存条目, 并执行 LRU 淘汰与 chunk 反向索引建立."""
        redis = await self._get_redis()
        if redis is None:
            return

        cache_key = self._cache_key(rewritten_query, department_id)
        now = time.time()
        entry = {
            "answer": answer,
            "citations": json.dumps(citations, ensure_ascii=False),
            "retrieved_chunk_ids": json.dumps(retrieved_chunk_ids, ensure_ascii=False),
            "created_at": str(now),
            "hit_count": "0",
        }

        try:
            pipe = redis.pipeline()
            pipe.hset(cache_key, mapping=entry)
            pipe.expire(cache_key, self._ttl)
            pipe.zadd(self.LRU_KEY, {cache_key: now})
            # 建立 chunk 反向索引 (用于文档更新时失效)
            for chunk_id in retrieved_chunk_ids:
                pipe.sadd(f"{self.CHUNKIDX_PREFIX}{chunk_id}", cache_key)
                pipe.expire(f"{self.CHUNKIDX_PREFIX}{chunk_id}", self._ttl)
            await pipe.execute()

            # LRU 淘汰: 超过上限时删除最久未访问的条目
            await self._evict_if_needed(redis)
        except Exception as e:
            logger.warning("QA 缓存写入失败: {}", str(e))

    async def _evict_if_needed(self, redis: Any) -> None:
        """超过 qa_cache_max_keys 时, 按访问时间升序淘汰最旧的条目."""
        try:
            total = await redis.zcard(self.LRU_KEY)
            if total <= self._max_keys:
                return

            # 需淘汰数量
            evict_count = total - self._max_keys
            # 取最早的 evict_count 个 key
            victims = await redis.zrange(self.LRU_KEY, 0, evict_count - 1)
            if not victims:
                return

            pipe = redis.pipeline()
            for key in victims:
                pipe.delete(key)
                pipe.zrem(self.LRU_KEY, key)
            await pipe.execute()
            logger.info("QA 缓存 LRU 淘汰: count={}", len(victims))
        except Exception as e:
            logger.warning("QA 缓存 LRU 淘汰失败: {}", str(e))

    # ======================== 失效 ========================
    async def invalidate_by_chunk(self, chunk_id: str) -> int:
        """
        文档更新/删除时, 按 chunk 反向失效所有引用过该 chunk 的缓存.

        Returns:
            被失效的缓存条目数.
        """
        redis = await self._get_redis()
        if redis is None:
            return 0

        idx_key = f"{self.CHUNKIDX_PREFIX}{chunk_id}"
        try:
            cache_keys = await redis.smembers(idx_key)
            if not cache_keys:
                return 0

            pipe = redis.pipeline()
            for key in cache_keys:
                pipe.delete(key)
                pipe.zrem(self.LRU_KEY, key)
            pipe.delete(idx_key)
            await pipe.execute()

            logger.info("QA 缓存按 chunk 失效: chunk_id={} count={}", chunk_id, len(cache_keys))
            return len(cache_keys)
        except Exception as e:
            logger.warning("QA 缓存 chunk 失效失败: {}", str(e))
            return 0

    # ======================== 统计 ========================
    async def stats(self) -> dict:
        """返回命中率统计: {total_keys, hits, misses, hit_rate}."""
        redis = await self._get_redis()
        if redis is None:
            return {"total_keys": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}

        try:
            pipe = redis.pipeline()
            pipe.zcard(self.LRU_KEY)
            pipe.hgetall(self.STATS_KEY)
            results = await pipe.execute()
            total_keys = results[0] or 0
            stats_raw = results[1] or {}
            hits = int(stats_raw.get("hits", 0))
            misses = int(stats_raw.get("misses", 0))
            total_req = hits + misses
            hit_rate = (hits / total_req) if total_req > 0 else 0.0
            return {
                "total_keys": total_keys,
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hit_rate, 4),
            }
        except Exception as e:
            logger.warning("QA 缓存统计失败: {}", str(e))
            return {"total_keys": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}

    async def _incr_stat(self, field: str) -> None:
        """累加命中/未命中计数."""
        redis = await self._get_redis()
        if redis is None:
            return
        try:
            await redis.hincrby(self.STATS_KEY, field, 1)
        except Exception:
            pass

    # ======================== 编解码 ========================
    @staticmethod
    def _decode_entry(raw: dict) -> dict:
        """Redis Hash 字符串 -> Python 类型."""
        citations_raw = raw.get("citations", "[]")
        chunks_raw = raw.get("retrieved_chunk_ids", "[]")
        try:
            citations = (
                json.loads(citations_raw) if isinstance(citations_raw, str) else citations_raw
            )
        except (json.JSONDecodeError, TypeError):
            citations = []
        try:
            chunk_ids = json.loads(chunks_raw) if isinstance(chunks_raw, str) else chunks_raw
        except (json.JSONDecodeError, TypeError):
            chunk_ids = []
        return {
            "answer": raw.get("answer", ""),
            "citations": citations,
            "retrieved_chunk_ids": chunk_ids,
            "created_at": float(raw.get("created_at", 0) or 0),
            "hit_count": int(raw.get("hit_count", 0) or 0),
        }
