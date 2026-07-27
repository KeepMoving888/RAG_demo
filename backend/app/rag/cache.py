"""app.rag.cache —— 检索结果缓存

设计要点
--------
企业 RAG 场景下，用户查询存在显著重复模式（FAQ、热门问题、操作手册查询）。
对完整检索结果做 Redis 缓存可大幅降低后端负载与检索延迟。

缓存策略
~~~~~~~~
- **Key 设计**：``retrieval:{md5(query + department_id + top_k)}``。纳入
  ``department_id`` 是因为权限隔离要求不同部门看到不同结果，不能共享缓存；
  纳入 ``top_k`` 是因为不同 top_k 返回的 chunk 列表不同。
- **存储结构**：Redis Hash，字段包括 ``result``（JSON 序列化的检索结果）、
  ``created_at``、``query``（便于排查）。使用 Hash 而非 String 便于附加
  元信息且支持部分字段更新。
- **TTL**：默认 3600s（1 小时），文档更新时通过 ``invalidate_pattern``
  批量失效。
- **降级**：Redis 不可用时所有操作降级为 no-op，检索链路正常走全量计算。

命中率统计
~~~~~~~~~~
可选上报到 ``app.metrics``，便于监控缓存效果与调优 TTL。
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from app.config import settings
from app.utils.logger import logger

try:
    import redis.asyncio as aioredis

    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore
    _REDIS_AVAILABLE = False

try:
    from app.metrics import record_cache_hit, record_cache_miss
except ImportError:  # noqa: F401  # metrics 可选
    record_cache_hit = None  # type: ignore
    record_cache_miss = None  # type: ignore


class RetrievalCache:
    """检索结果缓存（Redis Hash + 降级 no-op）。

    Parameters
    ----------
    redis_url : str | None
        Redis 连接地址，None 则不启用缓存。
    default_ttl : int
        默认缓存 TTL（秒）。
    """

    _KEY_PREFIX = "retrieval:"

    def __init__(
        self,
        redis_url: str | None = None,
        default_ttl: int = 3600,
    ) -> None:
        self._redis_url: str | None = redis_url or settings_redis_url()
        self._default_ttl: int = default_ttl
        self._redis = None
        self._connected: bool = False

    async def _get_redis(self):
        """懒获取 Redis 连接。"""
        if not _REDIS_AVAILABLE or not self._redis_url:
            return None
        if not self._connected:
            try:
                max_conn = int(getattr(settings, "redis_max_connections", 64))
                # 用 ConnectionPool (非阻塞) 而非 BlockingConnectionPool
                # async 上下文中 BlockingConnectionPool 可能阻塞事件循环
                pool = aioredis.ConnectionPool.from_url(
                    self._redis_url,
                    decode_responses=True,
                    max_connections=max_conn,
                )
                self._redis = aioredis.Redis(connection_pool=pool)
                await self._redis.ping()
                self._connected = True
                logger.info(
                    "RetrievalCache Redis 已连接: url={}, pool_max={}", self._redis_url, max_conn
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RetrievalCache Redis 连接失败，缓存降级: url={}, err={}", self._redis_url, exc
                )
                self._redis = None
                self._connected = False
        return self._redis

    # ------------------------------------------------------------------
    # Key 生成
    # ------------------------------------------------------------------
    def _make_key(self, query: str, department_id: int | None, top_k: int) -> str:
        """生成缓存 key。

        纳入 department_id 与 top_k 确保权限隔离与不同返回数量的结果不串。
        """
        raw = f"{query}|{department_id}|{top_k}"
        text_hash = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"{self._KEY_PREFIX}{text_hash}"

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------
    async def get(
        self,
        query: str,
        department_id: int | None,
        top_k: int,
    ) -> dict[str, Any] | None:
        """读取缓存。

        Returns
        -------
        dict | None
            命中则返回缓存的检索结果（含 ``cache_hit=True`` 标记），未命中或
            Redis 不可用返回 None。
        """
        redis = await self._get_redis()
        if redis is None:
            return None

        key = self._make_key(query, department_id, top_k)
        try:
            raw = await redis.hget(key, "result")
            if raw is not None:
                result = json.loads(raw)
                logger.debug("检索缓存命中: query='%s', dept=%s", query, department_id)
                if record_cache_hit is not None:
                    record_cache_hit("retrieval")
                return result
        except Exception as exc:  # noqa: BLE001
            logger.debug("检索缓存读取失败: %s", exc)

        if record_cache_miss is not None:
            record_cache_miss("retrieval")
        return None

    async def set(
        self,
        query: str,
        department_id: int | None,
        top_k: int,
        result: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """写入缓存。

        Parameters
        ----------
        query : str
            查询文本。
        department_id : int | None
            部门 ID。
        top_k : int
            返回数量。
        result : dict
            检索结果。
        ttl : int | None
            缓存 TTL，None 用默认值。
        """
        redis = await self._get_redis()
        if redis is None:
            return

        key = self._make_key(query, department_id, top_k)
        ttl_val = ttl if ttl is not None else self._default_ttl
        try:
            # 写入 Hash：result + 元信息
            await redis.hset(
                key,
                mapping={
                    "result": json.dumps(result, ensure_ascii=False),
                    "query": query,
                    "department_id": str(department_id),
                    "top_k": str(top_k),
                    "created_at": str(int(time.time())),
                },
            )
            await redis.expire(key, ttl_val)
            logger.debug(
                "检索缓存写入: query='%s', dept=%s, ttl=%ds", query, department_id, ttl_val
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("检索缓存写入失败: %s", exc)

    # ------------------------------------------------------------------
    # 失效
    # ------------------------------------------------------------------
    async def invalidate_pattern(self, pattern: str = "*") -> int:
        """按 pattern 批量失效缓存。

        文档更新 / 删除时调用，失效相关检索结果缓存。默认 pattern="*"
        清空所有检索缓存。

        Parameters
        ----------
        pattern : str
            匹配 key 的 glob 模式，作用于 ``retrieval:`` 前缀。

        Returns
        -------
        int
            删除的 key 数量。
        """
        redis = await self._get_redis()
        if redis is None:
            return 0

        full_pattern = f"{self._KEY_PREFIX}{pattern}"
        deleted = 0
        try:
            # 使用 SCAN 避免 KEYS 阻塞
            async for key in redis.scan_iter(match=full_pattern, count=100):
                await redis.delete(key)
                deleted += 1
            logger.info("检索缓存批量失效: pattern='%s', 删除=%d", full_pattern, deleted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("检索缓存失效失败: %s", exc)
        return deleted

    async def invalidate_all(self) -> int:
        """清空全部检索缓存。"""
        return await self.invalidate_pattern("*")

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        """缓存是否可用。"""
        return self._connected


def settings_redis_url() -> str:
    """从 settings 获取 redis_url，封装异常处理。"""
    try:
        from app.config import settings

        return getattr(settings, "redis_url", "") or ""
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_cache_instance: RetrievalCache | None = None


def get_retrieval_cache() -> RetrievalCache:
    """获取全局检索缓存单例。"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RetrievalCache()
    return _cache_instance
