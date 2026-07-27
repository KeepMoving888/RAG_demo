"""
接口限流: Redis 令牌桶算法

设计要点:
1. 令牌桶: 容量 = burst, 速率 = qpm/60 (每秒填充)
2. 按 user_id + endpoint 维度限流, 防止单用户击穿
3. 原子操作: Lua 脚本保证「取令牌」并发安全
4. 降级: Redis 不可用时放行, 不影响业务可用性
"""

import time
from typing import Optional

from fastapi import HTTPException, Request, status

from app.config import settings
from app.utils.logger import logger


# Lua 脚本: 原子取令牌 (避免竞态)
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])  -- tokens per second
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1]) or capacity
local last_refill = tonumber(bucket[2]) or now

-- 按时间填充令牌
local elapsed = math.max(0, now - last_refill)
tokens = math.min(capacity, tokens + elapsed * refill_rate)

local allowed = 0
if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
redis.call('EXPIRE', key, 120)
return allowed
"""


class RateLimiter:
    """Redis 令牌桶限流器"""

    _redis = None

    @classmethod
    async def _get_redis(cls):
        if cls._redis is None:
            try:
                import redis.asyncio as aioredis
                cls._redis = aioredis.from_url(
                    settings.redis_url, decode_responses=True
                )
            except Exception as e:
                logger.warning("Redis 不可用, 限流降级放行: {}", str(e))
                return None
        return cls._redis

    @classmethod
    async def acquire(
        cls,
        key: str,
        capacity: Optional[int] = None,
        refill_per_second: Optional[float] = None,
    ) -> bool:
        """
        尝试获取令牌

        Args:
            key: 限流 key (建议 user_id:endpoint)
            capacity: 桶容量 (突发上限), 默认 = rate_limit_burst
            refill_per_second: 每秒填充速率, 默认 = rate_limit_qpm/60

        Returns:
            True 允许 / False 拒绝
        """
        redis = await cls._get_redis()
        if redis is None:
            return True  # Redis 不可用, 放行

        capacity = capacity or settings.rate_limit_burst
        refill = refill_per_second or (settings.rate_limit_qpm / 60.0)

        try:
            allowed = await redis.eval(
                _TOKEN_BUCKET_LUA,
                1,
                f"ratelimit:{key}",
                capacity,
                refill,
                time.time(),
                1,
            )
            return bool(allowed)
        except Exception as e:
            logger.warning("限流脚本执行失败, 放行: {}", str(e))
            return True


async def rate_limit_dependency(
    request: Request,
    endpoint: Optional[str] = None,
) -> None:
    """FastAPI 限流依赖"""
    from app.core.context import request_context

    ctx = request_context.get()
    user_key = ctx.user_id or (request.client.host if request.client else "anonymous")
    endpoint = endpoint or request.url.path

    key = f"{user_key}:{endpoint}"

    allowed = await RateLimiter.acquire(key)
    if not allowed:
        from app.metrics import record_rate_limit
        record_rate_limit(endpoint=endpoint, user_id=str(user_key))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁, 请稍后重试",
            headers={"Retry-After": "1"},
        )
