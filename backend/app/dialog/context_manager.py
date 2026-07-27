"""
多轮对话上下文管理器 - 滑动窗口 + 长程记忆锚点

设计要点
========

1. 为何采用「滑动窗口」而非全量保留?
   - LLM 上下文窗口与成本约束: 全量历史会导致 prompt 膨胀, 既超出 token 预算,
     又稀释最新问题的注意力, 推高延迟与费用.
   - RAG 场景下, 绝大多数代词指代 (它/这个/那个) 只需最近 2~3 轮即可消解,
     远期对话的边际价值递减.
   - 因此以 dialog_window_size=6 (3 user + 3 assistant) 为热窗口, 既覆盖常见
     指代, 又把单次 prompt 控制在可预期范围.

2. 为何额外保留「首尾锚点」?
   - 纯滑动窗口会丢失会话开篇的话题设定 (如「我们在讨论 X 产品的部署」),
     导致长会话中后期出现主题漂移.
   - 保留首轮 user 消息作为「长程记忆锚点」, 让改写器与生成器始终感知会话主旨,
     兼顾短期指代消解与长期主题一致性.

3. 为何设 dialog_max_turns=20 上限并归档到 PostgreSQL?
   - Redis 是热数据存储, 不宜承载无限增长的会话; 超过上限后归档冷数据到 PG,
     Redis 重置为「首锚点 + 最近窗口」, 兼顾内存占用与历史可追溯.

4. 会话标识: 对外暴露 uuid 字符串 session_id (Redis key), 内部维护 pg_session_id
   映射到 PostgreSQL QASession 主键, 惰性创建以解耦热路径与数据库可用性.

5. 降级策略: Redis 不可用时退化为进程内 dict (仅单进程可用), 记录 warning,
   保证开发与离线模式全链路可运行.
"""

import json
import time
import uuid
from typing import Any, Optional

from app.config import settings
from app.utils.logger import logger


class DialogContextManager:
    """
    基于 Redis Hash 的多轮对话上下文管理器.

    Redis 数据结构 (key = dialog:{session_id}):
        {
            "turns": [<message>, ...],          # 消息列表 (JSON 字符串)
            "department_id": str,                # 部门快照 (权限隔离)
            "user_id": str,
            "created_at": str,                   # unix 时间戳
            "pg_session_id": str,                # QASession 主键 (惰性填充)
        }
    每条 message:
        {"role": "user", "content": "..."}
        {"role": "assistant", "content": "...", "citations": [...]}
    TTL = settings.dialog_session_ttl (秒)
    """

    _redis: Any = None
    _redis_broken: bool = False  # 标记 Redis 不可用, 避免每次重试连接触发异常
    _fallback_store: dict[str, dict] = {}  # Redis 不可用时的进程内降级存储

    KEY_PREFIX = "dialog:"

    def __init__(self) -> None:
        self._window_size: int = settings.dialog_window_size
        self._max_turns: int = settings.dialog_max_turns
        self._ttl: int = settings.dialog_session_ttl

    # ======================== Redis 连接 ========================
    @classmethod
    async def _get_redis(cls) -> Any:
        """惰性获取 Redis 连接 (单例), 失败返回 None 触发降级."""
        if cls._redis_broken:
            return None
        if cls._redis is None:
            try:
                import redis.asyncio as aioredis

                cls._redis = aioredis.from_url(
                    settings.redis_url, decode_responses=True
                )
            except Exception as e:  # pragma: no cover - 依赖环境
                cls._redis_broken = True
                logger.warning("Redis 连接失败, 对话上下文降级为进程内存储: {}", str(e))
                return None
        return cls._redis

    def _key(self, session_id: str) -> str:
        return f"{self.KEY_PREFIX}{session_id}"

    # ======================== 会话生命周期 ========================
    async def create_session(
        self, user_id: int, department_id: Optional[int]
    ) -> str:
        """
        创建新会话, 返回 uuid 形式的 session_id.

        会话元数据写入 Redis Hash; PostgreSQL QASession 记录采用惰性创建
        (首次持久化 QAMessage 时按需建立, 解耦热路径与数据库可用性).
        """
        session_id = str(uuid.uuid4())
        now = time.time()
        payload = {
            "turns": [],
            "department_id": department_id,
            "user_id": user_id,
            "created_at": now,
            "pg_session_id": None,  # 惰性填充
        }

        redis = await self._get_redis()
        if redis is not None:
            try:
                pipe = redis.pipeline()
                pipe.hset(self._key(session_id), mapping=self._encode_payload(payload))
                pipe.expire(self._key(session_id), self._ttl)
                await pipe.execute()
            except Exception as e:
                logger.warning("Redis 写入会话失败, 降级进程内存储: {}", str(e))
                self._fallback_store[session_id] = payload
        else:
            self._fallback_store[session_id] = payload

        logger.debug(
            "创建对话会话: session_id={} user_id={} dept={}",
            session_id, user_id, department_id,
        )
        return session_id

    async def _ensure_pg_session(self, session_id: str) -> Optional[int]:
        """
        惰性确保 PostgreSQL QASession 行存在, 返回其主键 id.
        失败返回 None (持久化将跳过, 但不影响在线问答).
        """
        data = await self._load(session_id)
        if data is None:
            return None

        pg_id = data.get("pg_session_id")
        if pg_id:
            return pg_id

        try:
            from app.database import db_session
            from app.models import QASession

            async with db_session() as db:
                session_row = QASession(
                    user_id=int(data["user_id"]),
                    department_id=(
                        int(data["department_id"])
                        if data.get("department_id") is not None
                        else None
                    ),
                    title=f"dialog:{session_id[:8]}",
                    turn_count=len(data.get("turns", [])),
                    is_archived=False,
                )
                db.add(session_row)
                await db.flush()
                pg_id = session_row.id

            # 回写 pg_session_id 到 Redis (持久化映射)
            await self._set_pg_session_id(session_id, pg_id)
            logger.debug(
                "惰性创建 QASession: session_id={} pg_id={}", session_id, pg_id
            )
            return pg_id
        except Exception as e:
            logger.warning("创建 QASession 失败, 跳过持久化: {}", str(e))
            return None

    async def archive_session(self, session_id: str) -> None:
        """
        归档会话: 更新 PostgreSQL QASession (turn_count / is_archived) 并清理 Redis 热数据.
        QAMessage 已在每轮 generate 中实时写入, 此处仅完成会话级收尾.
        """
        data = await self._load(session_id)
        if data is None:
            logger.warning("归档会话不存在: {}", session_id)
            return

        pg_id = data.get("pg_session_id")
        turn_count = len(data.get("turns", []))

        if pg_id:
            try:
                from sqlalchemy import update

                from app.database import db_session
                from app.models import QASession

                async with db_session() as db:
                    await db.execute(
                        update(QASession)
                        .where(QASession.id == int(pg_id))
                        .values(turn_count=turn_count, is_archived=True)
                    )
            except Exception as e:
                logger.warning("归档 QASession 更新失败: {}", str(e))

        # 清理 Redis 热数据
        redis = await self._get_redis()
        if redis is not None:
            try:
                await redis.delete(self._key(session_id))
            except Exception:
                pass
        self._fallback_store.pop(session_id, None)

        logger.info(
            "会话已归档: session_id={} turns={}", session_id, turn_count
        )

    # ======================== 消息追加 ========================
    async def add_user_message(self, session_id: str, query: str) -> None:
        """追加一轮用户消息."""
        await self._append_turn(
            session_id, {"role": "user", "content": query}
        )

    async def add_assistant_message(
        self, session_id: str, answer: str, citations: list
    ) -> None:
        """追加一轮助手消息 (含引用信息)."""
        await self._append_turn(
            session_id,
            {
                "role": "assistant",
                "content": answer,
                "citations": citations or [],
            },
        )

    async def _append_turn(self, session_id: str, message: dict) -> None:
        """
        追加一轮消息:
        1. 达到 max_turns 上限 -> 先落库当前轮数, 归档, 再以「首锚点 + 本轮」重建会话;
        2. 否则应用滑动窗口 + 锚点裁剪后写回.
        """
        data = await self._load(session_id)
        if data is None:
            logger.warning("会话不存在, 无法追加消息: {}", session_id)
            return

        turns: list[dict] = data.get("turns", [])
        turns.append(message)

        # 达到上限: 归档后重置
        if len(turns) >= self._max_turns:
            logger.info(
                "会话达到最大轮数上限 ({}), 触发归档: {}",
                self._max_turns, session_id,
            )
            # 先保存完整轮数, 使归档时的 turn_count 准确
            await self._save_turns(session_id, turns)
            await self.archive_session(session_id)

            # 以「首锚点 + 本轮」重建 (复用同一 session_id, 视为会话延续)
            first_anchor = turns[0] if turns else None
            new_turns: list[dict] = []
            if first_anchor and first_anchor.get("role") == "user":
                new_turns.append(first_anchor)
            new_turns.append(message)
            await self._reset_session(session_id, data, new_turns)
            return

        # 滑动窗口 + 锚点裁剪
        pruned = self._apply_window(turns)
        await self._save_turns(session_id, pruned)

    def _apply_window(self, turns: list[dict]) -> list[dict]:
        """
        滑动窗口裁剪:
        - 保留首轮 user 消息作为长程记忆锚点;
        - 保留最近 window_size 轮作为热窗口;
        - 中间轮次 LRU 淘汰.
        总轮数 <= window_size + 1 时原样返回.
        """
        if len(turns) <= self._window_size + 1:
            return turns

        # 锚点: 首条 user 消息 (会话开篇通常是用户提问)
        first_anchor = next(
            (t for t in turns if t.get("role") == "user"), turns[0]
        )
        recent = turns[-self._window_size:]

        # 拼接并去重 (避免锚点与 recent 末尾重复)
        pruned: list[dict] = [first_anchor]
        seen = {(first_anchor.get("role"), first_anchor.get("content"))}
        for t in recent:
            key = (t.get("role"), t.get("content"))
            if key not in seen:
                seen.add(key)
                pruned.append(t)
        return pruned

    # ======================== 上下文读取 ========================
    async def get_context(
        self, session_id: str, max_turns: Optional[int] = None
    ) -> list[dict]:
        """
        返回滑动窗口内的消息列表 (已含锚点).

        Args:
            max_turns: 额外上限, 限制返回条数; 优先保留首锚点 + 最近若干轮.
        """
        data = await self._load(session_id)
        if data is None:
            return []
        turns: list[dict] = data.get("turns", [])
        pruned = self._apply_window(turns)
        if max_turns is not None and max_turns > 0 and len(pruned) > max_turns:
            first = pruned[0]
            recent = pruned[-(max_turns - 1):] if max_turns > 1 else []
            pruned = [first] + recent
        return pruned

    async def get_context_text(self, session_id: str) -> str:
        """拼接为 LLM 可读的纯文本历史 (用户/助手 交替标注)."""
        turns = await self.get_context(session_id)
        if not turns:
            return ""
        lines: list[str] = []
        for t in turns:
            role = t.get("role", "user")
            content = t.get("content", "")
            label = "用户" if role == "user" else "助手"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    async def get_turn_count(self, session_id: str) -> int:
        """返回当前会话已记录的轮数 (用于指标上报)."""
        data = await self._load(session_id)
        if data is None:
            return 0
        return len(data.get("turns", []))

    async def get_department_id(self, session_id: str) -> Optional[int]:
        """返回会话的部门快照 (权限隔离用)."""
        data = await self._load(session_id)
        if data is None:
            return None
        return data.get("department_id")

    # ======================== 存储原语 ========================
    async def _load(self, session_id: str) -> Optional[dict]:
        """加载并解码会话 (Redis Hash 字符串 -> Python 类型; 降级读进程内 dict)."""
        redis = await self._get_redis()
        if redis is not None:
            try:
                raw = await redis.hgetall(self._key(session_id))
                if raw:
                    return self._decode_payload(raw)
            except Exception as e:
                logger.warning("Redis 读取会话失败, 降级进程内存储: {}", str(e))
        return self._fallback_store.get(session_id)

    async def _save_turns(self, session_id: str, turns: list[dict]) -> None:
        """写回消息列表, 续期 TTL."""
        redis = await self._get_redis()
        if redis is not None:
            try:
                pipe = redis.pipeline()
                pipe.hset(
                    self._key(session_id),
                    "turns",
                    json.dumps(turns, ensure_ascii=False),
                )
                pipe.expire(self._key(session_id), self._ttl)
                await pipe.execute()
                return
            except Exception as e:
                logger.warning("Redis 写入失败, 降级进程内: {}", str(e))
        # 降级: 进程内 dict (已是解码类型)
        data = self._fallback_store.get(session_id)
        if data is None:
            data = self._empty_payload()
            self._fallback_store[session_id] = data
        data["turns"] = turns

    async def _set_pg_session_id(
        self, session_id: str, pg_id: int
    ) -> None:
        """回写 pg_session_id 映射."""
        redis = await self._get_redis()
        if redis is not None:
            try:
                pipe = redis.pipeline()
                pipe.hset(self._key(session_id), "pg_session_id", str(pg_id))
                pipe.expire(self._key(session_id), self._ttl)
                await pipe.execute()
                return
            except Exception as e:
                logger.warning("Redis 写入失败, 降级进程内: {}", str(e))
        data = self._fallback_store.get(session_id)
        if data is None:
            data = self._empty_payload()
            self._fallback_store[session_id] = data
        data["pg_session_id"] = pg_id

    async def _reset_session(
        self, session_id: str, base_data: dict, turns: list[dict]
    ) -> None:
        """归档后以「首锚点 + 最近轮」重建会话 (复用 session_id, 重置 pg_session_id)."""
        payload = {
            "turns": turns,
            "department_id": base_data.get("department_id"),
            "user_id": base_data.get("user_id", 0),
            "created_at": base_data.get("created_at", time.time()),
            "pg_session_id": None,  # 新延续会话需重新惰性建 QASession
        }
        redis = await self._get_redis()
        if redis is not None:
            try:
                pipe = redis.pipeline()
                pipe.hset(self._key(session_id), mapping=self._encode_payload(payload))
                pipe.expire(self._key(session_id), self._ttl)
                await pipe.execute()
                return
            except Exception as e:
                logger.warning("Redis 重置会话失败, 降级进程内: {}", str(e))
        self._fallback_store[session_id] = payload

    # ======================== 编解码工具 ========================
    @staticmethod
    def _empty_payload() -> dict:
        return {
            "turns": [],
            "department_id": None,
            "user_id": 0,
            "created_at": time.time(),
            "pg_session_id": None,
        }

    @staticmethod
    def _decode_payload(raw: dict) -> dict:
        """Redis Hash 字符串值 -> Python 类型."""
        turns_raw = raw.get("turns", "[]")
        try:
            turns = (
                json.loads(turns_raw)
                if isinstance(turns_raw, str)
                else (turns_raw or [])
            )
        except (json.JSONDecodeError, TypeError):
            turns = []

        dept = raw.get("department_id")
        uid = raw.get("user_id")
        created = raw.get("created_at")
        pg = raw.get("pg_session_id")
        return {
            "turns": turns,
            "department_id": int(dept) if dept else None,
            "user_id": int(uid) if uid else 0,
            "created_at": float(created) if created else 0.0,
            "pg_session_id": int(pg) if pg else None,
        }

    @staticmethod
    def _encode_payload(data: dict) -> dict:
        """Python 类型 -> Redis Hash 字符串值."""
        dept = data.get("department_id")
        uid = data.get("user_id")
        created = data.get("created_at")
        pg = data.get("pg_session_id")
        return {
            "turns": json.dumps(data.get("turns", []), ensure_ascii=False),
            "department_id": str(dept) if dept is not None else "",
            "user_id": str(uid) if uid is not None else "",
            "created_at": str(created) if created is not None else "",
            "pg_session_id": str(pg) if pg is not None else "",
        }
