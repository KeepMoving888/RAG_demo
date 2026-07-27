"""
请求上下文: 基于 contextvars 存储当前请求作用域内的用户/部门/IP

使用场景:
1. 日志关联 (每条日志携带 user_id, 便于排查)
2. 业务代码深度调用时获取当前用户 (避免层层传参)
3. Prometheus 标签按 user 维度统计
"""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class RequestContext:
    """请求上下文"""

    user_id: int | None = None
    email: str | None = None
    department_id: int | None = None
    role: str | None = None
    ip_address: str | None = None
    request_id: str | None = None


# contextvars: 协程安全
request_context: ContextVar[RequestContext] = ContextVar(
    "request_context", default=RequestContext()
)


def set_request_context(ctx: RequestContext) -> None:
    """设置请求上下文 (中间件调用)"""
    request_context.set(ctx)


def get_current_user_id() -> int | None:
    """业务代码深度获取当前用户 ID"""
    return request_context.get().user_id


def get_current_department_id() -> int | None:
    """业务代码深度获取当前部门 ID"""
    return request_context.get().department_id


def get_current_request_id() -> str | None:
    """获取请求 ID (链路追踪)"""
    return request_context.get().request_id
