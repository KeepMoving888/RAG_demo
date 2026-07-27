"""
通用响应 Schema

统一响应格式:
{
  "code": 0,        # 0=成功, 非 0=错误码
  "message": "ok",
  "data": {...},
  "request_id": "uuid",
  "timestamp": "2025-..."
}
"""

from datetime import datetime
from typing import Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    """统一响应"""

    code: int = 0
    message: str = "ok"
    data: T | None = None
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SuccessResponse(BaseResponse[T]):
    """成功响应"""

    code: int = 0
    message: str = "ok"


class ErrorResponse(BaseResponse):
    """错误响应"""

    code: int = -1
    message: str = "error"


class MessageResponse(BaseResponse):
    """仅消息响应"""

    data: dict | None = None


class PaginatedResponse(BaseResponse, Generic[T]):
    """分页响应"""

    data: list[T] | None = None
    total: int = 0
    page: int = 1
    page_size: int = 20
    total_pages: int = 0


class PaginationParams(BaseModel):
    """分页参数"""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size
