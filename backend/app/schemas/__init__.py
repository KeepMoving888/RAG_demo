"""Pydantic Schema 聚合"""

from app.schemas.common import (
    BaseResponse,
    PaginatedResponse,
    ErrorResponse,
    SuccessResponse,
    MessageResponse,
)

__all__ = [
    "BaseResponse",
    "PaginatedResponse",
    "ErrorResponse",
    "SuccessResponse",
    "MessageResponse",
]
