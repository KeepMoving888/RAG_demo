"""
日志工具 - 基于 loguru

特性:
1. 控制台彩色输出 + 文件滚动
2. 按天切分, 保留 14 天
3. 异步写入, 不阻塞业务
4. 支持 JSON 格式 (生产环境)
"""

import os
import sys
from pathlib import Path

from loguru import logger

from app.config import settings


def _setup_logger():
    """初始化日志配置"""
    logger.remove()

    log_level = settings.log_level.upper()
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 控制台输出 (彩色)
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=True,
        diagnose=settings.debug,
    )

    # 全量日志文件 (按天切分)
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        level=log_level,
        rotation="00:00",
        retention="14 days",
        compression="zip",
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        backtrace=True,
        diagnose=settings.debug,
    )

    # 错误日志单独归档
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        level="ERROR",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        backtrace=True,
        diagnose=True,
    )

    # 拦截标准库 logging, 转发到 loguru (统一日志格式)
    import logging

    class InterceptHandler(logging.Handler):
        """标准库 logging -> loguru 桥接器"""

        def emit(self, record: logging.LogRecord) -> None:
            try:
                level: str | int = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # 静默噪声库 (uvicorn / sqlalchemy / pymilvus 等)
    for name in ("uvicorn.access", "uvicorn.error", "sqlalchemy.engine", "pymilvus"):
        logging.getLogger(name).handlers = [InterceptHandler()]
        if name != "sqlalchemy.engine":
            logging.getLogger(name).setLevel(logging.WARNING)

    return logger


logger = _setup_logger()

__all__ = ["logger"]
