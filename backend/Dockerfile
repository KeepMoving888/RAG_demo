# ============================================================
# Enterprise RAG Knowledge Base - Backend Dockerfile
# 多阶段构建: 减小最终镜像体积
# ============================================================

# ======================== Stage 1: 构建阶段 ========================
FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# 系统依赖 (PyMuPDF / PostgreSQL / 编译工具)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖文件, 利用 Docker 层缓存
COPY requirements-docker.txt .
RUN pip install --prefix=/install -r requirements-docker.txt

# ======================== Stage 2: 运行阶段 ========================
FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 运行时系统依赖 (精简: 仅保留运行所需, 排除编译工具)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 从 builder 拷贝已安装的依赖
COPY --from=builder /install /usr/local

# 拷贝应用代码 (.dockerignore 已排除 tests/__pycache__/.git 等)
COPY . /app

# 健康检查 (P0 优化后 /health 不检查 DB, 适合容器探针)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# 生产: 4 workers (避免 GIL 瓶颈, 见项目性能要求)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
