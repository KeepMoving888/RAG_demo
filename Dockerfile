FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 纯 Python 依赖优先，避免 apt 源波动导致构建失败

COPY requirements.txt ./
RUN /bin/sh -c 'for i in 1 2 3 4 5; do \
  python -m pip install --no-cache-dir --retries 5 --timeout 120 \
    -i https://pypi.org/simple \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt && exit 0; \
  echo "pip install failed, retry: $i"; \
  sleep 5; \
 done; \
 exit 1'

COPY . .

# 默认启动 FastAPI（docker-compose 会覆盖 web 服务命令）
EXPOSE 8001
CMD ["python", "main.py"]
