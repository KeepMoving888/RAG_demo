.PHONY: help up down build restart logs ps dev backend frontend worker flower test format lint eval seed graph-init init-db

help: ## 显示所有命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up: ## 启动全部服务 (PostgreSQL + Redis + RabbitMQ + Milvus + Neo4j + 后端 + 前端 + 监控)
	docker compose up -d

down: ## 停止全部服务
	docker compose down

build: ## 重新构建镜像
	docker compose build

restart: ## 重启后端与 Worker
	docker compose restart backend worker

logs: ## 查看后端日志
	docker compose logs -f backend

ps: ## 查看服务状态
	docker compose ps

dev: ## 启动本地基础依赖 (PostgreSQL + Redis + RabbitMQ + Milvus + Neo4j)
	docker compose up -d postgres redis rabbitmq milvus neo4j

backend: ## 本地热重载后端
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

prod: ## 生产模式多 Worker 启动 (默认 4 Worker, 可 UVICORN_WORKERS=N 调整)
	cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers $${UVICORN_WORKERS:-4}

worker: ## 本地启动 Celery Worker
	cd backend && celery -A app.celery_app worker --loglevel=info --concurrency=4

flower: ## 启动 Celery Flower 监控
	cd backend && celery -A app.celery_app flower --port=5555

frontend: ## 本地热重载前端
	cd frontend && npm run dev

test: ## 运行测试
	cd backend && pytest -v --cov=app --cov-report=term-missing

format: ## 代码格式化
	cd backend && ruff format app tests
	cd frontend && npm run format

lint: ## 代码检查
	cd backend && ruff check app tests
	cd frontend && npm run lint

seed: ## 灌入种子文档与术语词典
	cd backend && python scripts/seed_docs.py

eval: ## 运行 RAG 消融评估
	cd backend && python scripts/eval_rag.py

graph-init: ## 抽取并构建知识图谱
	cd backend && python scripts/extract_graph.py

init-db: ## 初始化数据库表结构
	cd backend && python scripts/init_db.py
