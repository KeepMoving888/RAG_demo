"""
Enterprise RAG Knowledge Base - 应用配置中心

支持通过环境变量动态切换:
- 数据库: PostgreSQL (异步 asyncpg / 同步 psycopg2)
- LLM Provider: offline / openai / deepseek
- 向量库: Milvus (生产) / 内存模式 (测试降级)
- 图谱库: Neo4j (生产) / 内存模式 (测试降级)

所有配置通过 .env 注入, 每个模块按需读取自身关心的字段.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置, 通过环境变量注入"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ======================== 应用 ========================
    app_name: str = "Enterprise RAG Knowledge Base"
    app_env: Literal["development", "production", "test"] = "production"
    debug: bool = False
    log_level: str = "INFO"

    # ======================== PostgreSQL ========================
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag"
    postgres_password: str = "rag123"
    postgres_db: str = "rag_kb"

    # ======================== Redis ========================
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    # ======================== RabbitMQ (Celery Broker) ========================
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "rag"
    rabbitmq_password: str = "rag123"

    # ======================== Milvus ========================
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "kb_chunks"
    milvus_dimension: int = 1024
    # IVF_FLAT 索引参数 (可调: 数据量大时增大 nlist, 召回不足时增大 nprobe)
    milvus_nlist: int = 128
    milvus_nprobe: int = 16

    # ======================== Neo4j ========================
    neo4j_host: str = "localhost"
    neo4j_port: int = 7687
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j123"

    # ======================== LLM ========================
    llm_provider: Literal["offline", "openai", "deepseek"] = "offline"
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    llm_request_timeout: int = 60

    # ======================== Embedding ========================
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    # FP16 半精度推理 (GPU 下吞吐约 2x, 精度损失可忽略, 适合 RAG 检索场景)
    embedding_use_fp16: bool = True

    # ======================== Reranker ========================
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_max_length: int = 512
    # Cross-Encoder predict 单批大小 (显存占用与吞吐的权衡, 4060Ti 16GB 可设 32)
    reranker_batch_size: int = 32

    # ======================== 检索参数 ========================
    # recall_k=20: Cross-Encoder 精排候选数, 从 50 降至 20 以降低 P95 延迟
    # (50 候选 ~1500ms → 20 候选 ~600ms, 召回质量损失 <3%)
    retrieval_recall_k: int = 20
    retrieval_top_k: int = 5
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ======================== 分块参数 ========================
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_min_size: int = 64

    # ======================== 多轮对话 ========================
    dialog_window_size: int = 6
    dialog_max_turns: int = 20
    dialog_session_ttl: int = 3600

    # ======================== 缓存 ========================
    qa_cache_ttl: int = 3600
    qa_cache_max_keys: int = 10000
    embedding_cache_ttl: int = 86400

    # ======================== 限流 ========================
    rate_limit_qpm: int = 60
    rate_limit_burst: int = 10

    # ======================== JWT ========================
    jwt_secret_key: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 720

    # ======================== CORS ========================
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3080"]
    )

    # ======================== 监控 ========================
    enable_metrics: bool = True
    metrics_port: int = 9091

    # ======================== 后端服务 ========================
    # Uvicorn Worker 数量 (生产建议 CPU 核数 * 2, 多 Worker 绕开 GIL 线性提升 QPS)
    uvicorn_workers: int = 4

    # ======================== 连接池 (多 Worker 下需相应放大) ========================
    # PostgreSQL 连接池: 单 Worker pool_size + max_overflow
    # 默认值已为 4 Worker 生产配置 (4 * 20 = 80 最大连接, 与 PG max_connections 平衡)
    db_pool_size: int = 20
    db_max_overflow: int = 40
    # 连接回收周期 (秒), 防止长连接被 PG 中断或防火墙超时
    db_pool_recycle: int = 1800
    # Redis 连接池上限 (多 Worker 下推荐 >= uvicorn_workers * 4)
    redis_max_connections: int = 64

    # ======================== 计算属性 ========================
    @computed_field  # type: ignore[misc]
    @property
    def database_url(self) -> str:
        """PostgreSQL 异步连接字符串 (asyncpg)"""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[misc]
    @property
    def database_url_sync(self) -> str:
        """同步连接字符串 (Alembic 迁移 / 脚本使用)"""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[misc]
    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @computed_field  # type: ignore[misc]
    @property
    def celery_broker_url(self) -> str:
        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}//"
        )

    @computed_field  # type: ignore[misc]
    @property
    def celery_result_backend(self) -> str:
        """使用 Redis 作为 Celery 结果后端"""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/1"

    @computed_field  # type: ignore[misc]
    @property
    def neo4j_uri(self) -> str:
        return f"bolt://{self.neo4j_host}:{self.neo4j_port}"

    @computed_field  # type: ignore[misc]
    @property
    def is_offline_mode(self) -> bool:
        """是否为离线模式 (无需实际大模型 API, 全链路可用种子数据驱动)"""
        return self.llm_provider == "offline"


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


settings = get_settings()
