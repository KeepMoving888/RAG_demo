"""
项目配置文件（云端部署优先）
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# 加载环境变量（可选）
load_dotenv()

# ==================== 基础路径配置（全部使用相对项目路径） ====================
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(DATA_DIR / "chroma_db")))
MODEL_DIR = BASE_DIR / "models"

# ==================== 运行环境 ====================
# 不在代码里强制设置 STREAMLIT_CLOUD，交给部署环境或用户显式配置
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_CLOUD", "").lower() in ("true", "1", "yes")

# ==================== 模型与 API 配置（默认在线，可部署） ====================
# 嵌入模型：默认在线 embedding，避免依赖本地大模型文件
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", ""))

# LLM：默认在线 OpenAI-compatible 服务
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-7b-instruct")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# ==================== RAG 参数 ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# 自动加载配置
AUTO_INGEST = os.getenv("AUTO_INGEST", "true").lower() in ("true", "1", "yes")
SUPPORTED_DOC_EXTS = {".pdf", ".docx", ".doc", ".txt"}
INGEST_STATE_FILE = DATA_DIR / "ingest_state.json"

# ==================== 服务配置 ====================
HOST = "0.0.0.0"
PORT = 8001

# ==================== 确保目录存在 ====================
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
