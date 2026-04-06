"""
项目配置文件 - 支持本地和 Streamlit Cloud 两种运行环境

功能说明：
- 自动检测运行环境（本地开发 or Streamlit Cloud 部署）
- 根据环境自动选择合适的嵌入模型（本地模型 or 在线 API）
- 配置路径、模型、RAG 参数等
"""
from pathlib import Path
import os
import tempfile
from dotenv import load_dotenv

# 加载环境变量（从 .env 文件读取配置）
load_dotenv()

# ==================== 基础路径配置 ====================
# 项目根目录（config.py 所在目录）
BASE_DIR = Path(__file__).parent.resolve()

# 数据目录：存放文档、向量数据库等
DATA_DIR = BASE_DIR / "data"

# 文档目录：存放待处理的原始文档
DOCUMENTS_DIR = DATA_DIR / "documents"

# ==================== 环境检测 ====================
# 判断是否在 Streamlit Cloud 上运行
# 通过环境变量 STREAMLIT_CLOUD 来区分
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_CLOUD", "").lower() in ("true", "1", "yes")

# ==================== 向量数据库路径配置 ====================
if IS_STREAMLIT_CLOUD:
    # Streamlit Cloud 环境：使用临时目录（因为云环境不支持持久化存储）
    CHROMA_DIR = Path(tempfile.gettempdir()) / "chroma_db"
else:
    # 本地环境：使用项目目录下的固定路径
    CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(DATA_DIR / "chroma_db")))

# ==================== 嵌入模型配置 ====================
if IS_STREAMLIT_CLOUD:
    # Streamlit Cloud 环境：使用在线 API（无法访问本地文件）
    # 默认使用通义千问的 text-embedding-v3 模型
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    # API 基础 URL
    EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    # API Key（用于调用嵌入模型 API）
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
else:
    # 本地环境：使用本地模型文件
    # 默认使用项目中的 bge-m3 模型
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", str(BASE_DIR / "models" / "bge-m3"))
    # 本地模式不需要 API
    EMBEDDING_API_BASE = ""
    EMBEDDING_API_KEY = ""

# ==================== LLM（大语言模型）配置 ====================
# 模型名称：使用通义千问 Qwen2.5-7B
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-7b-instruct")

# API 基础 URL
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# API Key：优先使用 LLM_API_KEY，如果没有则使用 DASHSCOPE_API_KEY
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))

# ==================== RAG 核心参数配置 ====================
# 文档分块大小（每个文本块的最大字符数）
CHUNK_SIZE = 512

# 分块重叠大小（相邻文本块之间的重叠部分，保持上下文连贯性）
CHUNK_OVERLAP = 100

# 检索时返回的最相关文档数量
TOP_K = 5

# 相似度阈值（低于此阈值的文档会被过滤掉）
SIMILARITY_THRESHOLD = 0.7

# ==================== 服务配置 ====================
# 本地运行时的服务器地址
HOST = "0.0.0.0"

# 本地运行时的服务器端口
PORT = 8001

# ==================== 创建必要的目录 ====================
# 本地环境才创建文档目录（云端不需要）
if not IS_STREAMLIT_CLOUD:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# 创建向量数据库目录（本地和云端都需要）
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
