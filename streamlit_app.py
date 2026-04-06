"""
Streamlit 前端界面（自动加载 data/documents + 缓存）
"""
import json
from pathlib import Path
from typing import Dict, List

import streamlit as st

import config

# 设置页面配置
st.set_page_config(page_title="企业知识库 RAG 系统", page_icon="📚", layout="wide")

st.markdown(
    """
<style>
    .main-header {font-size: 2.5rem; color: #1E88E5; text-align: center;}
    .sub-header {font-size: 1.2rem; color: #666; text-align: center;}
    .answer-box {background-color: #f0f7ff; padding: 20px; border-radius: 10px; border-left: 4px solid #1E88E5;}
    .source-box {background-color: #fff3e0; padding: 10px; border-radius: 5px; margin: 5px 0;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<h1 class='main-header'>📚 企业知识库智能问答系统</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>基于 RAG 技术的文档智能检索与问答（自动加载文档）</p>", unsafe_allow_html=True)
st.divider()


def _list_document_files() -> List[Path]:
    if not config.DOCUMENTS_DIR.exists():
        return []
    files = []
    for p in config.DOCUMENTS_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_DOC_EXTS:
            files.append(p)
    files.sort(key=lambda x: x.name)
    return files


def _build_docs_signature(files: List[Path]) -> Dict:
    return {
        "files": [
            {
                "name": f.name,
                "size": f.stat().st_size,
                "mtime": int(f.stat().st_mtime),
            }
            for f in files
        ]
    }


def _load_ingest_state() -> Dict:
    state_file = config.INGEST_STATE_FILE
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_ingest_state(state: Dict) -> None:
    config.INGEST_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.INGEST_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


@st.cache_resource
def init_rag_engine():
    from rag_llm import RAGEngine

    return RAGEngine()


def auto_ingest_if_needed(rag_engine) -> Dict:
    files = _list_document_files()
    current_signature = _build_docs_signature(files)
    state = _load_ingest_state()

    if not files:
        return {"status": "empty", "message": "文档目录为空，请将文件放入 data/documents", "stats": {}}

    if state.get("signature") == current_signature:
        return {"status": "cached", "message": "检测到文档未变化，已使用已有向量库缓存", "stats": {}}

    # 文档有变化：重建向量库，避免重复入库
    rag_engine.vector_store.clear_collection()
    stats = rag_engine.ingest_directory(config.DOCUMENTS_DIR)

    _save_ingest_state({"signature": current_signature, "last_stats": stats})
    return {"status": "reindexed", "message": "文档有更新，已自动重建向量索引", "stats": stats}


# 初始化系统
try:
    with st.spinner("正在初始化系统..."):
        rag_engine = init_rag_engine()
    init_success = True
except Exception as e:
    st.error(f"系统初始化失败：{e}")
    init_success = False
    rag_engine = None

# 启动自动加载
auto_ingest_result = None
if init_success and config.AUTO_INGEST:
    try:
        with st.spinner("正在自动检查并加载 data/documents 文档..."):
            auto_ingest_result = auto_ingest_if_needed(rag_engine)
    except Exception as e:
        auto_ingest_result = {"status": "error", "message": f"自动加载失败：{e}", "stats": {}}

# ==================== 侧边栏：状态 ====================
with st.sidebar:
    st.header("📁 文档状态")
    st.caption("文档已自动加载")
    st.caption("支持格式：pdf / docx / doc / txt")

    files = _list_document_files()
    st.metric("已发现文档", len(files))

    if auto_ingest_result:
        if auto_ingest_result["status"] in {"cached", "reindexed"}:
            st.success(auto_ingest_result["message"])
        elif auto_ingest_result["status"] == "empty":
            st.warning(auto_ingest_result["message"])
        else:
            st.error(auto_ingest_result["message"])

    st.divider()
    st.header("📊 系统状态")
    if init_success:
        try:
            stats = rag_engine.get_stats()
            st.metric("向量库文档块", stats.get("vector_store", {}).get("document_count", 0))
            st.metric("Top-K", stats.get("config", {}).get("top_k", 5))
        except Exception as e:
            st.warning(f"获取状态失败：{e}")
    else:
        st.warning("系统未初始化")

# ==================== 主界面：智能问答 ====================
col1, col2 = st.columns([3, 1])

with col1:
    st.header("🔍 智能问答")

    query = st.text_area("请输入您的问题", placeholder="例如：公司的年假政策是什么？", height=100)

    with st.expander("高级选项"):
        top_k = st.slider("检索文档数量", 1, 10, config.TOP_K)
        show_sources = st.checkbox("显示引用来源", value=True)

    if st.button("🚀 开始查询", type="primary", use_container_width=True):
        if not init_success:
            st.error("系统未初始化，无法查询")
        elif not query.strip():
            st.warning("请输入问题")
        else:
            old_top_k = config.TOP_K
            config.TOP_K = top_k
            try:
                with st.spinner("正在检索和生成答案..."):
                    result = rag_engine.query(question=query, with_sources=show_sources)

                st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                st.markdown("### 💡 答案")
                st.write(result.get("answer", ""))
                st.markdown("</div>", unsafe_allow_html=True)

                if show_sources and result.get("sources"):
                    st.markdown("### 📖 引用来源")
                    for source in result["sources"]:
                        st.markdown(
                            f"<div class='source-box'>📄 {source['file']} (相似度：{source['score']})</div>",
                            unsafe_allow_html=True,
                        )

                st.caption(f"检索了 {result.get('context_used', 0)} 个文档块")
            except Exception as e:
                st.error(f"查询失败：{e}")
            finally:
                config.TOP_K = old_top_k

with col2:
    st.header("💡 使用提示")
    st.info(
        """
    **部署前建议：**
    1. 把你的文档放入 `data/documents`
    2. 在环境变量/Secrets中配置在线模型 API
    3. 启动后会自动加载，后续未变更会走缓存

    **说明：**
    - 无需手动上传文档
    - 文档更新后会自动重建索引
    - 可直接开始问答
    """
    )

    st.divider()
    st.header("📝 示例问题")
    examples = ["公司的年假政策是什么？", "报销流程需要哪些材料？", "产品技术文档在哪里？"]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.query = ex

st.divider()
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "Powered by RAG + LangChain + Qwen2.5 | Offline-first"
    "</div>",
    unsafe_allow_html=True,
)
