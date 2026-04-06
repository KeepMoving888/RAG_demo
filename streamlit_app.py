"""
Streamlit 前端界面 - 独立运行版本

功能说明：
- 提供用户友好的 Web 界面
- 支持文档上传和管理
- 智能问答交互
- 显示引用来源和相似度
"""
import streamlit as st
import os
import tempfile
from pathlib import Path

# 设置页面配置
st.set_page_config(
    page_title="企业知识库 RAG 系统",
    page_icon="📚",
    layout="wide"  # 使用宽屏布局
)

# 自定义 CSS 样式，美化界面
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E88E5; text-align: center;}
    .sub-header {font-size: 1.2rem; color: #666; text-align: center;}
    .answer-box {background-color: #f0f7ff; padding: 20px; border-radius: 10px; border-left: 4px solid #1E88E5;}
    .source-box {background-color: #fff3e0; padding: 10px; border-radius: 5px; margin: 5px 0;}
</style>
""", unsafe_allow_html=True)

# 页面标题
st.markdown("<h1 class='main-header'>📚 企业知识库智能问答系统</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>基于 RAG 技术的文档智能检索与问答</p>", unsafe_allow_html=True)

st.divider()

# ==================== 初始化 RAG 引擎 ====================
@st.cache_resource
def init_rag_engine():
    """
    初始化 RAG 引擎（使用缓存避免重复初始化）
    
    Returns:
        RAGEngine: RAG 引擎实例
    """
    # 设置环境变量，告知代码运行在 Streamlit Cloud 环境
    os.environ["STREAMLIT_CLOUD"] = "true"
    from rag_llm import RAGEngine
    return RAGEngine()

# 尝试初始化系统
try:
    with st.spinner("正在初始化系统..."):
        rag_engine = init_rag_engine()
    init_success = True
except Exception as e:
    st.error(f"系统初始化失败：{e}")
    init_success = False

# ==================== 侧边栏：文档管理 ====================
with st.sidebar:
    st.header("📁 文档管理")
    
    # 文件上传组件
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["pdf", "docx", "txt"],  # 支持的文件格式
        accept_multiple_files=True  # 允许多选
    )
    
    # 上传按钮
    if st.button("上传到知识库", type="primary") and init_success:
        if uploaded_files:
            with st.spinner("正在处理文档..."):
                try:
                    # 创建临时目录保存上传的文件
                    temp_dir = tempfile.mkdtemp()
                    saved_paths = []
                    for f in uploaded_files:
                        file_path = os.path.join(temp_dir, f.name)
                        with open(file_path, "wb") as buffer:
                            buffer.write(f.getbuffer())
                        saved_paths.append(file_path)
                    
                    # 调用 RAG 引擎处理文档
                    stats = rag_engine.ingest_documents(saved_paths)
                    st.success(f"✅ 成功上传 {stats['success']} 个文件，共 {stats['total_chunks']} 个文档块")
                except Exception as e:
                    st.error(f"上传失败：{e}")
        else:
            st.warning("请先选择文件")
    
    st.divider()
    
    # 系统状态显示
    st.header("📊 系统状态")
    if init_success:
        try:
            stats = rag_engine.get_stats()
            st.metric("文档数量", stats.get("vector_store", {}).get("document_count", 0))
            st.metric("Top-K", stats.get("config", {}).get("top_k", 5))

            # 显示预加载文档提示
            st.success('✅ 已加载预置文档，可直接开始问答')
            with st.expander('查看预置文档列表'):
                st.markdown(''- 公司政策'')
                st.markdown(''- 财务制度'')
                st.markdown(''- 人力资源'')
                st.markdown(''- 信息安全'')
                st.markdown(''- 产品说明'')
                st.markdown(''- 企业培训/技术管理制度'')
        except Exception as e:
            st.warning(f"获取状态失败：{e}")
    else:
        st.warning("系统未初始化")

# ==================== 主界面：智能问答 ====================
# 创建两列布局（3:1 比例）
col1, col2 = st.columns([3, 1])

with col1:
    st.header("🔍 智能问答")
    
    # 问题输入框
    query = st.text_area(
        "请输入您的问题",
        placeholder="例如：公司的年假政策是什么？",
        height=100
    )
    
    # 高级选项
    with st.expander("高级选项"):
        top_k = st.slider("检索文档数量", 1, 10, 5)  # 默认检索 5 个文档
        show_sources = st.checkbox("显示引用来源", value=True)  # 默认显示来源
    
    # 查询按钮
    if st.button("🚀 开始查询", type="primary", use_container_width=True):
        if query and init_success:
            with st.spinner("正在检索和生成答案..."):
                try:
                    # 调用 RAG 引擎进行查询
                    result = rag_engine.query(
                        question=query,
                        with_sources=show_sources
                    )
                    
                    # 显示答案
                    st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                    st.markdown("### 💡 答案")
                    st.write(result["answer"])
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # 显示引用来源
                    if show_sources and result.get("sources"):
                        st.markdown("### 📖 引用来源")
                        for source in result["sources"]:
                            st.markdown(
                                f"<div class='source-box'>📄 {source['file']} "
                                f"(相似度：{source['score']})</div>",
                                unsafe_allow_html=True
                            )
                    
                    # 显示检索统计
                    st.caption(f"检索了 {result.get('context_used', 0)} 个文档块")
                    
                except Exception as e:
                    st.error(f"查询失败：{e}")
        elif not init_success:
            st.error("系统未初始化，无法查询")
        else:
            st.warning("请输入问题")

with col2:
    # 使用提示
    st.header("💡 使用提示")
    st.info("""
    **最佳实践：**
    1. 问题尽量具体明确
    2. 上传前确保文档质量
    3. 复杂问题可拆分成多个小问题
    
    **支持格式：**
    - PDF 文档
    - Word 文档
    - TXT 文本
    
    **典型场景：**
    - 员工手册查询
    - 产品文档检索
    - 制度政策问答
    """)
    
    st.divider()
    
    # 示例问题
    st.header("📝 示例问题")
    examples = [
        "公司的年假政策是什么？",
        "报销流程需要哪些材料？",
        "产品技术文档在哪里？"
    ]
    
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.query = ex

# ==================== 页脚 ====================
st.divider()
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "Powered by RAG + LangChain + Qwen2.5 | 2026"
    "</div>",
    unsafe_allow_html=True
)
