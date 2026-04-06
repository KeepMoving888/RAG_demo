"""
FastAPI主入口
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os

from rag_llm import rag_engine
import config


def _collect_supported_docs() -> List[str]:
    """收集documents目录下支持的文档"""
    supported_exts = {".pdf", ".docx", ".doc", ".txt"}
    doc_paths: List[str] = []

    if not config.DOCUMENTS_DIR.exists():
        return doc_paths

    for file_name in os.listdir(str(config.DOCUMENTS_DIR)):
        full_path = config.DOCUMENTS_DIR / file_name
        if full_path.is_file() and full_path.suffix.lower() in supported_exts:
            doc_paths.append(str(full_path))

    return doc_paths


app = FastAPI(
    title="企业知识库RAG系统",
    description="基于RAG技术的企业智能问答系统",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    with_sources: Optional[bool] = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    context_used: int
    query: str


@app.on_event("startup")
async def auto_ingest_on_startup():
    """服务启动时：若向量库为空，则自动导入data/documents中的文件"""
    try:
        current_count = rag_engine.vector_store.get_collection_stats().get("document_count", 0)
        if current_count and current_count > 0:
            print(f"✅ 向量库已有数据（{current_count}条），跳过自动导入")
            return

        doc_paths = _collect_supported_docs()
        if not doc_paths:
            print("ℹ️ documents目录无可导入文件，跳过自动导入")
            return

        stats = rag_engine.ingest_documents(doc_paths)
        print(f"✅ 启动自动导入完成：{stats}")
    except Exception as e:
        print(f"❌ 启动自动导入失败：{e}")


@app.get("/")
async def root():
    return {"message": "欢迎使用企业知识库RAG系统", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "stats": rag_engine.get_stats()}


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """上传文档到知识库"""
    saved_paths = []
    
    for file in files:
        file_path = os.path.join(str(config.DOCUMENTS_DIR), file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(file_path)
    
    # 导入到知识库
    stats = rag_engine.ingest_documents(saved_paths)
    
    return {
        "message": f"成功上传 {len(files)} 个文件",
        "stats": stats
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_knowledge(request: QueryRequest):
    """查询知识库"""
    result = rag_engine.query(
        question=request.question,
        with_sources=request.with_sources
    )
    return result


@app.get("/api/stats")
async def get_stats():
    """获取系统统计"""
    return rag_engine.get_stats()


@app.post("/api/rebuild")
async def rebuild_knowledge():
    """重建知识库：清空后重新导入data/documents目录"""
    rag_engine.vector_store.clear_collection()
    doc_paths = _collect_supported_docs()

    if not doc_paths:
        return {"message": "知识库已清空，documents目录无可导入文件", "stats": {"success": 0, "failed": 0, "total_chunks": 0, "vector_count": 0}}

    stats = rag_engine.ingest_documents(doc_paths)
    return {
        "message": f"重建完成，导入 {len(doc_paths)} 个文件",
        "stats": stats
    }


@app.delete("/api/clear")
async def clear_knowledge():
    """清空知识库"""
    rag_engine.vector_store.clear_collection()
    return {"message": "知识库已清空"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)