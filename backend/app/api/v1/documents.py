"""
文档管理 API 路由

提供文档分页查询 / 上传 / 状态查询 / 分块查看 / 失败重试 / 删除 / 统计接口.

权限模型:
- admin: 全公司文档可见, 可删除任意文档;
- staff: 仅可见本部门 + 公开 (department_id IS NULL) 文档, 只能删除自己上传的文档.

设计要点:
1. 上传: MD5 去重 + 格式白名单 + 单文件 100MB 上限 + 触发限流,
   落盘路径用 ``pathlib.Path`` 拼接 ``data/uploads/{yyyy/mm}/`` 防注入.
2. 删除: 软删除 (Document.is_deleted=True) + Milvus 删除 + 图谱按 chunk 清理,
   保证向量库与图谱不残留失效数据.
3. 解析: 上传 / 重试均调用 ``DocumentPipeline.submit / retry`` 异步入队, 立即返回 task_id.
"""
from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.context import request_context
from app.core.rate_limit import rate_limit_dependency
from app.core.security import get_current_user
from app.database import get_db
from app.ingestion import DocumentPipeline
from app.models import AuditLog, Document, DocumentChunk, User
from app.schemas.common import (
    PaginatedResponse,
    PaginationParams,
    SuccessResponse,
)
from app.utils.logger import logger

router = APIRouter()

# ======================== 常量 ========================
# 上传限制
_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
_ALLOWED_EXTENSIONS = {
    "pdf", "docx", "doc", "txt", "md",
    "png", "jpg", "jpeg", "tiff",
}
# 扩展名 -> file_format 映射
_EXT_TO_FORMAT = {
    "pdf": "pdf", "docx": "docx", "doc": "docx",
    "txt": "txt", "md": "md",
    "png": "image", "jpg": "image", "jpeg": "image", "tiff": "image",
}
# 上传根目录
_UPLOAD_ROOT = Path("data/uploads")


# ======================== Schemas ========================
class DocumentOut(BaseModel):
    """文档对外结构"""
    id: int
    title: str
    file_name: str
    file_size: int
    file_format: str
    mime_type: str
    md5_hash: str
    department_id: Optional[int] = None
    category: str
    status: str
    page_count: Optional[int] = None
    chunk_count: int = 0
    error_message: Optional[str] = None
    uploaded_by: Optional[int] = None
    parsed_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """上传响应"""
    document_id: int
    task_id: str
    status: str


class DocumentStatusOut(BaseModel):
    """文档解析状态"""
    document_id: int
    stage: Optional[str] = None
    progress: float = 0.0
    status: Optional[str] = None
    chunk_count: int = 0
    error_message: Optional[str] = None
    celery_task_id: Optional[str] = None


class ChunkOut(BaseModel):
    """分块对外结构"""
    id: int
    document_id: int
    chunk_index: int
    content: str
    token_count: int
    char_count: int
    heading_path: Optional[str] = None
    page_number: Optional[int] = None

    class Config:
        from_attributes = True


class StatsOut(BaseModel):
    """统计结果 (按维度分组的字典)"""
    by_department: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_status: dict[str, int] = {}
    total: int = 0


# ======================== 辅助 ========================
def _doc_to_out(doc: Document) -> DocumentOut:
    """ORM -> DocumentOut"""
    return DocumentOut(
        id=doc.id,
        title=doc.title,
        file_name=doc.file_name,
        file_size=doc.file_size,
        file_format=doc.file_format,
        mime_type=doc.mime_type,
        md5_hash=doc.md5_hash,
        department_id=doc.department_id,
        category=doc.category,
        status=doc.status,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        error_message=doc.error_message,
        uploaded_by=doc.uploaded_by,
        parsed_at=doc.parsed_at,
        created_at=doc.created_at,
    )


def _chunk_to_out(chunk: DocumentChunk) -> ChunkOut:
    """ORM -> ChunkOut"""
    return ChunkOut(
        id=chunk.id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        token_count=chunk.token_count,
        char_count=chunk.char_count,
        heading_path=chunk.heading_path,
        page_number=chunk.page_number,
    )


def _build_visibility_filter(user: User):
    """构建可见性过滤条件: admin 全部, staff 仅本部门 + 公开 (NULL)."""
    if user.role == "admin":
        return Document.is_deleted == False  # noqa: E712
    return (
        Document.is_deleted == False  # noqa: E712
    ) & (
        (Document.department_id == user.department_id)
        | (Document.department_id.is_(None))
    )


async def _write_audit(
    db: AsyncSession,
    action: str,
    user_id: Optional[int],
    status_: str = "success",
    error: Optional[str] = None,
    resource_id: Optional[str] = None,
    detail: Optional[dict] = None,
) -> None:
    """写入审计日志 (失败不阻断主流程)."""
    try:
        ctx = request_context.get()
        log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type="document",
            resource_id=resource_id,
            detail=detail or {},
            ip_address=ctx.ip_address,
            user_agent=None,
            status=status_,
            error=error,
        )
        db.add(log)
        await db.flush()
    except Exception as exc:  # noqa: BLE001
        logger.warning("审计日志写入失败 action={}: {}", action, str(exc))


def _safe_ext(file_name: str) -> str:
    """取扩展名 (小写), 防路径注入."""
    return Path(file_name).suffix.lower().lstrip(".")


def _upload_path(file_name: str) -> Path:
    """构建安全上传路径: data/uploads/{yyyy/mm}/{uuid}_{file_name}"""
    import uuid as _uuid

    now = datetime.utcnow()
    month_dir = _UPLOAD_ROOT / f"{now.year}/{now.month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file_name).name  # 去除任何路径分量
    unique_name = f"{_uuid.uuid4().hex}_{safe_name}"
    return month_dir / unique_name


def _md5_stream(stream: UploadFile) -> tuple[bytes, str]:
    """读取上传文件流, 返回 (raw_bytes, md5_hex)."""
    md5 = hashlib.md5()
    chunks: list[bytes] = []
    total = 0
    while True:
        buf = stream.file.read(1024 * 1024)  # 1MB
        if not buf:
            break
        total += len(buf)
        if total > _MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"文件超过上限 ({_MAX_FILE_SIZE // 1024 // 1024}MB)",
            )
        md5.update(buf)
        chunks.append(buf)
    return b"".join(chunks), md5.hexdigest()


# ======================== 路由 ========================
@router.get("", response_model=PaginatedResponse[DocumentOut])
@router.get("/", response_model=PaginatedResponse[DocumentOut])
async def list_documents(
    department_id: Optional[int] = Query(None, description="按部门过滤"),
    category: Optional[str] = Query(None, description="按分类过滤"),
    doc_status: Optional[str] = Query(None, alias="status", description="按状态过滤"),
    keyword: Optional[str] = Query(None, description="标题关键词"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """分页查询文档列表.

    权限:
    - admin: 看全部文档;
    - staff: 仅看本部门 + 公开 (department_id IS NULL) 文档.

    过滤维度: department_id / category / status / keyword (标题模糊匹配).
    """
    stmt = select(Document).where(_build_visibility_filter(current_user))

    if department_id is not None:
        stmt = stmt.where(Document.department_id == department_id)
    if category:
        stmt = stmt.where(Document.category == category)
    if doc_status:
        stmt = stmt.where(Document.status == doc_status)
    if keyword:
        stmt = stmt.where(Document.title.ilike(f"%{keyword}%"))

    # 总数
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    # 分页
    offset = (page - 1) * page_size
    stmt = stmt.order_by(Document.created_at.desc()).offset(offset).limit(page_size)
    docs = (await db.execute(stmt)).scalars().all()

    return PaginatedResponse[DocumentOut](
        data=[_doc_to_out(d) for d in docs],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if page_size else 0,
    )


@router.post(
    "/upload",
    response_model=SuccessResponse[UploadResponse],
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="待上传文件"),
    title: Optional[str] = Query(None, description="文档标题, 缺省取文件名"),
    department_id: Optional[int] = Query(
        None, description="归属部门 ID, None 表示全公司可见"
    ),
    category: str = Query("other", description="分类: product_manual/policy/faq/other"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """上传文档, 触发异步解析流水线.

    限制:
    - 单文件最大 100MB;
    - 格式白名单: pdf/docx/doc/txt/md/png/jpg/jpeg/tiff;
    - MD5 去重: 已存在相同 MD5 直接返回已有 document_id;
    - 触发限流 (rate_limit_dependency 在路由层注入, 此处显式调用避免双重依赖).

    返回: document_id + celery task_id + 初始 status.
    """
    # 限流
    await rate_limit_dependency(request, endpoint="documents.upload")

    ext = _safe_ext(file.filename or "")
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式: {ext}, 允许: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    # 读取 + MD5
    raw_bytes, md5 = _md5_stream(file)
    file_size = len(raw_bytes)

    # 去重: 已存在相同 MD5 (未删除) 则直接返回
    dup_result = await db.execute(
        select(Document).where(
            Document.md5_hash == md5,
            Document.is_deleted == False,  # noqa: E712
        )
    )
    existing = dup_result.scalar_one_or_none()
    if existing is not None:
        logger.info(
            "文件 MD5 重复, 跳过上传: md5={} doc_id={}", md5, existing.id
        )
        return SuccessResponse[UploadResponse](data=UploadResponse(
            document_id=existing.id,
            task_id="",
            status=existing.status,
        ))

    # 落盘
    save_path = _upload_path(file.filename or f"upload.{ext}")
    save_path.write_bytes(raw_bytes)

    file_format = _EXT_TO_FORMAT.get(ext, "other")
    mime_type = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"
    doc_title = title or Path(file.filename or "untitled").stem

    doc = Document(
        title=doc_title,
        file_name=file.filename or save_path.name,
        file_path=str(save_path),
        file_size=file_size,
        file_format=file_format,
        mime_type=mime_type,
        md5_hash=md5,
        department_id=department_id,
        category=category,
        status="pending",
        uploaded_by=current_user.id,
    )
    db.add(doc)
    await db.flush()

    await _write_audit(
        db, "doc.upload", current_user.id, status_="success",
        resource_id=str(doc.id),
        detail={
            "file_name": doc.file_name, "md5": md5,
            "size": file_size, "category": category,
            "department_id": department_id,
        },
    )

    # 提交解析任务
    pipeline = DocumentPipeline()
    try:
        task_id = await pipeline.submit(doc.id)
    except Exception as exc:
        logger.error("提交解析任务失败 doc_id={}: {}", doc.id, str(exc))
        doc.status = "failed"
        doc.error_message = f"提交解析失败: {exc}"
        await db.flush()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"提交解析任务失败: {exc}",
        )

    logger.info(
        "文档上传成功: id={} title={} size={} task_id={}",
        doc.id, doc.title, file_size, task_id,
    )

    return SuccessResponse[UploadResponse](data=UploadResponse(
        document_id=doc.id,
        task_id=task_id,
        status=doc.status,
    ))


@router.get("/stats/summary", response_model=SuccessResponse[StatsOut])
async def document_stats_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """文档统计: 按部门 / 分类 / 状态分组计数."""
    base = select(Document).where(_build_visibility_filter(current_user))

    # 按部门
    dept_stmt = (
        select(Document.department_id, func.count(Document.id))
        .where(_build_visibility_filter(current_user))
        .group_by(Document.department_id)
    )
    dept_rows = (await db.execute(dept_stmt)).all()
    by_department = {
        str(row[0]) if row[0] is not None else "public": row[1]
        for row in dept_rows
    }

    # 按分类
    cat_stmt = (
        select(Document.category, func.count(Document.id))
        .where(_build_visibility_filter(current_user))
        .group_by(Document.category)
    )
    cat_rows = (await db.execute(cat_stmt)).all()
    by_category = {row[0]: row[1] for row in cat_rows}

    # 按状态
    status_stmt = (
        select(Document.status, func.count(Document.id))
        .where(_build_visibility_filter(current_user))
        .group_by(Document.status)
    )
    status_rows = (await db.execute(status_stmt)).all()
    by_status = {row[0]: row[1] for row in status_rows}

    total = sum(by_status.values())

    return SuccessResponse[StatsOut](data=StatsOut(
        by_department=by_department,
        by_category=by_category,
        by_status=by_status,
        total=total,
    ))


@router.get("/{document_id}/status", response_model=SuccessResponse[DocumentStatusOut])
async def get_document_status(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """查询文档解析状态 (含 progress / stage / chunk_count)."""
    doc_result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.is_deleted == False,  # noqa: E712
        )
    )
    doc = doc_result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档不存在: {document_id}",
        )

    # 权限: staff 不可查看其他部门文档状态
    if current_user.role != "admin":
        if doc.department_id is not None and doc.department_id != current_user.department_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="无权查看该文档状态",
            )

    pipeline = DocumentPipeline()
    status_dict = await pipeline.get_status(document_id)

    out = DocumentStatusOut(
        document_id=document_id,
        stage=status_dict.get("stage"),
        progress=float(status_dict.get("progress", 0.0) or 0.0),
        status=status_dict.get("status") or doc.status,
        chunk_count=doc.chunk_count,
        error_message=status_dict.get("error_message") or doc.error_message,
        celery_task_id=status_dict.get("celery_task_id"),
    )
    return SuccessResponse[DocumentStatusOut](data=out)


@router.get("/{document_id}/chunks", response_model=PaginatedResponse[ChunkOut])
async def get_document_chunks(
    document_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """查看文档分块 (分页, 仅 status=ready 的文档)."""
    doc_result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.is_deleted == False,  # noqa: E712
        )
    )
    doc = doc_result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档不存在: {document_id}",
        )

    if doc.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"文档尚未就绪, 当前状态: {doc.status}",
        )

    # 权限
    if current_user.role != "admin":
        if doc.department_id is not None and doc.department_id != current_user.department_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="无权查看该文档分块",
            )

    count_stmt = select(func.count()).select_from(DocumentChunk).where(
        DocumentChunk.document_id == document_id
    )
    total = (await db.execute(count_stmt)).scalar_one()

    offset = (page - 1) * page_size
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .offset(offset)
        .limit(page_size)
    )
    chunks = (await db.execute(stmt)).scalars().all()

    return PaginatedResponse[ChunkOut](
        data=[_chunk_to_out(c) for c in chunks],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if page_size else 0,
    )


@router.post("/{document_id}/retry", response_model=SuccessResponse[UploadResponse])
async def retry_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """失败重试: 重置状态后重新投递解析任务."""
    doc_result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.is_deleted == False,  # noqa: E712
        )
    )
    doc = doc_result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档不存在: {document_id}",
        )

    # 权限: admin 或上传者
    if current_user.role != "admin" and doc.uploaded_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权重试该文档",
        )

    if doc.status not in ("failed", "cancelled"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"仅 failed / cancelled 状态可重试, 当前: {doc.status}",
        )

    await _write_audit(
        db, "doc.retry", current_user.id, status_="success",
        resource_id=str(document_id),
        detail={"prev_status": doc.status},
    )

    pipeline = DocumentPipeline()
    task_id = await pipeline.retry(document_id)

    logger.info("文档重试已提交: doc_id={} task_id={}", document_id, task_id)

    return SuccessResponse[UploadResponse](data=UploadResponse(
        document_id=document_id,
        task_id=task_id,
        status="processing",
    ))


@router.delete("/{document_id}", response_model=SuccessResponse[dict])
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除文档 (软删除 + Milvus 删除 + 图谱按 chunk 清理).

    权限: 仅 admin 或上传者可删.
    """
    doc_result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.is_deleted == False,  # noqa: E712
        )
    )
    doc = doc_result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档不存在: {document_id}",
        )

    if current_user.role != "admin" and doc.uploaded_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权删除该文档",
        )

    # 1. 软删除 Document
    doc.is_deleted = True
    doc.status = "deleted"
    await db.flush()

    # 2. 异步清理 Milvus + 图谱 (失败不阻断主流程)
    cleanup_detail: dict[str, Any] = {"milvus": "skipped", "graph": "skipped"}

    try:
        from app.rag.milvus_store import milvus_store

        chunk_ids_result = await db.execute(
            select(DocumentChunk.id).where(DocumentChunk.document_id == document_id)
        )
        chunk_ids = [row[0] for row in chunk_ids_result.all()]
        if chunk_ids and milvus_store.is_available:
            deleted = await milvus_store.delete_by_document(document_id)
            cleanup_detail["milvus"] = f"deleted={deleted}"
    except Exception as exc:  # noqa: BLE001
        cleanup_detail["milvus"] = f"error: {exc}"
        logger.warning("Milvus 清理失败 doc_id={}: {}", document_id, str(exc))

    try:
        from app.graphrag.neo4j_store import neo4j_store

        if neo4j_store.is_available:
            chunks_for_graph = await db.execute(
                select(DocumentChunk.id).where(DocumentChunk.document_id == document_id)
            )
            for (cid,) in chunks_for_graph.all():
                await neo4j_store.delete_by_source_chunk(int(cid))
            cleanup_detail["graph"] = "cleaned"
    except Exception as exc:  # noqa: BLE001
        cleanup_detail["graph"] = f"error: {exc}"
        logger.warning("图谱清理失败 doc_id={}: {}", document_id, str(exc))

    await _write_audit(
        db, "doc.delete", current_user.id, status_="success",
        resource_id=str(document_id),
        detail={
            "title": doc.title, "cleanup": cleanup_detail,
            "by_admin": current_user.role == "admin",
        },
    )

    logger.info(
        "文档已删除: doc_id={} title={} by={} cleanup={}",
        document_id, doc.title, current_user.id, cleanup_detail,
    )

    return SuccessResponse[dict](data={
        "document_id": document_id,
        "deleted": True,
        "cleanup": cleanup_detail,
    })
