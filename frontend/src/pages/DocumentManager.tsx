import { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  CheckCircle2,
  ChevronRight,
  Download,
  Eye,
  FileWarning,
  FileText,
  Loader2,
  RotateCw,
  Trash2,
  Upload,
  X,
} from 'lucide-react';
import { documentsApi, adminApi, type DocumentQuery } from '@/api';
import type { DocumentChunk, DocumentStatus, KBDocument } from '@/api/types';
import { cn, downloadCSV, formatBytes, formatDateTime } from '@/lib/utils';
import StatCard from '@/components/StatCard';
import Badge from '@/components/ui/Badge';
import Button from '@/components/ui/Button';
import { Input, Select } from '@/components/ui/Input';
import Skeleton, { SkeletonText } from '@/components/ui/Skeleton';

const STATUS_BADGE: Record<DocumentStatus, { variant: 'success' | 'warning' | 'danger' | 'neutral'; label: string }> = {
  ready: { variant: 'success', label: '就绪' },
  parsing: { variant: 'warning', label: '解析中' },
  pending: { variant: 'neutral', label: '排队中' },
  failed: { variant: 'danger', label: '失败' },
};

const ACCEPTED = '.pdf,.docx,.doc,.xlsx,.xls,.md,.txt';

/** Document management page: stats, filterable table, upload modal & chunk drawer. */
export default function DocumentManager() {
  const qc = useQueryClient();
  const [filters, setFilters] = useState<DocumentQuery>({});
  const [uploadOpen, setUploadOpen] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState<KBDocument | null>(null);

  const { data: stats } = useQuery({
    queryKey: ['doc-stats'],
    queryFn: documentsApi.stats,
  });

  const { data: departments } = useQuery({
    queryKey: ['departments'],
    queryFn: adminApi.departments,
  });

  const deptOptions = useMemo(() => {
    const out: { value: string; label: string }[] = [];
    const walk = (ds: typeof departments) => {
      ds?.forEach((d) => {
        out.push({ value: d.id, label: d.name });
        if (d.children) walk(d.children);
      });
    };
    walk(departments);
    return out;
  }, [departments]);

  const { data: docsRes, isLoading } = useQuery({
    queryKey: ['documents', filters],
    queryFn: () => documentsApi.list(filters),
  });

  const docs = docsRes?.items ?? [];

  // Poll while any doc is parsing/pending.
  const hasActive = docs.some(
    (d) => d.status === 'parsing' || d.status === 'pending'
  );
  useEffect(() => {
    if (!hasActive) return;
    const t = setInterval(() => {
      qc.invalidateQueries({ queryKey: ['documents', filters] });
      qc.invalidateQueries({ queryKey: ['doc-stats'] });
    }, 2000);
    return () => clearInterval(t);
  }, [hasActive, filters, qc]);

  const handleExport = () => {
    downloadCSV(
      'documents.csv',
      docs.map((d) => ({
        标题: d.title,
        部门: d.department_name,
        分类: d.category,
        格式: d.format,
        大小: formatBytes(d.size),
        状态: STATUS_BADGE[d.status].label,
        分块数: d.chunk_count,
        上传时间: formatDateTime(d.created_at),
      }))
    );
  };

  const onDelete = async (id: string) => {
    if (!confirm('确认删除该文档及其分块？此操作不可恢复。')) return;
    await documentsApi.delete(id);
    qc.invalidateQueries({ queryKey: ['documents'] });
    qc.invalidateQueries({ queryKey: ['doc-stats'] });
  };

  const onRetry = async (id: string) => {
    await documentsApi.retry(id);
    qc.invalidateQueries({ queryKey: ['documents'] });
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-800">文档管理</h1>
          <p className="mt-0.5 text-sm text-slate-500">
            管理企业知识库中的文档，支持上传、解析监控与分块查看
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" icon={Download} onClick={handleExport}>
            导出 CSV
          </Button>
          <Button icon={Upload} onClick={() => setUploadOpen(true)}>
            上传文档
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="总文档数" value={stats?.total ?? 0} icon={FileText} accent="primary" />
        <StatCard label="解析中" value={stats?.parsing ?? 0} icon={Loader2} accent="amber" />
        <StatCard label="就绪" value={stats?.ready ?? 0} icon={CheckCircle2} accent="emerald" />
        <StatCard label="失败" value={stats?.failed ?? 0} icon={FileWarning} accent="rose" />
      </div>

      {/* Filters */}
      <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-card">
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
          <Input
            placeholder="搜索标题 / 文件名"
            value={filters.keyword ?? ''}
            onChange={(e) =>
              setFilters((f) => ({ ...f, keyword: e.target.value || undefined }))
            }
          />
          <Select
            placeholder="全部部门"
            options={deptOptions}
            value={filters.department_id ?? ''}
            onChange={(e) =>
              setFilters((f) => ({
                ...f,
                department_id: e.target.value || undefined,
              }))
            }
          />
          <Select
            placeholder="全部分类"
            options={[
              { value: '技术规范', label: '技术规范' },
              { value: '采购标准', label: '采购标准' },
              { value: '国际标准', label: '国际标准' },
              { value: '审核报告', label: '审核报告' },
              { value: '发布说明', label: '发布说明' },
              { value: '管理制度', label: '管理制度' },
              { value: '认证资料', label: '认证资料' },
            ]}
            value={filters.category ?? ''}
            onChange={(e) =>
              setFilters((f) => ({ ...f, category: e.target.value || undefined }))
            }
          />
          <Select
            placeholder="全部状态"
            options={[
              { value: 'ready', label: '就绪' },
              { value: 'parsing', label: '解析中' },
              { value: 'pending', label: '排队中' },
              { value: 'failed', label: '失败' },
            ]}
            value={filters.status ?? ''}
            onChange={(e) =>
              setFilters((f) => ({ ...f, status: e.target.value || undefined }))
            }
          />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-card">
        <div className="overflow-x-auto scrollbar-thin">
          <table className="w-full min-w-[900px] text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="px-4 py-3 font-medium">标题</th>
                <th className="px-4 py-3 font-medium">部门</th>
                <th className="px-4 py-3 font-medium">分类</th>
                <th className="px-4 py-3 font-medium">格式</th>
                <th className="px-4 py-3 font-medium">大小</th>
                <th className="px-4 py-3 font-medium">状态</th>
                <th className="px-4 py-3 font-medium">分块</th>
                <th className="px-4 py-3 text-right font-medium">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {isLoading &&
                Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i}>
                    <td className="px-4 py-3" colSpan={8}>
                      <SkeletonText lines={1} />
                    </td>
                  </tr>
                ))}
              {!isLoading &&
                docs.map((doc) => (
                  <tr
                    key={doc.id}
                    className="cursor-pointer transition-colors hover:bg-slate-50"
                    onClick={() => setSelectedDoc(doc)}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 flex-shrink-0 text-slate-400" />
                        <span className="font-medium text-slate-800 line-clamp-1">
                          {doc.title}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-slate-600">{doc.department_name}</td>
                    <td className="px-4 py-3 text-slate-600">{doc.category}</td>
                    <td className="px-4 py-3">
                      <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs font-medium uppercase text-slate-600">
                        {doc.format}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate-500">{formatBytes(doc.size)}</td>
                    <td className="px-4 py-3">
                      <div className="space-y-1">
                        <Badge variant={STATUS_BADGE[doc.status].variant} dot>
                          {STATUS_BADGE[doc.status].label}
                        </Badge>
                        {(doc.status === 'parsing' || doc.status === 'pending') && (
                          <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-200">
                            <div
                              className="h-full rounded-full bg-amber-500 transition-all"
                              style={{ width: `${doc.progress}%` }}
                            />
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-slate-600">
                      {doc.chunk_count || '-'}
                    </td>
                    <td className="px-4 py-3">
                      <div
                        className="flex justify-end gap-1"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <IconBtn title="查看" onClick={() => setSelectedDoc(doc)}>
                          <Eye className="h-4 w-4" />
                        </IconBtn>
                        {doc.status === 'failed' && (
                          <IconBtn title="重试" onClick={() => onRetry(doc.id)}>
                            <RotateCw className="h-4 w-4" />
                          </IconBtn>
                        )}
                        <IconBtn
                          title="删除"
                          danger
                          onClick={() => onDelete(doc.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </IconBtn>
                      </div>
                    </td>
                  </tr>
                ))}
              {!isLoading && docs.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-4 py-16 text-center text-sm text-slate-400">
                    暂无匹配文档
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {uploadOpen && (
        <UploadModal
          deptOptions={deptOptions}
          onClose={() => setUploadOpen(false)}
          onUploaded={() => {
            setUploadOpen(false);
            qc.invalidateQueries({ queryKey: ['documents'] });
            qc.invalidateQueries({ queryKey: ['doc-stats'] });
          }}
        />
      )}

      {selectedDoc && (
        <ChunkDrawer
          doc={selectedDoc}
          onClose={() => setSelectedDoc(null)}
        />
      )}
    </div>
  );
}

function IconBtn({
  children,
  onClick,
  title,
  danger,
}: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  danger?: boolean;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={cn(
        'focus-ring rounded-md p-1.5 transition-colors',
        danger
          ? 'text-slate-400 hover:bg-rose-50 hover:text-rose-600'
          : 'text-slate-400 hover:bg-slate-100 hover:text-slate-700'
      )}
    >
      {children}
    </button>
  );
}

/* ----------------------------- Upload modal ----------------------------- */
function UploadModal({
  deptOptions,
  onClose,
  onUploaded,
}: {
  deptOptions: { value: string; label: string }[];
  onClose: () => void;
  onUploaded: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [departmentId, setDepartmentId] = useState(deptOptions[0]?.value ?? '');
  const [category, setCategory] = useState('技术规范');
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    if (!file) return;
    setLoading(true);
    const fd = new FormData();
    fd.append('file', file);
    fd.append('department_id', departmentId);
    fd.append('category', category);
    await documentsApi.upload(fd);
    setLoading(false);
    onUploaded();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg animate-fade-in rounded-xl bg-white shadow-card-hover">
        <div className="flex items-center justify-between border-b border-slate-100 px-5 py-3.5">
          <h3 className="text-sm font-semibold text-slate-800">上传文档</h3>
          <button onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-4 p-5">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
            }}
            className={cn(
              'flex flex-col items-center justify-center rounded-lg border-2 border-dashed py-8 transition-colors',
              dragging
                ? 'border-primary-400 bg-primary-50/50'
                : 'border-slate-300 bg-slate-50'
            )}
          >
            <Upload className="h-7 w-7 text-slate-400" />
            <p className="mt-2 text-sm text-slate-600">
              {file ? file.name : '拖拽文件到此处，或点击选择'}
            </p>
            <p className="mt-1 text-xs text-slate-400">
              支持 PDF / Word / Excel / Markdown / TXT
            </p>
            <label className="mt-3 cursor-pointer rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50">
              选择文件
              <input
                type="file"
                accept={ACCEPTED}
                className="hidden"
                onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
              />
            </label>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Select
              label="归属部门"
              options={deptOptions}
              value={departmentId}
              onChange={(e) => setDepartmentId(e.target.value)}
            />
            <Select
              label="文档分类"
              options={[
                { value: '技术规范', label: '技术规范' },
                { value: '采购标准', label: '采购标准' },
                { value: '审核报告', label: '审核报告' },
                { value: '管理制度', label: '管理制度' },
                { value: '认证资料', label: '认证资料' },
              ]}
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            />
          </div>
          <p className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700">
            格式白名单：pdf / docx / doc / xlsx / xls / md / txt，单文件 ≤ 50MB
          </p>
        </div>
        <div className="flex justify-end gap-2 border-t border-slate-100 px-5 py-3">
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button onClick={submit} loading={loading} disabled={!file}>
            开始上传
          </Button>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------ Chunk drawer ---------------------------- */
function ChunkDrawer({ doc, onClose }: { doc: KBDocument; onClose: () => void }) {
  const { data: chunks, isLoading } = useQuery({
    queryKey: ['chunks', doc.id],
    queryFn: () => documentsApi.chunks(doc.id),
  });

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-900/40 backdrop-blur-sm">
      <div className="flex w-full max-w-xl animate-slide-in-right flex-col bg-white shadow-card-hover">
        <div className="flex items-start justify-between border-b border-slate-100 px-5 py-4">
          <div className="min-w-0">
            <h3 className="truncate text-sm font-semibold text-slate-800">{doc.title}</h3>
            <p className="mt-0.5 text-xs text-slate-500">
              {doc.department_name} · {doc.category} · {formatBytes(doc.size)} ·{' '}
              {doc.chunk_count} 分块
            </p>
          </div>
          <button onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="scrollbar-thin flex-1 space-y-3 overflow-y-auto p-5">
          {isLoading &&
            Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-24 w-full" />
            ))}
          {!isLoading &&
            (chunks as DocumentChunk[])?.map((c) => (
              <div key={c.id} className="rounded-lg border border-slate-200 p-3">
                <div className="flex items-center gap-2 text-xs">
                  <ChevronRight className="h-3 w-3 text-slate-400" />
                  <span className="text-slate-400">
                    {c.heading_path.join(' › ')}
                  </span>
                  <span className="ml-auto rounded bg-slate-100 px-1.5 py-0.5 text-slate-500">
                    P{c.page_number}
                  </span>
                  <span className="text-slate-400">#{c.chunk_index}</span>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-slate-700">{c.content}</p>
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-[11px] text-slate-400">{c.token_count} tokens</span>
                  {c.bm25_terms?.map((t) => (
                    <span
                      key={t}
                      className="rounded bg-primary-50 px-1.5 py-0.5 text-[10px] text-primary-600"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          {!isLoading && (!chunks || chunks.length === 0) && (
            <p className="py-10 text-center text-sm text-slate-400">暂无分块数据</p>
          )}
        </div>
      </div>
    </div>
  );
}
