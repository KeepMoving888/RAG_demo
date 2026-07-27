import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts';
import { Activity, Database, Zap } from 'lucide-react';
import type { RetrievalExplain, RetrievedChunk } from '@/api/types';
import { cn, formatLatency, scoreBg, scoreColor } from '@/lib/utils';
import Tooltip from './ui/Tooltip';

export interface RetrievalDetailProps {
  retrieval: RetrievalExplain;
}

const STAGE_COLORS: Record<string, string> = {
  bm25: '#64748b',
  vector: '#6366f1',
  rrf: '#10b981',
  rerank: '#f59e0b',
};

const STAGE_LABEL: Record<string, string> = {
  bm25: 'BM25',
  vector: '向量',
  rrf: 'RRF 融合',
  rerank: '重排',
};

/**
 * Retrieval explainability panel: shows the rewritten query, per-stage
 * latency bar chart, cache-hit flag and recalled chunks with color-coded
 * scores.
 */
export default function RetrievalDetail({ retrieval }: RetrievalDetailProps) {
  const chartData = retrieval.stages.map((s) => ({
    name: STAGE_LABEL[s.stage] ?? s.stage,
    latency: Number(s.latency_ms.toFixed(1)),
    stage: s.stage,
  }));

  return (
    <div className="space-y-4">
      {/* Query rewrite */}
      <div className="rounded-lg border border-slate-200 bg-slate-50/60 p-3">
        <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
          原始查询
        </p>
        <p className="mt-1 text-sm text-slate-700">{retrieval.query}</p>
        <p className="mt-2 text-[11px] font-medium uppercase tracking-wide text-slate-400">
          重写查询
        </p>
        <p className="mt-1 text-sm font-medium text-primary-700">
          {retrieval.rewritten_query}
        </p>
      </div>

      {/* Cache + total latency */}
      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-lg border border-slate-200 p-3">
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <Zap className="h-3.5 w-3.5" /> 总延迟
          </div>
          <p className="mt-1 text-lg font-bold text-slate-800">
            {formatLatency(retrieval.total_latency_ms)}
          </p>
        </div>
        <div className="rounded-lg border border-slate-200 p-3">
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <Database className="h-3.5 w-3.5" /> 缓存命中
          </div>
          <p
            className={cn(
              'mt-1 text-lg font-bold',
              retrieval.cache_hit ? 'text-emerald-600' : 'text-slate-400'
            )}
          >
            {retrieval.cache_hit ? '是' : '否'}
          </p>
        </div>
      </div>

      {/* Stage latency chart */}
      <div>
        <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-slate-600">
          <Activity className="h-3.5 w-3.5" /> 各阶段延迟 (ms)
        </div>
        <ResponsiveContainer width="100%" height={140}>
          <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 11, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 11, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
            />
            <Bar dataKey="latency" radius={[4, 4, 0, 0]}>
              {chartData.map((d) => (
                <Cell key={d.stage} fill={STAGE_COLORS[d.stage] ?? '#94a3b8'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Recalled chunks */}
      <div>
        <p className="mb-2 text-xs font-medium text-slate-600">
          召回 Chunks（{retrieval.chunks.length}）
        </p>
        <div className="space-y-1.5">
          {retrieval.chunks.map((chunk, i) => (
            <ChunkRow key={chunk.chunk_id + i} chunk={chunk} />
          ))}
        </div>
      </div>
    </div>
  );
}

function ChunkRow({ chunk }: { chunk: RetrievedChunk }) {
  return (
    <Tooltip
      content={
        <div className="space-y-1">
          <p className="font-medium">{chunk.document_title}</p>
          <p className="text-slate-300">{chunk.snippet}</p>
          {chunk.bm25_score !== undefined && (
            <p>BM25 {chunk.bm25_score.toFixed(2)} · 向量 {chunk.vector_score?.toFixed(2)} · RRF {chunk.rrf_score?.toFixed(2)} · 重排 {chunk.rerank_score?.toFixed(2)}</p>
          )}
        </div>
      }
    >
      <div className="flex items-center gap-2 rounded-md border border-slate-200 px-2 py-1.5 transition-colors hover:bg-slate-50">
        <span
          className={cn(
            'inline-flex h-7 w-12 flex-shrink-0 items-center justify-center rounded text-xs font-bold',
            scoreBg(chunk.score)
          )}
        >
          {chunk.score.toFixed(2)}
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium text-slate-700">
            {chunk.document_title}
          </p>
          <p className="truncate text-[10px] text-slate-400">
            {chunk.heading_path.join(' › ')} · P{chunk.page_number}
          </p>
        </div>
        <span className={cn('text-xs font-semibold', scoreColor(chunk.score))}>
          #
        </span>
      </div>
    </Tooltip>
  );
}
