import { ChevronRight, FileText } from 'lucide-react';
import type { Citation } from '@/api/types';
import { cn, scoreBg } from '@/lib/utils';
import Tooltip from './ui/Tooltip';

export interface CitationCardProps {
  citation: Citation;
  index: number;
  onJump?: (documentId: string, chunkId: string) => void;
}

const STAGE_LABEL: Record<Citation['source'] & string, string> = {
  bm25: 'BM25',
  vector: '向量',
  rrf: 'RRF 融合',
  rerank: '重排',
};

/**
 * Answer-citation card: surfaces the source chunk snippet, heading path,
 * page number and a color-coded retrieval score. Used under assistant
 * answers for answer traceability.
 */
export default function CitationCard({ citation, index, onJump }: CitationCardProps) {
  return (
    <div className="group flex gap-3 rounded-lg border border-slate-200 bg-slate-50/60 p-3 transition-colors hover:border-primary-200 hover:bg-primary-50/40">
      <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-primary-600 text-xs font-semibold text-white">
        {index + 1}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <FileText className="h-3.5 w-3.5 flex-shrink-0 text-slate-400" />
          <p className="truncate text-xs font-medium text-slate-700">
            {citation.document_title}
          </p>
        </div>
        <p className="mt-1 text-[11px] text-slate-400">
          {citation.heading_path.join(' › ')} · 第 {citation.page_number} 页
        </p>
        <p className="mt-1.5 line-clamp-3 text-xs leading-relaxed text-slate-600">
          {citation.snippet}
        </p>
        <div className="mt-2 flex items-center gap-2">
          <Tooltip
            content={
              <div className="space-y-0.5">
                <p>检索得分：{citation.score.toFixed(3)}</p>
                {citation.source && <p>命中阶段：{STAGE_LABEL[citation.source]}</p>}
              </div>
            }
          >
            <span
              className={cn(
                'inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-semibold',
                scoreBg(citation.score)
              )}
            >
              {citation.score.toFixed(2)}
            </span>
          </Tooltip>
          {citation.source && (
            <span className="rounded bg-slate-200 px-1.5 py-0.5 text-[10px] font-medium text-slate-600">
              {STAGE_LABEL[citation.source]}
            </span>
          )}
          {onJump && (
            <button
              onClick={() => onJump(citation.document_id, citation.chunk_id)}
              className="ml-auto inline-flex items-center gap-0.5 text-[11px] font-medium text-primary-600 opacity-0 transition-opacity group-hover:opacity-100"
            >
              跳转原文 <ChevronRight className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
