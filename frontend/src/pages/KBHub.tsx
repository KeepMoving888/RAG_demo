import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowRight,
  Building2,
  FileText,
  Headphones,
  Library,
  Package,
  type LucideIcon,
} from 'lucide-react';
import { seedIndustries, type IndustryCategory } from '@/api/seed';
import { cn } from '@/lib/utils';

const ICON_MAP: Record<string, LucideIcon> = {
  Building2,
  Package,
  Headphones,
};

// 主题色 → Tailwind 类名映射 (单色系, 企业级简洁风格)
const ACCENT_CLASSES: Record<string, { bg: string; text: string; ring: string; bar: string }> = {
  indigo: {
    bg: 'bg-indigo-50',
    text: 'text-indigo-600',
    ring: 'hover:ring-indigo-200',
    bar: 'bg-indigo-500',
  },
  slate: {
    bg: 'bg-slate-100',
    text: 'text-slate-700',
    ring: 'hover:ring-slate-300',
    bar: 'bg-slate-600',
  },
  emerald: {
    bg: 'bg-emerald-50',
    text: 'text-emerald-600',
    ring: 'hover:ring-emerald-200',
    bar: 'bg-emerald-500',
  },
};

/** 知识库中心: 企业/产品/售后 三大知识库, 简洁企业级风格. */
export default function KBHub() {
  const navigate = useNavigate();

  const stats = useMemo(() => {
    const totalDocs = seedIndustries.reduce((s, i) => s + i.doc_count, 0);
    return { totalKb: seedIndustries.length, totalDocs };
  }, []);

  const goKb = (id: string) => navigate(`/documents?kb=${id}`);

  return (
    <div className="space-y-6">
      {/* 页头 */}
      <header className="flex flex-col gap-1">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-900 text-white shadow-sm">
            <Library className="h-5 w-5" strokeWidth={2} />
          </div>
          <h1 className="text-xl font-bold text-slate-800">知识库中心</h1>
        </div>
        <p className="text-sm text-slate-500">
          按业务域划分三大知识库, 快速进入对应资料库进行检索与问答.
        </p>
      </header>

      {/* 三大知识库卡片 - 等宽网格, 简洁白底 */}
      <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
        {seedIndustries.map((item) => (
          <KBCard key={item.id} item={item} onClick={() => goKb(item.id)} />
        ))}
      </div>

      {/* 底部统计概览条 */}
      <div className="grid grid-cols-2 gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-card">
        <StatInline label="知识库总数" value={`${stats.totalKb}`} suffix="个" icon={Library} />
        <StatInline label="文档总量" value={`${stats.totalDocs}`} suffix="篇" icon={FileText} />
      </div>
    </div>
  );
}

/** 知识库卡片: 白底 + 顶部色条 + 图标 + 名称 + 描述 + 标签 + 入口. */
function KBCard({ item, onClick }: { item: IndustryCategory; onClick: () => void }) {
  const Icon = ICON_MAP[item.icon] ?? FileText;
  const accent = ACCENT_CLASSES[item.accent] ?? ACCENT_CLASSES.indigo;

  return (
    <button
      onClick={onClick}
      className={cn(
        'group relative flex h-full flex-col overflow-hidden rounded-xl border border-slate-200 bg-white text-left shadow-card ring-1 ring-transparent transition-all duration-300 hover:-translate-y-0.5 hover:shadow-card-hover',
        accent.ring
      )}
    >
      {/* 顶部主题色条 */}
      <div className={cn('h-1 w-full', accent.bar)} />

      <div className="flex flex-1 flex-col p-5">
        {/* 图标 + 文档数 */}
        <div className="flex items-start justify-between">
          <div className={cn('flex h-11 w-11 items-center justify-center rounded-lg', accent.bg)}>
            <Icon className={cn('h-5 w-5', accent.text)} strokeWidth={2} />
          </div>
          <span className="rounded-md bg-slate-50 px-2 py-1 text-xs font-medium text-slate-500">
            {item.doc_count} 篇
          </span>
        </div>

        {/* 名称 + 描述 */}
        <h3 className="mt-4 text-base font-semibold text-slate-800">{item.name}</h3>
        <p className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-slate-500">
          {item.description}
        </p>

        {/* 标签 */}
        <div className="mt-3 flex flex-wrap gap-1.5">
          {item.tags.map((tag) => (
            <span
              key={tag}
              className="rounded bg-slate-50 px-2 py-0.5 text-[11px] font-medium text-slate-500"
            >
              {tag}
            </span>
          ))}
        </div>

        {/* 入口 */}
        <div className="mt-auto pt-4">
          <span className="inline-flex items-center gap-1 text-sm font-medium text-slate-600 transition-colors group-hover:text-slate-900">
            进入知识库
            <ArrowRight className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-0.5" />
          </span>
        </div>
      </div>
    </button>
  );
}

/** 统计概览条中的单行指标. */
function StatInline({
  label,
  value,
  suffix,
  icon: Icon,
}: {
  label: string;
  value: string;
  suffix: string;
  icon: LucideIcon;
}) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-slate-100 text-slate-600">
        <Icon className="h-5 w-5" strokeWidth={2} />
      </div>
      <div className="min-w-0">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
        <p className="mt-0.5 text-lg font-bold text-slate-800">
          {value}
          <span className="ml-1 text-xs font-normal text-slate-400">{suffix}</span>
        </p>
      </div>
    </div>
  );
}
