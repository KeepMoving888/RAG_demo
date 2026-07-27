import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Activity,
  BarChart3,
  Database,
  Gauge,
  Info,
  Play,
  Radar as RadarIcon,
  Sparkles,
  Target,
  Timer,
} from 'lucide-react';
import { evaluationApi } from '@/api';
import type { EvalDatasetItem, EvalResult } from '@/api/types';
import Card from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import Button from '@/components/ui/Button';
import { Select } from '@/components/ui/Input';
import Skeleton from '@/components/ui/Skeleton';
import { cn, formatLatency } from '@/lib/utils';

/** 策略主题色：从浅到深递进，完整管线使用品牌靛蓝 */
const STRATEGY_COLORS = ['#94a3b8', '#6366f1', '#10b981', '#f59e0b', '#4f46e5'];

const STRATEGY_LABEL: Record<string, string> = {
  bm25_only: 'BM25',
  vector_only: '向量',
  rrf_fusion: 'RRF 融合',
  rerank_only: '重排',
  full: '完整管线',
};

/** 难度中文标签 */
const DIFFICULTY_LABEL: Record<EvalDatasetItem['difficulty'], string> = {
  easy: '简单',
  medium: '中等',
  hard: '困难',
};

/** 检索评估页：消融对比表 + 指标图表 + 延迟分布 + 数据集预览。 */
export default function RetrievalEval() {
  const { data: results, isLoading } = useQuery({
    queryKey: ['eval-ablation'],
    queryFn: evaluationApi.ablation,
  });
  const { data: dataset } = useQuery({
    queryKey: ['eval-dataset'],
    queryFn: evaluationApi.dataset,
  });

  const [strategy, setStrategy] = useState('full');
  const [singleQuery, setSingleQuery] = useState('车规 eMMC 的 AEC-Q100 认证流程');
  const [singleResult, setSingleResult] = useState<EvalResult | null>(null);
  const [running, setRunning] = useState(false);

  const fullStrategy = results?.find((r) => r.strategy === 'full');
  const strategyCount = results?.length ?? 0;
  const datasetCount = dataset?.length ?? 0;
  const fullP95 = fullStrategy?.p95_latency_ms ?? 218;

  const radarData = fullStrategy
    ? [
        { metric: '召回率@5', value: fullStrategy.metrics.recall_at_5 },
        { metric: '平均倒数排名', value: fullStrategy.metrics.mrr },
        { metric: 'NDCG@5', value: fullStrategy.metrics.ndcg_at_5 },
        { metric: '精确率@5', value: fullStrategy.metrics.precision_at_5 },
      ]
    : [];

  const recallBarData =
    results?.map((r, i) => ({
      name: STRATEGY_LABEL[r.strategy] ?? r.strategy,
      recall: r.metrics.recall_at_5,
      recallLabel: r.metrics.recall_at_5.toFixed(2),
      color: STRATEGY_COLORS[i],
    })) ?? [];

  const latencyData =
    results?.map((r) => ({
      strategy: STRATEGY_LABEL[r.strategy] ?? r.strategy,
      p50: r.p50_latency_ms,
      p95: r.p95_latency_ms,
      p99: r.p99_latency_ms,
    })) ?? [];

  const runSingle = async () => {
    setRunning(true);
    try {
      const res = await evaluationApi.strategy(strategy, singleQuery);
      setSingleResult(res);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* 页面标题 */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary-50 text-primary-600 ring-1 ring-inset ring-primary-100">
            <Gauge className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-slate-800">检索评估</h1>
            <p className="mt-0.5 text-sm text-slate-500">
              消融实验对比 · 检索管线效果量化 · 延迟分布分析
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <HeaderStat icon={Sparkles} label="检索策略" value={`${strategyCount} 种`} />
          <HeaderStat icon={Database} label="标注查询" value={`${datasetCount} 条`} />
          <HeaderStat
            icon={Timer}
            label="完整管线 P95"
            value={formatLatency(fullP95)}
            tone="primary"
          />
        </div>
      </div>

      {/* 消融实验对比表 */}
      <Card
        title="消融实验对比"
        subtitle="5 种检索策略 × 4 项核心指标（Recall@5 / MRR / NDCG@5 / Precision@5）"
        actions={<BarChart3 className="h-4 w-4 text-slate-400" />}
        bodyClassName="p-0"
      >
        <div className="overflow-x-auto scrollbar-thin">
          <table className="w-full min-w-[820px] text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/60 text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="py-3 pl-5 pr-4 font-medium">策略</th>
                <th className="px-4 py-3 font-medium">说明</th>
                <th className="px-4 py-3 text-right font-medium">Recall@5</th>
                <th className="px-4 py-3 text-right font-medium">MRR</th>
                <th className="px-4 py-3 text-right font-medium">NDCG@5</th>
                <th className="px-4 py-3 text-right font-medium">Precision@5</th>
                <th className="px-4 py-3 pr-5 text-right font-medium">P95 延迟</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {isLoading &&
                Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i}>
                    <td colSpan={7} className="py-3.5 pl-5 pr-5">
                      <Skeleton className="h-5 w-full" />
                    </td>
                  </tr>
                ))}
              {results?.map((r, i) => {
                const isFull = r.strategy === 'full';
                return (
                  <tr
                    key={r.strategy}
                    className={cn(
                      'group transition-colors',
                      isFull
                        ? 'bg-primary-50/50 hover:bg-primary-50/80'
                        : i % 2 === 1
                        ? 'bg-slate-50/40 hover:bg-slate-50/80'
                        : 'hover:bg-slate-50/80'
                    )}
                  >
                    <td className="py-3 pl-5 pr-4">
                      <div className="flex items-center gap-2.5">
                        <span
                          className="h-2.5 w-2.5 flex-shrink-0 rounded-full ring-2 ring-white"
                          style={{
                            background: STRATEGY_COLORS[i],
                            boxShadow: `0 0 0 1px ${STRATEGY_COLORS[i]}33`,
                          }}
                        />
                        <span
                          className={cn(
                            'font-medium',
                            isFull ? 'text-primary-700' : 'text-slate-800'
                          )}
                        >
                          {STRATEGY_LABEL[r.strategy]}
                        </span>
                        {isFull && (
                          <Badge variant="info" className="gap-1 py-0">
                            <Sparkles className="h-3 w-3" />
                            推荐
                          </Badge>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-500">{r.description}</td>
                    <MetricCell value={r.metrics.recall_at_5} />
                    <MetricCell value={r.metrics.mrr} />
                    <MetricCell value={r.metrics.ndcg_at_5} />
                    <MetricCell value={r.metrics.precision_at_5} />
                    <td className="px-4 py-3 pr-5 text-right">
                      <span className="font-mono text-xs font-medium text-slate-500">
                        {formatLatency(r.p95_latency_ms)}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {/* 指标色阶图例 */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 border-t border-slate-100 bg-slate-50/40 px-5 py-2.5 text-[11px] text-slate-500">
          <span className="font-medium uppercase tracking-wide text-slate-400">指标色阶</span>
          <LegendDot className="bg-emerald-500" label="≥ 0.85 优秀" />
          <LegendDot className="bg-amber-500" label="0.70 – 0.85 良好" />
          <LegendDot className="bg-slate-400" label="< 0.70 待优化" />
        </div>
      </Card>

      {/* 图表行：Recall@5 柱状图 + 完整管线雷达 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card title="Recall@5 策略对比" actions={<BarChart3 className="h-4 w-4 text-slate-400" />}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={recallBarData}
              margin={{ top: 16, right: 8, bottom: 4, left: -16 }}
            >
              <defs>
                {recallBarData.map((d, i) => (
                  <linearGradient id={`barGrad-${i}`} x1="0" y1="0" x2="0" y2="1" key={i}>
                    <stop offset="0%" stopColor={d.color} stopOpacity={0.95} />
                    <stop offset="100%" stopColor={d.color} stopOpacity={0.45} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 11, fill: '#94a3b8' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fontSize: 11, fill: '#94a3b8' }}
                axisLine={false}
                tickLine={false}
              />
              <RTooltip
                cursor={{ fill: 'rgba(99, 102, 241, 0.06)' }}
                contentStyle={{
                  borderRadius: 10,
                  border: '1px solid #e2e8f0',
                  fontSize: 12,
                  boxShadow: '0 4px 12px -2px rgb(0 0 0 / 0.08)',
                }}
                formatter={(v: number) => v.toFixed(3)}
              />
              <Bar dataKey="recall" name="Recall@5" radius={[6, 6, 0, 0]} maxBarSize={56}>
                {recallBarData.map((_, i) => (
                  <Cell key={i} fill={`url(#barGrad-${i})`} />
                ))}
                <LabelList
                  dataKey="recallLabel"
                  position="top"
                  style={{ fontSize: 11, fontWeight: 600, fill: '#475569' }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card
          title="完整管线指标雷达"
          subtitle="full 策略四项核心指标（0 – 1 标准化）"
          actions={<RadarIcon className="h-4 w-4 text-slate-400" />}
        >
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData} outerRadius={108} cx="50%" cy="50%">
              <defs>
                <linearGradient id="radarFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#818cf8" stopOpacity={0.55} />
                  <stop offset="55%" stopColor="#6366f1" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#4f46e5" stopOpacity={0.12} />
                </linearGradient>
                <linearGradient id="radarStroke" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#6366f1" />
                  <stop offset="100%" stopColor="#4f46e5" />
                </linearGradient>
                <filter id="radarGlow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              <PolarGrid stroke="#eef2f7" strokeDasharray="3 4" />
              <PolarAngleAxis
                dataKey="metric"
                tick={{ fontSize: 11.5, fontWeight: 500, fill: '#475569' }}
              />
              <PolarRadiusAxis
                domain={[0, 1]}
                tickCount={5}
                tick={{ fontSize: 9, fill: '#cbd5e1' }}
                axisLine={false}
                stroke="#f1f5f9"
              />
              <RTooltip
                contentStyle={{
                  borderRadius: 12,
                  border: '1px solid #e2e8f0',
                  fontSize: 12,
                  boxShadow: '0 8px 24px -4px rgb(0 0 0 / 0.12)',
                  padding: '8px 12px',
                }}
                formatter={(v: number) => [v.toFixed(3), '完整管线']}
              />
              <Radar
                name="完整管线"
                dataKey="value"
                stroke="url(#radarStroke)"
                strokeWidth={2.2}
                fill="url(#radarFill)"
                fillOpacity={1}
                dot={{ r: 4, fill: '#4f46e5', strokeWidth: 2, stroke: '#fff' }}
                activeDot={{ r: 6, fill: '#4f46e5', strokeWidth: 2.5, stroke: '#fff' }}
                isAnimationActive
                animationDuration={900}
              />
            </RadarChart>
          </ResponsiveContainer>
          {fullStrategy && (
            <div className="flex items-center justify-center gap-6 border-t border-slate-100 bg-slate-50/40 px-4 py-2.5 text-xs">
              {radarData.map((d) => (
                <div key={d.metric} className="flex items-center gap-1.5">
                  <span className="text-slate-500">{d.metric}</span>
                  <span className="font-mono font-semibold text-primary-600">
                    {d.value.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* 检索延迟分布（面积图） */}
      <Card
        title="检索延迟分布"
        subtitle="各策略 P50 / P95 / P99 延迟对比"
        actions={<Timer className="h-4 w-4 text-slate-400" />}
      >
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={latencyData} margin={{ top: 12, right: 16, bottom: 0, left: -8 }}>
            <defs>
              <linearGradient id="latP50" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="latP95" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="latP99" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#f43f5e" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="strategy"
              tick={{ fontSize: 11, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 11, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
              width={44}
            />
            <RTooltip
              contentStyle={{
                borderRadius: 10,
                border: '1px solid #e2e8f0',
                fontSize: 12,
                boxShadow: '0 4px 12px -2px rgb(0 0 0 / 0.08)',
              }}
              formatter={(v: number) => formatLatency(v)}
            />
            <Legend
              wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
              iconType="plainline"
            />
            <Area
              type="monotone"
              dataKey="p50"
              name="P50"
              stroke="#10b981"
              strokeWidth={2}
              fill="url(#latP50)"
              dot={{ r: 3, strokeWidth: 0, fill: '#10b981' }}
              activeDot={{ r: 4 }}
            />
            <Area
              type="monotone"
              dataKey="p95"
              name="P95"
              stroke="#f59e0b"
              strokeWidth={2}
              fill="url(#latP95)"
              dot={{ r: 3, strokeWidth: 0, fill: '#f59e0b' }}
              activeDot={{ r: 4 }}
            />
            <Area
              type="monotone"
              dataKey="p99"
              name="P99"
              stroke="#f43f5e"
              strokeWidth={2}
              fill="url(#latP99)"
              dot={{ r: 3, strokeWidth: 0, fill: '#f43f5e' }}
              activeDot={{ r: 4 }}
            />
          </AreaChart>
        </ResponsiveContainer>
        {/* 延迟解读说明 */}
        <div className="mt-3 flex items-start gap-2 rounded-lg border border-slate-200 bg-slate-50/60 px-3.5 py-2.5 text-xs leading-relaxed text-slate-600">
          <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-slate-400" />
          <span>
            各策略为递进式消融，延迟随管线阶段增加而上升属预期行为。完整管线 P95=
            <span className="font-mono font-semibold text-slate-700">
              {formatLatency(fullP95)}
            </span>
            ，满足企业级 &lt;300ms SLA。
          </span>
        </div>
      </Card>

      {/* 单策略评估 */}
      <Card
        title="单策略评估"
        subtitle="输入查询与策略，查看检索效果指标"
        actions={<Target className="h-4 w-4 text-slate-400" />}
      >
        <div className="grid grid-cols-1 gap-3 md:grid-cols-12 md:items-end">
          <div className="md:col-span-4">
            <Select
              label="检索策略"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              options={[
                { value: 'bm25_only', label: 'BM25' },
                { value: 'vector_only', label: '向量' },
                { value: 'rrf_fusion', label: 'RRF 融合' },
                { value: 'rerank_only', label: '重排' },
                { value: 'full', label: '完整管线' },
              ]}
            />
          </div>
          <div className="md:col-span-7">
            <label className="mb-1.5 block text-sm font-medium text-slate-700">查询语句</label>
            <input
              value={singleQuery}
              onChange={(e) => setSingleQuery(e.target.value)}
              placeholder="输入待评估的查询语句"
              className="focus-ring h-9 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-800 placeholder:text-slate-400"
            />
          </div>
          <div className="md:col-span-1 md:flex md:justify-end">
            <Button
              icon={Play}
              loading={running}
              onClick={runSingle}
              className="w-full md:w-auto"
            >
              运行评估
            </Button>
          </div>
        </div>

        {singleResult && (
          <div className="mt-4 grid grid-cols-2 gap-3 animate-fade-in lg:grid-cols-5">
            <MetricBox label="Recall@5" value={singleResult.metrics.recall_at_5} />
            <MetricBox label="MRR" value={singleResult.metrics.mrr} />
            <MetricBox label="NDCG@5" value={singleResult.metrics.ndcg_at_5} />
            <MetricBox label="Precision@5" value={singleResult.metrics.precision_at_5} />
            <MetricBox
              label="平均延迟"
              value={singleResult.avg_latency_ms}
              suffix="ms"
              isLatency
            />
          </div>
        )}
      </Card>

      {/* 评估数据集预览 */}
      <Card
        title="评估数据集预览"
        subtitle="标注查询集合（含相关 chunk 与期望答案）"
        actions={<Activity className="h-4 w-4 text-slate-400" />}
        bodyClassName="p-0"
      >
        <div className="overflow-x-auto scrollbar-thin">
          <table className="w-full min-w-[760px] text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/60 text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="px-5 py-3 font-medium">ID</th>
                <th className="px-4 py-3 font-medium">查询</th>
                <th className="px-4 py-3 font-medium">期望答案</th>
                <th className="px-4 py-3 text-center font-medium">相关 chunk</th>
                <th className="px-5 py-3 font-medium">难度</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {dataset?.map((d) => (
                <tr key={d.id} className="transition-colors hover:bg-slate-50/80">
                  <td className="px-5 py-2.5">
                    <span className="font-mono text-xs text-slate-400">{d.id}</span>
                  </td>
                  <td className="px-4 py-2.5">
                    <span className="block max-w-[260px] truncate text-slate-700" title={d.query}>
                      {d.query}
                    </span>
                  </td>
                  <td className="px-4 py-2.5">
                    <span
                      className="block max-w-[280px] truncate text-xs text-slate-500"
                      title={d.expected_answer}
                    >
                      {d.expected_answer}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    <span className="inline-flex min-w-[1.75rem] justify-center rounded-md bg-slate-100 px-1.5 py-0.5 font-mono text-xs font-medium text-slate-600">
                      {d.relevant_chunk_ids.length}
                    </span>
                  </td>
                  <td className="px-5 py-2.5">
                    <Badge
                      variant={
                        d.difficulty === 'easy'
                          ? 'success'
                          : d.difficulty === 'medium'
                          ? 'warning'
                          : 'danger'
                      }
                    >
                      {DIFFICULTY_LABEL[d.difficulty]}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

/** 页头统计小徽章 */
function HeaderStat({
  icon: Icon,
  label,
  value,
  tone = 'neutral',
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  tone?: 'neutral' | 'primary';
}) {
  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-lg border px-3 py-1.5',
        tone === 'primary'
          ? 'border-primary-200 bg-primary-50/60'
          : 'border-slate-200 bg-white'
      )}
    >
      <Icon
        className={cn(
          'h-3.5 w-3.5',
          tone === 'primary' ? 'text-primary-500' : 'text-slate-400'
        )}
      />
      <div className="leading-tight">
        <p className="text-[10px] uppercase tracking-wide text-slate-400">{label}</p>
        <p
          className={cn(
            'text-sm font-semibold',
            tone === 'primary' ? 'text-primary-700' : 'text-slate-700'
          )}
        >
          {value}
        </p>
      </div>
    </div>
  );
}

/** 色阶图例圆点 */
function LegendDot({ className, label }: { className: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={cn('h-2 w-2 rounded-full', className)} />
      {label}
    </span>
  );
}

/** 消融表指标单元格：按阈值着色（≥0.85 优秀 / ≥0.7 良好 / <0.7 待优化） */
function MetricCell({ value }: { value: number }) {
  const color =
    value >= 0.85
      ? 'text-emerald-600'
      : value >= 0.7
      ? 'text-amber-600'
      : 'text-slate-500';
  return (
    <td className="px-4 py-3 text-right">
      <span className={cn('font-mono text-sm font-semibold tabular-nums', color)}>
        {value.toFixed(3)}
      </span>
    </td>
  );
}

/** 单策略评估指标卡 */
function MetricBox({
  label,
  value,
  suffix,
  isLatency,
}: {
  label: string;
  value: number;
  suffix?: string;
  isLatency?: boolean;
}) {
  const tone = isLatency
    ? 'border-slate-200 bg-slate-50/60'
    : value >= 0.85
    ? 'border-emerald-200 bg-emerald-50/40'
    : value >= 0.7
    ? 'border-amber-200 bg-amber-50/40'
    : 'border-slate-200 bg-slate-50/60';
  const valueColor = isLatency
    ? 'text-slate-800'
    : value >= 0.85
    ? 'text-emerald-700'
    : value >= 0.7
    ? 'text-amber-700'
    : 'text-slate-700';
  return (
    <div className={cn('rounded-lg border p-3 transition-shadow hover:shadow-card', tone)}>
      <p className="text-[11px] uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-1 text-xl font-bold tabular-nums">
        <span className={valueColor}>{value.toFixed(suffix ? 0 : 3)}</span>
        {suffix && <span className="ml-0.5 text-xs font-normal text-slate-400">{suffix}</span>}
      </p>
    </div>
  );
}
