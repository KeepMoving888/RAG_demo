import { useMemo, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Copy,
  Database,
  GitBranch,
  Layers,
  Network,
  Play,
  Search,
  Share2,
  Sparkles,
  X,
} from 'lucide-react';
import { graphApi } from '@/api';
import type {
  EntityType,
  GraphData,
  GraphNode,
  GraphPathResult,
  RelationType,
} from '@/api/types';
import { cn } from '@/lib/utils';
import StatCard from '@/components/StatCard';
import ForceGraph from '@/components/ForceGraph';
import Skeleton from '@/components/ui/Skeleton';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';

/** 实体类型可选项：value 为后端枚举，label 为中文展示 */
const TYPE_OPTIONS: { value: EntityType; label: string }[] = [
  { value: 'Product', label: '产品' },
  { value: 'Department', label: '部门' },
  { value: 'Person', label: '人员' },
  { value: 'Policy', label: '制度' },
  { value: 'Supplier', label: '供应商' },
  { value: 'Certification', label: '认证' },
  { value: 'Patent', label: '专利' },
];

/** 实体类型配色（与 ForceGraph 保持一致） */
const TYPE_COLORS: Record<EntityType, string> = {
  Product: '#4f46e5',
  Department: '#10b981',
  Person: '#f59e0b',
  Policy: '#f43f5e',
  Supplier: '#06b6d4',
  Certification: '#8b5cf6',
  Patent: '#0ea5e9',
};

/** 关系类型中文标签 */
const RELATION_LABELS: Record<RelationType, string> = {
  BELONGS_TO: '隶属于',
  MANUFACTURES: '生产',
  CERTIFIED_BY: '认证于',
  SUPPLIES: '供应',
  AUTHORED_BY: '撰写',
  REFERENCES: '引用',
  GOVERNED_BY: '受管于',
  AUDITED_BY: '审计于',
  PARTICIPATES_IN: '参与',
  INVENTED_BY: '发明于',
};

/** StatCard 强调色对应的十六进制值，用于 sparkline 装饰 */
const ACCENT_HEX: Record<'primary' | 'emerald' | 'amber' | 'cyan' | 'rose' | 'slate', string> = {
  primary: '#6366f1',
  emerald: '#10b981',
  amber: '#f59e0b',
  cyan: '#06b6d4',
  rose: '#f43f5e',
  slate: '#64748b',
};

/** Cypher 关键字集合（用于轻量语法高亮） */
const CYPHER_KEYWORDS = new Set([
  'MATCH', 'OPTIONAL', 'WHERE', 'RETURN', 'WITH', 'AS', 'LIMIT', 'ORDER', 'BY',
  'SKIP', 'DISTINCT', 'UNION', 'CREATE', 'MERGE', 'DELETE', 'DETACH', 'SET',
  'AND', 'OR', 'NOT', 'IN', 'CONTAINS', 'STARTS', 'ENDS', 'COUNT', 'COLLECT',
  'UNWIND', 'NODES', 'RELATIONSHIPS', 'ON', 'WHEN', 'THEN', 'ELSE', 'END',
  'CASE', 'EXISTS', 'SIZE', 'TOLOWER', 'TOUPPER', 'TRIM', 'HEAD', 'TAIL',
  'LABELS', 'TYPE', 'ID', 'PROPERTIES', 'KEYS', 'ALL', 'ANY', 'NONE',
  'SINGLE', 'XOR', 'DESC', 'ASC', 'DISTINCT',
]);

/** 轻量 Cypher 语法高亮：将语句拆分为带颜色的 token 节点 */
function highlightCypher(cypher: string): ReactNode[] {
  const tokenRe = /('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|--[^\n]*|\b\d+(?:\.\d+)?\b|:[A-Za-z_]\w*|[A-Za-z_]\w*|[()[\]{}.,;:=<>+\-*/|]+|\s+)/g;
  const nodes: ReactNode[] = [];
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = tokenRe.exec(cypher)) !== null) {
    const tok = m[0];
    if (/^['"]/.test(tok)) {
      nodes.push(<span key={i} className="text-emerald-300">{tok}</span>);
    } else if (/^--/.test(tok)) {
      nodes.push(<span key={i} className="text-slate-500">{tok}</span>);
    } else if (/^\d/.test(tok)) {
      nodes.push(<span key={i} className="text-amber-300">{tok}</span>);
    } else if (/^:/.test(tok)) {
      nodes.push(<span key={i} className="text-cyan-300">{tok}</span>);
    } else if (/^[A-Za-z_]/.test(tok)) {
      if (CYPHER_KEYWORDS.has(tok.toUpperCase())) {
        nodes.push(<span key={i} className="font-semibold text-indigo-300">{tok}</span>);
      } else {
        nodes.push(<span key={i} className="text-slate-200">{tok}</span>);
      }
    } else if (/^\s/.test(tok)) {
      nodes.push(tok);
    } else {
      nodes.push(<span key={i} className="text-slate-400">{tok}</span>);
    }
    i++;
  }
  return nodes;
}

/** 迷你 sparkline 装饰条：放置于 StatCard 右下角作为数据律动点缀 */
function Sparkline({ color, values }: { color: string; values: number[] }) {
  const max = Math.max(...values, 1);
  return (
    <div className="flex h-6 items-end gap-[2px]">
      {values.map((v, i) => (
        <span
          key={i}
          className="w-[3px] rounded-full"
          style={{ height: `${(v / max) * 100}%`, background: color, opacity: 0.35 }}
        />
      ))}
    </div>
  );
}

/** 知识图谱页面：力导向可视化 + 自然语言转 Cypher 查询 + 实体筛选与节点详情 */
export default function KnowledgeGraph() {
  const [query, setQuery] = useState('通过 CE 认证的产品有哪些');
  const [pathResult, setPathResult] = useState<GraphPathResult | null>(null);
  const [querying, setQuerying] = useState(false);
  const [copied, setCopied] = useState(false);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [hiddenTypes, setHiddenTypes] = useState<Set<EntityType>>(new Set());

  const statsQuery = useQuery({ queryKey: ['graph-stats'], queryFn: graphApi.stats });
  const stats = statsQuery.data;
  const { data: rawData, isLoading } = useQuery({
    queryKey: ['graph-data'],
    queryFn: graphApi.data,
  });

  const data: GraphData = useMemo(() => {
    if (!rawData) return { nodes: [], links: [] };
    const visibleNodes = rawData.nodes.filter((n) => !hiddenTypes.has(n.type));
    const visibleIds = new Set(visibleNodes.map((n) => n.id));
    const visibleLinks = rawData.links.filter(
      (l) => visibleIds.has(l.source) && visibleIds.has(l.target)
    );
    return { nodes: visibleNodes, links: visibleLinks };
  }, [rawData, hiddenTypes]);

  const runQuery = async () => {
    if (!query.trim()) return;
    setQuerying(true);
    try {
      const res = await graphApi.query(query);
      setPathResult(res);
    } finally {
      setQuerying(false);
    }
  };

  const copyCypher = async () => {
    if (!pathResult) return;
    try {
      await navigator.clipboard.writeText(pathResult.cypher);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* 剪贴板不可用时静默忽略 */
    }
  };

  const neighbors = useMemo(() => {
    if (!selected || !rawData) return [];
    return rawData.links
      .filter((l) => l.source === selected.id || l.target === selected.id)
      .map((l) => {
        const otherId = l.source === selected.id ? l.target : l.source;
        return {
          node: rawData.nodes.find((n) => n.id === otherId),
          type: l.type,
          direction: (l.source === selected.id ? 'out' : 'in') as 'out' | 'in',
        };
      })
      .filter(
        (x): x is { node: GraphNode; type: RelationType; direction: 'out' | 'in' } =>
          x.node !== undefined
      );
  }, [selected, rawData]);

  const toggleType = (t: EntityType) => {
    setHiddenTypes((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });
  };

  const avgDegree = stats
    ? ((stats.relation_count / Math.max(1, stats.node_count)) * 2).toFixed(1)
    : '0';
  const activeTypeCount = stats
    ? Object.values(stats.type_distribution).filter((c) => c > 0).length
    : 0;

  // Neo4j 连接状态：依据 stats 查询阶段推导
  const neoStatus: { variant: 'success' | 'warning' | 'info'; label: string } = statsQuery.isError
    ? { variant: 'warning', label: 'Neo4j · 连接降级' }
    : statsQuery.isPending
      ? { variant: 'info', label: 'Neo4j · 连接中' }
      : { variant: 'success', label: 'Neo4j · 已连接' };

  return (
    <div className="space-y-5">
      {/* 页头：图标 + 标题 + 副标题 + Neo4j 连接状态徽标 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary-50 text-primary-600">
            <Network className="h-5 w-5" strokeWidth={2} />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-slate-800">知识图谱</h1>
            <p className="text-xs text-slate-500">
              实体关系可视化 · 自然语言转 Cypher · 路径推理
            </p>
          </div>
        </div>
        <Badge variant={neoStatus.variant} dot>
          {neoStatus.label}
        </Badge>
      </div>

      {/* 统计卡片：四宫格 + sparkline 装饰 */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <div className="group relative">
          <StatCard
            label="节点总数"
            value={stats?.node_count ?? 0}
            icon={Network}
            accent="primary"
            hint="覆盖全部实体类型"
          />
          <div className="pointer-events-none absolute bottom-5 right-5 transition-transform duration-200 group-hover:-translate-y-0.5">
            <Sparkline color={ACCENT_HEX.primary} values={[3, 5, 4, 6, 5, 7, 6, 8]} />
          </div>
        </div>
        <div className="group relative">
          <StatCard
            label="关系总数"
            value={stats?.relation_count ?? 0}
            icon={GitBranch}
            accent="emerald"
            hint="实体间连接边"
          />
          <div className="pointer-events-none absolute bottom-5 right-5 transition-transform duration-200 group-hover:-translate-y-0.5">
            <Sparkline color={ACCENT_HEX.emerald} values={[4, 6, 5, 7, 6, 8, 7, 9]} />
          </div>
        </div>
        <div className="group relative">
          <StatCard
            label="实体类型"
            value={activeTypeCount}
            icon={Layers}
            accent="amber"
            hint="已启用类型数"
          />
          <div className="pointer-events-none absolute bottom-5 right-5 transition-transform duration-200 group-hover:-translate-y-0.5">
            <Sparkline color={ACCENT_HEX.amber} values={[2, 3, 3, 4, 4, 5, 5, 6]} />
          </div>
        </div>
        <div className="group relative">
          <StatCard
            label="平均度数"
            value={avgDegree}
            icon={Share2}
            accent="cyan"
            hint="每节点平均连接数"
          />
          <div className="pointer-events-none absolute bottom-5 right-5 transition-transform duration-200 group-hover:-translate-y-0.5">
            <Sparkline color={ACCENT_HEX.cyan} values={[5, 4, 6, 5, 7, 6, 8, 7]} />
          </div>
        </div>
      </div>

      {/* 自然语言查询控制台 */}
      <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-card">
        <div className="flex items-center gap-2 border-b border-slate-100 px-4 py-2.5">
          <Sparkles className="h-4 w-4 text-primary-500" />
          <span className="text-sm font-semibold text-slate-800">自然语言查询</span>
          <Badge variant="info" className="ml-auto gap-1 py-0">
            <Network className="h-3 w-3" />
            存储行业知识图谱
          </Badge>
        </div>
        <div className="p-4">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && runQuery()}
                placeholder="如：通过 CE 认证的产品有哪些"
                className="focus-ring h-10 w-full rounded-lg border border-slate-300 pl-9 pr-3 text-sm placeholder:text-slate-400"
              />
            </div>
            <Button icon={Play} loading={querying} onClick={runQuery}>
              执行查询
            </Button>
          </div>
          <p className="mt-1.5 text-xs text-slate-400">
            按 Enter 执行 · 支持自然语言描述实体关系与路径
          </p>

          {pathResult && (
            <div className="mt-4 space-y-3 animate-fade-in">
              {/* 生成的 Cypher：深色代码块 + 轻量语法高亮 */}
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-500">生成 Cypher</span>
                  <button
                    onClick={copyCypher}
                    className="focus-ring inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
                  >
                    {copied ? (
                      <>
                        <Check className="h-3 w-3" /> 已复制
                      </>
                    ) : (
                      <>
                        <Copy className="h-3 w-3" /> 复制
                      </>
                    )}
                  </button>
                </div>
                <pre className="scrollbar-dark mt-1.5 overflow-x-auto rounded-lg bg-slate-900 p-3 text-xs leading-relaxed">
                  <code className="font-mono">{highlightCypher(pathResult.cypher)}</code>
                </pre>
              </div>

              {/* 解释说明 */}
              <div className="flex items-start gap-2 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800">
                <Database className="mt-0.5 h-4 w-4 flex-shrink-0" />
                <p>{pathResult.explanation}</p>
              </div>

              {/* 查询结果表格 */}
              {pathResult.records.length > 0 && (
                <div className="overflow-hidden rounded-lg border border-slate-200">
                  <div className="border-b border-slate-100 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-500">
                    查询结果 · {pathResult.records.length} 条记录
                  </div>
                  <div className="scrollbar-thin overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-200 bg-slate-50/60 text-left text-xs text-slate-500">
                          {Object.keys(pathResult.records[0]).map((k) => (
                            <th key={k} className="px-3 py-2 font-medium">{k}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {pathResult.records.map((r, i) => (
                          <tr key={i} className="transition-colors hover:bg-slate-50">
                            {Object.values(r).map((v, j) => (
                              <td key={j} className="px-3 py-2 text-slate-700">{String(v)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* 图谱 + 侧边栏：3/4 + 1/4 布局 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
        {/* 图谱区域 */}
        <div className="lg:col-span-3">
          <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-card">
            {/* 头部条 */}
            <div className="flex items-center justify-between border-b border-slate-100 px-4 py-2.5">
              <div className="flex items-baseline gap-2">
                <h3 className="text-sm font-semibold text-slate-800">关系图谱</h3>
                <span className="text-xs text-slate-400">
                  {data.nodes.length} 节点 · {data.links.length} 关系
                </span>
              </div>
              <span className="text-xs text-slate-400">点击节点查看详情</span>
            </div>
            {/* 实体筛选胶囊 */}
            <div className="flex flex-wrap items-center gap-2 border-b border-slate-100 px-4 py-2.5">
              <span className="text-xs font-medium text-slate-500">实体筛选</span>
              {TYPE_OPTIONS.map((t) => {
                const hidden = hiddenTypes.has(t.value);
                const count = stats?.type_distribution[t.value] ?? 0;
                return (
                  <button
                    key={t.value}
                    onClick={() => toggleType(t.value)}
                    className={cn(
                      'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-all duration-200',
                      hidden
                        ? 'border-slate-200 bg-slate-50 text-slate-400 hover:border-slate-300 hover:text-slate-500'
                        : 'border-slate-200 bg-white text-slate-700 shadow-sm hover:border-slate-300 hover:shadow'
                    )}
                  >
                    <span
                      className="h-2 w-2 rounded-full transition-all"
                      style={{
                        background: hidden ? '#cbd5e1' : TYPE_COLORS[t.value],
                        boxShadow: hidden ? 'none' : `0 0 0 2px ${TYPE_COLORS[t.value]}25`,
                      }}
                    />
                    {t.label}
                    <span className={cn('text-[10px]', hidden ? 'text-slate-300' : 'text-slate-400')}>
                      {count}
                    </span>
                  </button>
                );
              })}
            </div>
            {/* 力导向图 */}
            <div className="relative">
              {isLoading ? (
                <Skeleton className="h-[520px] w-full" />
              ) : data.nodes.length === 0 ? (
                <div className="flex h-[520px] flex-col items-center justify-center text-center">
                  <Network className="h-8 w-8 text-slate-300" />
                  <p className="mt-2 text-sm text-slate-400">暂无可显示的节点</p>
                  <p className="text-xs text-slate-400">请调整实体筛选条件</p>
                </div>
              ) : (
                <ForceGraph
                  data={data}
                  onNodeClick={setSelected}
                  height={520}
                  className="border-0 rounded-none"
                />
              )}
            </div>
          </div>
        </div>

        {/* 侧边栏：实体类型分布 + 关系类型分布 */}
        <div className="space-y-4 lg:col-span-1">
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-card">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold text-slate-700">实体类型分布</p>
              <span className="text-[11px] text-slate-400">{stats?.node_count ?? 0} 节点</span>
            </div>
            <div className="mt-3 space-y-2.5">
              {stats &&
                (Object.entries(stats.type_distribution) as [EntityType, number][])
                  .filter(([, c]) => c > 0)
                  .map(([t, count]) => {
                    const total = stats.node_count || 1;
                    return (
                      <div key={t}>
                        <div className="flex items-center justify-between text-xs">
                          <span className="flex items-center gap-1.5 text-slate-600">
                            <span
                              className="h-2 w-2 rounded-full"
                              style={{ background: TYPE_COLORS[t] }}
                            />
                            {TYPE_OPTIONS.find((o) => o.value === t)?.label ?? t}
                          </span>
                          <span className="font-medium text-slate-700">{count}</span>
                        </div>
                        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-100">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{
                              width: `${(count / total) * 100}%`,
                              background: TYPE_COLORS[t],
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
              {!stats && <Skeleton className="h-24 w-full" />}
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-card">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold text-slate-700">关系类型分布</p>
              <span className="text-[11px] text-slate-400">{stats?.relation_count ?? 0} 关系</span>
            </div>
            <div className="mt-3 space-y-2.5">
              {stats &&
                (Object.entries(stats.relation_distribution) as [RelationType, number][])
                  .filter(([, c]) => c > 0)
                  .map(([r, count]) => {
                    const total = stats.relation_count || 1;
                    return (
                      <div key={r}>
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-slate-600">{RELATION_LABELS[r] ?? r}</span>
                          <span className="font-medium text-slate-700">{count}</span>
                        </div>
                        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-100">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{
                              width: `${(count / total) * 100}%`,
                              background: 'linear-gradient(90deg, #c7d2fe, #6366f1)',
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
              {!stats && <Skeleton className="h-24 w-full" />}
            </div>
          </div>
        </div>
      </div>

      {/* 节点详情抽屉 */}
      {selected && (
        <div className="fixed inset-0 z-50 flex justify-end">
          <div
            className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm"
            onClick={() => setSelected(null)}
          />
          <div className="relative flex w-full max-w-md flex-col bg-white shadow-card-hover animate-slide-in-right">
            {/* 顶部类型色带 */}
            <div className="h-1" style={{ background: TYPE_COLORS[selected.type] }} />
            {/* 抽屉头部 */}
            <div className="flex items-start justify-between border-b border-slate-100 px-5 py-4">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ background: TYPE_COLORS[selected.type] }}
                  />
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">
                    {TYPE_OPTIONS.find((o) => o.value === selected.type)?.label}
                  </span>
                </div>
                <h3 className="mt-1.5 truncate text-base font-semibold text-slate-800">
                  {selected.label}
                </h3>
                <p className="mt-0.5 text-xs text-slate-500">
                  来源 chunk {selected.source_chunks} · ID {selected.id}
                </p>
              </div>
              <button
                onClick={() => setSelected(null)}
                className="focus-ring rounded-md p-1 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            {/* 抽屉主体 */}
            <div className="scrollbar-thin flex-1 overflow-y-auto p-5">
              {/* 属性表 */}
              <section>
                <p className="text-xs font-medium text-slate-500">属性</p>
                {Object.keys(selected.properties).length === 0 ? (
                  <p className="mt-2 text-xs text-slate-400">暂无属性</p>
                ) : (
                  <div className="mt-2 overflow-hidden rounded-lg border border-slate-200">
                    <table className="w-full text-sm">
                      <tbody className="divide-y divide-slate-100">
                        {Object.entries(selected.properties).map(([k, v]) => (
                          <tr key={k} className="transition-colors hover:bg-slate-50">
                            <td className="w-1/3 bg-slate-50/60 px-3 py-2 align-top text-xs font-medium text-slate-500">
                              {k}
                            </td>
                            <td className="px-3 py-2 text-slate-800">{String(v)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>

              {/* 邻居关系 */}
              <section className="mt-5">
                <p className="text-xs font-medium text-slate-500">
                  邻居关系（{neighbors.length}）
                </p>
                {neighbors.length === 0 ? (
                  <p className="mt-2 text-xs text-slate-400">暂无邻居关系</p>
                ) : (
                  <div className="mt-2 space-y-1.5">
                    {neighbors.map((nb, i) => (
                      <button
                        key={i}
                        onClick={() => setSelected(nb.node)}
                        className="group flex w-full items-center gap-2.5 rounded-lg border border-slate-200 px-3 py-2 text-left transition-colors hover:border-slate-300 hover:bg-slate-50"
                      >
                        <span
                          className="h-2 w-2 flex-shrink-0 rounded-full"
                          style={{ background: TYPE_COLORS[nb.node.type] }}
                        />
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-medium text-slate-700">
                            {nb.node.label}
                          </p>
                          <p className="text-[11px] text-slate-400">
                            {TYPE_OPTIONS.find((o) => o.value === nb.node.type)?.label}
                          </p>
                        </div>
                        <span className="flex flex-shrink-0 items-center gap-1 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-500">
                          {nb.direction === 'out' ? (
                            <ArrowRight className="h-2.5 w-2.5" />
                          ) : (
                            <ArrowLeft className="h-2.5 w-2.5" />
                          )}
                          {RELATION_LABELS[nb.type] ?? nb.type}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </section>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
