import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Activity,
  AlertTriangle,
  Building2,
  FileText,
  Gauge,
  MessagesSquare,
  Network,
  ScrollText,
  Search,
  Timer,
  UserCog,
  Users,
  Zap,
} from 'lucide-react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { adminApi } from '@/api';
import type { AuditLog, Department, User } from '@/api/types';
import StatCard from '@/components/StatCard';
import Card from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import Button from '@/components/ui/Button';
import { Select } from '@/components/ui/Input';
import Skeleton from '@/components/ui/Skeleton';
import { cn, formatDateTime } from '@/lib/utils';

type Tab = 'users' | 'departments' | 'stats' | 'audit';

/** Admin console: users / departments / system stats / audit logs. */
export default function Admin() {
  const [tab, setTab] = useState<Tab>('stats');

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-bold text-slate-800">管理后台</h1>
        <p className="mt-0.5 text-sm text-slate-500">
          用户与部门管理、系统运行统计、审计日志查询
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 rounded-xl border border-slate-200 bg-white p-1 shadow-card">
        {(
          [
            { key: 'stats', label: '系统统计', icon: Gauge },
            { key: 'users', label: '用户管理', icon: UserCog },
            { key: 'departments', label: '部门管理', icon: Building2 },
            { key: 'audit', label: '审计日志', icon: ScrollText },
          ] as { key: Tab; label: string; icon: typeof Gauge }[]
        ).map((t) => {
          const Icon = t.icon;
          return (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={cn(
                'flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                tab === t.key
                  ? 'bg-primary-600 text-white shadow-sm'
                  : 'text-slate-600 hover:bg-slate-100'
              )}
            >
              <Icon className="h-4 w-4" />
              {t.label}
            </button>
          );
        })}
      </div>

      {tab === 'stats' && <StatsTab />}
      {tab === 'users' && <UsersTab />}
      {tab === 'departments' && <DepartmentsTab />}
      {tab === 'audit' && <AuditTab />}
    </div>
  );
}

/* ------------------------------- Stats --------------------------------- */
function StatsTab() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['admin-stats'],
    queryFn: adminApi.stats,
  });

  const trend = stats?.trend ?? [];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="用户总数" value={stats?.user_count ?? 0} icon={Users} accent="primary" trend="up" trendValue="+4" loading={isLoading} />
        <StatCard label="文档总数" value={stats?.document_count ?? 0} icon={FileText} accent="emerald" trend="up" trendValue="+23" loading={isLoading} />
        <StatCard label="会话总数" value={stats?.session_count ?? 0} icon={MessagesSquare} accent="cyan" trend="up" trendValue="+128" loading={isLoading} />
        <StatCard label="检索次数" value={stats?.retrieval_count ?? 0} icon={Search} accent="amber" trend="up" trendValue="+1.2k" loading={isLoading} />
        <StatCard
          label="缓存命中率"
          value={stats ? `${(stats.cache_hit_rate * 100).toFixed(1)}%` : '-'}
          icon={Zap}
          accent="emerald"
          trend="up"
          trendValue="+2.1%"
          loading={isLoading}
        />
        <StatCard label="图谱节点" value={stats?.graph_node_count ?? 0} icon={Network} accent="primary" trend="up" trendValue="+12" loading={isLoading} />
        <StatCard label="解析失败" value={stats?.parse_failed_count ?? 0} icon={AlertTriangle} accent="rose" trend="down" trendValue="-3" loading={isLoading} />
        <StatCard label="限流次数" value={stats?.rate_limited_count ?? 0} icon={Timer} accent="amber" trend="flat" trendValue="0" loading={isLoading} />
      </div>

      <Card title="近 14 日检索趋势" actions={<Activity className="h-4 w-4 text-slate-400" />}>
        {isLoading ? (
          <Skeleton className="h-64 w-full" />
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={trend} margin={{ top: 8, right: 16, bottom: 0, left: -8 }}>
              <defs>
                <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#4f46e5" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#4f46e5" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <RTooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12 }}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#4f46e5"
                strokeWidth={2.5}
                dot={{ r: 2 }}
                activeDot={{ r: 5 }}
                name="检索次数"
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Card>
    </div>
  );
}

/* ------------------------------- Users --------------------------------- */
function UsersTab() {
  const qc = useQueryClient();
  const { data: users, isLoading } = useQuery({
    queryKey: ['admin-users'],
    queryFn: adminApi.users,
  });

  const ROLE_BADGE = {
    admin: { variant: 'danger' as const, label: '管理员' },
    editor: { variant: 'info' as const, label: '编辑' },
    viewer: { variant: 'neutral' as const, label: '查看者' },
  };

  return (
    <Card
      title="用户管理"
      actions={
        <Button size="sm" icon={UserCog}>
          新增用户
        </Button>
      }
      noPadding
    >
      <div className="overflow-x-auto scrollbar-thin">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="px-4 py-2.5 font-medium">用户</th>
              <th className="px-4 py-2.5 font-medium">邮箱</th>
              <th className="px-4 py-2.5 font-medium">部门</th>
              <th className="px-4 py-2.5 font-medium">角色</th>
              <th className="px-4 py-2.5 font-medium">状态</th>
              <th className="px-4 py-2.5 font-medium">最近登录</th>
              <th className="px-4 py-2.5 text-right font-medium">操作</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {isLoading &&
              Array.from({ length: 4 }).map((_, i) => (
                <tr key={i}>
                  <td colSpan={7} className="px-4 py-3">
                    <Skeleton className="h-5 w-full" />
                  </td>
                </tr>
              ))}
            {users?.map((u: User) => (
              <tr key={u.id} className="hover:bg-slate-50">
                <td className="px-4 py-2.5">
                  <div className="flex items-center gap-2">
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary-600 text-xs font-semibold text-white">
                      {u.name[0]}
                    </div>
                    <span className="font-medium text-slate-800">{u.name}</span>
                  </div>
                </td>
                <td className="px-4 py-2.5 text-slate-500">{u.email}</td>
                <td className="px-4 py-2.5 text-slate-600">{u.department_name ?? '-'}</td>
                <td className="px-4 py-2.5">
                  <Badge variant={ROLE_BADGE[u.role].variant}>{ROLE_BADGE[u.role].label}</Badge>
                </td>
                <td className="px-4 py-2.5">
                  <Badge variant={u.is_active ? 'success' : 'neutral'} dot>
                    {u.is_active ? '启用' : '禁用'}
                  </Badge>
                </td>
                <td className="px-4 py-2.5 text-xs text-slate-500">
                  {formatDateTime(u.last_login_at)}
                </td>
                <td className="px-4 py-2.5 text-right">
                  <button
                    onClick={() => qc.invalidateQueries({ queryKey: ['admin-users'] })}
                    className="text-xs font-medium text-primary-600 hover:text-primary-700"
                  >
                    编辑
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/* --------------------------- Departments ------------------------------- */
function DepartmentsTab() {
  const { data: departments, isLoading } = useQuery({
    queryKey: ['admin-departments'],
    queryFn: adminApi.departments,
  });

  return (
    <Card
      title="部门管理"
      subtitle="组织架构树"
      actions={
        <Button size="sm" icon={Building2}>
          新增部门
        </Button>
      }
    >
      {isLoading ? (
        <Skeleton className="h-48 w-full" />
      ) : (
        <div className="space-y-1">
          {departments?.map((d) => (
            <DeptNode key={d.id} dept={d} depth={0} />
          ))}
        </div>
      )}
    </Card>
  );
}

function DeptNode({ dept, depth }: { dept: Department; depth: number }) {
  return (
    <>
      <div
        className="flex items-center gap-2 rounded-lg py-2 hover:bg-slate-50"
        style={{ paddingLeft: depth * 20 }}
      >
        <Building2 className="h-4 w-4 text-slate-400" />
        <span className="text-sm font-medium text-slate-700">{dept.name}</span>
        <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">
          {dept.code}
        </span>
        <Badge variant="neutral">{dept.member_count} 人</Badge>
      </div>
      {dept.children?.map((c) => (
        <DeptNode key={c.id} dept={c} depth={depth + 1} />
      ))}
    </>
  );
}

/* -------------------------------- Audit -------------------------------- */
function AuditTab() {
  const [action, setAction] = useState('');
  const { data: logs, isLoading } = useQuery({
    queryKey: ['admin-audit', action],
    queryFn: () => adminApi.auditLogs({ action: action || undefined }),
  });

  return (
    <Card
      title="审计日志"
      actions={
        <div className="w-48">
          <Select
            placeholder="全部操作"
            value={action}
            onChange={(e) => setAction(e.target.value)}
            options={[
              { value: 'document', label: '文档操作' },
              { value: 'qa', label: '问答操作' },
              { value: 'user', label: '用户操作' },
              { value: 'graph', label: '图谱操作' },
            ]}
          />
        </div>
      }
      noPadding
    >
      <div className="overflow-x-auto scrollbar-thin">
        <table className="w-full min-w-[760px] text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="px-4 py-2.5 font-medium">时间</th>
              <th className="px-4 py-2.5 font-medium">用户</th>
              <th className="px-4 py-2.5 font-medium">操作</th>
              <th className="px-4 py-2.5 font-medium">详情</th>
              <th className="px-4 py-2.5 font-medium">IP</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {isLoading &&
              Array.from({ length: 4 }).map((_, i) => (
                <tr key={i}>
                  <td colSpan={5} className="px-4 py-3">
                    <Skeleton className="h-5 w-full" />
                  </td>
                </tr>
              ))}
            {logs?.map((log: AuditLog) => (
              <tr key={log.id} className="hover:bg-slate-50">
                <td className="px-4 py-2.5 text-xs text-slate-500">
                  {formatDateTime(log.created_at)}
                </td>
                <td className="px-4 py-2.5 text-slate-700">{log.user_name}</td>
                <td className="px-4 py-2.5">
                  <code className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] text-primary-700">
                    {log.action}
                  </code>
                </td>
                <td className="px-4 py-2.5 text-xs text-slate-600">{log.detail}</td>
                <td className="px-4 py-2.5 text-xs text-slate-400">{log.ip}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
