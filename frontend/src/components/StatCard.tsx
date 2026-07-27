import { ArrowDownRight, ArrowUpRight, type LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import Skeleton from './ui/Skeleton';

export type Trend = 'up' | 'down' | 'flat';

export interface StatCardProps {
  label: string;
  value: string | number;
  icon: LucideIcon;
  accent?: 'primary' | 'emerald' | 'amber' | 'rose' | 'cyan' | 'slate';
  trend?: Trend;
  trendValue?: string;
  hint?: string;
  loading?: boolean;
}

const ACCENT_CLASSES: Record<NonNullable<StatCardProps['accent']>, string> = {
  primary: 'bg-primary-50 text-primary-600',
  emerald: 'bg-emerald-50 text-emerald-600',
  amber: 'bg-amber-50 text-amber-600',
  rose: 'bg-rose-50 text-rose-600',
  cyan: 'bg-cyan-50 text-cyan-600',
  slate: 'bg-slate-100 text-slate-600',
};

/** KPI statistic card: large value, label, icon chip and trend indicator. */
export default function StatCard({
  label,
  value,
  icon: Icon,
  accent = 'primary',
  trend,
  trendValue,
  hint,
  loading,
}: StatCardProps) {
  return (
    <div className="group rounded-xl border border-slate-200 bg-white p-5 shadow-card transition-all duration-200 hover:-translate-y-0.5 hover:shadow-card-hover">
      <div className="flex items-start justify-between">
        <div className="min-w-0">
          <p className="truncate text-xs font-medium uppercase tracking-wide text-slate-500">
            {label}
          </p>
          {loading ? (
            <Skeleton className="mt-2 h-7 w-24" />
          ) : (
            <p className="mt-2 text-2xl font-bold text-slate-800">{value}</p>
          )}
          {hint && <p className="mt-1 text-xs text-slate-400">{hint}</p>}
        </div>
        <div
          className={cn(
            'flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg',
            ACCENT_CLASSES[accent]
          )}
        >
          <Icon className="h-5 w-5" strokeWidth={2} />
        </div>
      </div>
      {trend && trendValue && (
        <div className="mt-3 flex items-center gap-1 text-xs">
          {trend === 'up' ? (
            <ArrowUpRight className="h-3.5 w-3.5 text-emerald-500" />
          ) : trend === 'down' ? (
            <ArrowDownRight className="h-3.5 w-3.5 text-rose-500" />
          ) : null}
          <span
            className={cn(
              'font-medium',
              trend === 'up' && 'text-emerald-600',
              trend === 'down' && 'text-rose-600',
              trend === 'flat' && 'text-slate-500'
            )}
          >
            {trendValue}
          </span>
          <span className="text-slate-400">较上周</span>
        </div>
      )}
    </div>
  );
}
