import { cn } from '@/lib/utils';

export type BadgeVariant = 'success' | 'warning' | 'danger' | 'info' | 'neutral';

export interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  className?: string;
  dot?: boolean;
}

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  success: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  warning: 'bg-amber-50 text-amber-700 border-amber-200',
  danger: 'bg-rose-50 text-rose-700 border-rose-200',
  info: 'bg-primary-50 text-primary-700 border-primary-200',
  neutral: 'bg-slate-100 text-slate-600 border-slate-200',
};

const DOT_CLASSES: Record<BadgeVariant, string> = {
  success: 'bg-emerald-500',
  warning: 'bg-amber-500',
  danger: 'bg-rose-500',
  info: 'bg-primary-500',
  neutral: 'bg-slate-400',
};

/** Compact status pill with optional status dot. */
export default function Badge({
  variant = 'neutral',
  children,
  className,
  dot,
}: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium',
        VARIANT_CLASSES[variant],
        className
      )}
    >
      {dot && <span className={cn('h-1.5 w-1.5 rounded-full', DOT_CLASSES[variant])} />}
      {children}
    </span>
  );
}
