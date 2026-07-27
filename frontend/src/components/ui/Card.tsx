import { cn } from '@/lib/utils';

interface CardProps {
  title?: string;
  subtitle?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  noPadding?: boolean;
}

/** Surface container with title row, optional actions, and shadow + hover lift. */
export default function Card({
  title,
  subtitle,
  actions,
  children,
  className,
  bodyClassName,
  noPadding,
}: CardProps) {
  return (
    <div
      className={cn(
        'overflow-hidden rounded-xl border border-slate-200 bg-white shadow-card transition-shadow hover:shadow-card-hover',
        className
      )}
    >
      {(title || actions) && (
        <div className="flex items-center justify-between border-b border-slate-100 px-5 py-3.5">
          <div className="min-w-0">
            {title && (
              <h3 className="truncate text-sm font-semibold text-slate-800">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="mt-0.5 truncate text-xs text-slate-500">{subtitle}</p>
            )}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className={cn(!noPadding && 'p-5', bodyClassName)}>{children}</div>
    </div>
  );
}
