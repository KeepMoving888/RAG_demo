import { useLayoutEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';

export interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  placement?: 'top' | 'bottom';
}

/**
 * Hover tooltip with a dark slate-900 surface, rounded corners and a 320px
 * max width. Positioned via measurement to stay within the viewport.
 */
export default function Tooltip({
  content,
  children,
  className,
  bodyClassName,
  placement = 'top',
}: TooltipProps) {
  const [open, setOpen] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(null);
  const triggerRef = useRef<HTMLSpanElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (!open || !triggerRef.current || !tipRef.current) return;
    const tr = triggerRef.current.getBoundingClientRect();
    const tip = tipRef.current.getBoundingClientRect();
    const left = Math.max(
      8,
      Math.min(tr.left + tr.width / 2 - tip.width / 2, window.innerWidth - tip.width - 8)
    );
    const top =
      placement === 'top'
        ? Math.max(8, tr.top - tip.height - 8)
        : tr.bottom + 8;
    setCoords({ top, left });
  }, [open, placement]);

  return (
    <span
      ref={triggerRef}
      className={cn('relative inline-flex', className)}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children}
      {open && (
        <div
          ref={tipRef}
          role="tooltip"
          style={coords ? { top: coords.top, left: coords.left } : undefined}
          className={cn(
            'fixed z-50 max-w-[320px] rounded-lg bg-slate-900 px-3 py-2 text-xs leading-relaxed text-slate-100 shadow-xl',
            'animate-fade-in',
            bodyClassName
          )}
        >
          {content}
        </div>
      )}
    </span>
  );
}
