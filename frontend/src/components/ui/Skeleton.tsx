import { cn } from '@/lib/utils';

export interface SkeletonProps {
  className?: string;
}

/** Shimmer placeholder block used during data loading. */
export default function Skeleton({ className }: SkeletonProps) {
  return <div className={cn('skeleton-shimmer rounded-md', className)} />;
}

/** Text-shaped skeleton with adjustable width. */
export function SkeletonText({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={cn('space-y-2', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className={cn('h-3', i === lines - 1 ? 'w-2/3' : 'w-full')}
        />
      ))}
    </div>
  );
}
