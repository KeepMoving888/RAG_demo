import { useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { GraphData, GraphNode, EntityType } from '@/api/types';
import { cn } from '@/lib/utils';

export interface ForceGraphProps {
  data: GraphData;
  onNodeClick?: (node: GraphNode) => void;
  height?: number;
  className?: string;
}

const TYPE_COLORS: Record<EntityType, string> = {
  Product: '#4f46e5',
  Department: '#10b981',
  Person: '#f59e0b',
  Policy: '#f43f5e',
  Supplier: '#06b6d4',
  Certification: '#8b5cf6',
  Patent: '#0ea5e9',
};

const TYPE_LABEL: Record<EntityType, string> = {
  Product: '产品',
  Department: '部门',
  Person: '人员',
  Policy: '制度',
  Supplier: '供应商',
  Certification: '认证',
  Patent: '专利',
};

// Neo4j-style: small uniform nodes, differentiated by subtle size hints.
// Hub nodes (Product/Department) are slightly larger but still compact.
const TYPE_RADIUS: Record<EntityType, number> = {
  Product: 7,
  Department: 6.5,
  Person: 5,
  Policy: 5.5,
  Supplier: 5,
  Certification: 6,
  Patent: 5,
};

interface FGNode {
  id: string;
  label: string;
  type: EntityType;
  properties: Record<string, string | number>;
  source_chunks: number;
  color?: string;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number;
  fy?: number;
}

interface FGLink {
  source: string | FGNode;
  target: string | FGNode;
  type?: string;
  weight?: number;
}

/**
 * Neo4j-inspired force-directed knowledge graph.
 *
 * Design principles (referencing Neo4j Bloom / neovis.js aesthetics):
 * 1. **Small, uniform nodes** — circles are compact (5-7px), not oversized.
 *    Node importance is conveyed by subtle size differences, not dramatic scaling.
 * 2. **Labels beside nodes** — text floats to the right of the circle, not inside
 *    or below, matching Neo4j's default rendering where labels don't overlap circles.
 * 3. **Clean relationship lines** — thin, curved-free straight lines with subtle
 *    directional arrows. Line weight is uniform (1-1.5px), not thick.
 * 4. **Generous spacing** — strong repulsion force + collision detection ensures
 *    nodes never overlap, even with 30+ entities.
 * 5. **Minimal color palette** — each entity type gets one distinct color,
 *    used consistently for node fill, label, and legend.
 */
export default function ForceGraph({
  data,
  onNodeClick,
  height = 560,
  className,
}: ForceGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const [width, setWidth] = useState(800);
  const [hoverNode, setHoverNode] = useState<FGNode | null>(null);
  const [mouse, setMouse] = useState({ x: 0, y: 0 });

  const graphData = useMemo(
    () => ({
      nodes: data.nodes.map((n) => ({
        ...n,
        color: TYPE_COLORS[n.type],
      })),
      links: data.links.map((l) => ({ ...l })),
    }),
    [data]
  );

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setWidth(w);
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Neo4j-style layout: strong repulsion + collision + moderate link distance
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;

    // Strong charge: pushes all nodes apart aggressively (Neo4j default ~-300 to -600)
    fg.d3Force('charge').strength(-450);

    // Link distance: fixed moderate length for clean visual rhythm
    fg.d3Force('link').distance((l: FGLink) => {
      const w = (l.weight as number) ?? 1;
      return Math.max(80, 120 - w * 6);
    });
    fg.d3Force('link').strength(0.3);

    // Centering: gentle pull toward center
    fg.d3Force('center').strength(0.05);

    // Add collision detection to prevent overlap (critical for small nodes)
    fg.d3Force('collide', (fg as any).d3Force('collide') || undefined);
    // Use d3-force collide with radius based on node size + label width
    const collide = fg.d3Force('collide');
    if (collide) {
      collide.radius((n: FGNode) => (TYPE_RADIUS[n.type] ?? 5) + 28);
    }

    fg.d3ReheatSimulation();
  }, [graphData]);

  const handleZoomFit = () => {
    fgRef.current?.zoomToFit(400, 50);
  };

  return (
    <div
      ref={containerRef}
      className={cn('relative overflow-hidden rounded-xl border border-slate-200 bg-white', className)}
      style={{ height }}
      onMouseMove={(e) => setMouse({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY })}
    >
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        width={width}
        height={height}
        nodeRelSize={1}
        backgroundColor="#ffffff"
        // Neo4j-style: thin, subtle relationship lines
        linkColor={() => '#cbd5e1'}
        linkWidth={1}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        linkDirectionalArrowColor="#94a3b8"
        cooldownTicks={300}
        enableNodeDrag
        nodeCanvasObject={(node, ctx, globalScale) => {
          const n = node as FGNode;
          const label = n.label;
          const baseR = TYPE_RADIUS[n.type] ?? 5;
          const color = n.color ?? '#6366f1';
          const isHub = n.type === 'Product' || n.type === 'Department';
          const isHovered = hoverNode?.id === n.id;

          // Neo4j-style: small filled circle with thin white border
          const r = isHovered ? baseR + 2 : baseR;

          // Subtle outer ring for hub nodes (very faint)
          if (isHub) {
            ctx.beginPath();
            ctx.arc(n.x ?? 0, n.y ?? 0, r + 3, 0, 2 * Math.PI);
            ctx.fillStyle = color + '20';
            ctx.fill();
          }

          // Node circle — solid fill, Neo4j style
          ctx.beginPath();
          ctx.arc(n.x ?? 0, n.y ?? 0, r, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
          // Thin white border for separation from links
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = '#ffffff';
          ctx.stroke();

          // Label to the RIGHT of node (Neo4j default layout)
          const fontSize = Math.max(9, Math.min(12, 11 / globalScale));
          ctx.font = `${isHub ? '600' : '400'} ${fontSize}px Inter, system-ui, sans-serif`;
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';

          const labelX = (n.x ?? 0) + r + 4;
          const labelY = n.y ?? 0;

          // Subtle text shadow for readability on white bg
          ctx.fillStyle = 'rgba(255,255,255,0.85)';
          ctx.fillRect(
            labelX - 1,
            labelY - fontSize / 2 - 1,
            ctx.measureText(label).width + 2,
            fontSize + 2
          );

          ctx.fillStyle = isHub ? '#1e293b' : '#475569';
          ctx.fillText(label, labelX, labelY);
        }}
        onNodeHover={(node) => {
          setHoverNode((node as FGNode) ?? null);
          document.body.style.cursor = node ? 'pointer' : 'default';
        }}
        onNodeClick={(node) => {
          if (onNodeClick) onNodeClick(node as unknown as GraphNode);
        }}
        onBackgroundClick={handleZoomFit}
      />

      {/* Hover tooltip — Neo4j-style compact info card */}
      {hoverNode && (
        <div
          className="pointer-events-none absolute z-10 max-w-[260px] rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg"
          style={{
            left: Math.min(mouse.x + 14, width - 240),
            top: Math.min(mouse.y + 14, height - 140),
          }}
        >
          <div className="flex items-center gap-2 border-b border-slate-100 pb-1.5">
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ background: TYPE_COLORS[hoverNode.type] }}
            />
            <span className="font-semibold text-slate-800">{hoverNode.label}</span>
            <span className="ml-auto rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-500">
              {TYPE_LABEL[hoverNode.type]}
            </span>
          </div>
          {Object.keys(hoverNode.properties).length > 0 && (
            <div className="mt-1.5 space-y-0.5">
              {Object.entries(hoverNode.properties).slice(0, 4).map(([k, v]) => (
                <div key={k} className="flex justify-between gap-3">
                  <span className="text-slate-400">{k}</span>
                  <span className="font-medium text-slate-600">{String(v)}</span>
                </div>
              ))}
            </div>
          )}
          <p className="mt-1.5 border-t border-slate-100 pt-1 text-[10px] text-slate-400">
            来源 chunk：{hoverNode.source_chunks}
          </p>
        </div>
      )}

      {/* Legend — Neo4j-style compact bottom-left */}
      <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap gap-x-3 gap-y-1 rounded-lg border border-slate-200 bg-white/95 px-2.5 py-1.5 shadow-sm backdrop-blur">
        {(Object.keys(TYPE_LABEL) as EntityType[]).map((t) => (
          <div key={t} className="flex items-center gap-1.5">
            <span
              className="h-2 w-2 rounded-full"
              style={{ background: TYPE_COLORS[t] }}
            />
            <span className="text-[11px] text-slate-600">{TYPE_LABEL[t]}</span>
          </div>
        ))}
      </div>

      {/* Zoom controls — Neo4j-style top-right */}
      <div className="absolute right-3 top-3 flex gap-1">
        <button
          onClick={() => fgRef.current?.zoom(1.3)}
          className="flex h-7 w-7 items-center justify-center rounded-md border border-slate-200 bg-white/95 text-slate-500 shadow-sm backdrop-blur hover:bg-slate-50 hover:text-slate-700"
          title="放大"
        >
          <span className="text-sm font-medium">+</span>
        </button>
        <button
          onClick={() => fgRef.current?.zoom(0.7)}
          className="flex h-7 w-7 items-center justify-center rounded-md border border-slate-200 bg-white/95 text-slate-500 shadow-sm backdrop-blur hover:bg-slate-50 hover:text-slate-700"
          title="缩小"
        >
          <span className="text-sm font-medium">−</span>
        </button>
        <button
          onClick={handleZoomFit}
          className="rounded-md border border-slate-200 bg-white/95 px-2 py-1 text-[11px] text-slate-500 shadow-sm backdrop-blur hover:bg-slate-50 hover:text-slate-700"
          title="适应画布"
        >
          适应
        </button>
      </div>
    </div>
  );
}
