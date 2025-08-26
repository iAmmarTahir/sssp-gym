import React, { useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MarkerType,
  Position,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import ELK from "elkjs/lib/elk.bundled.js";

type NodeId = string;

type RFNodeData = {
  label: string;
  dist?: number;
  status?: "unseen" | "frontier" | "settled" | "current" | "pivot" | "u-set";
};

type RFNode = {
  id: NodeId;
  type?: string;
  position: { x: number; y: number };
  data: RFNodeData;
  sourcePosition?: Position;
  targetPosition?: Position;
  style?: React.CSSProperties;
};

type RFEdge = {
  id: string;
  source: NodeId;
  target: NodeId;
  label: string;
  data?: { w: number };
  markerEnd?: any;
  style?: React.CSSProperties;
  animated?: boolean;
};

type Graph = { nodes: RFNode[]; edges: RFEdge[] };

interface StepSnapshot {
  step: number;
  description: string;
  current?: NodeId; // extracted/processing vertex u
  settled: Set<NodeId>;
  frontier: Set<NodeId>;
  dist: Record<NodeId, number>;
  relaxing?: { u: NodeId; v: NodeId; w: number; improved: boolean };
  pred: Record<NodeId, NodeId | undefined>; // parent pointers for shortest-path tree
}

interface PaperSnapshot extends StepSnapshot {
  S: Set<NodeId>;
  P?: Set<NodeId>;
  Uchunk?: Set<NodeId>;
  level?: number;
  B?: number;
  Bp?: number;
}

// ---------------- Utilities ----------------
function seededRandom(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function generateGraph(n = 10, density = 0.18, seed = 42): Graph {
  const rand = seededRandom(seed);
  const nodes: RFNode[] = Array.from({ length: n }, (_, i) => ({
    id: String(i),
    position: { x: 0, y: 0 }, // ELK will position
    data: { label: `v${i}`, status: "unseen", dist: i === 0 ? 0 : Infinity },
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
  }));

  const edges: RFEdge[] = [];
  for (let u = 0; u < n; u++) {
    for (let v = 0; v < n; v++) {
      if (u === v) continue;
      if (rand() < density) {
        const w = 1 + Math.floor(rand() * 9);
        edges.push({
          id: `${u}-${v}`,
          source: String(u),
          target: String(v),
          label: String(w),
          data: { w },
          markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
        });
      }
    }
  }

  if (!edges.some((e) => e.source === "0")) {
    if (n > 1)
      edges.push({
        id: `0-1`,
        source: "0",
        target: "1",
        label: "1",
        data: { w: 1 },
        markerEnd: { type: MarkerType.ArrowClosed },
      });
    if (n > 2)
      edges.push({
        id: `0-2`,
        source: "0",
        target: "2",
        label: "4",
        data: { w: 4 },
        markerEnd: { type: MarkerType.ArrowClosed },
      });
  }

  return { nodes, edges };
}

function cloneSet<T>(s: Set<T>): Set<T> {
  return new Set(Array.from(s));
}
function clonePred(
  pred: Record<NodeId, NodeId | undefined>
): Record<NodeId, NodeId | undefined> {
  const out: Record<NodeId, NodeId | undefined> = {};
  for (const k in pred) out[k] = pred[k];
  return out;
}

// Build path from pred to end
function buildPathPairs(
  pred: Record<NodeId, NodeId | undefined>,
  endId: NodeId,
  startId: NodeId
): { pairs: Set<string>; nodes: Set<NodeId> } {
  const pairs = new Set<string>();
  const nodes = new Set<NodeId>();
  let v: NodeId | undefined = endId;
  const seen = new Set<NodeId>();
  while (v && pred[v] !== undefined) {
    if (seen.has(v)) break;
    seen.add(v);
    const u: any = pred[v]!;
    pairs.add(`${u}|${v}`);
    nodes.add(u);
    nodes.add(v);
    if (u === startId) break;
    v = u;
  }
  return { pairs, nodes };
}

// ---------------- Dijkstra steps ----------------
function runDijkstraSteps(graph: Graph, src: NodeId): StepSnapshot[] {
  const { nodes, edges } = graph;
  const adj: Record<NodeId, { v: NodeId; w: number }[]> = {};
  nodes.forEach((n) => (adj[n.id] = []));
  edges.forEach((e) =>
    adj[e.source]?.push({ v: e.target, w: e.data?.w ?? Number(e.label) })
  );

  const dist: Record<NodeId, number> = {};
  nodes.forEach((n) => (dist[n.id] = Infinity));
  dist[src] = 0;

  const pred: Record<NodeId, NodeId | undefined> = {};

  const settled = new Set<NodeId>();
  const frontier = new Set<NodeId>([src]);
  const pq: { id: NodeId; d: number }[] = [{ id: src, d: 0 }];

  const steps: StepSnapshot[] = [];
  let step = 0;
  const pushStep = (snap: Partial<StepSnapshot>) => {
    steps.push({
      step: step++,
      description: snap.description ?? "",
      current: snap.current,
      settled: cloneSet(settled),
      frontier: cloneSet(frontier),
      dist: { ...dist },
      relaxing: snap.relaxing,
      pred: clonePred(pred),
    });
  };

  pushStep({
    description: `Init: dist(${src}) = 0; PQ ← {${src}}; frontier ← {${src}}`,
  });

  while (pq.length > 0) {
    let minIdx = 0;
    for (let i = 1; i < pq.length; i++) if (pq[i].d < pq[minIdx].d) minIdx = i;
    const { id: u } = pq.splice(minIdx, 1)[0];
    if (settled.has(u)) continue;

    settled.add(u);
    frontier.delete(u);
    pushStep({ description: `Extract-min: settle ${u}`, current: u });

    for (const { v, w } of adj[u] ?? []) {
      const cand = dist[u] + w;
      const improved = cand < dist[v];
      if (improved) {
        dist[v] = cand;
        pred[v] = u;
        pq.push({ id: v, d: cand });
        frontier.add(v);
      }
      pushStep({
        description: improved
          ? `Relax (${u} → ${v}, w=${w}): dist(${v}) ← ${cand}`
          : `Relax (${u} → ${v}, w=${w}): no improvement`,
        current: u,
        relaxing: { u, v, w, improved },
      });
    }
  }

  return steps;
}

// ---------------- Paper BMSSP-style steps ----------------
function runPaperSteps(graph: Graph, src: NodeId): PaperSnapshot[] {
  const { nodes, edges } = graph;
  const adj: Record<NodeId, { v: NodeId; w: number }[]> = {};
  nodes.forEach((n) => (adj[n.id] = []));
  edges.forEach((e) =>
    adj[e.source]?.push({ v: e.target, w: e.data?.w ?? Number(e.label) })
  );

  const n = nodes.length;
  const k = Math.max(2, Math.floor(Math.pow(Math.log2(Math.max(4, n)), 1 / 3)));
  const t = Math.max(2, Math.floor(Math.pow(Math.log2(Math.max(4, n)), 2 / 3)));
  const L = Math.max(1, Math.ceil(Math.log2(n) / Math.max(1, t)));

  const dist: Record<NodeId, number> = {};
  nodes.forEach((nn) => (dist[nn.id] = Infinity));
  dist[src] = 0;

  const pred: Record<NodeId, NodeId | undefined> = {};

  const settled = new Set<NodeId>();
  const frontier = new Set<NodeId>([src]);

  let B = Number.POSITIVE_INFINITY;
  let level = L;
  let chunkIndex = 0;

  const steps: PaperSnapshot[] = [];
  let step = 0;
  const push = (snap: Partial<PaperSnapshot>) => {
    steps.push({
      step: step++,
      description: snap.description ?? "",
      current: snap.current,
      settled: cloneSet(settled),
      frontier: cloneSet(frontier),
      dist: { ...dist },
      relaxing: snap.relaxing,
      pred: clonePred(pred),
      S: new Set(snap.S ? Array.from(snap.S) : Array.from(frontier)),
      P: snap.P ? new Set(Array.from(snap.P)) : undefined,
      Uchunk: snap.Uchunk ? new Set(Array.from(snap.Uchunk)) : undefined,
      level: snap.level ?? level,
      B: snap.B ?? B,
      Bp: snap.Bp,
    });
  };

  while (true) {
    const remaining = Array.from(nodes.map((n) => n.id)).some(
      (id) => !settled.has(id) && dist[id] < Infinity
    );
    if (!remaining && frontier.size === 0) break;

    const Sset = new Set<NodeId>(frontier);
    const Ssorted = Array.from(Sset).sort((a, b) => dist[a] - dist[b]);
    const P = new Set<NodeId>(
      Ssorted.slice(0, Math.max(1, Math.ceil(Ssorted.length / Math.max(1, k))))
    );

    push({
      description: `Level ${level}: FindPivots on S. Choose pivots P.`,
      S: Sset,
      P,
      level,
      B,
    });

    type PQItem = { id: NodeId; d: number; parent?: NodeId };
    const pq: PQItem[] = [];
    const localExtracted: NodeId[] = [];
    const U = new Set<NodeId>();
    for (const p of P) pq.push({ id: p, d: dist[p] });

    // eslint-disable-next-line no-loop-func
    const relaxEdge = (u: NodeId, v: NodeId, w: number) => {
      const cand = dist[u] + w;
      const improved = cand < dist[v] && cand < B;
      if (improved) {
        dist[v] = cand;
        pred[v] = u;
        pq.push({ id: v, d: cand, parent: u });
        frontier.add(v);
      }
      push({
        description: improved
          ? `BaseCase relax (${u} → ${v}, w=${w}): dist(${v}) ← ${cand}`
          : `BaseCase relax (${u} → ${v}): no improvement (or ≥ B)`,
        relaxing: { u, v, w, improved },
        S: Sset,
        P,
      });
    };

    while (pq.length > 0 && localExtracted.length < k) {
      let minIdx = 0;
      for (let i = 1; i < pq.length; i++)
        if (pq[i].d < pq[minIdx].d) minIdx = i;
      const { id: u } = pq.splice(minIdx, 1)[0];
      if (settled.has(u)) continue;
      localExtracted.push(u);
      settled.add(u);
      frontier.delete(u);
      U.add(u);
      push({
        description: `BaseCase extract ${u} (chunk ${chunkIndex + 1})`,
        current: u,
        S: Sset,
        P,
        Uchunk: U,
      });

      const neigh = adj[u] ?? [];
      for (const { v, w } of neigh) relaxEdge(u, v, w);
    }

    let Bp = B;
    if (pq.length > 0 && localExtracted.length >= k) {
      let minIdx2 = 0;
      for (let i = 1; i < pq.length; i++)
        if (pq[i].d < pq[minIdx2].d) minIdx2 = i;
      Bp = pq[minIdx2].d;
    }

    push({
      description: `End of chunk ${chunkIndex + 1}: U complete under B′`,
      S: Sset,
      P,
      Uchunk: U,
      Bp,
    });

    chunkIndex++;
    if (Bp < B) {
      B = Bp;
    } else {
      B = Number.POSITIVE_INFINITY;
      level = Math.max(0, level - 1);
    }

    const allSettled = nodes.every(
      (n) => settled.has(n.id) || dist[n.id] === Infinity
    );
    if (allSettled) break;
  }

  return steps as PaperSnapshot[];
}

// ---------------- ELK layout ----------------
const elk = new ELK();

async function applyElkLayout(
  graph: Graph,
  direction: "RIGHT" | "DOWN" | "LEFT" | "UP" = "RIGHT"
): Promise<Graph> {
  const elkGraph: any = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": direction,
      "elk.spacing.nodeNode": "60",
      "elk.layered.spacing.nodeNodeBetweenLayers": "80",
      "elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
    },
    children: graph.nodes.map((n) => ({ id: n.id, width: 48, height: 48 })),
    edges: graph.edges.map((e) => ({
      id: e.id,
      sources: [e.source],
      targets: [e.target],
    })),
  };

  const res = await elk.layout(elkGraph);
  const posById: Record<string, { x: number; y: number }> = {};
  (res.children || []).forEach((c: any) => {
    posById[c.id] = { x: c.x ?? 0, y: c.y ?? 0 };
  });

  const nodes = graph.nodes.map((n) => ({
    ...n,
    position: posById[n.id] ?? n.position,
  }));
  return { nodes, edges: graph.edges };
}

// ---------------- Small UI bits (no Tailwind) ----------------
const styles = {
  page: {
    minHeight: "100vh",
    width: "100%",
    background: "#f8fafc",
    color: "#0f172a",
    fontFamily:
      "system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif",
  },
  container: { maxWidth: 1200, margin: "0 auto", padding: 16 },
  h1: { fontSize: 22, fontWeight: 700, margin: 0 },
  p: { marginTop: 6, fontSize: 13, color: "#475569" },
  grid3: {
    display: "grid",
    gridTemplateColumns: "repeat(3, minmax(0,1fr))",
    gap: 16,
  },
  card: {
    border: "1px solid #e2e8f0",
    background: "#fff",
    borderRadius: 12,
    padding: 12,
  },
  labelCol: { display: "flex", flexDirection: "column", gap: 6, fontSize: 13 },
  button: {
    padding: "6px 10px",
    borderRadius: 8,
    border: "1px solid #cbd5e1",
    background: "#e2e8f0",
    cursor: "pointer" as const,
  },
  buttonPrimary: {
    padding: "6px 10px",
    borderRadius: 8,
    border: "1px solid #065f46",
    background: "#059669",
    color: "#fff",
    cursor: "pointer" as const,
  },
  checkboxRow: { display: "flex", alignItems: "center", gap: 8 },
  small: { fontSize: 12, color: "#64748b" },
  flowWrap: { height: 420, width: "100%" },
  twoCols: {
    display: "grid",
    gridTemplateColumns: "repeat(2, minmax(0,1fr))",
    gap: 16,
    marginTop: 16,
  },
  legendRow: {
    display: "grid",
    gridTemplateColumns: "repeat(3, minmax(0,1fr))",
    gap: 8,
    fontSize: 12,
  },
  legendItem: { display: "flex", alignItems: "center", gap: 8 },
  dot: (bg: string) => ({
    width: 12,
    height: 12,
    borderRadius: 4,
    background: bg,
    display: "inline-block",
  }),
  table: { width: "100%", borderCollapse: "collapse" as const, fontSize: 12 },
  th: {
    textAlign: "left" as const,
    background: "#f1f5f9",
    border: "1px solid #e2e8f0",
    padding: "6px 8px",
  },
  td: { border: "1px solid #e2e8f0", padding: "6px 8px" },
};

// ---------------- Main Component ----------------
export default function App() {
  const [n, setN] = useState(10);
  const [density, setDensity] = useState(0.18);
  const [seed, setSeed] = useState(42);
  const [customGraph, setCustomGraph] = useState<Graph | null>(null);

  const baseGraph = useMemo(
    () => customGraph ?? generateGraph(n, density, seed),
    [n, density, seed, customGraph]
  );

  // ELK layout — recompute positions whenever the graph changes
  const [layouted, setLayouted] = useState<Graph>(baseGraph);
  useEffect(() => {
    let alive = true;
    (async () => {
      const g = await applyElkLayout(baseGraph, "RIGHT");
      if (alive) setLayouted(g);
    })();
    return () => {
      alive = false;
    };
  }, [baseGraph]);

  const [rfLeft, setRfLeft] = useState<any>(null);
  const [rfRight, setRfRight] = useState<any>(null);

  useEffect(() => {
    // small timeout lets ReactFlow finish measuring container size
    const id = setTimeout(() => {
      rfLeft?.fitView({
        padding: 0.1,
        includeHiddenNodes: true,
        duration: 300,
      });
      rfRight?.fitView({
        padding: 0.1,
        includeHiddenNodes: true,
        duration: 300,
      });
    }, 0);
    return () => clearTimeout(id);
  }, [layouted, rfLeft, rfRight]);

  // Start / End node selection
  const [startId, setStartId] = useState<NodeId>(
    () => baseGraph.nodes[0]?.id ?? "0"
  );
  const [endId, setEndId] = useState<NodeId>(
    () => baseGraph.nodes[baseGraph.nodes.length - 1]?.id ?? "0"
  );

  useEffect(() => {
    if (!baseGraph.nodes.find((x) => x.id === startId))
      setStartId(baseGraph.nodes[0]?.id ?? "0");
    if (!baseGraph.nodes.find((x) => x.id === endId))
      setEndId(
        baseGraph.nodes[baseGraph.nodes.length - 1]?.id ??
          baseGraph.nodes[0]?.id ??
          "0"
      );
  }, [baseGraph]);

  const dijkSteps = useMemo(
    () => runDijkstraSteps(baseGraph, startId),
    [baseGraph, startId]
  );
  const paperSteps = useMemo(
    () => runPaperSteps(baseGraph, startId),
    [baseGraph, startId]
  );

  const [iA, setIA] = useState(0);
  const [iB, setIB] = useState(0);
  const [locked, setLocked] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(600);

  const snapA = dijkSteps[Math.min(iA, dijkSteps.length - 1)] ?? dijkSteps[0];
  const snapB =
    paperSteps[Math.min(iB, paperSteps.length - 1)] ?? paperSteps[0];

  // ---------- PATH SETS ----------
  const pathA = useMemo(
    () =>
      snapA?.dist?.[endId] !== Infinity
        ? buildPathPairs(snapA.pred, endId, startId)
        : { pairs: new Set<string>(), nodes: new Set<NodeId>() },
    [snapA, startId, endId]
  );
  const pathB = useMemo(
    () =>
      snapB?.dist?.[endId] !== Infinity
        ? buildPathPairs(snapB.pred, endId, startId)
        : { pairs: new Set<string>(), nodes: new Set<NodeId>() },
    [snapB, startId, endId]
  );

  // ---------- unified auto-advance + proper stopping ----------
  const endReachedA = !!snapA?.settled?.has?.(endId);
  const endReachedB = !!snapB?.settled?.has?.(endId);
  const doneA = endReachedA || iA >= dijkSteps.length - 1;
  const doneB = endReachedB || iB >= paperSteps.length - 1;

  useEffect(() => {
    if (!playing) return;

    const id = setInterval(() => {
      if (locked) {
        // advance both together as long as both can still move
        if (!doneA && !doneB) {
          setIA((i) => Math.min(i + 1, dijkSteps.length - 1));
          setIB((i) => Math.min(i + 1, paperSteps.length - 1));
        } else {
          setPlaying(false);
        }
      } else {
        // advance each independently; freeze any finished side
        setIA((i) => (doneA ? i : Math.min(i + 1, dijkSteps.length - 1)));
        setIB((i) => (doneB ? i : Math.min(i + 1, paperSteps.length - 1)));

        // stop when both are done
        if (doneA && doneB) setPlaying(false);
      }
    }, Math.max(120, speedMs));

    return () => clearInterval(id);
  }, [
    playing,
    speedMs,
    locked,
    doneA,
    doneB,
    dijkSteps.length,
    paperSteps.length,
  ]);

  const reset = () => {
    setIA(0);
    setIB(0);
    setPlaying(false);
  };
  const regenerate = () => {
    setSeed((s) => s + 37);
    setPlaying(false);
    setIA(0);
    setIB(0);
    setCustomGraph(null);
  };

  // -------- JSON import/export --------
  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const obj = JSON.parse(String(reader.result));
        if (!obj || !Array.isArray(obj.nodes) || !Array.isArray(obj.edges))
          throw new Error("Invalid format");
        const nodes: RFNode[] = obj.nodes.map((n: any, i: number) => ({
          id: String(n.id ?? i),
          position: { x: 0, y: 0 }, // ELK will reposition
          data: {
            label: String(n.data?.label ?? `v${i}`),
            status: "unseen",
            dist: n.data?.dist ?? (i === 0 ? 0 : Infinity),
          },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
        }));
        const edges: RFEdge[] = obj.edges.map((e: any, j: number) => ({
          id: String(e.id ?? `${e.source}-${e.target}-${j}`),
          source: String(e.source),
          target: String(e.target),
          label: String(e.label ?? e.data?.w ?? 1),
          data: { w: Number(e.data?.w ?? e.label ?? 1) },
          markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
        }));
        setCustomGraph({ nodes, edges });
        reset();
      } catch (err) {
        alert(
          "Failed to parse JSON. Expected { nodes: [], edges: [] } with positions."
        );
      }
    };
    reader.readAsText(file);
  }

  function exportJSON() {
    const data = JSON.stringify(baseGraph, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "graph.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  // ---------- Helpers to render STATE TABLES ----------
  function fmtDist(x: number) {
    return x === Infinity ? "∞" : x;
  }
  function statusFor(
    nid: NodeId,
    snap: StepSnapshot | PaperSnapshot,
    extra?: { startId: NodeId; endId: NodeId; pathNodes?: Set<NodeId> }
  ) {
    const tags: string[] = [];
    if (nid === extra?.startId) tags.push("start");
    if (nid === extra?.endId) tags.push("end");
    if (snap.current === nid) tags.push("current");
    if ((snap as any).S?.has?.(nid)) tags.push("S");
    if ((snap as any).P?.has?.(nid)) tags.push("P");
    if ((snap as any).Uchunk?.has?.(nid)) tags.push("U");
    if (snap.settled.has(nid)) tags.push("settled");
    else if (snap.frontier.has(nid)) tags.push("frontier");
    if (extra?.pathNodes?.has(nid)) tags.push("path");
    return tags.join(" · ");
  }

  function InternalTable({
    title,
    snap,
    isPaper,
    pathNodes,
  }: {
    title: string;
    snap: StepSnapshot | PaperSnapshot;
    isPaper?: boolean;
    pathNodes?: Set<NodeId>;
  }) {
    return (
      <div style={styles.card}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
          }}
        >
          <div style={{ fontWeight: 600, fontSize: 14 }}>{title}</div>
          <div style={styles.small}>
            Step {snap.step}: {snap.description}
          </div>
        </div>
        <div style={{ marginTop: 8, overflowX: "auto" }}>
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Node</th>
                <th style={styles.th}>dist</th>
                <th style={styles.th}>pred</th>
                <th style={styles.th}>status / sets</th>
              </tr>
            </thead>
            <tbody>
              {layouted.nodes.map((n) => (
                <tr key={`${title}-row-${n.id}`}>
                  <td style={styles.td}>
                    <code>{n.data.label}</code>
                  </td>
                  <td style={styles.td}>
                    <code>{fmtDist(snap.dist[n.id])}</code>
                  </td>
                  <td style={styles.td}>
                    <code>{snap.pred[n.id] ?? "—"}</code>
                  </td>
                  <td style={styles.td}>
                    {statusFor(n.id, snap, { startId, endId, pathNodes })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: 8, fontSize: 12, color: "#334155" }}>
          <div>
            <b>Settled:</b> {[...snap.settled].join(", ") || "—"}
          </div>
          <div>
            <b>Frontier:</b> {[...snap.frontier].join(", ") || "—"}
          </div>
          {isPaper && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, minmax(0,1fr))",
                gap: 8,
              }}
            >
              <div>
                <b>S:</b> {[...(snap as PaperSnapshot).S].join(", ") || "—"}
              </div>
              <div>
                <b>P:</b>{" "}
                {[...((snap as PaperSnapshot).P ?? new Set()).values()].join(
                  ", "
                ) || "—"}
              </div>
              <div>
                <b>U:</b>{" "}
                {[
                  ...((snap as PaperSnapshot).Uchunk ?? new Set()).values(),
                ].join(", ") || "—"}
              </div>
              <div>
                <b>Level:</b> {(snap as PaperSnapshot).level}
              </div>
              <div>
                <b>B:</b>{" "}
                {(snap as PaperSnapshot).B === Infinity
                  ? "∞"
                  : (snap as PaperSnapshot).B}
              </div>
              <div>
                <b>B′:</b> {(snap as PaperSnapshot).Bp ?? "—"}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // ---------- node/edge styling ----------
  const circleBase: React.CSSProperties = {
    width: 44,
    height: 44,
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    border: "2px solid #cbd5e1",
    background: "#ffffff",
    boxShadow: "0 1px 0 rgba(0,0,0,0.04)",
  };

  function styleNodesFromSnapshot(
    baseNodes: RFNode[],
    snap: StepSnapshot | PaperSnapshot,
    extra?: {
      pivots?: Set<NodeId>;
      U?: Set<NodeId>;
      startId?: NodeId;
      endId?: NodeId;
      pathNodes?: Set<NodeId>;
    }
  ) {
    const activeU = snap.relaxing?.u;
    const activeV = snap.relaxing?.v;

    return baseNodes.map((n) => {
      const status: RFNodeData["status"] =
        snap.current === n.id
          ? "current"
          : extra?.pivots?.has(n.id)
          ? "pivot"
          : extra?.U?.has(n.id)
          ? "u-set"
          : snap.settled.has(n.id)
          ? "settled"
          : snap.frontier.has(n.id)
          ? "frontier"
          : "unseen";

      let style: React.CSSProperties = { ...circleBase };

      // Active consideration (light blue)
      if (n.id === activeU || n.id === activeV || n.id === snap.current) {
        style = {
          ...style,
          background: "#dbeafe",
          border: "2px solid #60a5fa",
        };
      }

      // Shortest path (green outline)
      if (extra?.pathNodes?.has(n.id)) {
        style = { ...style, border: "2px solid #10b981" };
      }

      // Start / End overrides
      if (n.id === extra?.startId) {
        style = {
          ...style,
          background: "#fee2e2",
          border: "2px solid #ef4444",
        };
      }
      if (n.id === extra?.endId) {
        style = {
          ...style,
          background: "#dcfce7",
          border: "2px solid #10b981",
        };
      }

      return {
        ...n,
        data: { ...n.data, status, dist: snap.dist[n.id] },
        style,
      };
    });
  }

  function styleEdgesFromSnapshot(
    baseEdges: RFEdge[],
    snap: StepSnapshot | PaperSnapshot,
    pathPairs?: Set<string>
  ) {
    const { relaxing } = snap;
    return baseEdges.map((e) => {
      const isActive =
        relaxing && e.source === relaxing.u && e.target === relaxing.v;
      const onPath = pathPairs?.has?.(`${e.source}|${e.target}`);
      let style: React.CSSProperties = { strokeWidth: 1.5, opacity: 0.9 };
      if (onPath) style = { ...style, stroke: "#10b981", strokeWidth: 3 };
      if (isActive) style = { ...style, stroke: "#60a5fa", strokeWidth: 3 };
      return { ...e, animated: !!isActive, style };
    });
  }

  const leftNodes = useMemo(
    () =>
      styleNodesFromSnapshot(layouted.nodes, snapA, {
        startId,
        endId,
        pathNodes: pathA.nodes,
      }),
    [layouted.nodes, snapA, startId, endId, pathA.nodes]
  );
  const leftEdges = useMemo(
    () => styleEdgesFromSnapshot(layouted.edges, snapA, pathA.pairs),
    [layouted.edges, snapA, pathA.pairs]
  );
  const rightNodes = useMemo(
    () =>
      styleNodesFromSnapshot(layouted.nodes, snapB, {
        pivots: (snapB as PaperSnapshot).P,
        U: (snapB as PaperSnapshot).Uchunk,
        startId,
        endId,
        pathNodes: pathB.nodes,
      }),
    [layouted.nodes, snapB, startId, endId, pathB.nodes]
  );
  const rightEdges = useMemo(
    () => styleEdgesFromSnapshot(layouted.edges, snapB, pathB.pairs),
    [layouted.edges, snapB, pathB.pairs]
  );

  // ---------- Unreachable flags ----------
  const unreachableA = useMemo(() => {
    const last = dijkSteps[dijkSteps.length - 1];
    return (last?.dist?.[endId] ?? Infinity) === Infinity;
  }, [dijkSteps, endId]);
  const unreachableB = useMemo(() => {
    const last = paperSteps[paperSteps.length - 1];
    return (last?.dist?.[endId] ?? Infinity) === Infinity;
  }, [paperSteps, endId]);

  // ------------- UI -------------
  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <h1 style={styles.h1}>SSSP Gym — BMSPP vs. Dijkstra </h1>
        <a
          href="https://arxiv.org/pdf/2504.17033"
          target="_blank"
          rel="noreferrer"
        >
          Breaking the Sorting Barrier for Directed Single-Source Shortest Paths
        </a>
        <p style={styles.p}>
          Active relaxations are{" "}
          <span style={{ color: "#2563eb" }}>light blue</span>. Shortest path is{" "}
          <span style={{ color: "#10b981" }}>green</span>. Start is{" "}
          <span style={{ color: "#ef4444" }}>red</span>; End is{" "}
          <span style={{ color: "#10b981" }}>green</span>.
        </p>

        <div style={styles.grid3 as React.CSSProperties}>
          {/* Graph settings */}
          <div style={styles.card}>
            <div style={{ fontWeight: 600, fontSize: 14 }}>Graph Settings</div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, minmax(0,1fr))",
                gap: 12,
                marginTop: 8,
              }}
            >
              <label style={styles.labelCol as React.CSSProperties}>
                <span>Nodes: {n}</span>
                <input
                  type="range"
                  min={5}
                  max={18}
                  value={n}
                  onChange={(e) =>
                    setN(parseInt((e.target as HTMLInputElement).value))
                  }
                />
              </label>
              <label style={styles.labelCol as React.CSSProperties}>
                <span>Density: {density.toFixed(2)}</span>
                <input
                  type="range"
                  min={0.15}
                  max={0.6}
                  step={0.01}
                  value={density}
                  onChange={(e) =>
                    setDensity(parseFloat((e.target as HTMLInputElement).value))
                  }
                />
              </label>

              <label style={styles.labelCol as React.CSSProperties}>
                <span>Start node</span>
                <select
                  value={startId}
                  onChange={(e) => {
                    setStartId(e.target.value);
                    setIA(0);
                    setIB(0);
                  }}
                >
                  {layouted.nodes.map((n) => (
                    <option key={`start-${n.id}`} value={n.id}>
                      {n.data.label}
                    </option>
                  ))}
                </select>
              </label>
              <label style={styles.labelCol as React.CSSProperties}>
                <span>End node</span>
                <select
                  value={endId}
                  onChange={(e) => setEndId(e.target.value)}
                >
                  {layouted.nodes.map((n) => (
                    <option key={`end-${n.id}`} value={n.id}>
                      {n.data.label}
                    </option>
                  ))}
                </select>
              </label>

              <button style={styles.buttonPrimary} onClick={regenerate}>
                Generate New Graph
              </button>
            </div>
          </div>

          {/* Playback */}
          <div style={styles.card}>
            <div style={{ fontWeight: 600, fontSize: 14 }}>Playback</div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, minmax(0,1fr))",
                gap: 8,
                marginTop: 8,
              }}
            >
              <div
                style={{
                  gridColumn: "1 / span 2",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <button style={styles.button} onClick={reset}>
                  Reset
                </button>
                <button
                  style={styles.buttonPrimary}
                  onClick={() => setPlaying((p) => !p)}
                >
                  {playing ? "Pause" : "Play"}
                </button>
                <label style={styles.checkboxRow as React.CSSProperties}>
                  <input
                    type="checkbox"
                    checked={locked}
                    onChange={(e) => setLocked(e.target.checked)}
                  />{" "}
                  Lock steps (sync)
                </label>
              </div>
              <label
                style={{
                  gridColumn: "1 / span 2",
                  display: "flex",
                  flexDirection: "column",
                  gap: 6,
                }}
              >
                <span>Speed: {speedMs} ms/step</span>
                <input
                  type="range"
                  min={120}
                  max={1500}
                  step={20}
                  value={speedMs}
                  onChange={(e) =>
                    setSpeedMs(parseInt((e.target as HTMLInputElement).value))
                  }
                />
              </label>

              <div style={{ display: "flex", gap: 6 }}>
                <button
                  style={styles.button}
                  onClick={() => setIB((i) => Math.max(0, i - 1))}
                >
                  ◀︎ BMSPP
                </button>
                <button
                  style={styles.button}
                  onClick={() =>
                    setIB((i) => Math.min(paperSteps.length - 1, i + 1))
                  }
                >
                  BMSPP ▶︎
                </button>
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                <button
                  style={styles.button}
                  onClick={() => setIA((i) => Math.max(0, i - 1))}
                >
                  ◀︎ Dijkstra
                </button>
                <button
                  style={styles.button}
                  onClick={() =>
                    setIA((i) => Math.min(dijkSteps.length - 1, i + 1))
                  }
                >
                  Dijkstra ▶︎
                </button>
              </div>
            </div>
          </div>

          {/* Import / Export */}
          <div style={styles.card}>
            <div style={{ fontWeight: 600, fontSize: 14 }}>Graph JSON</div>
            <div style={{ marginTop: 8, display: "grid", gap: 8 }}>
              <input
                type="file"
                accept="application/json,.json"
                onChange={onFileChange}
              />
              <div style={styles.small}>
                Expected:{" "}
                {`{ nodes:[{id, position:{x,y}, data:{label}}], edges:[{source,target,label|data:{w}}] }`}
              </div>
              <div
                style={{ display: "flex", gap: 8, flexWrap: "wrap" as const }}
              >
                <button
                  style={styles.button}
                  onClick={() => setCustomGraph(null)}
                >
                  Clear custom graph
                </button>
                <button style={styles.button} onClick={exportJSON}>
                  Export current graph
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Panels */}
        <div style={styles.twoCols as React.CSSProperties}>
          <div style={styles.card}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div style={{ fontWeight: 600, fontSize: 14 }}>BMSPP</div>
              <div style={styles.small}>
                ≈ O(m · log^(2 / 3) · n)
                {unreachableB ? " • end unreachable (∞)" : ""}
              </div>
            </div>
            <div style={{ height: 420, width: "100%" }}>
              <ReactFlow
                nodes={rightNodes as any}
                edges={rightEdges as any}
                fitView
                defaultEdgeOptions={{ type: "straight" }}
                onInit={(inst) => {
                  setRfRight(inst);
                  // optional initial fit for first mount
                  inst.fitView({ padding: 0.1, includeHiddenNodes: true });
                }}
              >
                <Background />
                <Controls />
              </ReactFlow>
            </div>
          </div>
          <div style={styles.card}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div style={{ fontWeight: 600, fontSize: 14 }}>Dijkstra</div>
              <div style={styles.small}>
                O(m + n log n)
                {unreachableA ? " • end unreachable (∞)" : ""}
              </div>
            </div>
            <div style={{ height: 420, width: "100%" }}>
              <ReactFlow
                nodes={leftNodes as any}
                edges={leftEdges as any}
                fitView
                defaultEdgeOptions={{ type: "straight" }}
                onInit={(inst) => {
                  setRfLeft(inst);
                  // optional initial fit for first mount
                  inst.fitView({ padding: 0.1, includeHiddenNodes: true });
                }}
              >
                <Background />
                <Controls />
              </ReactFlow>
            </div>
          </div>
        </div>

        {/* INTERNAL STATE TABLES */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(2, minmax(0,1fr))",
            gap: 16,
            marginTop: 16,
          }}
        >
          <InternalTable
            title="BMSPP — Internal State"
            snap={snapB}
            isPaper
            pathNodes={pathB.nodes}
          />
          <InternalTable
            title="Dijkstra — Internal State"
            snap={snapA}
            pathNodes={pathA.nodes}
          />
        </div>
      </div>
    </div>
  );
}
