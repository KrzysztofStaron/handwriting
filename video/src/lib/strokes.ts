import raw from '../strokes.json';
import tl from '../timeline.json';
export type Pt = [number, number, number]; // x, y (screen-down), pen_up after this point
export const timeline = tl;
export const RATE = tl.rate;
const cache: Record<string, Pt[]> = {};
// API y grows upward -> flip so y grows down on screen. Order is the model's generation order.
export const getPts = (key: string): Pt[] => {
  if (!cache[key]) {
    const p = (raw as any)[key].points as number[][];
    cache[key] = p.map((q) => [q[0], -q[1], q[2]] as Pt);
  }
  return cache[key];
};
export const bbox = (p: Pt[]) => {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of p) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  return {x0, y0, x1, y1, w: x1 - x0, h: y1 - y0};
};
export const pathOf = (p: Pt[], n: number, s: number, ox: number, oy: number) => {
  let d = '';
  let lift = true;
  for (let i = 0; i < Math.min(n, p.length); i++) {
    const X = (ox + p[i][0] * s).toFixed(1), Y = (oy + p[i][1] * s).toFixed(1);
    d += (lift ? 'M' : 'L') + X + ' ' + Y;
    lift = p[i][2] === 1;
  }
  return d;
};
// Resample a pen sequence to N points along its drawing order (for morphs).
export const resample = (p: Pt[], N: number): Pt[] => {
  const cum = [0];
  for (let i = 1; i < p.length; i++) cum.push(cum[i - 1] + Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]));
  const L = cum[cum.length - 1];
  const out: Pt[] = [];
  let j = 0;
  for (let k = 0; k < N; k++) {
    const t = (k / (N - 1)) * L;
    while (j < p.length - 2 && cum[j + 1] < t) j++;
    const a = p[j], b = p[j + 1];
    const f = cum[j + 1] > cum[j] ? (t - cum[j]) / (cum[j + 1] - cum[j]) : 0;
    out.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2]]);
  }
  return out;
};
export const centered = (p: Pt[]): Pt[] => {
  const b = bbox(p);
  const cx = b.x0 + b.w / 2, cy = b.y0 + b.h / 2;
  return p.map(([x, y, u]) => [x - cx, y - cy, u] as Pt);
};
