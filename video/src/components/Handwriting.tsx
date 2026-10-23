import React from 'react';
import {useCurrentFrame} from 'remotion';
import {getPts, bbox, pathOf, RATE} from '../lib/strokes';

type Props = {k: string; start: number; x: number; y: number; scale: number; align?: 'left' | 'center';
  color?: string; width?: number; tip?: boolean; rate?: number; opacity?: number};
// Draws model strokes in generation order at a constant step rate (RATE model steps per frame).
export const Handwriting: React.FC<Props> = ({k, start, x, y, scale, align = 'center', color = '#141210', width = 3.2, tip = true, rate = RATE, opacity = 1}) => {
  const f = useCurrentFrame();
  const p = getPts(k);
  const b = bbox(p);
  const n = Math.max(0, Math.floor((f - start) * rate));
  if (n <= 0) return null;
  const ox = align === 'center' ? x - (b.x0 + b.w / 2) * scale : x - b.x0 * scale;
  const oy = y - (b.y0 + b.h / 2) * scale;
  const d = pathOf(p, n, scale, ox, oy);
  const last = p[Math.min(n, p.length) - 1];
  const writing = n < p.length;
  return (
    <g opacity={opacity}>
      <path d={d} fill="none" stroke={color} strokeOpacity={0.18} strokeWidth={width * 2.2} strokeLinecap="round" strokeLinejoin="round" filter="url(#bleed)" />
      <path d={d} fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" strokeLinejoin="round" />
      {tip && writing && last[2] === 0 ? <circle cx={ox + last[0] * scale} cy={oy + last[1] * scale} r={width * 1.6} fill={color} opacity={0.85} /> : null}
    </g>
  );
};
