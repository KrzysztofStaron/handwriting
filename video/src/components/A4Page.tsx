import React from 'react';
import {Handwriting} from './Handwriting';
export const PAGE = {w: 960, h: 1358, x: 60, y: 281}; // portrait A4 (1:1.414) centered in 1080x1920
// A ruled A4 sheet whose lines are written by the model, each line its own pen pass.
export const A4Page: React.FC<{lines: {k: string; start: number}[]; top: number; spacing: number; scale: number; tilt?: number[]}> = ({lines, top, spacing, scale, tilt}) => (
  <g>
    <rect x={PAGE.x + 8} y={PAGE.y + 14} width={PAGE.w} height={PAGE.h} fill="rgba(0,0,0,0.25)" filter="url(#bleed)" />
    <rect x={PAGE.x} y={PAGE.y} width={PAGE.w} height={PAGE.h} fill="#f4f1ea" />
    <rect x={PAGE.x} y={PAGE.y} width={PAGE.w} height={PAGE.h} filter="url(#fibre)" />
    {lines.map((_, i) => <line key={'r' + i} x1={PAGE.x + 40} x2={PAGE.x + PAGE.w - 40} y1={PAGE.y + top + i * spacing + 22} y2={PAGE.y + top + i * spacing + 22} stroke="#9fb3c8" strokeOpacity={0.35} strokeWidth={1.4} />)}
    <line x1={PAGE.x + 96} x2={PAGE.x + 96} y1={PAGE.y} y2={PAGE.y + PAGE.h} stroke="#c98a8a" strokeOpacity={0.35} strokeWidth={1.4} />
    {lines.map((l, i) => (
      <g key={i} transform={tilt ? `rotate(${tilt[i] ?? 0} ${PAGE.x + 110} ${PAGE.y + top + i * spacing})` : undefined}>
        <Handwriting k={l.k} start={l.start} x={PAGE.x + 112} y={PAGE.y + top + i * spacing} scale={scale} align="left" width={2.4} tip={false} />
      </g>
    ))}
  </g>
);
