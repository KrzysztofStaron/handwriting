import React from 'react';
import {AbsoluteFill, Img, interpolate, Easing, staticFile, useCurrentFrame, random} from 'remotion';
import {Paper, Camera, Ink, Finish, W, H, Defs} from '../components/Stage';
import {Handwriting} from '../components/Handwriting';
import {A4Page, PAGE} from '../components/A4Page';
import {timeline, getPts, bbox, centered, resample, Pt} from '../lib/strokes';

const shot = (id: string) => timeline.shots.find((s) => s.id === id)!;
const ev = (id: string, k: string) => shot(id).events.find((e) => e.key === k)!.start;
const fit = (k: string, maxW: number, maxH = 400) => { const b = bbox(getPts(k)); return Math.min(maxW / b.w, maxH / b.h); };
const ease = Easing.bezier(0.45, 0, 0.2, 1);
const fadeIn = (f: number, n = 8) => interpolate(f, [0, n], [1, 0], {extrapolateRight: 'clamp'});
const Black: React.FC<{o: number}> = ({o}) => <AbsoluteFill style={{background: '#000', opacity: o}} />;

// One centered spoken line (or a stack of lines sharing one scale).
const Lines: React.FC<{id: string; keys: string[]; maxW: number; gap?: number; push?: [number, number]; y?: number}> = ({id, keys, maxW, gap = 150, push = [1, 1.07], y = H * 0.47}) => {
  const f = useCurrentFrame();
  const d = shot(id).duration;
  const s = Math.min(...keys.map((k) => fit(k, maxW)));
  const sc = interpolate(f, [0, d], push, {easing: ease});
  return (
    <Paper>
      <Camera scale={sc}><Ink>{keys.map((k, i) => <Handwriting key={k} k={k} start={ev(id, k)} x={W / 2} y={y + (i - (keys.length - 1) / 2) * gap} scale={s} />)}</Ink></Camera>
      <Finish /><Black o={fadeIn(f)} />
    </Paper>
  );
};

export const S1 = () => { const f = useCurrentFrame(); return <><Lines id="S1" keys={['s1']} maxW={860} push={[1.12, 1.0]} /><Black o={interpolate(f, [0, 20], [1, 0], {extrapolateRight: 'clamp'})} /></>; };
export const S2 = () => <Lines id="S2" keys={['s2']} maxW={760} />;
export const S3 = () => <Lines id="S3" keys={['s3a', 's3b']} maxW={900} gap={160} push={[1.0, 1.1]} />;

export const S4 = () => {
  const f = useCurrentFrame();
  const id = 'S4'; const keys = ['s4a', 's4b'];
  const s = Math.min(...keys.map((k) => fit(k, 900)));
  // "brothers": flickering faint doubles of the line, jittered like rejected samples
  const ghosts = f > 60 ? [0, 1, 2, 3].map((g) => {
    const on = random(`gh${g}-${Math.floor(f / 3)}`) > 0.55;
    if (!on) return null;
    const jx = (random(`jx${g}${Math.floor(f / 3)}`) - 0.5) * 60, jy = (random(`jy${g}${Math.floor(f / 3)}`) - 0.5) * 300;
    return <g key={g} transform={`translate(${jx} ${jy})`} opacity={0.1}>{keys.map((k, i) => <Handwriting key={k} k={k} start={ev(id, k)} x={W / 2} y={H * 0.47 + (i - 0.5) * 160} scale={s} tip={false} />)}</g>;
  }) : null;
  return (
    <Paper>
      <Camera scale={interpolate(f, [0, 150], [1.05, 0.97])}><Ink>{ghosts}{keys.map((k, i) => <Handwriting key={k} k={k} start={ev(id, k)} x={W / 2} y={H * 0.47 + (i - 0.5) * 160} scale={s} />)}</Ink></Camera>
      <Finish dark={0.7} /><Black o={fadeIn(f)} />
    </Paper>
  );
};

export const S5 = () => (
  <Paper tone="#efebe3">
    <Camera shake={0}><Ink><Handwriting k="s5" start={ev('S5', 's5')} x={W / 2} y={H * 0.48} scale={fit('s5', 640, 420)} width={7} /></Ink></Camera>
    <Finish dark={0.85} />
  </Paper>
);

const PageScene: React.FC<{id: string; head: string; body: string; scale: number; top: number; spacing: number; cam: (f: number) => {scale: number; x: number; y: number; rot: number}; tilt?: boolean}> = ({id, head, body, scale, top, spacing, cam, tilt}) => {
  const f = useCurrentFrame();
  const evs = shot(id).events;
  const lines = [{k: head, start: ev(id, head)}, ...evs.filter((e) => e.key.startsWith(body)).map((e) => ({k: e.key, start: e.start}))];
  const c = cam(f);
  const tilts = tilt ? lines.map((_, i) => (random(`t${i}`) - 0.5) * 2.2 * (i / lines.length)) : undefined;
  return (
    <Paper tone="#d9d3c7">
      <Camera scale={c.scale} x={c.x} y={c.y} rot={c.rot}><Ink><A4Page lines={lines} top={top} spacing={spacing} scale={scale} tilt={tilts} /></Ink></Camera>
      <Finish dark={0.7} /><Black o={fadeIn(f, 6)} />
    </Paper>
  );
};
// Heading sits at PAGE.y+top; camera starts tight on it, then pulls back to the whole sheet.
const headFocus = (top: number, z: number) => ({x: -(PAGE.x + 112 + 330 - W / 2) * z, y: -(PAGE.y + top - H / 2) * z});
export const S6 = () => <PageScene id="S6" head="s6" body="cons" scale={3.2} top={80} spacing={52.5} cam={(f) => {
  const t = interpolate(f, [56, 150], [0, 1], {easing: ease, extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const z = 2.3 + (1.06 - 2.3) * t; const h = headFocus(80, z);
  return {scale: z, x: h.x * (1 - t), y: h.y * (1 - t), rot: 0};
}} />;
export const S7 = () => <PageScene id="S7" head="s7" body="scary" scale={3.0} top={70} spacing={49.5} tilt cam={(f) => {
  const t = interpolate(f, [56, 150], [0, 1], {easing: ease, extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const z = 2.2 + (1.18 - 2.2) * t; const h = headFocus(70, z);
  const pan = interpolate(f, [120, 210], [0, -260], {extrapolateLeft: 'clamp'});
  return {scale: z, x: h.x * (1 - t) + 40 * t, y: h.y * (1 - t) + 200 * t + pan, rot: interpolate(f, [0, 210], [-1.2, 2.0])};
}} />;
export const S8 = () => <Lines id="S8" keys={['s8']} maxW={640} push={[1.0, 1.06]} />;

export const S9 = () => {
  const f = useCurrentFrame();
  const m = timeline.morph; const N = 900;
  const A = resample(centered(getPts('s9a')), N), B = resample(centered(getPts('scrib')), N), C = resample(centered(getPts('s9s')), N);
  const sA = fit('s9a', 640), sB = fit('scrib', 980, 900), sC = fit('s9s', 700);
  const cx = W / 2, cy = H * 0.47;
  let content: React.ReactNode;
  if (f < m.start) {
    content = <Handwriting k="s9a" start={ev('S9', 's9a')} x={cx} y={cy} scale={sA} />;
  } else {
    const ph1 = f < m.mid;
    const t = ph1 ? (f - m.start) / (m.mid - m.start) : Math.min(1, (f - m.mid) / (m.end - m.mid));
    const e = ease(t);
    const [P, Q, sp, sq] = ph1 ? [A, B, sA, sB] : [B, C, sB, sC];
    const pts: Pt[] = P.map((p, i) => [p[0] * sp + (Q[i][0] * sq - p[0] * sp) * e, p[1] * sp + (Q[i][1] * sq - p[1] * sp) * e, e < 0.5 ? p[2] : Q[i][2]]);
    let d = ''; let lift = true;
    for (const p of pts) { d += (lift ? 'M' : 'L') + (cx + p[0]).toFixed(1) + ' ' + (cy + p[1]).toFixed(1); lift = p[2] === 1; }
    const frantic = f < m.end + 4;
    const w = frantic ? 3.2 + 2.5 * Math.sin(t * Math.PI) : 3.2;
    content = <g><path d={d} fill="none" stroke="#141210" strokeOpacity={0.2} strokeWidth={w * 2.2} strokeLinecap="round" strokeLinejoin="round" filter="url(#bleed)" /><path d={d} fill="none" stroke="#141210" strokeWidth={w} strokeLinecap="round" strokeLinejoin="round" /></g>;
  }
  const shaking = f >= m.start && f < m.end + 6;
  const jx = shaking ? (random(`sx${f}`) - 0.5) * 40 : 0, jy = shaking ? (random(`sy${f}`) - 0.5) * 40 : 0;
  const flash = f >= m.start && f < m.start + 3 ? 0.5 : 0;
  return (
    <Paper>
      <Camera x={jx} y={jy} scale={interpolate(f, [0, m.start, m.end, 120], [1, 1.04, 1.18, 1.24])}><Ink>{content}</Ink></Camera>
      <AbsoluteFill style={{background: '#fff', opacity: flash}} />
      <Finish dark={shaking ? 0.9 : 0.6} /><Black o={fadeIn(f, 5)} />
    </Paper>
  );
};

// Wall photo (1536x2048) cover-fit to 1080x1920 -> 1440x1920, offset -180px. Text sits on the blank wall.
export const S10 = () => {
  const f = useCurrentFrame();
  const d = shot('S10').duration;
  const z = interpolate(f, [0, d], [1.0, 1.16], {easing: Easing.out(Easing.quad)});
  const yT = 520;
  return (
    <AbsoluteFill style={{background: '#000'}}>
      <AbsoluteFill style={{transform: `scale(${z})`, transformOrigin: `50% ${(yT / H) * 100}%`}}>
        <Img src={staticFile('wall.png')} style={{position: 'absolute', left: -180, top: 0, width: 1440, height: 1920, filter: 'saturate(0.75) contrast(1.05) brightness(0.92)'}} />
        <svg width={W} height={H} style={{position: 'absolute', mixBlendMode: 'multiply', opacity: 0.86, filter: 'blur(0.7px)'}}><Defs />
          <Handwriting k="s10" start={ev('S10', 's10')} x={W / 2} y={yT} scale={fit('s10', 700)} color="#2b2621" width={4.2} />
        </svg>
      </AbsoluteFill>
      <Finish dark={0.8} />
      <Black o={interpolate(f, [0, 10, d - 40, d], [1, 0, 0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})} />
    </AbsoluteFill>
  );
};
export const SCENES: Record<string, React.FC> = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10};
