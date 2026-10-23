import React from 'react';
import {AbsoluteFill, useCurrentFrame, random} from 'remotion';
export const W = 1080, H = 1920;
// Shared SVG defs: ink bleed + paper fibre texture.
export const Defs: React.FC = () => (
  <defs>
    <filter id="bleed" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="1.6" /></filter>
    <filter id="fibre"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="3" seed="3" /><feColorMatrix values="0 0 0 0 0.35  0 0 0 0 0.32  0 0 0 0 0.28  0 0 0 0.09 0" /></filter>
  </defs>
);
export const Paper: React.FC<{tone?: string; children?: React.ReactNode}> = ({tone = '#e9e4da', children}) => (
  <AbsoluteFill style={{background: `radial-gradient(ellipse at 50% 42%, ${tone} 0%, #cfc8bb 70%, #8f887c 125%)`}}>
    <svg width={W} height={H} style={{position: 'absolute'}}><Defs /><rect width={W} height={H} filter="url(#fibre)" /></svg>
    {children}
  </AbsoluteFill>
);
// Handheld-ish camera: slow drift + tiny breathing, plus explicit push/pan.
export const Camera: React.FC<{scale?: number; x?: number; y?: number; rot?: number; shake?: number; children: React.ReactNode}> = ({scale = 1, x = 0, y = 0, rot = 0, shake = 1, children}) => {
  const f = useCurrentFrame();
  const dx = (Math.sin(f / 37) * 4 + Math.sin(f / 13.3) * 1.5) * shake;
  const dy = (Math.cos(f / 41) * 4 + Math.sin(f / 17.1) * 1.2) * shake;
  const dr = Math.sin(f / 53) * 0.25 * shake;
  return <AbsoluteFill style={{transform: `translate(${x + dx}px, ${y + dy}px) scale(${scale}) rotate(${rot + dr}deg)`, transformOrigin: '50% 50%'}}>{children}</AbsoluteFill>;
};
export const Ink: React.FC<{children: React.ReactNode}> = ({children}) => (
  <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{position: 'absolute', overflow: 'visible'}}><Defs />{children}</svg>
);
// Film finish: grain, vignette, flicker, letterbox-free.
export const Finish: React.FC<{dark?: number}> = ({dark = 0.55}) => {
  const f = useCurrentFrame();
  const flick = 1 - random(`fl${f}`) * 0.05;
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <svg width={W} height={H} style={{position: 'absolute', mixBlendMode: 'overlay', opacity: 0.55}}>
        <filter id={`g${f % 6}`}><feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" seed={f % 6} /><feColorMatrix type="saturate" values="0" /></filter>
        <rect width={W} height={H} filter={`url(#g${f % 6})`} />
      </svg>
      <AbsoluteFill style={{background: `radial-gradient(ellipse at 50% 45%, transparent 45%, rgba(0,0,0,${dark}) 120%)`}} />
      <AbsoluteFill style={{background: '#000', opacity: 1 - flick}} />
    </AbsoluteFill>
  );
};
