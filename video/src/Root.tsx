import React from 'react';
import {Composition, Sequence, Audio, staticFile, AbsoluteFill} from 'remotion';
import {SCENES} from './scenes/Scenes';
import {timeline} from './lib/strokes';

export const Master: React.FC = () => (
  <AbsoluteFill style={{background: '#000'}}>
    {timeline.shots.map((s) => { const C = SCENES[s.id]; return <Sequence key={s.id} from={s.from} durationInFrames={s.duration} name={s.id}><C /></Sequence>; })}
    <Audio src={staticFile('audio.wav')} />
  </AbsoluteFill>
);
export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="Master" component={Master} durationInFrames={timeline.total} fps={timeline.fps} width={1080} height={1920} />
    {timeline.shots.map((s) => <Composition key={s.id} id={s.id} component={SCENES[s.id]} durationInFrames={s.duration} fps={timeline.fps} width={1080} height={1920} />)}
  </>
);
