# "extended" launch video (Remotion)

Vertical 1080x1920, 30 fps, ~45 s. All handwriting is sampled from the `extended` model (live API `/generate/stream`) and cached in `src/strokes.json`; the pen is animated in generation order at 10 model steps per frame.

## Render on a Mac

Requires Node.js 18+ (e.g. `brew install node`).

```bash
cd video
npm install
npx remotion render Master out/extended.mp4
```

Output: `video/out/extended.mp4`. Remotion downloads its own headless Chrome on first render.

Preview / edit: `npx remotion studio`. Single shots render as compositions `S1`..`S10`.

## Layout

- `src/Root.tsx`: `Master` timeline plus one composition per shot
- `src/scenes/Scenes.tsx`: shots S1-S10
- `src/components/`: `Handwriting` (stroke animation), `A4Page`, `Stage` (paper, camera, grain)
- `src/strokes.json`: cached model strokes; `src/timeline.json`: shot timings and line start frames (shared with audio)
- `public/wall.png`, `public/audio.wav`: wall photo and generated audio
- `scripts/fetch_strokes.py`: re-fetch strokes from the API (overwrites curated picks)
- `scripts/build_timeline_audio.py`: rebuild `timeline.json` + `audio.wav` (Python 3 + numpy)
