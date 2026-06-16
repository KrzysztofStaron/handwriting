# Conversation summary — collector review & dataset QA

## Goal

Build a dataset quality-assurance workflow in the SvelteKit **collector** app: review training samples (label + handwriting on canvas), reject bad ones without deleting files, support the full dataset (IAM + collected), work over the local network, and feed rejected prompts back into the collector for re-recording.

---

## 1. Review subpage (`/review`)

Created a Tinder-style QA page at `/review`:

- Shows **label** prominently and **handwriting** rendered on a scaled canvas
- **Reject** (red, ← or D) and **Keep** (green, → or K) buttons
- Prev/Next navigation, progress counter (e.g. `159 / 6275`)
- Source badge: **IAM**, **Collected**, **Collected 2**
- Flash feedback on card border (green keep / red reject)
- Link back to main collector via `← Collector`
- Optional `?index=N` query param to jump to a specific sample
- Lazy stroke loading with prefetch of next sample

### Files added/updated

| Path | Role |
|------|------|
| `collector/src/routes/review/+page.svelte` | Review UI |
| `collector/src/routes/api/review/+server.ts` | List samples, reject endpoint |
| `collector/src/routes/api/review/strokes/+server.ts` | Fetch strokes on demand |
| `collector/src/lib/review.ts` | Read/write `data/rejected_review.json` |
| `iam_helper.py` | Python helper for IAM dataset access |

---

## 2. Full dataset coverage (not just collected)

The review API serves **three sources**:

1. `collected/*.json` — hand-drawn canvas samples
2. `collected2/*.json` — second batch of hand-drawn samples
3. **IAM** — ~6117 lines from `data/dataset.npz` via `iam_helper.py`

Total sample count at peak: **~6275–6280** (IAM + collected + collected2, minus rejected).

IAM strokes are loaded lazily (list metadata upfront, fetch strokes per sample) because the npz is ~45MB.

### `iam_helper.py` commands

```
python3 iam_helper.py list              → JSON [{index, text}, ...] (excludes rejected)
python3 iam_helper.py stroke <index>    → JSON {strokes: [[[x,y],...], ...]}
python3 iam_helper.py text <index>      → JSON {text: "..."}
python3 iam_helper.py reject <index>    → flags in data/rejected_review.json
python3 iam_helper.py unreject <index>  → removes flag
```

---

## 3. IAM stroke rendering bug (fixed)

IAM samples initially rendered **stacked vertically** — wrong column semantics.

- `data/dataset.npz` stores strokes as **`(dx, dy, pen_up)`** (already std-normalised, reordered from raw IAM)
- `iam_helper.py` incorrectly assumed **`(pen_up, dx, dy)`** (raw `strokes.npy` format)

**Fix:** parse `dx, dy, pen_up = row[0], row[1], row[2]`, integrate offsets with `std`, flip y for canvas coordinates.

After fix, IAM lines render as horizontal text (e.g. “the Congo-Katanga dispute”).

---

## 4. Reject instead of delete

User asked to **flag** bad samples, not delete files.

- Unified rejection file: **`data/rejected_review.json`**
- Format: array of string IDs, e.g.:
  - `"iam:158"`
  - `"collected2:1781289289849-hello_world.json"`
- Collected files stay on disk; rejected IDs are filtered from the review queue
- IAM rejections use the same file (via `iam_helper.py reject`)

### Empty JSON crash (fixed)

An empty `rejected_review.json` caused `JSON.parse` to throw → **500 on GET /api/review**.

**Fix:** treat missing/empty/invalid file as no rejections (in both `review.ts` and `iam_helper.py`). Reset file to `[]`.

---

## 5. Network access (`--host`)

User wanted other devices on the LAN to access `/review` and the collector.

- `package.json`: `"dev": "vite dev --host"`
- `vite.config.ts`: `server.host: true`, `preview.host: true`
- Dev server prints **Network URL** (e.g. `http://192.168.88.243:5174/`)

### Svelte warnings fixed on review page

- `canvas` changed to `$state<HTMLCanvasElement | undefined>()` for correct reactivity with `bind:this`

---

## 6. Dev server config crash (fixed)

After adding `bodySizeLimit` and `csrf` inside `vite.config.ts`:

```
Error loading SvelteKit options from Vite config: Unexpected option config.kit.kit
```

**Cause:** nested `kit: { ... }` inside `sveltekit()` plugin (double `kit.kit`).

**Fix:** split config:

- **`svelte.config.js`** — adapter, `bodySizeLimit: 50MB`, `csrf.checkOrigin` (off in dev, on in production)
- **`vite.config.ts`** — Vite server settings + Svelte compiler options only

Note: terminal may warn `svelte.config.js is ignored when options are passed via your Vite config` if both overlap; server still starts.

---

## 7. Submit button: `TypeError: Load failed` (collector root)

User hit **Load failed** when pressing Submit on the main collector page (especially from another device over the network).

### Fixes applied

| Change | Why |
|--------|-----|
| `canvas` / `wrap` → `$state` + `$effect` for ctx init | `bind:this` can run after `onMount`; submit used `canvas.width` when undefined |
| Stroke **downsampling** (~1.5px min distance) | Smaller payloads for slow Wi‑Fi / mobile |
| `$lib/api.ts` — `apiFetch()` using `window.location.origin` | Ensures API hits same host:port as the page |
| Better error message via `formatFetchError()` | “Load failed” → network hint with origin |
| `type="button"` on all buttons | Avoid accidental form behaviour |
| Footer shows **connected origin** | Debug wrong URL (e.g. `localhost` on remote device) |
| Review page also uses `apiFetch` | Consistent network behaviour |

### Important for remote collectors

On another device, use the **Network URL** from the terminal (not `localhost`). Footer should show e.g. `http://192.168.88.243:5174`.

Submit API writes to **`collected/`** (not `collected2/`).

---

## 8. Rejected prompts → collector prompts

User asked to add rejected prompts to the collector rotation so bad samples can be re-recorded.

### Implementation

- **`collector/src/lib/rejected-prompts.ts`** — resolves rejected IDs to text (collected JSON files + IAM via `iam_helper.py text`)
- **`collector/src/routes/+page.server.ts`** — merges `PROMPTS` + rejected-only texts (deduped)
- **`+page.svelte`** — uses `data.prompts` from server load instead of static import only

At time of implementation: **52 base prompts + 113 rejected-only = 165 total**. Prompts already in the base list are not duplicated. New rejections in `/review` appear after page refresh.

---

## Key paths

```
handwriting/
├── collector/                    # SvelteKit app
│   ├── src/routes/
│   │   ├── +page.svelte          # Handwriting collector
│   │   ├── +page.server.ts       # Merged prompts load
│   │   └── review/+page.svelte   # Dataset QA
│   ├── src/lib/
│   │   ├── prompts.ts            # Base prompt list
│   │   ├── rejected-prompts.ts   # Rejected → text
│   │   ├── review.ts             # rejected_review.json I/O
│   │   └── api.ts                # apiFetch helper
│   ├── svelte.config.js          # Kit config (adapter, csrf, body limit)
│   └── vite.config.ts            # Vite dev server (--host)
├── collected/                    # Submit target + training data
├── collected2/                   # Second batch of hand-drawn samples
├── data/
│   ├── dataset.npz               # IAM + collected merged for training
│   └── rejected_review.json      # Flagged bad samples (IDs)
└── iam_helper.py                 # IAM read/reject helper for Node API
```

---

## How to run

```bash
cd collector
pnpm dev
```

- **Local:** `http://localhost:5173/` (or next free port, e.g. 5174)
- **Network:** use the Network URL printed in the terminal
- **Review:** `/review`
- **Collector:** `/`

---

## Workflow

1. **Collect** handwriting on `/` (prompts include base list + rejected-only texts)
2. **Review** all sources on `/review` — keep or reject
3. Rejected IDs go to `data/rejected_review.json` (files not deleted)
4. Rejected prompts cycle back into the collector for fresh recordings
5. Training can skip rejected IDs (IAM already filtered in `iam_helper.py list`; collected filtering in review API GET)

---

## Test artifacts created during debugging

Some junk files landed in `collected/` from API tests (`test.json`, `net_test.json`, `browser_test.json`, large `1-test.json` from payload size tests). Safe to delete if not needed.
