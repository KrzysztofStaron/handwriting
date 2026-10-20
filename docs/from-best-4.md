# Run the recognizer from best-4

This starts a new training series from `models/best-4.pth` with the current pipeline. The weights, the optimizer, and the charts go in their own directory. `models/best-4.pth` is the frozen KL anchor and is never written. How the loop scores and credits a line is in `docs/rl.md`.

An empty save file is what makes the first launch load `best-4`. The same command later resumes the new weights.

## Copy

Copy the repository. These tracked files have to come with it:

- `data/romeo_juliet.txt`
- `vcer/table.json`
- `uv.lock`

`models/` is gitignored. Copy `models/best-4.pth` (about 14 MB) by itself into `models/` on the new machine. Leave the other checkpoints behind. Leave `samples/` behind. An old `rl-trocr.pth` or `events.jsonl` in the new directory would resume that old run.

The word list is not in the repo. The default path is `/usr/share/dict/words`. macOS has it. On Debian or Ubuntu, `sudo apt install wamerican` provides it. Another list is `--dict /path/to/words`, one word per line.

## Install

[uv](https://docs.astral.sh/uv/) installs Python 3.12 for this repo. Run the trainer with `uv run`. The system Python is not the one that loads the tokenizer.

```bash
uv sync --extra rl
```

The first launch downloads `microsoft/trocr-large-handwritten`. That needs a network and a few gigabytes of Hugging Face cache. `HF_HUB_DISABLE_XET=1` keeps that download off the Xet transfer path.

## Start

From the repository root, with `samples/rl-best4/` absent or empty:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 \
  uv run train_rl.py --no-open \
  --save samples/rl-best4/rl-trocr.pth \
  --sheet-dir samples/rl-best4
```

On a Mac, wrap it so the machine stays awake:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 \
  caffeinate -dimsu uv run train_rl.py --no-open \
  --save samples/rl-best4/rl-trocr.pth \
  --sheet-dir samples/rl-best4
```

No other flags. The defaults are the pipeline.

The first log lines should include:

```text
policy: models/best-4.pth   (base model)
KL anchor: models/best-4.pth
play beam 16, tree auto   dictionary 4-8   novel 64
straightness 0.05  spacing 0  char penalty 0.05
```

`(base model)` means it loaded `best-4`. `(resuming)` means `--save` already existed and this is a continuation.

Where the models sit:

| Machine | Policy | TrOCR |
|---|---|---|
| Apple Silicon | CPU | MPS |
| NVIDIA | CUDA | CUDA |
| Anything else | CPU | CPU |

`--device` and `--reward-device` override that. On Apple Silicon the pen loop is faster on CPU. `PYTORCH_ENABLE_MPS_FALLBACK=1` covers the few TrOCR ops that have no MPS kernel.

## What one update does

One update is one line. The policy draws it, TrOCR scores the finished ink, and one Adam step follows (1e-5, gradient clip 1, KL 0.1 toward `best-4`).

| Source | Share | Drawings |
|---|---|---|
| Romeo and Juliet, 35–48 characters | 0.85 | 16 finished lines, one shared pen state. Half the time the shape is 4×2×2. The rest are 4×4, 2×8, or 16 separate lines. |
| Dictionary words, lowercase, no punctuation | 0.10 | 4 to 8 separate lines. |
| Ten random short words | 0.05 | Up to 64 separate lines, in waves of 16. An exact read stops the search and is the demonstration. |

A root draws one temperature from a normal centered at 0.5, kept inside 0.1–0.9. Forks of that root keep it. The log-probability uses that same temperature. Window sharpen is 2.

The logged reward is

```text
0.6 * (1 - min(CER, 1)) + 0.2 * (1 - min(wCER, 1)) + 0.2 * confidence
    - 0.05 * straightness
```

Confidence only adds. Straightness is the end-to-end ink shift divided by three letter heights, so one letter of tilt scores about 1/3 and the score reaches 1 at three letter heights. Spacing is logged and is not in the score.

After that score becomes an advantage, a pen step whose window sat on a character TrOCR substituted or deleted loses 0.05, faded across the two neighboring characters. The chart reward does not include that penalty.

## Files this run writes

All of them are under `samples/rl-best4/`:

| File | What it is |
|---|---|
| `rl-trocr.pth` | Latest weights. The name has to stay `rl-trocr.pth` so the optimizer sidecar is found. |
| `rl-trocr-best.pth` | Best mean reward on the fixed sentence `the old man`. |
| `latest.optim.pth` | Adam state for the latest weights. |
| `ckpt-NNNN.pth` and `.png` | Checkpoint and 10 drawings of `the old man`, about every 10 minutes. |
| `events.jsonl` | One JSON line per update. The charts read this. |
| `run.json` | Best review reward for the series. |

`models/best-4.pth`, `models/rl-trocr.pth`, and `models/rl-trocr-best.pth` are not written.

## Resume

Run the same command. Do not pass `--ckpt`. Do not pass `--ckpt models/best-4.pth` again: that ignores the new weights and starts over, then overwrites `samples/rl-best4/rl-trocr.pth`. A process loads the code when it starts, so a later edit is invisible until the next start.

## Dashboard

In another terminal:

```bash
uv run python rl_dashboard.py --dir samples/rl-best4
```

The page is <http://127.0.0.1:8765/>. It binds to localhost. From another computer:

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<host>
```

Reward and CER can show Novel, Words, R&J, or All. The green line is the mean of that update. The pale band is the lowest and highest drawing. The blue dashed line is a straight fit. The rust dots are the review sentence.
