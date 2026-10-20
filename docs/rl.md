# Recognizer RL

`train_rl.py` is the post-training loop that makes the Graves policy easier for TrOCR to read. One update is one line of text. The policy draws that line, TrOCR scores the finished ink, and a REINFORCE step follows. `models/best-4.pth` is the frozen KL anchor and is never written.

Supervised training (`train.py`) comes first. Human best-of-k (`train_rlhf.py`) is a separate later loop on the same estimator. A process loads this code when it starts. A trainer already running keeps the sampler it started with until it is started again.

The serving API is one pass: the full string goes in, pen steps come out, and the line stops when the attention window passes the last character and the pen lifts. Training uses that same call.

## Pieces

```
text  →  policy (3 LSTMs, mixture of 20 Gaussians, monotonic window)
      →  pen steps (Δx, Δy, pen up)
      →  white line image
      →  TrOCR-Large
      →  reward
      →  advantage on the new pen steps
      →  one Adam step, KL toward models/best-4.pth
```

The policy stays on CPU. TrOCR stays on MPS when Apple Silicon is available, or on CUDA when the policy is on CUDA. Sampling and the log-probability use the same temperature and the same window sharpen (default 2).

The copyable pen state is both LSTM memories, the third layer, the window weights, kappa, and the last point. Earlier ink is only appended.

## Which line, and how many drawings

Each update picks one source. The rest of the probability is the play.

| Source | Default share | Drawings | Shape |
|---|---|---|---|
| Play (`data/romeo_juliet.txt`) | 0.85 | `--beam` 16 | Shared pen state. Half the updates are 4×2×2. The rest are 4×4, 2×8, or 16 separate lines. |
| Words (`/usr/share/dict/words`) | `--words` 0.10 | 4–8 (`--easy-leaves` 0) | Separate full lines. Regression check. |
| Novel (10 random words) | `--novel` 0.05 | up to `--novel-leaves` 64 | Separate full lines, in waves of 16. |

`--text` pins one sentence and skips the mix. `--novel` and `--words` together cannot pass 1. `-k`, `--batch`, `--keep`, and `--factor` are accepted and do not change this.

Play lines are the file, 35–48 characters, about 9 words. Word lines are lowercase dictionary words with no digits or punctuation, packed up to `--max-chars` (48). Novel lines are 10 words of 1–4 letters or digits from the model alphabet.

`--tree 4x2x2` or `--tree 16` pins the play shape. The product has to equal `--beam`. Dictionary and novel lines ignore `--tree`.

## Drawing a play line

Every segment is conditioned on the full string.

1. Start 4 independent lines (or 2, or 16, for the other shapes).
2. After a fixed number of pen steps, copy the pen state and continue the same string. A 4×2×2 line does this twice.
3. The last segment runs until the pen stops, the same stop as deployment.
4. TrOCR scores only the finished lines.

The cut length is about 30 pen steps per character, split evenly across the segments before the last one. A 45-character line with three segments cuts at step 450 and again at step 900. A row that already stopped is not forked. If every row finishes before a fork, the update is a flat group of whatever finished.

## Score

TrOCR reads a 384×384 rendering. The default `--reward both` is

```
0.6 * (1 - min(CER, 1)) + 0.2 * (1 - min(wCER, 1)) + 0.2 * confidence
    - 0.05 * straightness
```

Plain CER is case-sensitive Levenshtein over the normalized reference. vCER is the visual distance in `docs/vcer.md`: a lookalike such as `d`→`ol` costs less than a full edit. Confidence is the geometric mean of the probabilities on the tokens TrOCR chose. It never subtracts. Straightness compares the ink band at each end of the line, and the middle against the ends, so tilt and arch both count. `--straightness` 0.05 is the most a crooked line can lose.

Word spacing is measured and logged. `--spacing` defaults to 0, so it is not in the score.

`--reward cer` and `--reward vcer` keep an exact-match term inside the string score, then the same confidence and straightness. `--reward exp` is `exp(-CER)`. The charts plot plain CER either way.

KL is not inside this number. It is applied in the update below.

## Credit

The value of a pen state is the mean reward of the finished lines under it. A chunk's advantage is that value minus the value of the state it grew from. A root is compared with the mean of every finished line. On a flat group the advantage is the reward minus the group mean. One drawing falls back to an exponential moving average (0.9 / 0.1).

Around a fork, the advantage fades across 8 pen steps. The start of a chunk fades in from the parent advantage. The end of a chunk fades toward the mean of its children's advantages. A flat drawing has no neighbor, so every pen step keeps the same advantage.

Temperature is drawn from a normal centered at 0.5, kept inside 0.1–0.9. The log-probability uses that drawing's temperature. A temperature of 0 is the mode: its reward still enters the baseline, and it stays out of the gradient.

## Update

For one segment of one drawing the loss is

```
-mean_t (A_t - 0.1 * stopgrad(mean log π / π_ref)) * log π_t
```

`A_t` is the advantage at that pen step. The KL term is the mean log-ratio to `models/best-4.pth` over the segment, and it is stopped, so the backward is only the score function. The loss is then multiplied by `temperature / 0.5`, so a draw at 0.5 keeps the reference step size. Each segment level contributes equally. Adam (1e-5) takes one step after every segment has been differentiated, with gradient norm clipped at 1.

An exact novel read (CER 0) is the exception. The search stops, that drawing is trained with advantage 1, and the misses in that wave are left out of the gradient. The KL term still applies to the exact drawing. If the wave of 64 has no exact read, the update is the ordinary group advantage on the continuous score.

The gradient is only on the new pen steps. The first segment is scored from a blank page. A continuation is scored from the saved pen state. The anchor replays the earlier segments under the same full string before it scores the new points.

## What is written

About every `--block-minutes` (10), the loop saves weights, an optimizer sidecar, and 10 drawings of the fixed sentence `the old man`. Latest weights go to `--save` (default `models/rl-trocr.pth`). Numbered checkpoints go to `--sheet-dir` (default `samples/rl/`). `models/rl-trocr-best.pth` updates when the review sentence's mean reward improves. It is that one sentence, not a held-out set.

Every update is appended to `events.jsonl` in the sheet directory. The dashboard is `uv run python rl_dashboard.py` at <http://127.0.0.1:8765/>. Reward and CER can be filtered to Novel, Words, R&J, or All. KL and stroke length stay on every update. Sample rows show the shape that was drawn (`4×2×2`, a flat count, or `sft` when a novel line was exact).

A from-base run keeps its files out of `models/`:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 \
  caffeinate -dimsu uv run train_rl.py --no-open \
  --save samples/rl-from-base/rl-trocr.pth \
  --sheet-dir samples/rl-from-base
```

Do not point `--save` at `models/rl-trocr.pth` for that run. The best-checkpoint name is derived from `--save`, and a save inside `models/` would replace `models/rl-trocr-best.pth`.

## Human loop

`train_rlhf.py` draws `--k` 3 samples of one prompt at one temperature. The chosen sample scores 1 and the others score 0. The same REINFORCE step runs, with no tree and no novel or dictionary mix. Picks are appended to `data/rlhf_prefs.jsonl` before the step. Weights go to `models/rl-human.pth`. Resume order is that file, else `models/rl-trocr.pth`, else the base model.

## Not in this loop

Adaptive extra leaves, a learned value head, a PPO clip, and per-step pruning are not built. There is no prefix curriculum and no shorter copy of a novel line. A novel pattern that the policy never writes exactly gets only the ranking among its misses.
