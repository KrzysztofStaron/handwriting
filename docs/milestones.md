# Pipeline milestones

Each row is the trainer you get by checking out that commit. Later rows replace earlier behavior. `models/best-4.pth` is the frozen anchor for every row below and is not in git. A running process keeps the code it loaded until it is started again.

```bash
git switch --detach 7efa8d1   # the prefix beam with ESS resampling
git switch master             # back to the current pipeline
```

The trainer on `master` is `5cf0d6e`. Commits after that one do not change the update. How to launch it from `best-4` is `docs/from-best-4.md`. What that loop does is `docs/rl.md`. The series under `samples/rl-from-base` is written up in `docs/from-base-report.md`. Those three starts loaded `1d3be2c`, `8e524cc`, and `7efa8d1`.

## Trainer

| Commit | Date | Milestone | What that checkout trains |
|---|---|---|---|
| `27e970f` | 2026-10-19 | Recognizer enters the repo | Visual CER is the reward. Prompts are the whole play. One flat rollout per update. |
| `56283d4` | 2026-10-19 | Lines capped at 48 characters | Same reward. Prompts long enough to squash under TrOCR's square resize are dropped. |
| `8ab1da6` | 2026-10-19 | Paper stop | The sampler stops once the window passes the last character and the pen lifts. Every later row uses this stop. |
| `89dd233` | 2026-10-19 | `uv sync --extra rl` | Same trainer. Pillow, sentencepiece, and transformers are the recognizer extra. |
| `c624618` | 2026-10-19 | Per-draw temperature | One batched sample. Each drawing takes a temperature from a normal around 0.5, kept in 0.1–0.9. Confidence raises the reward when CER is low and lowers it when the read is wrong. |
| `de59401` | 2026-10-19 | First tilt penalty | Straightness is one slope through the ink. A flat line loses nothing. A crooked line loses at most 0.05. |
| `319c595` | 2026-10-19 | Space before punctuation | Plain CER and vCER drop a space that sits immediately before punctuation. A space after punctuation still counts. |
| `4f276d4` | 2026-10-19 | Plain CER is the reward | Default score is `0.8 * (1 - min(CER, 1)) + 0.2 * exact`. `--reward vcer` still loads the visual table. |
| `ae54de7` | 2026-10-19 | Confidence only adds | Confidence is a share of the reward. The sign flip at CER 0.5 is gone. |
| `371bd14` | 2026-10-19 | Prefix beam | Each stage draws the next word onto ink already kept. TrOCR scores that prefix. The gradient is the new strokes. One log line per word. |
| `4d87881` | 2026-10-19 | Novel lines, three words | One draw in four is three random words from the model alphabet. `--novel 0` keeps the play. |
| `552bf6b` | 2026-10-19 | One log line per sentence | Still a prefix beam. Reward, CER, KL, and length on the chart are the mean of every prefix of that line. |
| `79eae9c` | 2026-10-19 | Novel lines, ten short words | A novel prompt is ten words of at most four letters or digits. |
| `79a0a82` | 2026-10-19 | 16 draw, keep 4 | Each stage draws 16 prefixes and continues 4 of them, four ways each. |
| `973aeae` | 2026-10-19 | Dictionary lines | A quarter of draws are lowercase dictionary words with no punctuation, no longer than a play line. Novel stays a quarter. The rest is the play. |
| `931c1e8` | 2026-10-19 | Random beam factor | Each line picks a divisor `f` of the beam. It draws the full width, keeps `width/f` prefixes, and continues each one `f` times. |
| `d3c0506` | 2026-10-19 | vCER is the reward again | The default string score is the visual distance. `--reward cer` is plain CER. |
| `b32d425` | 2026-10-19 | CER and vCER split evenly | The default string score is half plain CER and half vCER. |
| `16045d5` | 2026-10-19 | Four-term reward | The exact-match bonus leaves the default. CER and vCER share the string score, confidence adds, and straightness subtracts. |
| `c8ed085` | 2026-10-19 | Ink-overlap vCER | A different character costs one minus its ink overlap on Bradley Hand, Marker Felt, and Noteworthy. Eight hand digraphs stay. No new two-letter pair is added. Table hash `fc2253e9098154c1`. |
| `983d835` | 2026-10-20 | 60/20/20 | Plain CER is 60% of the score, vCER 20%, confidence 20%. Straightness still subtracts at most 0.05. |
| `4c18be4` | 2026-10-20 | Spacing penalty | Uneven gaps between words subtract at most 0.05. A gap inside a word does not count. |
| `1d3be2c` | 2026-10-20 | End-band tilt | The slope fit is gone. Each end is the band from the 10th to the 90th percentile. The middle is compared with the ends. One letter of shift scores 1. |
| `8e524cc` | 2026-10-20 | Spacing logged only | `--spacing` defaults to 0. The measurement stays on the sample table. |
| `7efa8d1` | 2026-10-20 | ESS prefix beam | Run `20261020-022557` loaded this. Per word, the beam is resampled when the effective sample size falls below half the width. The random divisor is gone. Mix is still 25% novel and 25% words. |
| `c9fd6b7` | 2026-10-20 | Full-line tree | ESS and the prefix beam are gone. Play lines are 16 finished drawings of the full string from a copied pen state, half of them 4×2×2. Words are 4–8 separate drawings. Novel lines draw up to 64 and an exact read is the demonstration. One Adam step covers every chunk. Mix is 5% novel and 10% words. |
| `93f3ea1` | 2026-10-20 | One temperature per branch | A root draws the temperature. Every fork of that root keeps it, and the log-probability uses it. |
| `8cc2539` | 2026-10-20 | Per-character penalty | Pen steps whose window sat on a substituted or deleted character lose 0.05, faded across the two neighbors. The logged line reward is unchanged. |
| `5cf0d6e` | 2026-10-20 | Gentler tilt | Straightness is divided by 3. One letter of tilt scores about 1/3. The score reaches 1 at three letter heights. The 0.05 weight stays. This is the trainer on current `master`. |

## Dashboard, page, and docs

These commits do not change the update. The Trainer column is the pipeline that checkout still runs.

| Commit | Date | Trainer | What changed |
|---|---|---|---|
| `37e60b9` | 2026-10-19 | `c624618` | Charts fit a line to the batch means. Samples walk one update at a time. |
| `5bd96df` | 2026-10-19 | `c624618` | The fitted slope becomes a time-to-perfection estimate. |
| `6fdccae` | 2026-10-19 | `de59401` | That estimate sits in the Reward and CER titles, in hours and minutes. |
| `d964988` | 2026-10-19 | `de59401` | The clock shows seconds. |
| `ec0d9e1` | 2026-10-19 | `319c595` | `/vcer` lists the substitution rules and scores a pair. |
| `00232b8` | 2026-10-19 | `552bf6b` | The sample page shows one prefix length at a time. Useful on the prefix beam. |
| `0be5983` | 2026-10-19 | `79a0a82` | Reward and CER can show novel lines, the play, or both. |
| `6dac966` | 2026-10-19 | `c8ed085` | The vCER page prints the stored ink cost instead of a rounded one. |
| `0f76803` | 2026-10-20 | `983d835` | `run_aliases.json` shows a resumed process on the earlier run id. |
| `819d37c` | 2026-10-20 | `4c18be4` | Sample tables add tilt, vCER, and spacing. |
| `a0ff4c0` | 2026-10-20 | `c9fd6b7` | `docs/rl.md` describes the full-line tree. |
| `04a1312` | 2026-10-20 | `5cf0d6e` | `docs/from-best-4.md`. |

## Restoring one

Check the hash out, then start the trainer the way `docs/from-best-4.md` describes, with a new `--save` and `--sheet-dir`. An old code load will not see files written by a newer trainer as its own series. `models/best-4.pth` stays the KL anchor on every row. Point `--save` at a directory under `samples/`, so the best-checkpoint name does not land on `models/rl-trocr-best.pth`.
