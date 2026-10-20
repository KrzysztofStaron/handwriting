# From-base run

Snapshot of `samples/rl-from-base`, taken 2026-10-20 14:59. The trainer was still appending. The tables cover every update on disk at 14:55, through update 866 of the third start. By 14:59 the log had reached update 871, including a new review at update 868. Those extra updates sit in the same late-hour pattern and do not move the medians below.

The series is three starts of the recognizer loop, all saving to `samples/rl-from-base/rl-trocr.pth`, all anchored to `models/best-4.pth`. The command line of the process that was still running is:

```bash
uv run train_rl.py --no-open \
  --save samples/rl-from-base/rl-trocr.pth \
  --sheet-dir samples/rl-from-base
```

No other flags. Each start picked up the defaults of the commit it loaded. A later commit stays out of a process that is already running.

## The three starts

| | Start A | Start B | Start C, still running |
|---|---|---|---|
| Run id | `20261020-003843` | `20261020-012914` | `20261020-022557` |
| Wall clock | 00:39–01:30 | 01:31–02:25 | 02:27–14:55 in the tables |
| Updates | 1–44 (44) | 45–109 (65) | 106–866 (761) |
| Commit loaded | `1d3be2c` | `8e524cc` | `7efa8d1` |
| How it ended | restarted | interrupted at update 109 | still writing at 14:59 |
| Sampler | divisor prefix beam | divisor prefix beam | ESS prefix beam |
| Spacing in the reward | 0.05 | 0 | 0 |
| Tilt scale | 1 at one letter height | 1 at one letter height | 1 at one letter height |

Start A began at 00:38, after the end-band tilt commit (`1d3be2c`, 00:32) and before spacing left the reward (`8e524cc`, 01:28). Its session record says `spacing: 0.05` and `factors: [1, 2, 4, 8, 16]`. Rebuilding the logged reward with that spacing term matches to a residual of 0.

Start B has no session record of its own. The run id is 01:29, one minute after `8e524cc`. Every update still logs a divisor `factor`, and the reward matches with the spacing weight at 0. It was interrupted at 02:25. The save it left had finished update 105, so updates 106–109 of this start were dropped. `run_aliases.json` maps this run id onto `20261020-003843`, which is why the dashboard shows A and B as one series.

Start C is pid 64911, started 02:25:50, after ESS landed (`7efa8d1`, 02:16) and before the full-line tree (`c9fd6b7`, 03:03). The session says `resuming: true`, `updates: 105`, `spacing: 0`, `select_min: 0.05`, `select_max: 0.4`. The update counter therefore starts again at 106. Updates 106–109 exist on both B and C and are different lines. The dashboard lists them under different run ids, so each chart is consistent inside itself.

Commits after `7efa8d1` are outside this process. That includes the full-line tree (`c9fd6b7`), one temperature per branch (`93f3ea1`), the per-character penalty (`8cc2539`), and the tilt scale that reaches 1 at three letter heights (`5cf0d6e`).

## What every start had in common

- Policy on CPU, TrOCR on MPS. Frozen KL anchor `models/best-4.pth`, coefficient 0.1.
- Reward `both`. vCER table hash `fc2253e9098154c1`.
- Beam 16. Mix 25% novel, 25% dictionary words, 50% play. The ESS start landed at 48% / 24% / 28% over 761 updates.
- Play lines from `data/romeo_juliet.txt` (3033 prompts, at most 48 characters). Dictionary lines are lowercase words with no punctuation, packed to that same cap. A novel line is ten tokens of at most four letters or digits.
- Temperature drawn per drawing from a normal around 0.5, kept inside 0.1–0.9. Window sharpen 2.
- Adam `1e-5`, gradient clipped at norm 1. The logged norm is the value before that clip.
- Review sentence `the old man`, ten drawings, about once per ten-minute block.
- One Adam step per word. The gradient covers the new strokes of that word. The logged update averages every prefix of the line.

The score on a drawing, after the string blend and the confidence mix, is:

```text
0.60 * (1 - min(CER, 1))
+ 0.20 * (1 - min(vCER, 1))
+ 0.20 * confidence
- 0.05 * straightness
```

Confidence is in `[0, 1]`, so that term only adds. Straightness is in `[0, 1]`. On this process a shift of one letter height between the end bands already scores 1, and so does a matching arch in the middle, and the score stops there. A pegged line loses 0.05.

Start A also subtracts `0.05 * spacing`. On its finished play lines that term had median 0.004. CER is the column that compares across the three starts. The raw reward on start A is a few thousandths lower because of spacing.

KL is applied later, inside the update: the advantage loses `0.1 *` the per-step KL against `best-4`. It is absent from the logged reward and from the reward chart.

## Finished line versus the chart

Each update draws the line one word at a time, 16 copies of each prefix. The logged `reward_mean` and `cer_mean` average every prefix. That is the green line on the dashboard. The pale band is the min and max of those prefixes. The rust dots are the review sentence.

The finished line is the longest text in the update. Every update in this log had 16 copies of one finished text. The numbers below are those 16, unless a column says chart.

A play update has a median of 9 prefixes, a dictionary update 5, a novel update 10. Finished play lines in the ESS run have median 46 characters and 1384 pen steps, about 30 steps per character. The chart's length is the mean of the prefixes, median 758 steps, so it reads as a shorter line than the one TrOCR scored at the end.

| | Finished CER, play | Chart CER, play | Finished CER, novel | Chart CER, novel |
|---|---|---|---|---|
| A | 0.132 | 0.213 | 0.594 | 0.748 |
| B | 0.108 | 0.156 | 0.644 | 0.795 |
| C | 0.171 | 0.186 | 0.694 | 0.814 |

On the ESS run the play chart is close to the finished line. On novel lines, and on start A, the chart is worse, because the short prefixes are in the average. The finished-line table is the one to use for a sentence.

## Finished lines

Median of the 16 finished copies, then the rate across those copies. "At cap" means straightness at least 0.99.

| Start | Source | Lines | Reward | CER | Exact | CER ≤ 0.1 | CER ≤ 0.3 | vCER | Conf | Tilt | At cap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | play | 336 | 0.814 | 0.132 | 4.8% | 33.9% | 86.0% | 0.071 | 0.782 | 1.00 | 62% |
| B | play | 480 | 0.864 | 0.108 | 2.9% | 49.6% | 92.9% | 0.056 | 0.828 | 0.56 | 26% |
| C | play | 5856 | 0.783 | 0.171 | 1.5% | 23.3% | 85.7% | 0.093 | 0.747 | 1.00 | 74% |
| A | words | 160 | 0.820 | 0.109 | 5.6% | 41.3% | 96.9% | 0.061 | 0.722 | 0.73 | 38% |
| B | words | 320 | 0.832 | 0.106 | 9.4% | 42.5% | 96.3% | 0.061 | 0.745 | 0.87 | 47% |
| C | words | 2944 | 0.789 | 0.146 | 1.8% | 26.5% | 90.8% | 0.079 | 0.691 | 1.00 | 78% |
| A | novel | 208 | 0.456 | 0.594 | 0 | 0 | 7.7% | 0.300 | 0.554 | 1.00 | 59% |
| B | novel | 240 | 0.418 | 0.644 | 0 | 0 | 0 | 0.331 | 0.518 | 0.75 | 34% |
| C | novel | 3376 | 0.365 | 0.694 | 0 | 0 | 0.1% | 0.349 | 0.484 | 1.00 | 71% |

The 16 finished copies of one play sentence land close together. The within-update standard deviation of their reward has median 0.034 in start B and 0.060 in start C.

Temperatures on finished lines sit on the normal that was asked for. In the ESS run the median is 0.501, the 10th percentile 0.42, the 90th 0.59. None of the 12,176 finished lines in that start drew a temperature under 0.05.

### Play

Start B is the best hour of play in the series, on 30 updates. Median CER 0.108, and half the copies were under 0.10. The whole distribution sits lower than the ESS run:

| | p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|---|
| B, 480 lines | 0.022 | 0.060 | 0.108 | 0.167 | 0.262 |
| C, 5856 lines | 0.063 | 0.106 | 0.171 | 0.255 | 0.333 |

The drop is at the restart. The first two hours of ESS play (55 updates, from 02:27) have median CER 0.200. In the first dozen play lines, some that kept every drawing finished between 0.23 and 0.39, and some finished near 0.10. After that the finished play CER stays near 0.17 for the rest of the day. It does not climb further, and it does not return to 0.11.

| From | Play updates | Play CER | At cap | Play KL | Words CER |
|---|---|---|---|---|---|
| 02:27 | 55 | 0.200 | 71% | 0.34 | 0.208 |
| 04:27 | 67 | 0.167 | 54% | 0.33 | 0.146 |
| 06:27 | 50 | 0.167 | 83% | 0.41 | 0.128 |
| 08:27 | 58 | 0.167 | 84% | 0.45 | 0.149 |
| 10:27 | 68 | 0.152 | 72% | 0.47 | 0.109 |
| 12:27 | 53 | 0.190 | 81% | 0.54 | 0.130 |
| 14:27 (28 min) | 15 | 0.170 | 100% | 0.61 | 0.174 |

The sampler is the change that lines up with the drop. The divisor beam picks a factor `f` in `{1, 2, 4, 8, 16}`, keeps `16/f` prefixes, and continues each one `f` times. In the B hour the narrowed settings scored better, on a small sample: factor 1 (4 updates) had median CER 0.149, factor 8 (6 updates) 0.071, factor 16 (4 updates) 0.087.

ESS draws one selection temperature per line, log-uniform on 0.05–0.40 (median 0.141), and turns the 16 scores into weights with `softmax(score / τ)`. ESS is `1 / sum(w²)`, so 16 means the weights are equal and 1 means one drawing has all the mass. At ESS 8 or above, every drawing continues once. Below 8, systematic resampling copies drawings in proportion to the weights. Median ESS in this run is 12.4 of 16 (10th percentile 8.7, 90th 15.0). At least one word was resampled on 325 of 761 updates (43%). Across word boundaries the rate is 753 of 5,413 (14%). Most words keep all 16 drawings, which is the factor-1 setting, and that was the weaker setting in the hour before.

The reward drop from B to C on play is 0.864 to 0.783. Splitting the median of each term:

| Term | Weight | B | C | Change in the reward |
|---|---|---|---|---|
| `1 - min(CER, 1)` | 0.60 | 0.536 | 0.497 | −0.038 |
| `1 - min(vCER, 1)` | 0.20 | 0.189 | 0.181 | −0.007 |
| confidence | 0.20 | 0.166 | 0.149 | −0.016 |
| straightness | −0.05 | 0.028 | 0.050 | −0.022 |

The unrounded medians sum to −0.084, against a −0.081 drop in the median reward. Adding the rounded column comes to −0.083. The gap is the difference between a median of sums and a sum of medians. Every term got worse. The CER term is about half of the drop. Tilt is about a quarter. Once straightness sits at 1 on most copies, that quarter stops ranking them: the first word is centered on the 16 drawings, and a later word that continues each drawing once is centered on an EMA of the reward. A penalty shared by the whole group is inside the baseline.

A late play line, update 866: `God forgive me! Marry and amen. How sound is she` was read as `God forgive me ! Marry and aimen . How sound is she-`.

### Dictionary words

Same shape as play at the restart, then a partial return. Finished CER is 0.167 in the first third of the ESS run (updates 106–358), 0.146 in the middle (359–612), and 0.128 in the last third (613–866). Exact copies go from 0.7% to 3.3%. The 10:27 hour matches the B hour (CER 0.109) and the hours around it sit a little worse. The late third as a whole (0.128, exact 3.3%, CER ≤ 0.3 on 93.5%) is the fair summary.

These lines are long rare dictionary words, packed to 48 characters. Update 864 asked for `unspectacularly sacculoutricular osteofibroma ox` and was read as `unspectacularly Saccultricular osteofibroma on`.

### Novel lines

No finished novel line was an exact read. That is 208 + 240 + 3,376 lines. Finished CER stays in a band from 0.59 to 0.70, and the ESS run is at the top of it (median 0.694, mean 0.697). The two-hour bins stay between 0.667 and 0.706. CER ≤ 0.3 is 0.1% of the ESS copies. Reward stays near 0.37.

The prompt is ten short random tokens, and TrOCR reads them as words. Update 855 asked for `hf Wo MB4 t JW mr p9Ly Mf HOQl zD` and was read as `hof wo mbunt jw mr pglya Mit Hotel of`. A quarter of the ESS updates are this source, and the KL on them is the highest in the run (chart median 0.583, 0.743 in the last third, max 1.029).

## The review sentence

`the old man` is eleven characters, drawn as ten finished copies, about every ten minutes. It is a different distribution from a 46-character play line. It got better in the first hour and then sat still, while play lines got worse and tilt on the long lines pegged.

| Update | Start | Reward | CER | Exact copies | Tilt | Stored as best |
|---|---|---|---|---|---|---|
| 10 | A | 0.838 | 0.136 | 2/10 | 0.69 | yes |
| 20 | A | 0.911 | 0.055 | 5/10 | 0.59 | yes |
| 30 | A | 0.933 | 0.036 | 6/10 | 0.63 | yes |
| 43 | A | 0.905 | 0.064 | 4/10 | 0.42 | |
| 55 | B | 0.953 | 0.009 | 9/10 | 0.76 | yes |
| 68 | B | 0.923 | 0.045 | 5/10 | 0.63 | |
| 80 | B | 0.918 | 0.055 | 4/10 | 0.60 | |
| 93 | B | 0.892 | 0.073 | 4/10 | 0.87 | |
| 105 | B | 0.942 | 0.027 | 7/10 | 0.58 | |

ESS reviews, mean of each group of ten checkpoints:

| Updates | Reward | CER | Exact copies | Tilt |
|---|---|---|---|---|
| 115–198 | 0.889 | 0.086 | 41% | 0.60 |
| 210–311 | 0.901 | 0.074 | 35% | 0.59 |
| 322–423 | 0.899 | 0.075 | 44% | 0.54 |
| 435–532 | 0.900 | 0.070 | 40% | 0.66 |
| 542–644 | 0.916 | 0.047 | 57% | 0.79 |
| 658–759 | 0.906 | 0.058 | 48% | 0.70 |
| 769–857 | 0.906 | 0.057 | 47% | 0.81 |

The stored best is update 345, reward 0.961, CER 0.009, tilt 0.52, sheet `ckpt-0032.png`. `run.json` still records that reward. Update 588 was a clean read of all ten copies (CER 0, reward 0.947) and stayed under it. The review at update 868, written after the tables above, scored 0.954 with CER 0.009 and did not replace it either. The file `rl-trocr-best.pth` is dated 06:30, the same minute as `ckpt-0032.pth`.

The sheets are already sloped in the good hour. `ckpt-0009.png` (update 105, mean CER 0.03) drops from `the` toward `man`, with wide gaps, and TrOCR still reads it. `ckpt-0032.png` (the stored best, mean CER 0.01) and `ckpt-0079.png` (update 857, mean CER 0.04) do the same. Later review groups are more tilted (0.8 against 0.6) while the CER stays near 0.05. The short sentence remains readable on a slope that the long lines have pegged.

## Tilt

Straightness in this process reaches 1 at a one-letter-height shift of the end bands, or an arch of that size. The weight is 0.05, so the reward loses at most 0.05. The log stores the capped score, so a line that has fallen three letter heights looks the same as a line that has fallen one.

Finished play lines at the cap: 62% in A, 26% in B, 74% in C. The middle third of C is 86%. The last half hour in the table, 15 play updates, is 100%. The same rise is on the dictionary and novel lines (about 70–78% at the cap in C).

Start A was already often pegged and still had play CER 0.132. Start B had the low CER and the only stretch where most play lines sat below the cap (median straightness 0.56). The ESS run pegged again, and the CER did not come back. The visible slope on the review sheets is the same penalty with the score stuck at the top of its range. Commit `5cf0d6e` divides the score by 3 before the cap. This process started before that commit.

## KL and the step size

Logged KL is the mean, over the strokes that entered the gradient, of `(log p − log p_anchor) / steps`. The advantage then loses 0.1 times that number. At the ESS median of 0.45, a typical step loses 0.045 of advantage. The spread of finished-line rewards inside one update is about 0.06, so the anchor penalty is the same size as the gap between drawings.

| Start | Median KL | 90th | Max |
|---|---|---|---|
| A | 0.175 | 0.405 | 0.509 |
| B | 0.234 | 0.501 | 0.829 |
| C | 0.450 | 0.662 | 1.029 |

Inside the ESS run the play KL climbs through the day: 0.33 in the first hours, 0.47 by late morning, 0.61 after 14:27. Play CER stays near 0.17 while that distance from `best-4` grows. Novel KL is higher still.

The pre-clip gradient norm has median 0.56 in the B hour and 1.06 in the ESS run. About half of the ESS updates are at or above the clip of 1, and the 90th percentile is 2.6. Those steps are cut back to norm 1 before Adam. The learning rate is `1e-5` either way.

## Weights on disk

| File | What it is |
|---|---|
| `samples/rl-from-base/rl-trocr.pth` | Latest weights. The running process rewrites this file. |
| `samples/rl-from-base/rl-trocr-best.pth` | Review best, update 345, reward 0.961. Written at 06:30. Play CER had already settled near 0.18 by then. |
| `samples/rl-from-base/ckpt-0009.pth` | Last save of the divisor beam, update 105, the end of the best play hour. |
| `samples/rl-from-base/ckpt-0032.png` | Review sheet for the stored best. |
| `samples/rl-from-base/ckpt-0080.png` | Review at update 868, reward 0.954, CER 0.009. |
| `models/best-4.pth` | Frozen anchor. This series does not write it. |

The best file is chosen by the review sentence. A perfect read of `the old man` with more tilt scored 0.947 and lost to update 345.

## Reading the dashboard

The run menu has two ids. `20261020-022557` is the ESS process and the one the page opens. `20261020-003843` is starts A and B stitched by the alias. The x-axis is the update counter, so the ESS chart begins at 106.

The green line is the prefix average. For play on the ESS chart that is close to the finished line (CER 0.186 against 0.171). For novel it is worse (0.814 against 0.694). The rust dots are `the old man`. They improved in the first hour and then held near reward 0.90 and CER 0.06, which is a real result for that sentence and a different result from the play lines.
