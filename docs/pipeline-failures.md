# Where the recognizer training fails

Two weight files, read from their event logs on 2026-10-20.

- `samples/rl/` is the earlier line. It started from `models/best-4.pth`, saved to `models/rl-trocr.pth`, and ran from 10 October through 00:29 on 20 October. The surviving log has 2,600 updates and 77 reviews. A cold-temperature episode between updates 2231 and the resume at 2230 was discarded and is not in this file. Nothing that remains has a KL above 1.03.
- `samples/rl-from-base/` is the fresh start from `best-4` the same morning. Its hour-by-hour numbers are `docs/from-base-report.md`. The counts below for that series go through update 877.

Finished-line CER is the median over the drawings of the full text. On a prefix beam that is the longest text in the update. Chart means average every prefix and sit higher. Reward is not comparable across phases: the formula changed under the same weights.

## The earlier line

One command, then whatever defaults the commit had when the process was started again. Update numbers overlap across restarts, because each resume continued from the last saved checkpoint and the log kept the abandoned tail.

| Phase | Updates | When | Reward | Lines | Sampler | Finished CER |
|---|---|---|---|---|---|---|
| P0 | 1–192 | recovered stdout | batch mean only | IAM, 4,138 prompts | flat | chart 0.176 |
| P1 | 182–2155 | 10-19 03:20–11:31 | `exp(-CER)`, exact match to the log | IAM, median 27 characters | flat, k=2, T=0.5 | 0.103 |
| P2 | 2147–2206 | 16:09–16:43 | vCER, table `217d3835`, confidence 0.2 | Romeo, median 46 characters | flat, k=2, T=0.5 | 0.267 |
| P3 | 2200–2231 | 16:44–17:06 | same | Romeo | flat, k=8, T=0.5 | 0.255 |
| P4 | 2231–2540 | 17:47–20:49 | same | Romeo | flat, k=8, temperature normal 0.1–0.9 | 0.167 |
| P5 | 2541–2572 | 23:52–00:29 | 60/20/20, straightness 0.05, new table `fc2253e9` | 25% novel, 25% words, rest play | divisor beam 16 | play 0.065 |

P0 has no per-draw rows. Its chart CER moved 0.200, 0.167, 0.161, 0.179 across four quarters. P1's per-draw reward matches `exp(-CER)` with residual 0.

Shared settings once they were logged: policy on CPU, TrOCR on MPS, anchor `models/best-4.pth`, KL coefficient 0.1, Adam `1e-5`, clip 1, sharpen 2. P5 adds the beam, the mix, and the tilt weight. It is the same sampler as from-base starts A and B, on the trained weights rather than on `best-4`.

## What actually moved

The stretches that held one objective still are the ones where finished CER fell.

P1, IAM lines, 15,792 drawings. Quartiles of the per-update median: 0.130, 0.104, 0.100, 0.097. Exact copies 12.4%. Half the drawings were under CER 0.10. KL median 0.281, and 0.9% of updates had a pre-clip gradient at or above 1. Within-update reward spread median 0.069.

The switch to Romeo is a different problem. Length goes from 27 characters to 46, speaker names arrive in capitals, and the reward becomes visual CER. P2 and P3 sit at 0.26–0.27 for ninety updates and do not come back. Exact copies in P2: 0 of 480.

P4, same Romeo lines, k=8, temperatures clustered at 0.42–0.59. Quartiles: 0.211, 0.165, 0.149, 0.130. That is about four fewer character edits per line (9.7 down to 6.0 on a 46-character line) in five hours. Exact copies are 0.4%. KL median 0.539. Gradient median 0.155, none of the 310 updates clipped. The correlation of temperature with reward inside an update is −0.09. Temperature is not deciding the ranking.

P5 is thirty-two beam updates on those weights. Play, 13 updates: finished CER 0.065. Words, 12 updates: 0.074. Novel, 7 updates: 0.514, no exact line. Tilt median 1.0 on every source, 89% of the 512 finished lines at the cap, and 19 of 32 updates have every copy at the cap. This is a short sample. It is also the same divisor beam as from-base start B, whose play CER from fresh `best-4` was 0.108.

The review sentence `the old man` does not track the training lines.

| Phase | Reviews | Median reward | Median CER | Best CER |
|---|---|---|---|---|
| P0 | 5 | 0.891 | 0.117 | 0.063 |
| P1 | 48 | 0.853 | 0.164 | 0.064 |
| P2 | 3 | 0.933 | 0.064 | 0.018 |
| P3 | 2 | 0.866 | 0.364 | 0.118 |
| P4 | 17 | 0.940 | 0.027 | 0.000 |
| P5 | 2 | 0.905 | 0.055 | 0.018 |

During the IAM run the training lines (median CER 0.103) were cleaner than the review (0.164). During P2 the review stayed near 0.06 while Romeo lines were at 0.27. During P4 the review reached a perfect read several times. The stored best, `models/rl-trocr-best.pth`, is update 2529, reward 0.999956, sheet `samples/rl/ckpt-0075.png`. All ten copies read `the old man` with CER 0. The ink on that sheet already drops from `the` toward `man`, with wide gaps between the words. Straightness was not in the reward yet. TrOCR gave that slope a perfect score, and the best-file rule kept it.

From-base then started over from `best-4`. Its divisor hour matched the old line's P4 finish (play CER 0.108 against 0.130). Its ESS run, twelve hours later, sits at play CER 0.174 (5,952 lines through update 877). The weights that had already reached 0.13, and 0.065 under the beam, are the other file.

## The errors on the page

Levenshtein on the normalized strings. Normalization already drops a space before punctuation, so a read of `will ,` for `will,` is not an edit. What remains:

| Lines | Edits per line | Share sub | Share ins | Share del | Lines that end on an insertion |
|---|---|---|---|---|---|
| P1 IAM, CER 0.103 | 2.3 | 48% | 45% | 7% | 54% |
| P2 Romeo, CER 0.267 | 11.5 | 70% | 22% | 9% | 12% |
| P4 Romeo, CER 0.167 | 6.7 | 69% | 22% | 9% | 14% |
| B play, CER 0.108 | 5.6 | 57% | 36% | 7% | 19% |
| C play, CER 0.174 | 8.6 | 61% | 32% | 7% | 12% |
| C words, CER 0.146 | 7.9 | 64% | 24% | 13% | 10% |
| C novel, CER 0.694 | 24 | 74% | 22% | 4% | 8% |

Deletions are the small share on every phase. The log does not show a policy that is mostly skipping letters. Insertions outnumber deletions about four to one on the from-base play lines, and six to one on the IAM run.

On IAM, half the lines end with an extra character, and insertions are almost as common as substitutions. On Romeo, the tail is uncommon (12–14% of lines) and the edits are inside the sentence. Of the insertions on from-base play lines, 71% are letters or digits and 13% are punctuation. The extra stroke is usually another letter, not a trailing dot.

The substitutions repeat. On from-base play, the top pairs are `o→a`, `a→o`, `W→w`, `T→t`, `n→r`, `o→e`, `h→n`, `a→e`. The same family is on top in P1, P2, P4, and the good divisor hour. Case pairs are 7% of play substitutions. Letter-for-letter pairs are 69%. Capitals of speaker names (`W`, `T`, `O`) are read as lowercase often enough to make the top eight, and they are not free: the ink table prices `W→w` at 0.69, `T→t` at 0.57, `O→o` at 0.79. The 0.2 case fallback applies only when the pair is absent from the table.

Novel lines are a different error. The top alignment steps are a word gap read as a period, and insertions of `i`, `t`, `r`, `e`, `o`. A prompt such as `hf Wo MB4 t JW mr p9Ly Mf HOQl zD` comes back as English words. There is no exact finished novel line in P5 (112 lines) or in from-base (3,824 lines).

## Why the update does not clear these

**A defect shared by the whole group has no advantage.** The baseline is the mean of the copies, or an EMA when each beam parent continues once. Tilt is the measured case. On P5 the median within-update tilt spread is 0, and 19 of 32 updates are entirely at the cap. On from-base play, 142 of 372 ESS updates have every finished copy at the cap, and 75% of the 5,952 lines score 1. The first from-base hour, sampling `best-4`, already had 62% of play lines at the cap and 7 of 21 updates fully pegged. The slope is in the base samples. The long run, with no tilt term for 2,500 updates, made it uniform: the perfect review sheet slopes, and the first logged straightness is pegged. The score in both of these processes reaches 1 at one letter height, so a steeper line looks the same. The weight is 0.05. Once every copy is at 1, that 0.05 is inside the baseline.

**One number is spread over the whole trajectory.** A single substitution on a 46-character line moves the 60% CER term by about 0.013. P4 and the ESS run put that advantage on every pen step of the line or the word. A finished play line is about 1,400 steps. The within-group reward spread is 0.06 to 0.07, which is the entire signal the baseline can see. The KL term subtracts `0.1 × KL` from that same advantage: 0.028 in P1, 0.054 in P4, 0.045 in the ESS run. P1 and P4 still moved the average CER, because the spread was larger than the KL term and the gradient was small (median 0.15–0.26, rarely clipped). They moved it by a few characters per line, and the identity of the mistakes did not change. The stroke that turns `n` into something TrOCR reads as `r` is one step out of a thousand, and the advantage cannot name it.

**Plain CER charges the common confusions almost in full.** The string score is three quarters plain CER and one quarter vCER. An insertion costs 0.60 visually and 1.00 in plain CER, so 90% of a full edit remains in the string score. `o↔a` is priced at 0.40 and still leaves 85%. `n→r`, the workhorse one-way error, is priced at 0.64 and leaves 91%. `h→t` leaves 94%. The eight hand digraphs (`rn↔m` at 0.25 and the rest near 0.30) are the cheap pairs, and they are not the pairs the log makes. The median stored visual cost is 0.78. A policy that writes a slightly wrong vowel is scored almost as if it had written an unrelated character, on 60% of the reward, on every step of the line.

**Novel prompts ask TrOCR to read a string it will turn into English.** A quarter of P5 and of the ESS run is ten random tokens of at most four characters. The continuous score still differs between copies, so those updates take gradient steps. The direction of the score is "look more like the English word the decoder guessed," and the KL on that source is the highest in the ESS run (chart median 0.58, 0.74 in the last third). Confidence never subtracts. A line with CER above 0.5 still has confidence about 0.50, which adds 0.10. That floor is a large piece of the novel reward, whose median is 0.37.

**The best file is the short sentence.** Both series store `rl-trocr-best` from `the old man` only. The old best is a perfect read of a sloping line at update 2529, while Romeo CER in that phase was 0.13. The from-base best is reward 0.961 at update 345, after play CER had already settled near 0.18. An all-exact review later in the ESS run scored 0.947 and lost to it on tilt and confidence. The file the trainer is writing is newer than either best and is not selected by the training CER.

**The ESS run clips, and it usually does not narrow.** P4's gradient median was 0.16. The ESS run's is 1.06, and about half the updates are at or above the clip of 1, with KL on play climbing from 0.33 to 0.61 while finished play CER stays near 0.17. ESS keeps all 16 drawings unless the effective sample size falls below 8, which is 14% of word boundaries. In the divisor hour, the updates that kept all 16 (factor 1, four updates) had play CER 0.149, and the updates that narrowed to one or two parents had 0.07–0.09. The sampler that is running is the setting that scored worse in the only hour where both were measured. Each word is its own Adam step, so a nine-word line is nine steps, including words whose copies already agree.

## What the two series agree on

The average CER can move when one reward stays in place for a few hundred updates and the gradient is smaller than the clip. It moved on IAM from about 0.13 to 0.10, and on Romeo from about 0.21 to 0.13. It did not change which mistakes those lines make.

The mistakes are repeated letter confusions and extra letters inside the line. On the play lines, deletions are about one edit in fourteen. The visual table discounts some of those pairs and the plain-CER term, at 60% of the reward, puts the discount back. The advantage then copies one line-level number onto every pen step, and any defect shared by all the copies, including the slope, is removed by the baseline. The review sentence is short enough that TrOCR reads it anyway, including when it slopes, and that read is what gets saved as best.
