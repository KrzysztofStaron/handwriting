# vCER validation

The ordering set is proposed by the hand seed, not labeled by a person.
Spot-check `vcer/eval/sanity_pairs.jsonl` before reading the rate as a human study.
Hypothesis A is the one that should look closer. A metric agrees when it strictly prefers A.

- Pairs: 60
- vCER agreement: 60/60 (1.000)
- Plain CER agreement: 35/60 (0.583)
- Pairs vCER gets and plain CER misses: 25

## Hacking suite

Bar: reward for `the old man` -> `the 0ld man` is 0.782.
Every degenerate string has to land strictly below that.

| case | reward | wCER | below bar |
|---|---|---|---|
| thin_spam | 0.000 | 2.255 | yes |
| only_i | 0.396 | 0.505 | yes |
| repeat_i | 0.276 | 0.655 | yes |
| very_short | 0.335 | 0.582 | yes |
| very_long | 0.000 | 4.127 | yes |
| neutral_dots | 0.364 | 0.545 | yes |
| quotes | 0.364 | 0.545 | yes |
| dropped_word | 0.629 | 0.214 | yes |
| empty | 0.320 | 0.600 | yes |

The hacking suite passed.

## Ablation to run later

Do not start a long job for this without a go-ahead. The live trainer is a separate process.

1. Plain CER (`--reward exp`) on `data/romeo_juliet.txt`.
2. vCER (`--reward vcer`) on the same file.
3. vCER with annealing: start from this table, then linearly raise substitution costs toward 1 so the end of training is close to exact match.

Score each run with plain CER and exact match on text that was not in the prompt file. vCER is only the training signal.
