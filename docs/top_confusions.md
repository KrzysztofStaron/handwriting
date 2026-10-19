# Cheapest non-identical substitutions

One-character costs are the share of ink that does not overlap.
A hand-listed pair can only make that cheaper. Two-letter pairs are only the hand list.
Phase 3a (baseline confusions) was not run: the recognizer trainer already holds TrOCR.

| ref | hyp | cost |
|---|---|---|
| `0` | `O` | 0.250 |
| `0` | `o` | 0.250 |
| `O` | `0` | 0.250 |
| `d` | `ol` | 0.250 |
| `m` | `rn` | 0.250 |
| `o` | `0` | 0.250 |
| `ol` | `d` | 0.250 |
| `rn` | `m` | 0.250 |
| `I` | `l` | 0.288 |
| `l` | `I` | 0.288 |
| `1` | `I` | 0.300 |
| `1` | `l` | 0.300 |
| `5` | `S` | 0.300 |
| `5` | `s` | 0.300 |
| `9` | `g` | 0.300 |
| `I` | `1` | 0.300 |
| `S` | `5` | 0.300 |
| `cl` | `d` | 0.300 |
| `d` | `cl` | 0.300 |
| `g` | `9` | 0.300 |
| `l` | `1` | 0.300 |
| `s` | `5` | 0.300 |
| `vv` | `w` | 0.300 |
| `w` | `vv` | 0.300 |
| `!` | `I` | 0.326 |
| `I` | `!` | 0.326 |
| `c` | `e` | 0.347 |
| `e` | `c` | 0.347 |
| `i` | `l` | 0.348 |
| `l` | `i` | 0.348 |
| `2` | `Z` | 0.350 |
| `2` | `z` | 0.350 |
| `9` | `q` | 0.350 |
| `Z` | `2` | 0.350 |
| `b` | `h` | 0.350 |
| `h` | `b` | 0.350 |
| `q` | `9` | 0.350 |
| `u` | `v` | 0.350 |
| `v` | `u` | 0.350 |
| `z` | `2` | 0.350 |
| `O` | `Q` | 0.357 |
| `Q` | `O` | 0.357 |
| `c` | `o` | 0.371 |
| `o` | `c` | 0.371 |
| `!` | `l` | 0.375 |
| `l` | `!` | 0.375 |
| `g` | `q` | 0.378 |
| `q` | `g` | 0.378 |
| `E` | `F` | 0.389 |
| `F` | `E` | 0.389 |
