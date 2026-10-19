# vCER sensitivity

Rewards use the merged table. `K=1` cannot substitute a digraph in one edit,
so `d -> ol` stops being cheaper than `d -> e` until `K >= 2`.

## K and alpha

| ref | hyp | K=1 a=0.6 | K=1 a=0.8 | K=1 a=1.0 | K=2 a=0.6 | K=2 a=0.8 | K=2 a=1.0 | K=3 a=0.6 | K=3 a=0.8 | K=3 a=1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| `d` | `ol` | 0.000 | 0.000 | 0.000 | 0.450 | 0.600 | 0.750 | 0.450 | 0.600 | 0.750 |
| `d` | `e` | 0.252 | 0.336 | 0.420 | 0.252 | 0.336 | 0.420 | 0.252 | 0.336 | 0.420 |
| `world` | `vvorld` | 0.433 | 0.577 | 0.721 | 0.564 | 0.752 | 0.940 | 0.564 | 0.752 | 0.940 |
| `world` | `xorld` | 0.513 | 0.684 | 0.855 | 0.513 | 0.684 | 0.855 | 0.513 | 0.684 | 0.855 |
| `hello` | `he11o` | 0.528 | 0.704 | 0.880 | 0.528 | 0.704 | 0.880 | 0.528 | 0.704 | 0.880 |
| `hello` | `hexlo` | 0.503 | 0.671 | 0.838 | 0.503 | 0.671 | 0.838 | 0.503 | 0.671 | 0.838 |
| `the old man` | `the old man` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `the old man` | `the old man.` | 0.567 | 0.756 | 0.945 | 0.567 | 0.756 | 0.945 | 0.567 | 0.756 | 0.945 |
| `the old man` | `the old mar` | 0.565 | 0.753 | 0.942 | 0.565 | 0.753 | 0.942 | 0.565 | 0.753 | 0.942 |
| `the old man` | `iiiiiiiiiii` | 0.142 | 0.190 | 0.237 | 0.302 | 0.402 | 0.503 | 0.382 | 0.509 | 0.636 |

## Substitution floor

Stored costs are lifted when the substitution floor rises. One-character costs are the unmatched share of the ink.

| floor | d→ol | d→e | trailing dot | one wrong letter | all i |
|---|---|---|---|---|---|
| 0.15 | 0.600 | 0.336 | 0.756 | 0.753 | 0.402 |
| 0.30 | 0.560 | 0.336 | 0.756 | 0.753 | 0.402 |
| 0.50 | 0.400 | 0.336 | 0.756 | 0.753 | 0.400 |

## Indel floor

| indel | trailing dot | dropped word `the man` | reasonable `the 0ld man` |
|---|---|---|---|
| 0.4 | 0.771 | 0.684 | 0.782 |
| 0.6 | 0.756 | 0.625 | 0.782 |
| 0.8 | 0.742 | 0.567 | 0.782 |
| 1.0 | 0.727 | 0.509 | 0.782 |
