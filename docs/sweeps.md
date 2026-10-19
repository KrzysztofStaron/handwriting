# vCER sensitivity

Rewards use the merged table. `K=1` cannot substitute a digraph in one edit,
so `d -> ol` stops being cheaper than `d -> e` until `K >= 2`.

## K and alpha

| ref | hyp | K=1 a=0.6 | K=1 a=0.8 | K=1 a=1.0 | K=2 a=0.6 | K=2 a=0.8 | K=2 a=1.0 | K=3 a=0.6 | K=3 a=0.8 | K=3 a=1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| `d` | `ol` | 0.000 | 0.000 | 0.000 | 0.450 | 0.600 | 0.750 | 0.450 | 0.600 | 0.750 |
| `d` | `e` | 0.269 | 0.359 | 0.448 | 0.269 | 0.359 | 0.448 | 0.269 | 0.359 | 0.448 |
| `world` | `vvorld` | 0.408 | 0.544 | 0.680 | 0.564 | 0.752 | 0.940 | 0.564 | 0.752 | 0.940 |
| `world` | `xorld` | 0.520 | 0.694 | 0.867 | 0.520 | 0.694 | 0.867 | 0.520 | 0.694 | 0.867 |
| `hello` | `he11o` | 0.528 | 0.704 | 0.880 | 0.528 | 0.704 | 0.880 | 0.528 | 0.704 | 0.880 |
| `hello` | `hexlo` | 0.480 | 0.640 | 0.800 | 0.480 | 0.640 | 0.800 | 0.480 | 0.640 | 0.800 |
| `the old man` | `the old man` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `the old man` | `the old man.` | 0.567 | 0.756 | 0.945 | 0.567 | 0.756 | 0.945 | 0.567 | 0.756 | 0.945 |
| `the old man` | `the old mar` | 0.558 | 0.744 | 0.930 | 0.558 | 0.744 | 0.930 | 0.558 | 0.744 | 0.930 |
| `the old man` | `iiiiiiiiiii` | 0.074 | 0.098 | 0.123 | 0.297 | 0.396 | 0.495 | 0.382 | 0.509 | 0.636 |

## Substitution floor

Stored hand costs are lifted when the floor rises. Render-only pairs are already at least 0.55.

| floor | d→ol | d→e | trailing dot | one wrong letter | all i |
|---|---|---|---|---|---|
| 0.15 | 0.600 | 0.359 | 0.756 | 0.744 | 0.396 |
| 0.30 | 0.560 | 0.359 | 0.756 | 0.744 | 0.396 |
| 0.50 | 0.400 | 0.359 | 0.756 | 0.744 | 0.396 |

## Indel floor

| indel | trailing dot | dropped word `the man` | reasonable `the 0ld man` |
|---|---|---|---|
| 0.4 | 0.771 | 0.684 | 0.782 |
| 0.6 | 0.756 | 0.629 | 0.782 |
| 0.8 | 0.742 | 0.585 | 0.782 |
| 1.0 | 0.727 | 0.542 | 0.782 |
