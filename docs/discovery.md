# vCER discovery

The recognizer reward lives in `rl.py`. `TrocrScorer.score` renders a stroke, asks frozen TrOCR-Large for a string, and turns character error rate into a reward. The error rate is case-sensitive Levenshtein over the reference length, after whitespace is collapsed, curly quotes are straightened, and a space before punctuation is dropped. The training default was `exp(-CER)`, capped at a CER of 2. `train_rl.py` calls that once per draw. Nothing in the reward looked at whether the mistaken glyphs look alike.

Targets are whole lines, not characters or words. Supervised training uses `data/sentences.txt`, the IAM transcriptions, one line per stroke sequence. The recognizer run was sampling the 4,138 of those lines that are 8–32 characters and use the 77-character alphabet in `stoi.json`: space, the punctuation `!"# '()+,-./:;?`, digits, and `A-Z` `a-z`. There are no diacritics in the alphabet. TrOCR is decoded with `batch_decode(..., skip_special_tokens=True)` to a Unicode string. The reward never sees tokens.

A rollout is 4 prompts times 2 draws, so 8 reward calls. An update in the current run takes about 20 seconds, almost all of it sampling and the TrOCR forward. A pure-Python edit distance is inside that budget. The 20k-pairs-per-second figure is not the bar for this trainer.

`models/best-4.pth` is the supervised baseline, and TrOCR-Large is the reader. A confusion harvest would run that reader over a held-out split. The recognizer run already has TrOCR loaded on MPS, so that harvest is not done in this pass. The cost table prices every alphabet symbol by how much ink it shares with every other symbol, drawn from Bradley Hand, Marker Felt, and Noteworthy on one baseline. A hand-listed one-character pair can only make that cheaper. The two-letter pairs stay the hand list (`d -> ol` and the other three, both ways), so a font similarity cannot invent a new digraph.

Romeo and Juliet is not in the IAM lines. `data/romeo_juliet.txt` is the whole play, cut into lines of at most 48 characters, and the trainer samples that file at random. `--max-chars` is 48 so TrOCR's square resize does not squash the line.
