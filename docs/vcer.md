# Visually weighted CER

The recognizer reward can score a decoded string by how the mistakes look, not only by how many edits they take. Reading `d` as `ol` costs less than reading `d` as `e`.

This is a training signal. Held-out quality stays plain CER and exact match.

The running trainer does not pick this up until it is started again. `models/best-4.pth` is still never written.

## Reward

```
wcer   = distance(ref, hyp) / max(len(ref), 1)
reward = alpha * (1 - min(wcer, 1)) + (1 - alpha) * exact
```

`alpha` defaults to 0.8. An exact normalized match scores 1. An empty reference scores 1 only against an empty hypothesis.

Strings are NFC, curly quotes and dashes are straightened, and whitespace is stripped and collapsed. A space immediately before punctuation is dropped, so TrOCR's " :" scores the same as ":". A space after punctuation stays. That matches the existing CER normalizer. The distance sees characters, not model tokens.

Substitution spans are 1..K characters, K defaults to 2 and can be 1, 2, or 3. Insertion and deletion cost 0.6, so a stray dot is cheaper than a random substitution and a deleted word is not free. Substitution costs are at least 0.15, so nothing visual is free. Same letters in different case cost 0.2 when no exact pair is stored.

## Cost table

`vcer/table.json` is the table training loads. It has a version and a sha256 prefix. The loader rejects a mismatched hash. The rule page at [http://127.0.0.1:8765/vcer](http://127.0.0.1:8765/vcer) lists every discounted pair and scores a prompt against a read.

Each symbol is drawn once in Bradley Hand, Marker Felt, and Noteworthy, at the same size on a shared baseline. A pixel stamp is the fallback when those fonts do not load. The cost against another symbol is the fraction of ink that does not overlap, after trying a shift of a few pixels, averaged over the fonts that actually drew both marks. A period stays a dot. Glyphs are not stretched to the same height. A blank mark, including space, costs 1 against every symbol.

The hand list in `vcer/table_build/hand_seed.yaml` is an upper bound on the one-character pairs it names: the stored cost is the cheaper of the ink overlap and the listed cost. The two-letter pairs are only the ones in that list (`rn`/`m`, `cl`/`d`, `d`/`ol`, `vv`/`w`, both ways). No other two-letter substitution is added.

Confusion counts from a baseline TrOCR pass are implemented in `vcer.table_build.from_confusions` and were not run. The recognizer trainer already has TrOCR loaded. A later harvest has to use a held-out split, not the RL prompts.

Rebuild, from the repo root:

```bash
uv run python -m vcer.table_build.build
```

## Trainer

`train_rl.py` defaults to `--reward both`. That score is 60% plain CER and 20% this visual distance, then 20% confidence. Straightness is subtracted. Word spacing is not part of the score. `--reward vcer` uses the visual distance alone, plus an exact-match term. `--reward cer` is plain CER and does not load this table. `--reward exp` is `exp(-CER)`.

| flag | default | meaning |
|---|---|---|
| `--reward` | `both` | `both`, `vcer`, `cer`, `exp`, `linear`, or `soft` |
| `--vcer-alpha` | `0.8` | weight on `1 - wCER`; the rest is the exact-match bit |
| `--vcer-k` | `2` | longest substitution span |
| `--vcer-table` | `vcer/table.json` | versioned cost table |
| `--prompts` | `data/romeo_juliet.txt` | line pool, sampled at random |

Each vCER update logs plain CER, wCER, TrOCR token confidence, the reward spread, mean output length, and the share of thin glyphs `i l 1 . '`.

After that mix, a crooked line loses a little more. The left quarter and the right quarter of the ink each have a band from the 10th to the 90th percentile of height. The score is how far those two bands have moved apart, divided by the letter height. The middle quarter is compared with the ends in the same way, so an arch counts. A level line scores 0. The score stops at 1, and `--straightness` (default 0.05) is the most that can come off the reward. `--straightness 0` turns it off.

Word spacing is logged on each drawing and is not part of the reward. `--spacing` defaults to 0.

TrOCR has no separate clarity score. In greedy decoding the confidence of a read is the geometric mean of the probabilities it assigned to the tokens it chose. `--confidence` (default 0.2) is that term's share of the reward: `(1 - weight) * string + weight * confidence`. High confidence raises the reward. `--confidence 0` turns it off. It needs `--beams 1`.

Prompts are the whole of Romeo and Juliet (Project Gutenberg ebook 1513), cut on word boundaries into lines of 8–48 characters in the model alphabet. Repeated lines are kept once. The review sentence stays `the old man`, which is not in that file. `--max-chars` defaults to 48. TrOCR resizes every rendering to a 384-pixel square, and a longer line is squashed too thin to read.

```bash
uv run python scripts/build_romeo_prompts.py   # needs /tmp/romeo_raw.txt
uv run train_rl.py --no-open                    # vCER, Romeo and Juliet
uv run train_rl.py --reward exp --prompts data/sentences.txt
```

## Checks

`docs/validation.md` is the proposed ordering rate and the hacking suite. The ordering file is `vcer/eval/sanity_pairs.jsonl`. Those labels were proposed from the hand seed, not collected from a person. `docs/sweeps.md` is K, alpha, and the floors.

An ablation to run only with a go-ahead: plain CER, vCER, and vCER annealed toward cost 1. Score all three with plain CER and exact match on text outside the prompt file.
