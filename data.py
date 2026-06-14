import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


def build_vocab(sentences):
    chars = sorted(set("".join(sentences)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    return stoi

def load_data(strokes_path="data/strokes.npy", sentences_path="data/sentences.txt"):
    strokes = np.load(strokes_path, allow_pickle=True, encoding="latin1")
    sentences = open(sentences_path).read().splitlines()
    assert len(strokes) == len(sentences)

    # compute std of offsets for normalisation (paper sec. 4.2)
    all_offsets = np.concatenate([s[:, 1:].reshape(-1) for s in strokes])
    std = float(all_offsets.std())

    return strokes, sentences, std


class StrokeDataset(Dataset):
    """One item = one handwritten line.

    Each sequence row in the file is (pen_up, dx, dy).
    We reorder to (dx, dy, pen_up) to match the paper's eq. (15): x_t in R x R x {0,1}.

    The model is trained to predict x_{t+1} given x_{1:t}, so we return:
      input  = zero start point + points 0 .. T-2
      target = points 0 .. T-1
    """

    def __init__(self, strokes, sentences, stoi, std, max_len=None,
                 collected_dir=None):
        self.std = std
        self.stoi = stoi
        self.seqs = []
        self.texts = []

        for s, sent in zip(strokes, sentences):
            s = s.astype(np.float32).copy()
            s[:, 1:] /= std                    # normalise offsets
            s = s[:, [1, 2, 0]]                # reorder to (dx, dy, pen_up)
            idx = [self.stoi[c] for c in sent]

            # cap sequence length to keep the layer-1 loop short. Truncate the text
            # proportionally so text/stroke alignment stays consistent (paper sec. 5.2
            # trains on fixed-length chunks, not whole lines).
            if max_len is not None and len(s) > max_len:
                n_chars = max(1, round(len(idx) * max_len / len(s)))
                s = s[:max_len]
                idx = idx[:n_chars]

            self.seqs.append(torch.from_numpy(s))
            self.texts.append(torch.tensor(idx, dtype=torch.long))

        if collected_dir:
            self._append_collected(collected_dir, max_len)

    def _append_collected(self, collected_dir, max_len):
        """Convert collected/*.json canvas samples on the fly and append them.

        Reuses import_collected.file_to_array -- the single source of truth for the
        canvas->IAM normalisation (flip-y, robust rescale, arc-length resample, std
        divide, reorder to (dx, dy, pen_up)). So dropping JSONs in collected/ is all
        that's needed before training: no import_collected / export_dataset run.
        """
        from glob import glob
        from pathlib import Path
        from import_collected import file_to_array

        files = sorted(glob(os.path.join(collected_dir, "*.json")))
        if not files:
            print(f"StrokeDataset: no canvas samples in {collected_dir}/")
            return
        added = skipped = 0
        for f in files:
            result = file_to_array(Path(f), self.std)        # already normalised
            if result is None:                                # degenerate (empty / too small)
                skipped += 1
                continue
            arr, text, _ = result
            if any(ch not in self.stoi for ch in text):       # out-of-vocab guard
                skipped += 1
                continue
            idx = [self.stoi[c] for c in text]
            if max_len is not None and len(arr) > max_len:
                n_chars = max(1, round(len(idx) * max_len / len(arr)))
                arr = arr[:max_len]
                idx = idx[:n_chars]
            self.seqs.append(torch.from_numpy(np.asarray(arr, dtype=np.float32)))
            self.texts.append(torch.tensor(idx, dtype=torch.long))
            added += 1
        msg = f"StrokeDataset: +{added} collected canvas samples (total {len(self.seqs)})"
        if skipped:
            msg += f", skipped {skipped} (degenerate/out-of-vocab)"
        print(msg)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        seq = self.seqs[i]                     # (T, 3)
        start = torch.zeros(1, 3)              # null first input (paper sec. 2)
        inp = torch.cat([start, seq[:-1]])     # shifted right by one
        return inp, seq, self.texts[i]         # input, target, text


def collate(batch, vocab_size):
    """Pad variable-length sequences.

    Returns (x, y, mask, c, c_mask):
      x, y    : (B, T, 3)  stroke input / target
      mask    : (B, T)     1 where a real stroke point exists
      c       : (B, U, V)  one-hot text the writing is conditioned on
      c_mask  : (B, U)     1 where a real character exists (0 = padding)
    """
    inputs, targets, texts = zip(*batch)
    T = max(len(x) for x in inputs)
    U = max(len(t) for t in texts)             # longest sentence in batch
    B = len(inputs)
    x = torch.zeros(B, T, 3)
    y = torch.zeros(B, T, 3)
    mask = torch.zeros(B, T)
    c = torch.zeros(B, U, vocab_size)          # one-hot text
    c_mask = torch.zeros(B, U)                 # 1 = real char, 0 = padding
    for i, (inp, tgt, txt) in enumerate(zip(inputs, targets, texts)):
        x[i, :len(inp)] = inp
        y[i, :len(tgt)] = tgt
        mask[i, :len(tgt)] = 1.0
        c[i, torch.arange(len(txt)), txt] = 1.0
        c_mask[i, :len(txt)] = 1.0

    return x, y, mask, c, c_mask


class BucketBatchSampler(Sampler):
    """Group similar-length sequences into batches to minimise padding waste.

    collate() pads every sequence in a batch up to the batch's LONGEST member, so a
    random batch is dominated by its longest outlier -- measured ~40% of the compute
    on the IAM+collected set goes to padding. Here we split the sequences at the
    MEDIAN length into a short bucket (<= median) and a long bucket (> median), and
    draw each batch from a SINGLE bucket. The pad target then tracks the bucket, not a
    global outlier, so short batches pad to ~median and long batches stay together.

    Within each bucket we further SORT by length inside megabatches (sort_megabatch
    batches' worth at a time) before cutting into batches, so each batch is nearly
    length-homogeneous and pads to almost nothing. The shuffle-then-sort-in-chunks
    order keeps membership varying epoch to epoch (a full global sort would fix it).

    Sequences are never split or truncated -- only the grouping into batches changes.
    """

    def __init__(self, lengths, batch_size, shuffle=True, sort_megabatch=50):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sort_megabatch = sort_megabatch
        median = float(np.median(lengths))
        short = [i for i, L in enumerate(lengths) if L <= median]
        long_ = [i for i, L in enumerate(lengths) if L > median]
        self.buckets = [b for b in (short, long_) if b]   # drop an empty bucket
        self.median = median

    def _make_batches(self):
        batches = []
        mega = self.batch_size * self.sort_megabatch
        for bucket in self.buckets:
            idx = list(bucket)
            if self.shuffle:
                random.shuffle(idx)
            # sort by length within each megabatch so consecutive batches contain
            # near-identical lengths (minimal padding), with the per-epoch shuffle
            # above still reshuffling which sequences land in which megabatch.
            for m in range(0, len(idx), mega):
                chunk = sorted(idx[m:m + mega], key=lambda i: self.lengths[i])
                batches += [chunk[k:k + self.batch_size]
                            for k in range(0, len(chunk), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)        # mix short/long batches across the epoch
        return batches

    def __iter__(self):
        return iter(self._make_batches())

    def __len__(self):
        return sum((len(b) + self.batch_size - 1) // self.batch_size
                   for b in self.buckets)


def get_loader(batch_size=32, num_workers=4, pin_memory=True, max_len=700,
               collected_dir="collected", bucket=True):
    # IAM (raw) + the hand-collected canvas samples, converted straight from
    # collected/*.json. No pre-processing scripts: drop JSONs in, run train.py.
    strokes, sentences, std = load_data()
    stoi = build_vocab(sentences)
    dataset = StrokeDataset(strokes, sentences, stoi, std, max_len=max_len,
                            collected_dir=collected_dir)
    collate_fn = partial(collate, vocab_size=len(stoi))
    if bucket:
        lengths = [len(s) for s in dataset.seqs]
        sampler = BucketBatchSampler(lengths, batch_size, shuffle=True)
        print(f"BucketBatchSampler: median len {sampler.median:.0f}, "
              f"{len(sampler.buckets)} buckets, {len(sampler)} batches/epoch")
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=num_workers > 0)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=num_workers > 0)
    return loader, std, stoi


if __name__ == "__main__":
    loader, _, stoi = get_loader(batch_size=32)
    print(f"batches: {len(loader)}   vocab size: {len(stoi)}")
    x, y, mask, c, c_mask = next(iter(loader))
    print(f"input  shape: {x.shape}   (batch, time, 3)")
    print(f"target shape: {y.shape}")
    print(f"mask   shape: {mask.shape}")
    print(f"text   shape: {c.shape}   (batch, chars, vocab)")
    print(f"c_mask shape: {c_mask.shape}")
    print(f"one-hots per row sum to 1? {c.sum(-1).max().item()}  "
          f"total chars in batch: {int(c.sum().item())}")
    print(f"example input row (dx, dy, pen_up): {x[0, 5]}")
