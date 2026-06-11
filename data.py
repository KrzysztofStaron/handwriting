from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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

    def __init__(self, strokes, sentences, stoi, std):
        self.std = std
        self.stoi = stoi
        self.seqs = []
        self.texts = []


        for s, sent in zip(strokes, sentences):
            s = s.astype(np.float32).copy()
            s[:, 1:] /= std                    # normalise offsets
            s = s[:, [1, 2, 0]]                # reorder to (dx, dy, pen_up)
            self.seqs.append(torch.from_numpy(s))
            idx = [self.stoi[c] for c in sent]
            self.texts.append(torch.tensor(idx, dtype=torch.long))

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


def get_loader(batch_size=32, num_workers=4, pin_memory=True):
    strokes, sentences, std = load_data()
    stoi = build_vocab(sentences)
    dataset = StrokeDataset(strokes, sentences, stoi, std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=partial(collate, vocab_size=len(stoi)),
                        num_workers=num_workers, pin_memory=pin_memory,
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
