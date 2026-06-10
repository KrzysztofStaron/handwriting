import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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

    def __init__(self, strokes, std):
        self.std = std
        self.seqs = []
        for s in strokes:
            s = s.astype(np.float32).copy()
            s[:, 1:] /= std                    # normalise offsets
            s = s[:, [1, 2, 0]]                # reorder to (dx, dy, pen_up)
            self.seqs.append(torch.from_numpy(s))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        seq = self.seqs[i]                     # (T, 3)
        start = torch.zeros(1, 3)              # null first input (paper sec. 2)
        inp = torch.cat([start, seq[:-1]])     # shifted right by one
        return inp, seq                        # input, target


def collate(batch):
    """Pad variable-length sequences. Returns (inputs, targets, mask)."""
    inputs, targets = zip(*batch)
    lengths = [len(x) for x in inputs]
    T = max(lengths)
    B = len(inputs)
    x = torch.zeros(B, T, 3)
    y = torch.zeros(B, T, 3)
    mask = torch.zeros(B, T)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        x[i, :len(inp)] = inp
        y[i, :len(tgt)] = tgt
        mask[i, :len(tgt)] = 1.0
        
    return x, y, mask


def get_loader(batch_size=32, num_workers=4, pin_memory=True):
    strokes, sentences, std = load_data()
    dataset = StrokeDataset(strokes, std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate, num_workers=num_workers,
                        pin_memory=pin_memory, persistent_workers=num_workers > 0)
    return loader, std


if __name__ == "__main__":
    loader, _ = get_loader(batch_size=32)
    print(f"batches: {len(loader)}")
    x, y, mask = next(iter(loader))
    print(f"input  shape: {x.shape}   (batch, time, 3)")
    print(f"target shape: {y.shape}")
    print(f"mask   shape: {mask.shape}")
    print(f"example input row (dx, dy, pen_up): {x[0, 5]}")
