"""Generate a grid of handwriting samples across varying difficulty. Throwaway."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from generate import sample, load_model

TEXTS = [
    "hi",                                            # trivial
    "hello world",                                   # easy
    "good morning",                                  # easy
    "practice makes perfect",                        # medium phrase
    "2026",                                          # digits
    "-0-1-2-3-4-5-6-7-8-9",                           # the fine-tuned edge case
    "it's a test, isn't it?",                        # punctuation
    "extraordinary",                                 # hard single word
    "the quick brown fox jumps over the lazy dog",   # long pangram
    "Saturday, January 7th 2026",                    # mixed case + digits + punct
]
TEMP = 0.4

def main():
    torch.manual_seed(0)
    model, stoi = load_model("cpu", ckpt="best.pth")
    std = 1.9816  # IAM offset std (same constant training used)

    n = len(TEXTS)
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.7 * n))
    for ax, text in zip(axes, TEXTS):
        seq, _ = sample(model, text, stoi, "cpu", temperature=TEMP)
        xy = np.cumsum(seq[:, :2].numpy() * std, axis=0)
        pen_up = seq[:, 2].numpy()
        start = 0
        for j in range(len(seq)):
            if pen_up[j] == 1 or j == len(seq) - 1:
                ax.plot(xy[start:j + 1, 0], xy[start:j + 1, 1], "k", linewidth=1.8)
                start = j + 1
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f'"{text}"   ({len(seq)} steps)', fontsize=10, loc="left")

    fig.suptitle(f"best.pth handwriting samples  (temperature {TEMP})", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig("samples_grid.png", bbox_inches="tight", dpi=130)
    print("saved samples_grid.png")

if __name__ == "__main__":
    main()
