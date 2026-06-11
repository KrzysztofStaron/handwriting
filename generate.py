import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import HandwritingModel, split_params
from data import get_loader

device = "mps" if torch.backends.mps.is_available() else "cpu"

# the vocab must be the SAME mapping used during training (saved by train.py)
stoi = torch.load("stoi.pth")

model = HandwritingModel(vocab_size=len(stoi)).to(device)
model.load_state_dict(torch.load("best.pth", map_location=device))
model.eval()

_, std, _ = get_loader()   # normalisation factor used during training


def sample(text, temperature=1.0, max_steps=2000):
    """Generate handwriting for `text`, one point at a time.

    Returns (seq, phis):
      seq  : (steps, 3)  the (dx, dy, pen_up) trajectory
      phis : (steps, U)  window attention over the text at each step (for Fig. 13)
    """
    U = len(text)
    c = torch.zeros(1, U, len(stoi)).to(device)
    c[0, torch.arange(U), [stoi[ch] for ch in text]] = 1.0   # one-hot the text (fixed)
    c_mask = torch.ones(1, U).to(device)

    x = torch.zeros(1, 1, 3).to(device)   # null start point
    hidden = None                          # fresh: pen at origin, window at char 0
    pts, phis = [], []

    with torch.no_grad():
        for _ in range(max_steps):
            out, hidden, phi = model(x, c, c_mask, hidden)   # ONE step; hidden carries kappa
            phi = phi[0, 0]                                  # (U,) attention this step

            pi, mu_x, mu_y, sig_x, sig_y, rho, pen = split_params(out)

            j = torch.multinomial(pi[0, 0], 1).item()          # pick a Gaussian
            mx, my = mu_x[0, 0, j].item(), mu_y[0, 0, j].item()
            sx = sig_x[0, 0, j].item() * temperature
            sy = sig_y[0, 0, j].item() * temperature
            r = rho[0, 0, j].item()

            u1, u2 = torch.randn(1).item(), torch.randn(1).item()  # sample the blob
            dx = mx + sx * u1
            dy = my + sy * (r * u1 + math.sqrt(max(1 - r**2, 1e-8)) * u2)

            pen_up = 1.0 if torch.rand(1).item() < pen[0, 0].item() else 0.0
            pts.append([dx, dy, pen_up])
            phis.append(phi.cpu().numpy())
            x = torch.tensor([[[dx, dy, pen_up]]]).to(device)    # feed the point back

            # stop once the window has slid onto the last character (paper sec. 5.2):
            # the model has finished reading the text.
            if phi.argmax().item() == U - 1:
                break

    return torch.tensor(pts), np.array(phis)


def save_plot(seq, path, title):
    """Denormalise -> integrate offsets -> split on pen lifts -> draw."""
    xy = np.cumsum(seq[:, :2].numpy() * std, axis=0)
    pen_up = seq[:, 2].numpy()
    fig, ax = plt.subplots(figsize=(12, 3))
    start = 0
    for j in range(len(seq)):
        if pen_up[j] == 1 or j == len(seq) - 1:
            ax.plot(xy[start:j+1, 0], xy[start:j+1, 1], "k", linewidth=2)
            start = j + 1
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {path}")


def save_alignment(phis, text, path):
    """Plot the window weights phi(t, u) — the diagonal stripe of Fig. 13."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.imshow(phis.T, aspect="auto", origin="lower", cmap="Blues",
              interpolation="nearest")
    ax.set_yticks(range(len(text)))
    ax.set_yticklabels(list(text))
    ax.set_xlabel("pen step t")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {path}")


if __name__ == "__main__":
    text = "hello world"
    for temp in [0.3, 0.5, 0.8]:
        seq, phis = sample(text, temperature=temp)
        save_plot(seq, f"sample_temp_{temp}.png", f"'{text}'  temperature={temp}")
    save_alignment(phis, text, "alignment.png")
