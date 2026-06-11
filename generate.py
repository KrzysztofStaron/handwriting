import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import HandwritingModel, split_params
from data import get_loader

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = HandwritingModel().to(device)
model.load_state_dict(torch.load("best.pth", map_location=device))
model.eval()

_, std = get_loader()   # normalisation factor used during training


def sample(steps=700, temperature=1.0):
    """Autoregressively generate one stroke sequence (T, 3) of (dx, dy, pen_up)."""
    pts = []
    x = torch.zeros(1, 1, 3).to(device)   # null start point
    hidden = None
    with torch.no_grad():
        for _ in range(steps):
            out, hidden = model(x, hidden)
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
            x = torch.tensor([[[dx, dy, pen_up]]]).to(device)    # feed back in
    return torch.tensor(pts)


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


for temp in [0.3, 0.5, 0.8]:
    seq = sample(steps=700, temperature=temp)
    save_plot(seq, f"sample_temp_{temp}.png", f"temperature={temp}")
