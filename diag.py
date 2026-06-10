import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from model import HandwritingModel, split_params
from data import get_loader, load_data

device = "mps" if torch.backends.mps.is_available() else "cpu"
_, std = get_loader()


def plot_strokes(ax, stroke_seq, title):
    """stroke_seq: (T,3) tensor of (dx, dy, pen_up) in NORMALISED units."""
    xy = np.cumsum(stroke_seq[:, :2].cpu().numpy() * std, axis=0)
    pen_up = stroke_seq[:, 2].cpu().numpy()
    start = 0
    for j in range(len(stroke_seq)):
        if pen_up[j] == 1 or j == len(stroke_seq) - 1:
            ax.plot(xy[start:j+1, 0], xy[start:j+1, 1], "k", linewidth=1.5)
            start = j + 1
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=9)


def sample(model, steps=700, temperature=1.0):
    model.eval()
    pts = []
    x = torch.zeros(1, 1, 3).to(device)
    hidden = None
    with torch.no_grad():
        for _ in range(steps):
            out, hidden = model(x, hidden)
            pi, mu_x, mu_y, sig_x, sig_y, rho, pen = split_params(out)
            idx = torch.multinomial(pi[0, 0], 1).item()
            mx, my = mu_x[0, 0, idx].item(), mu_y[0, 0, idx].item()
            sx = sig_x[0, 0, idx].item() * temperature
            sy = sig_y[0, 0, idx].item() * temperature
            r = rho[0, 0, idx].item()
            u1, u2 = torch.randn(1).item(), torch.randn(1).item()
            dx = mx + sx * u1
            dy = my + sy * (r * u1 + math.sqrt(max(1 - r**2, 1e-8)) * u2)
            pen_up = 1.0 if torch.rand(1).item() < pen[0, 0].item() else 0.0
            pts.append([dx, dy, pen_up])
            x = torch.tensor([[[dx, dy, pen_up]]]).to(device)
    return torch.tensor(pts)


# --- Diagnostic 1: render a REAL data sample through the same plotting code ---
strokes, sentences, _ = load_data()
real = strokes[0].astype(np.float32).copy()
real[:, 1:] /= std
real = real[:, [1, 2, 0]]          # (dx, dy, pen_up), normalised
real_t = torch.from_numpy(real)

# --- Diagnostic 2: sample from several checkpoints ---
ckpts = ["10_564.pth", "10_752.pth", "10_846.pth", "10_940.pth"]

fig, axes = plt.subplots(len(ckpts) + 1, 1, figsize=(12, 2.2 * (len(ckpts) + 1)))
plot_strokes(axes[0], real_t, f"REAL data sample: {sentences[0]!r}")

for ax, ck in zip(axes[1:], ckpts):
    m = HandwritingModel().to(device)
    m.load_state_dict(torch.load(ck, map_location=device))
    seq = sample(m, steps=700, temperature=1.0)
    # report the sigma scale the model emits, to see collapse
    with torch.no_grad():
        out, _ = m(torch.zeros(1, 5, 3).to(device))
        _, _, _, sx, sy, _, pen = split_params(out)
    plot_strokes(ax, seq, f"{ck}   mean_sig={sx.mean():.3f}  mean_pen={pen.mean():.3f}")

plt.tight_layout()
plt.savefig("diag.png", dpi=130, bbox_inches="tight")
print("saved diag.png")
