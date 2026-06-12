import math
import torch
import numpy as np
from model import HandwritingModel, split_params

# NOTE: matplotlib and data.get_loader are imported lazily (inside the functions
# that need them) so the serving image can skip those deps entirely. Only the
# training/debug plotters and the __main__ block use them.


def load_model(device, ckpt="best-4.pth", stoi_path="stoi.pth"):
    """Load a trained synthesis model and its vocab."""
    stoi = torch.load(stoi_path)
    model = HandwritingModel(vocab_size=len(stoi)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, stoi


def sample_iter(model, text, stoi, device, temperature=1.0, max_steps=2000, tail_steps=25):
    """Generator core: yield one pen point at a time as it is drawn.

    Yields (dx, dy, pen_up, phi) per step, where phi is the (U,) window attention.
    Used both by sample() (collects everything) and by the streaming API.
    """
    U = len(text)
    c = torch.zeros(1, U, len(stoi)).to(device)
    c[0, torch.arange(U), [stoi[ch] for ch in text]] = 1.0   # one-hot the text (fixed)
    c_mask = torch.ones(1, U).to(device)

    x = torch.zeros(1, 1, 3).to(device)   # null start point
    hidden = None                          # fresh: pen at origin, window at char 0
    reached_end_at = None                  # step where attention first hit the last char

    with torch.no_grad():
        for t in range(max_steps):
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
            x = torch.tensor([[[dx, dy, pen_up]]]).to(device)    # feed the point back
            yield dx, dy, pen_up, phi.cpu().numpy()

            # Stop heuristic. The window attention phi is the reliable signal (kappa
            # is polluted by parked/unused mixture components). But don't quit the
            # instant attention reaches the last char -- that gives it ~0 steps to be
            # drawn. Instead, once it arrives, draw a grace period (~one character,
            # the paper's ~25 steps/char) so the final letter is actually rendered.
            if reached_end_at is None and phi.argmax().item() == U - 1:
                reached_end_at = t
            if reached_end_at is not None and t - reached_end_at >= tail_steps:
                break


def sample(model, text, stoi, device, temperature=1.0, max_steps=2000, tail_steps=25):
    """Generate handwriting for `text`, collecting the whole trajectory.

    Returns (seq, phis):
      seq  : (steps, 3)  the (dx, dy, pen_up) trajectory
      phis : (steps, U)  window attention over the text at each step (for Fig. 13)
    """
    pts, phis = [], []
    for dx, dy, pen_up, phi in sample_iter(model, text, stoi, device,
                                           temperature, max_steps, tail_steps):
        pts.append([dx, dy, pen_up])
        phis.append(phi)
    return torch.tensor(pts), np.array(phis)


def save_plot(seq, std, path, title):
    """Denormalise -> integrate offsets -> split on pen lifts -> draw."""
    import matplotlib.pyplot as plt
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
    import matplotlib.pyplot as plt
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
    from data import get_loader
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    model, stoi = load_model(device)
    _, std, _ = get_loader()   # normalisation factor used during training

    text = "Hello world"
    for temp in [0.0, 0.3, 0.5, 0.8]:
        seq, phis = sample(model, text, stoi, device, temperature=temp)
        save_plot(seq, std, f"sample_temp_{temp}.png", f"'{text}'  temperature={temp}")
    save_alignment(phis, text, "alignment.png")
