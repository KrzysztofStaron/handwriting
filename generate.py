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


def sample_iter(model, text, stoi, device, temperature=1.0, max_steps=None,
                steps_per_char=50, tail_steps=12, min_steps=20, window_sharpen=2.0):
    """Generator core: yield one pen point at a time as it is drawn.

    Yields (dx, dy, pen_up, phi) per step, where phi is the (U,) window attention.
    Used both by sample() (collects everything) and by the streaming API.

    max_steps is only a RUNAWAY GUARD (in case the stop trigger never fires); normal
    generation ends well before it. When None it scales with the text length --
    400 + steps_per_char * len(text) -- so the budget tracks the input instead of a
    flat cap that silently clips long sentences. Real data runs ~22 steps/char (p95
    ~31), so steps_per_char=50 leaves comfortable headroom for any input.

    Stopping -- the paper's criterion (sec. 5.3), a TRIGGER then a short GRACE:
      - Trigger: phi at the phantom position U+1 (one past the last char) outweighs
        phi at EVERY real char -- i.e. the soft window has slid off the end of the
        text, so the model is done. This is the model's own alignment telling us it
        has finished; there is no learned end-of-sequence flag. Armed after min_steps
        so the window passing through char 0 at the very start can't trip it.
      - Grace: we then draw tail_steps MORE before stopping, because the window clears
        the last char slightly before its final strokes are drawn; without it the last
        letter can be clipped. Kept small since this trigger fires later (window fully
        PAST the end) than attention merely arriving on the last char.
    """
    # re-sharpen the soft window (beta scaling) to stop attention smearing across
    # characters late in long words; see model.compute_window. 2.0 is the sweet spot.
    model.window_sharpen = window_sharpen

    if max_steps is None:
        max_steps = 400 + steps_per_char * len(text)   # budget tracks input length

    U = len(text)
    c = torch.zeros(1, U, len(stoi)).to(device)
    c[0, torch.arange(U), [stoi[ch] for ch in text]] = 1.0   # one-hot the text (fixed)
    c_mask = torch.ones(1, U).to(device)

    x = torch.zeros(1, 1, 3).to(device)   # null start point
    hidden = None                          # fresh: pen at origin, window at char 0
    reached_end_at = None                  # step the stop was triggered

    with torch.no_grad():
        for t in range(max_steps):
            out, hidden, phi, phi_end = model(x, c, c_mask, hidden)  # ONE step; hidden carries kappa
            phi = phi[0, 0]                                  # (U,) attention this step
            phi_end = phi_end[0, 0].item()                  # phi at phantom pos U+1

            pi, mu_x, mu_y, sig_x, sig_y, rho, pen = split_params(out)

            # --- paper's probability bias b (sec. 5.4, eq. 61-62) ---
            # The public knob is `temperature` = exp(-b): temp=1 -> b=0 (unbiased,
            # eq.62/61 reduce to identity), temp->0 -> b->inf (network emits the mode).
            # A SINGLE b biases two things together, and we must apply BOTH:
            #   eq.62  the CHOICE of component:  pi <- softmax(pi_hat*(1+b)) = pi^(1+b)/Z
            #   eq.61  the VARIANCE within it:   sigma <- sigma*exp(-b)
            # The old code only did eq.61 (sigma*temperature) and sampled the component
            # from the UNbiased pi -- so low temp shrank blobs but never sharpened the
            # letter choice. That half-measure is the bug being fixed here.
            p = pi[0, 0]
            if temperature <= 0:                                   # b -> inf: pure mode
                j = int(torch.argmax(p).item())
                sigma_scale = 0.0
            else:
                b = -math.log(temperature)
                # softmax(log(pi)*(1+b)) == pi^(1+b)/Z; log(pi) differs from pi_hat only
                # by a per-step constant, which softmax cancels -- so this is exactly eq.62.
                p = torch.softmax(torch.log(p + 1e-12) * (1.0 + b), dim=-1)
                j = torch.multinomial(p, 1).item()                 # pick a Gaussian (eq.62)
                sigma_scale = temperature                          # exp(-b), eq.61
            mx, my = mu_x[0, 0, j].item(), mu_y[0, 0, j].item()
            sx = sig_x[0, 0, j].item() * sigma_scale
            sy = sig_y[0, 0, j].item() * sigma_scale
            r = rho[0, 0, j].item()

            u1, u2 = torch.randn(1).item(), torch.randn(1).item()  # sample the blob
            dx = mx + sx * u1
            dy = my + sy * (r * u1 + math.sqrt(max(1 - r**2, 1e-8)) * u2)

            pen_up = 1.0 if torch.rand(1).item() < pen[0, 0].item() else 0.0
            x = torch.tensor([[[dx, dy, pen_up]]]).to(device)    # feed the point back
            yield dx, dy, pen_up, phi.cpu().numpy()

            # --- stopping (paper sec. 5.3) ---
            # Trigger: phi at the phantom position U+1 outweighs phi at every real
            # char -- the window has slid off the end of the text. Then draw a short
            # grace (tail_steps) so the last letter's strokes finish.
            if reached_end_at is None and t >= min_steps and phi_end > phi.max().item():
                reached_end_at = t
            if reached_end_at is not None and t - reached_end_at >= tail_steps:
                break


def sample(model, text, stoi, device, temperature=1.0, max_steps=None,
           steps_per_char=50, tail_steps=12, min_steps=20, window_sharpen=2.0):
    """Generate handwriting for `text`, collecting the whole trajectory.

    Returns (seq, phis):
      seq  : (steps, 3)  the (dx, dy, pen_up) trajectory
      phis : (steps, U)  window attention over the text at each step (for Fig. 13)
    """
    pts, phis = [], []
    for dx, dy, pen_up, phi in sample_iter(model, text, stoi, device,
                                           temperature=temperature, max_steps=max_steps,
                                           steps_per_char=steps_per_char,
                                           tail_steps=tail_steps, min_steps=min_steps,
                                           window_sharpen=window_sharpen):
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
