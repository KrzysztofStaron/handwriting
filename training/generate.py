import argparse
VERBOSE = True  # train_rl --quiet turns this off
import json
import math
import torch
import numpy as np
from pathlib import Path
from training.model import HandwritingModel, split_params

STD_PATH = Path("std.json")

# NOTE: matplotlib and data.get_loader are imported lazily (inside the functions
# that need them) so the serving image can skip those deps entirely. Only the
# training/debug plotters and the __main__ block use them.


def load_std(path=STD_PATH):
    """Load the offset normalisation constant used at training and serving time."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["std"]


def load_model(device, ckpt="models/post-trained.pth", stoi_path="stoi.pth"):
    """Load a trained synthesis model and its vocab."""
    p = Path(stoi_path)
    if p.exists():
        stoi = torch.load(stoi_path)
    else:
        stoi = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    model = HandwritingModel(vocab_size=len(stoi)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, stoi


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stop_after_point(reached_end_at, pen_up, phi_end, phi_max, t, min_steps, tail_steps):
    """Paper stop, after this point is already kept.

    reached_end_at stays None until phi at the phantom position beats every
    real character. That trigger waits until min_steps. Then the current stroke
    finishes: stop when the pen lifts, or after tail_steps if it never does.
    """
    if reached_end_at is None and t >= min_steps and phi_end > phi_max:
        reached_end_at = t
    done = reached_end_at is not None and (
        pen_up == 1.0 or t - reached_end_at >= tail_steps
    )
    return reached_end_at, done


def stop_rows(reached, take, pen_up, phi_end, phi_max, t, min_steps, tail_steps):
    """Vector form of stop_after_point. reached is -1 until the trigger.

    take is false for a row whose point was not kept this step. Those rows
    are left unchanged. done is true only for a kept point that should end
    the trajectory.
    """
    step_reached = reached.clone()
    arm = take & (step_reached < 0) & (t >= min_steps) & (phi_end > phi_max)
    step_reached = torch.where(arm, torch.full_like(step_reached, t), step_reached)
    done = take & (step_reached >= 0) & (
        (pen_up == 1.0) | ((t - step_reached) >= tail_steps)
    )
    reached = torch.where(take, step_reached, reached)
    return reached, done


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

    Stopping -- the paper's criterion (sec. 5.3), a TRIGGER then the end of the stroke:
      - Trigger: phi at the phantom position U (one past the last char, which sits
        at U-1) outweighs phi at EVERY real char. The soft window has slid off the
        text. Armed after min_steps so the window passing char 0 at the start can't
        trip it.
      - Then finish the stroke that is already on the page and stop when the pen
        lifts. A fixed tail of 12 steps kept drawing after that lift, and the start
        of the next character was read as a period, comma, or exclamation mark.
        tail_steps is only the cap for a pen that never lifts.
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

            # Stop once the window has slid off the text, but only after the pen
            # lifts, so the last letter finishes and the next character never starts.
            # tail_steps caps a pen that stays down.
            reached_end_at, done = stop_after_point(
                reached_end_at, pen_up, phi_end, float(phi.max()), t, min_steps, tail_steps)
            if done:
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


def _temperature_list(temperature, count):
    """One float per row. A single number is used for every row."""
    if isinstance(temperature, (int, float)):
        values = [float(temperature)] * count
    else:
        values = [float(value) for value in temperature]
    if len(values) != count:
        raise ValueError(f"expected {count} temperatures, got {len(values)}")
    return values


def _row_state(hidden, last_point, index):
    """One row's pen state, detached, at the step that produced last_point[index]."""
    h1, c1, s2, s3, w, kappa = hidden

    def lstm(state):
        if state is None:
            return None
        h, c = state
        return (h[:, index:index + 1].detach().clone(), c[:, index:index + 1].detach().clone())

    return {
        "h1": h1[index:index + 1].detach().clone(),
        "c1": c1[index:index + 1].detach().clone(),
        "s2": lstm(s2),
        "s3": lstm(s3),
        "w": w[index:index + 1].detach().clone(),
        "kappa": kappa[index:index + 1].detach().clone(),
        "x": last_point[index].detach().reshape(3).clone(),
    }


def stack_states(states, device):
    """Batch the per-row states saved by sample_batch."""
    def lstm(key):
        if states[0][key] is None:
            return None
        h = torch.cat([item[key][0].to(device) for item in states], dim=1)
        c = torch.cat([item[key][1].to(device) for item in states], dim=1)
        return (h, c)

    hidden = (
        torch.cat([item["h1"].to(device) for item in states], 0),
        torch.cat([item["c1"].to(device) for item in states], 0),
        lstm("s2"),
        lstm("s3"),
        torch.cat([item["w"].to(device) for item in states], 0),
        torch.cat([item["kappa"].to(device) for item in states], 0),
    )
    x = torch.stack([item["x"].to(device).reshape(3) for item in states]).view(len(states), 1, 3)
    return hidden, x


def sample_batch(model, texts, stoi, device, temperature=1.0, max_steps=None,
                 steps_per_char=50, tail_steps=12, min_steps=20, window_sharpen=2.0,
                 states=None, return_state=False):
    """Sample one trajectory per text in one autoregressive loop.

    `temperature` is a float, or one float per text. Each row uses that
    temperature, the shared sharpen, and the same stop rule as sample().
    Temperature 0 draws the mode. A row that has already stopped stays in the
    batch so the others can keep stepping, and its later points are discarded.
    `states` resumes each row from a pen state saved earlier; the returned
    points are only the new ones. With `return_state`, also return the pen
    state at the step each row stopped. That state includes `chars`, the
    window's character index at each new pen step, or -1 once the window
    has left the text.
    Returns a list of (T, 3) CPU tensors, one per text.
    """
    if not texts:
        return []
    for text in texts:
        missing = [ch for ch in text if ch not in stoi]
        if missing:
            raise ValueError(f"text has characters outside the vocab: {sorted(set(missing))}")

    model.window_sharpen = window_sharpen
    lengths = [len(text) for text in texts]
    count = len(texts)
    temperatures = _temperature_list(temperature, count)
    width = max(lengths)
    vocab = len(stoi)
    if max_steps is None:
        budgets = [400 + steps_per_char * n for n in lengths]
    else:
        budgets = [int(max_steps)] * count
    limit = max(budgets)

    c = torch.zeros(count, width, vocab, device=device)
    c_mask = torch.zeros(count, width, device=device)
    for i, text in enumerate(texts):
        index = torch.tensor([stoi[ch] for ch in text], dtype=torch.long, device=device)
        c[i, torch.arange(lengths[i], device=device), index] = 1.0
        c_mask[i, :lengths[i]] = 1.0

    if states is None:
        x = torch.zeros(count, 1, 3, device=device)
        hidden = None
    else:
        if len(states) != count:
            raise ValueError(f"expected {count} pen states, got {len(states)}")
        hidden, x = stack_states(states, device)
    active = torch.ones(count, dtype=torch.bool)
    reached = torch.full((count,), -1, dtype=torch.long)
    budget = torch.tensor(budgets, dtype=torch.long)
    points = [[] for _ in range(count)]
    saved = [None] * count
    char_rows = [[] for _ in range(count)] if return_state else None
    row_temp = torch.tensor(temperatures, dtype=torch.float32, device=device)
    last_hidden = None
    last_step = None

    def remember(mask, pen_hidden, pen_step, stopped):
        # `stopped` is the paper rule (the window passed the last character).
        # A budget cut leaves it false so a caller can resume the same text.
        for i in range(count):
            if saved[i] is None and bool(mask[i]):
                state = _row_state(pen_hidden, pen_step, i)
                state["stopped"] = bool(stopped)
                if char_rows is not None:
                    state["chars"] = list(char_rows[i])
                saved[i] = state

    with torch.no_grad():
        for t in range(limit):
            take = active & (t < budget)
            if not bool(take.any()):
                break
            out, hidden, phi, phi_end = model(x, c, c_mask, hidden)
            pi, mu_x, mu_y, sig_x, sig_y, rho, pen = split_params(out)
            pi, mu_x, mu_y = pi[:, 0], mu_x[:, 0], mu_y[:, 0]
            sig_x, sig_y, rho, pen = sig_x[:, 0], sig_y[:, 0], rho[:, 0], pen[:, 0]
            temps = row_temp.to(dtype=pi.dtype)
            cold = temps <= 0
            bias = torch.zeros(count, dtype=pi.dtype, device=device)
            warm = ~cold
            if bool(warm.any()):
                bias[warm] = -torch.log(temps[warm])
            probs = torch.softmax(torch.log(pi.clamp_min(1e-12)) * (1.0 + bias.unsqueeze(-1)), dim=-1)
            choice = torch.multinomial(probs, 1).squeeze(-1)
            if bool(cold.any()):
                choice = torch.where(cold, torch.argmax(pi, dim=-1), choice)
            sigma_scale = torch.where(cold, torch.zeros_like(temps), temps)
            picked = choice.view(count, 1)
            mx = mu_x.gather(1, picked).squeeze(1)
            my = mu_y.gather(1, picked).squeeze(1)
            sx = sig_x.gather(1, picked).squeeze(1) * sigma_scale
            sy = sig_y.gather(1, picked).squeeze(1) * sigma_scale
            corr = rho.gather(1, picked).squeeze(1)
            u1 = torch.randn(count, device=device)
            u2 = torch.randn(count, device=device)
            dx = mx + sx * u1
            dy = my + sy * (corr * u1 + torch.sqrt((1.0 - corr * corr).clamp_min(1e-8)) * u2)
            pen_up = (torch.rand(count, device=device) < pen).float()
            step = torch.stack([dx, dy, pen_up], dim=-1)
            window = None
            if char_rows is not None:
                # The character this pen step is writing. Once phi at the
                # phantom position beats every real character, the window
                # has left the text and the step belongs to no character.
                phi_row = phi[:, 0]
                on_text = phi_end[:, 0] <= phi_row.amax(dim=-1)
                window = torch.where(
                    on_text,
                    phi_row.argmax(dim=-1),
                    torch.full((count,), -1, dtype=torch.long, device=phi.device),
                ).detach().cpu()
            moved = torch.cat([
                step,
                phi_end[:, 0].unsqueeze(1),
                phi[:, 0].amax(dim=-1).unsqueeze(1),
            ], dim=1).detach().cpu()
            take = take.cpu()
            for i in range(count):
                if bool(take[i]):
                    points[i].append(moved[i, :3].clone())
                    if window is not None:
                        index = int(window[i])
                        if index >= lengths[i]:
                            index = -1
                        char_rows[i].append(index)
            reached, done = stop_rows(
                reached, take, moved[:, 2], moved[:, 3], moved[:, 4],
                t, min_steps, tail_steps)
            last_hidden, last_step = hidden, step
            remember(done, hidden, step, True)
            active = active & ~done & ((t + 1) < budget)
            x = step.view(count, 1, 3)
            if count > 1 and t > 0 and t % 200 == 0:
                if VERBOSE: print(f"      step {t}  finished {int((~active).sum())}/{count}", flush=True)

    sequences = []
    for text, pts in zip(texts, points):
        if not pts:
            raise RuntimeError(f"model produced an empty trajectory for {text!r}")
        sequences.append(torch.stack(pts, dim=0))
    if not return_state:
        return sequences
    missing = torch.tensor([item is None for item in saved])
    if bool(missing.any()):
        if last_hidden is None:
            raise RuntimeError("sampler stopped before any pen step")
        remember(missing, last_hidden, last_step, False)
    return sequences, saved


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
    parser = argparse.ArgumentParser(description="Generate handwriting PNGs from a checkpoint.")
    parser.add_argument("text", nargs="?", default="Hello world", help="Text to synthesize")
    parser.add_argument(
        "-t", "--temperature", type=float, default=None,
        help="Sampling temperature (default: sweep 0.0, 0.3, 0.5, 0.8)",
    )
    parser.add_argument("-o", "--output", help="Output PNG path (single temperature only)")
    parser.add_argument("--alignment", help="Save window-alignment plot to this path")
    parser.add_argument("--ckpt", default="models/post-trained.pth", help="Model checkpoint")
    parser.add_argument("--device", default=default_device(), help="Torch device")
    args = parser.parse_args()

    model, stoi = load_model(args.device, ckpt=args.ckpt)
    std = load_std()

    if args.temperature is not None:
        seq, phis = sample(model, args.text, stoi, args.device, temperature=args.temperature)
        out = args.output or f"sample_temp_{args.temperature}.png"
        save_plot(seq, std, out, f"'{args.text}'  temperature={args.temperature}")
        if args.alignment:
            save_alignment(phis, args.text, args.alignment)
    else:
        last_phis = None
        for temp in [0.0, 0.3, 0.5, 0.8]:
            seq, phis = sample(model, args.text, stoi, args.device, temperature=temp)
            save_plot(seq, std, f"sample_temp_{temp}.png", f"'{args.text}'  temperature={temp}")
            last_phis = phis
        if last_phis is not None:
            save_alignment(last_phis, args.text, args.alignment or "alignment.png")
