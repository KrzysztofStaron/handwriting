"""Post-training pieces shared by the recognizer loop and the human loop.

The policy is the Graves handwriting model. A trajectory is a stroke sequence
sampled the same way as generate.py (temperature bias and window sharpening
included). The reward is one number for the whole trajectory, so the update is
sequence-level REINFORCE:

    grad = advantage * grad mean_t log pi(a_t | history, text)

Length-normalized, not the raw sum, so a long line does not take a step
proportional to its length. When several samples share a prompt, the advantage
subtracts that prompt's mean reward. A KL penalty toward a frozen copy of
models/best-4.pth is folded into the advantage with a stop-gradient (the
InstructGPT form). The anchor is the base model; other checkpoints are
experiment runs and this module refuses to write best-4.pth.

Recognizer reward, from the character error rate of TrOCR-Large:

    linear    1 - CER
    exp       exp(-CER)                         default
    soft      1 if CER <= tau, else (1 - CER) / (1 - tau) until CER = 1, then 0

CER is capped at 2 before that map.
"""

import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from data import collate
from generate import _row_state, load_model, sample_batch, stack_states
from model import split_params
from vcer.reward import tighten_punctuation

BASE_CKPT = "models/best-4.pth"
CER_CAP = 2.0

_QUOTES = str.maketrans({
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2014": "-", "\u2013": "-",
})


def refuse_overwrite_base(path):
    """best-4.pth is the base model. Post-training writes somewhere else."""
    if Path(path).expanduser().resolve() == Path(BASE_CKPT).resolve():
        raise SystemExit(f"refusing to write {path}: {BASE_CKPT} is the base model")


def save_policy(model, path):
    refuse_overwrite_base(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def draw_temperatures(count, low, high):
    """Truncated normal on [low, high], centered on the midpoint.

    The interval is six standard deviations each side of the midpoint, so the
    density is piled on the middle and the ends almost never appear. Samples
    that fall outside are rejected. A zero-width range returns that temperature.
    """
    low = float(low)
    high = float(high)
    if count < 0:
        raise ValueError("count must be >= 0")
    if high < low:
        raise ValueError("temperature max is below min")
    if count == 0:
        return []
    if low == high:
        return [low] * count
    mean = (low + high) / 2.0
    sigma = (high - low) / 12.0
    draws = []
    while len(draws) < count:
        value = random.gauss(mean, sigma)
        if low <= value <= high:
            draws.append(value)
    return draws


def normalize_text(text):
    """Case-sensitive, with whitespace collapsed and curly quotes straightened.

    TrOCR sometimes emits typographic quotes for the straight ones in the prompt.
    It also writes a space before punctuation (" :"). Neither is a handwriting
    error. A space after punctuation is kept.
    """
    if text is None:
        return ""
    collapsed = " ".join(str(text).translate(_QUOTES).strip().split())
    return tighten_punctuation(collapsed)


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def cer(reference, hypothesis):
    """Character error rate, edit distance divided by the reference length.

    Can exceed 1 when the hypothesis is longer than the reference. An empty
    reference scores 0 against an empty hypothesis and 1 otherwise.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def string_reward(reference, hypothesis, alpha=0.8):
    """Plain CER plus an exact-match bonus.

    `0.8 * (1 - min(CER, 1)) + 0.2 * exact`. A lookalike mistake costs the
    same as any other edit. CER can exceed 1; the bonus uses the capped value.
    """
    value = cer(reference, hypothesis)
    exact = normalize_text(reference) == normalize_text(hypothesis)
    capped = min(max(value, 0.0), 1.0)
    reward = alpha * (1.0 - capped) + (1.0 - alpha) * float(exact)
    return value, reward


def blend_string_reward(cer_value, wcer_value):
    """Plain CER takes three quarters of the string score, visual CER one quarter.

    `0.75 * (1 - min(CER, 1)) + 0.25 * (1 - min(wCER, 1))`.
    Confidence then keeps 80% of this and adds itself at 0.2, so the full
    score is 60% CER, 20% vCER, and 20% confidence. Straightness is subtracted.
    A distance above 1 is capped, so a long hypothesis cannot push either
    term below zero.
    """
    cer_c = min(max(float(cer_value), 0.0), 1.0)
    wcer_c = min(max(float(wcer_value), 0.0), 1.0)
    return 0.75 * (1.0 - cer_c) + 0.25 * (1.0 - wcer_c)


def reward_from_cer(value, kind="exp", tau=0.1):
    """Map a (possibly capped) CER to a scalar reward.

    linear  1 - CER. Proportional, and negative once CER exceeds 1.
    exp     exp(-CER). Always in (0, 1]. The first errors cost more than later ones.
    soft    Thresholded. CER at or below tau scores 1, so recognizer noise and a
            slipped character on a long line are not punished. Above tau the score
            falls linearly to 0 at CER = 1, and stays 0 after that.
    """
    if kind == "linear":
        return 1.0 - value
    if kind == "exp":
        return math.exp(-value)
    if kind == "soft":
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be between 0 and 1")
        if value <= tau:
            return 1.0
        if value >= 1.0:
            return 0.0
        return (1.0 - value) / (1.0 - tau)
    raise ValueError(f"unknown reward {kind!r} (expected linear, exp, soft)")


def token_confidence(scores, sequences, skip_ids):
    """Geometric mean of the probabilities TrOCR assigned to the tokens it chose.

    `scores[t]` is the logit row that picked `sequences[:, t + 1]`. End-of-sequence
    and padding tokens are left out, so a short confident stop does not look like
    a clear transcription. No content tokens scores 0.
    """
    if not scores:
        return [0.0] * len(sequences)
    steps = min(len(scores), sequences.shape[1] - 1)
    confidences = []
    for row in range(sequences.shape[0]):
        chosen = []
        for step in range(steps):
            token = int(sequences[row, step + 1])
            if token in skip_ids:
                continue
            log_probs = torch.log_softmax(scores[step][row].float(), dim=-1)
            chosen.append(log_probs[token])
        if not chosen:
            confidences.append(0.0)
        else:
            confidences.append(float(torch.exp(torch.stack(chosen).mean())))
    return confidences


def mix_confidence(reward, confidence, weight):
    """Keep the string score in charge and add TrOCR confidence on top.

    `weight` is confidence's share of the reward. The rest stays the string
    reward. Confidence is in [0, 1], so this term never subtracts.
    """
    if confidence is None or weight <= 0:
        return float(reward)
    return (1.0 - weight) * float(reward) + weight * float(confidence)


def line_straightness(seq, std, side=0.25):
    """How far the writing on the right sits from the writing on the left. 0 is level.

    Each end is the band from the 10th to the 90th percentile of the ink there.
    The score is how far the right band has moved from the left band, divided
    by the height of those bands. A small rise is a small score. The middle
    band is compared with the two ends the same way, so an arch counts too.
    The total stops at 1. A short or empty stroke scores 0.
    """
    if torch.is_tensor(seq):
        seq = seq.detach().cpu().numpy()
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[0] < 8 or seq.shape[1] < 2 or not np.isfinite(seq).all():
        return 0.0
    xy = np.cumsum(seq[:, :2] * float(std), axis=0)
    xs = xy[:, 0]
    ys = xy[:, 1]
    width = float(xs.max() - xs.min())
    if width < 1e-6:
        return 0.0
    side = float(side)
    left = _ink_band(xs, ys, float(xs.min()), float(xs.min()) + side * width)
    right = _ink_band(xs, ys, float(xs.max()) - side * width, float(xs.max()))
    if left is None or right is None:
        return 0.0
    height = 0.5 * ((left[1] - left[0]) + (right[1] - right[0]))
    height = max(height, 1e-3)
    tilt = _band_shift(left, right) / height
    middle = _ink_band(
        xs, ys,
        float(xs.min()) + (0.5 - side / 2.0) * width,
        float(xs.min()) + (0.5 + side / 2.0) * width,
    )
    arch = 0.0
    if middle is not None:
        ends = (
            0.5 * (left[0] + right[0]),
            0.5 * (left[1] + right[1]),
        )
        arch = _band_shift(middle, ends) / height
    return float(min(1.0, tilt + arch))


def _ink_band(xs, ys, x0, x1):
    """10th and 90th percentile of y for the ink between x0 and x1."""
    chosen = ys[(xs >= x0) & (xs <= x1)]
    if chosen.size < 4:
        return None
    return float(np.percentile(chosen, 10)), float(np.percentile(chosen, 90))


def _band_shift(one, other):
    """How far two ink bands are from each other, top edge and bottom edge."""
    return 0.5 * (abs(one[0] - other[0]) + abs(one[1] - other[1]))


def spacing_inconsistency(seq, std, text):
    """How uneven the gaps between words are. 0 is even. Caps at 1.

    The text says how many word gaps there should be. Those gaps are the
    widest blanks between pen strokes, one per space. A gap inside a word is
    smaller and is left out. The score is the mean squared distance of those
    widths from their median, divided by that median, so a small wobble is
    almost nothing and a mix of tiny and huge spaces goes to 1. Fewer than
    two word gaps cannot be uneven.
    """
    words = str(text or "").split()
    needed = len(words) - 1
    if needed < 2:
        return 0.0
    gaps = _word_gap_widths(seq, std)
    if not gaps:
        return 0.0
    chosen = sorted(gaps, reverse=True)[:needed]
    if len(chosen) < needed:
        chosen.extend(0.0 for _ in range(needed - len(chosen)))
    widths = np.asarray(chosen, dtype=np.float64)
    typical = float(np.median(widths))
    if typical < 1e-3:
        return 0.0 if float(widths.max()) < 1e-3 else 1.0
    relative = (widths - typical) / typical
    return float(min(1.0, np.mean(relative ** 2)))


def _word_gap_widths(seq, std):
    """Positive horizontal blanks between successive pen strokes."""
    if torch.is_tensor(seq):
        seq = seq.detach().cpu().numpy()
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[0] < 2 or seq.shape[1] < 3 or not np.isfinite(seq).all():
        return []
    xy = np.cumsum(seq[:, :2] * float(std), axis=0)
    pen = seq[:, 2] > 0.5
    spans = []
    start = 0
    for index in range(len(seq)):
        if pen[index] or index == len(seq) - 1:
            xs = xy[start:index + 1, 0]
            if len(xs):
                spans.append((float(xs.min()), float(xs.max())))
            start = index + 1
    gaps = []
    for (left_min, left_max), (right_min, right_max) in zip(spans, spans[1:]):
        gap = right_min - left_max
        if gap > 0.0:
            gaps.append(gap)
    return gaps


def cer_and_reward(reference, hypothesis, kind="exp", tau=0.1, cer_cap=CER_CAP):
    """Return (raw CER, reward). The reward sees CER capped at cer_cap."""
    raw = cer(reference, hypothesis)
    return raw, reward_from_cer(min(raw, cer_cap), kind=kind, tau=tau)


def apply_temperature(pi, sig_x, sig_y, temperature):
    """Graves probability bias (sec. 5.4). temperature = exp(-b).

    Mixture weights become pi^(1+b) / Z and each std is multiplied by temperature.
    This is the density generate.sample_iter actually draws from, so the REINFORCE
    score has to use it too. temperature <= 0 is a point mass and has no density.
    A 1-D tensor gives each batch row its own temperature.
    """
    if torch.is_tensor(temperature):
        temp = temperature.to(device=pi.device, dtype=pi.dtype)
        if torch.any(temp <= 0):
            raise ValueError("temperature must be > 0; 0 is a point mass and has no density")
        while temp.ndim < pi.ndim:
            temp = temp.unsqueeze(-1)
        bias = -torch.log(temp)
        log_pi = torch.log(pi.clamp_min(1e-12)) * (1.0 + bias)
        return torch.softmax(log_pi, dim=-1), sig_x * temp, sig_y * temp
    if temperature <= 0:
        raise ValueError("temperature must be > 0; 0 is a point mass and has no density")
    if abs(temperature - 1.0) < 1e-8:
        return pi, sig_x, sig_y
    b = -math.log(temperature)
    log_pi = torch.log(pi.clamp_min(1e-12)) * (1.0 + b)
    return torch.softmax(log_pi, dim=-1), sig_x * temperature, sig_y * temperature


def _log_gaussian_2d(dx, dy, mu_x, mu_y, sig_x, sig_y, rho):
    """Log of model.gaussian_2d. Log-domain so a tight component does not overflow."""
    one_m_rho2 = (1.0 - rho ** 2).clamp(min=1e-6)
    z1 = (dx - mu_x) ** 2 / sig_x ** 2
    z2 = (dy - mu_y) ** 2 / sig_y ** 2
    z3 = 2.0 * rho * (dx - mu_x) * (dy - mu_y) / (sig_x * sig_y)
    log_norm = (
        -math.log(2.0 * math.pi)
        - torch.log(sig_x)
        - torch.log(sig_y)
        - 0.5 * torch.log(one_m_rho2)
    )
    return log_norm - (z1 + z2 - z3) / (2.0 * one_m_rho2)


def step_logprob(out, y, temperature=1.0):
    """Per-step log pi(y_t | model output). out (B, T, 121), y (B, T, 3) -> (B, T).

    The action is the pen offset and the pen-up bit. The mixture component is a
    latent, so this is the marginal density, which is what was sampled.
    """
    pi, mu_x, mu_y, sig_x, sig_y, rho, pen = split_params(out)
    pi, sig_x, sig_y = apply_temperature(pi, sig_x, sig_y, temperature)
    dx = y[:, :, 0].unsqueeze(-1)
    dy = y[:, :, 1].unsqueeze(-1)
    log_xy = torch.logsumexp(torch.log(pi.clamp_min(1e-12)) + _log_gaussian_2d(
        dx, dy, mu_x, mu_y, sig_x, sig_y, rho), dim=-1)
    pen_up = y[:, :, 2]
    log_pen = (
        pen_up * torch.log(pen.clamp_min(1e-8))
        + (1.0 - pen_up) * torch.log((1.0 - pen).clamp_min(1e-8))
    )
    return log_xy + log_pen


def _replay_batch(seqs, texts, stoi):
    """Teacher-forced (input, target, char index) rows for collate."""
    rows = []
    for seq, text in zip(seqs, texts):
        missing = [ch for ch in text if ch not in stoi]
        if missing:
            raise ValueError(f"text has characters outside the vocab: {sorted(set(missing))}")
        seq = seq.detach().float().cpu()
        if seq.ndim != 2 or seq.shape[-1] != 3 or seq.shape[0] == 0:
            raise ValueError("seq must be (T, 3) with T > 0")
        start = torch.zeros(1, 3)
        inp = torch.cat([start, seq[:-1]], dim=0)
        idx = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
        rows.append((inp, seq, idx))
    return rows


def trajectory_logprobs(model, seqs, texts, stoi, device, temperature, sharpen,
                        temperatures=None, return_steps=False):
    """Teacher-forced sum of log pi for a batch. Returns (logp (B,), length (B,)).

    Sampling feeds each point back as the next input. Replaying those points
    through forward() is the same autoregressive density, and it is the graph
    REINFORCE differentiates. `sharpen` has to match the value used while
    sampling: it scales beta inside compute_window, so it is part of the policy.
    Pass `temperatures` for a different temperature on each row; otherwise every
    row uses `temperature`. Shorter rows are padded. The mask drops those steps,
    and the window's phantom position stays on the real last character.
    """
    if temperatures is None:
        temperatures = [float(temperature)] * len(seqs)
    if len(temperatures) != len(seqs):
        raise ValueError("one temperature per trajectory")
    if any(float(value) <= 0 for value in temperatures):
        raise ValueError("temperature must be > 0; 0 is a point mass and has no density")
    x, y, mask, c, c_mask = collate(_replay_batch(seqs, texts, stoi), vocab_size=len(stoi))
    x, y, mask, c, c_mask = [t.to(device) for t in (x, y, mask, c, c_mask)]
    model.window_sharpen = sharpen
    out, _, _, _ = model(x, c, c_mask)
    temp = torch.tensor([float(value) for value in temperatures], dtype=out.dtype, device=device)
    per = step_logprob(out, y, temperature=temp) * mask
    if return_steps:
        return per, mask
    return per.sum(dim=1), mask.sum(dim=1)


def _detach_hidden(hidden):
    h1, c1, s2, s3, w, kappa = hidden

    def lstm(state):
        if state is None:
            return None
        h, c = state
        return (h.detach(), c.detach())

    return (h1.detach(), c1.detach(), lstm(s2), lstm(s3), w.detach(), kappa.detach())


def _replay_hidden(model, seq, text, stoi, device, sharpen):
    """Pen state after teacher-forcing `seq` under `text`. The next input is seq[-1]."""
    model.window_sharpen = sharpen
    x, _y, _mask, c, c_mask = collate(
        _replay_batch([seq], [text], stoi), vocab_size=len(stoi))
    x, c, c_mask = x.to(device), c.to(device), c_mask.to(device)
    with torch.no_grad():
        _out, hidden, _phi, _phi_end = model(x, c, c_mask)
    return _detach_hidden(hidden)


def _advance_hidden(model, hidden, last_point, seq, text, stoi, device, sharpen):
    """Carry `hidden` across `seq`. `last_point` is the pen step already on the page."""
    model.window_sharpen = sharpen
    seq = seq.detach().float().cpu()
    start = last_point.detach().float().cpu().reshape(1, 3)
    inp = torch.cat([start, seq[:-1]], dim=0)
    idx = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    x, _y, _mask, c, c_mask = collate([(inp, seq, idx)], vocab_size=len(stoi))
    x, c, c_mask = x.to(device), c.to(device), c_mask.to(device)
    with torch.no_grad():
        _out, nxt, _phi, _phi_end = model(x, c, c_mask, hidden)
    return _detach_hidden(nxt)


def history_state(model, segments, stoi, device, sharpen):
    """Pen state after each past word, under the text that word was sampled with."""
    text0, seq0 = segments[0]
    hidden = _replay_hidden(model, seq0, text0, stoi, device, sharpen)
    last = seq0[-1]
    for text, seq in segments[1:]:
        hidden = _advance_hidden(model, hidden, last, seq, text, stoi, device, sharpen)
        last = seq[-1]
    return _row_state(hidden, last.detach().cpu().unsqueeze(0), 0)


def _segment_rows(last_points, news, texts, stoi):
    """Teacher-forced rows whose first input is the last point already drawn."""
    rows = []
    for last, new, text in zip(last_points, news, texts):
        missing = [ch for ch in text if ch not in stoi]
        if missing:
            raise ValueError(f"text has characters outside the vocab: {sorted(set(missing))}")
        new = new.detach().float().cpu()
        if new.ndim != 2 or new.shape[-1] != 3 or new.shape[0] == 0:
            raise ValueError("new segment must be (T, 3) with T > 0")
        start = last.detach().float().cpu().reshape(1, 3)
        inp = torch.cat([start, new[:-1]], dim=0)
        idx = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
        rows.append((inp, new, idx))
    return rows


def logprobs_from_states(model, states, news, texts, stoi, device, temperature, sharpen,
                         temperatures=None, return_steps=False):
    """Log-density of new points given the pen states they were sampled from.

    The state is a constant, so the gradient stays on the word just added.
    """
    if temperatures is None:
        temperatures = [float(temperature)] * len(news)
    if not (len(states) == len(news) == len(texts) == len(temperatures)):
        raise ValueError("one state, text, and temperature per new segment")
    if any(float(value) <= 0 for value in temperatures):
        raise ValueError("temperature must be > 0; 0 is a point mass and has no density")
    hidden, _x = stack_states(states, device)
    hidden = _detach_hidden(hidden)
    last_points = [item["x"] for item in states]
    x, y, mask, c, c_mask = collate(
        _segment_rows(last_points, news, texts, stoi), vocab_size=len(stoi))
    x, y, mask, c, c_mask = [t.to(device) for t in (x, y, mask, c, c_mask)]
    model.window_sharpen = sharpen
    out, _, _, _ = model(x, c, c_mask, hidden)
    temp = torch.tensor([float(value) for value in temperatures], dtype=out.dtype, device=device)
    per = step_logprob(out, y, temperature=temp) * mask
    if return_steps:
        return per, mask
    return per.sum(dim=1), mask.sum(dim=1)


def trajectory_logprob(model, seq, text, stoi, device, temperature, sharpen):
    """Teacher-forced sum of log pi(seq | text), and the number of real steps."""
    total, length = trajectory_logprobs(
        model, [seq], [text], stoi, device, temperature, sharpen)
    return total[0], length[0]


def reinforce_loss(logp_sum, logp_ref_sum, length, advantage, kl_coef):
    """Scalar loss to minimize for one trajectory.

    advantage is treated as a constant (the reward already happened).
    kl_coef * per-step KL is subtracted from it, with the KL stopped so the
    backward is only the score function. The direct gradient of the KL has
    expectation zero under the policy, so leaving it out does not bias the update.
    """
    if not torch.is_tensor(length):
        length = torch.tensor(float(length), dtype=logp_sum.dtype, device=logp_sum.device)
    length = length.clamp(min=1)
    mean_logp = logp_sum / length
    step_kl = (logp_sum.detach() - logp_ref_sum.detach()) / length
    shaped = advantage - kl_coef * step_kl
    if torch.is_tensor(shaped):
        shaped = shaped.detach()
    return -(shaped * mean_logp)


def token_reinforce_loss(per, per_ref, advantages, kl_coef):
    """Mean over pen steps of -(A_t - kl * mean_kl) * log π_t.

    `per` and `per_ref` are the step log-densities of one segment, with no
    padding. `advantages` is a scalar or one value per step. A constant
    advantage matches `reinforce_loss` on the sums. The KL term is the mean
    over the segment and is stopped, same as the scalar loss.
    """
    per = per.reshape(-1)
    per_ref = per_ref.reshape(-1)
    if per.shape[0] != per_ref.shape[0] or per.shape[0] < 1:
        raise ValueError("one reference log-density per pen step")
    if isinstance(advantages, (list, tuple)):
        adv = torch.tensor(list(advantages), dtype=per.dtype, device=per.device)
    elif torch.is_tensor(advantages) and advantages.ndim > 0:
        adv = advantages.reshape(-1).to(dtype=per.dtype, device=per.device)
    else:
        adv = per.new_full(per.shape, float(advantages))
    if adv.shape[0] != per.shape[0]:
        raise ValueError("one advantage per pen step")
    adv = adv.detach()
    length = per.new_tensor(float(per.shape[0]))
    step_kl = (per.detach().sum() - per_ref.detach().sum()) / length
    shaped = adv - kl_coef * step_kl.detach()
    return -((shaped * per).sum() / length)


def group_advantages(rewards, k):
    """Subtract each prompt's mean reward. rewards are grouped in blocks of k."""
    if k < 2:
        raise ValueError("a within-group baseline needs k >= 2")
    if len(rewards) % k != 0:
        raise ValueError("number of rewards must be a multiple of k")
    out = []
    for i in range(0, len(rewards), k):
        group = rewards[i:i + k]
        base = sum(group) / k
        out.extend(r - base for r in group)
    return out


def advantages(rewards, k, ema):
    """Return (advantages, new_ema).

    k > 1 compares the samples of one prompt with each other, so a hard word
    is not punished just for being hard. k == 1 falls back to an EMA of the
    reward. The first batch centers on its own mean.
    """
    if k > 1:
        return group_advantages(rewards, k), ema
    mean = sum(rewards) / len(rewards)
    baseline = mean if ema is None else ema
    new_ema = mean if ema is None else (0.9 * ema + 0.1 * mean)
    return [r - baseline for r in rewards], new_ema


# Eight lines at a time keeps the backward graph on a laptop while the
# sampler still draws the whole update in one loop.
_GRAD_CHUNK = 8
# A colder gaussian has a larger score. Scaling the loss by T / 0.5 keeps a
# draw at the old fixed temperature the same size, and stops a near-zero draw
# from owning the whole update.
_TEMP_REFERENCE = 0.5


def reinforce_step(model, ref, rollouts, advs, stoi, device, temperature,
                   sharpen, kl_coef, optimizer, clip, apply=True, normalizer=None):
    """On-policy update over a list of {text, seq} rollouts.

    The score-function backward runs in chunks of variable-length lines. Each
    chunk is one padded forward. Gradients are averaged over `normalizer`
    rows, or over this call when that is omitted, including rows that were
    skipped. `apply` False leaves the gradients in place so a caller can
    backward every chunk of one line and then take a single step.
    A rollout may carry `adv_steps`, one advantage per new pen step. Without
    it the scalar advantage is used on every step.
    """
    if apply:
        optimizer.zero_grad(set_to_none=True)
    n = len(rollouts) if normalizer is None else int(normalizer)
    if n < 1:
        n = 1
    used = 0
    loss_acc = 0.0
    kl_acc = 0.0
    traces = []
    pending = []
    model.train()
    for index, (item, adv) in enumerate(zip(rollouts, advs)):
        adv_f = float(adv)
        temp = float(item.get("temperature", temperature))
        trace = {
            "advantage": adv_f,
            "temperature": temp,
            "logp": None,
            "kl": None,
            "length": None,
            "skipped": None,
        }
        traces.append(trace)
        if item.get("skip"):
            reason = item.get("skip")
            trace["skipped"] = reason if isinstance(reason, str) else "rejected"
            continue
        if not math.isfinite(adv_f):
            trace["skipped"] = "nonfinite_advantage"
            continue
        if temp <= 0:
            # The mode has no density. Its reward still sits in the baseline.
            trace["skipped"] = "zero_temperature"
            continue
        pending.append(index)
    for start in range(0, len(pending), _GRAD_CHUNK):
        chosen = pending[start:start + _GRAD_CHUNK]
        seqs = [rollouts[index]["seq"] for index in chosen]
        texts = [rollouts[index]["text"] for index in chosen]
        temps = [traces[index]["temperature"] for index in chosen]
        continued = rollouts[chosen[0]].get("state") is not None
        if any((rollouts[index].get("state") is not None) != continued for index in chosen):
            raise RuntimeError("a gradient chunk mixed a fresh line with a continued word")
        if continued:
            pen_states = [rollouts[index]["state"] for index in chosen]
            per, mask = logprobs_from_states(
                model, pen_states, seqs, texts, stoi, device,
                temperature, sharpen, temperatures=temps, return_steps=True)
        else:
            per, mask = trajectory_logprobs(
                model, seqs, texts, stoi, device, temperature, sharpen,
                temperatures=temps, return_steps=True)
        if kl_coef:
            with torch.no_grad():
                if continued:
                    ref_states = [
                        history_state(ref, rollouts[index]["history"], stoi, device, sharpen)
                        for index in chosen
                    ]
                    per_ref, _ = logprobs_from_states(
                        ref, ref_states, seqs, texts, stoi, device,
                        temperature, sharpen, temperatures=temps, return_steps=True)
                else:
                    per_ref, _ = trajectory_logprobs(
                        ref, seqs, texts, stoi, device, temperature, sharpen,
                        temperatures=temps, return_steps=True)
        else:
            per_ref = torch.zeros_like(per)
        losses = []
        for local, index in enumerate(chosen):
            trace = traces[index]
            item = rollouts[index]
            real = int(mask[local].sum().item())
            if real < 1:
                trace["skipped"] = "empty"
                continue
            step_logp = per[local, :real]
            step_ref = per_ref[local, :real]
            if not torch.isfinite(step_logp).all() or not torch.isfinite(step_ref).all():
                print(f"  non-finite logprob for {item['text']!r}, skipping", flush=True)
                trace["skipped"] = "nonfinite_logprob"
                continue
            advantages = item.get("adv_steps")
            if advantages is None:
                advantages = trace["advantage"]
            loss = token_reinforce_loss(step_logp, step_ref, advantages, kl_coef)
            scale = trace["temperature"] / _TEMP_REFERENCE
            losses.append(loss * scale / n)
            used += 1
            loss_acc += float(loss.detach())
            total = step_logp.detach().sum()
            total_ref = step_ref.detach().sum()
            trace["logp"] = float(total)
            trace["length"] = real
            if kl_coef:
                step_kl = float((total - total_ref) / real)
                kl_acc += step_kl
                trace["kl"] = step_kl
            else:
                trace["kl"] = 0.0
        if losses:
            torch.stack(losses).sum().backward()
    if not apply:
        return {
            "loss": loss_acc / used if used else float("nan"),
            "kl": kl_acc / used if used else float("nan"),
            "used": used,
            "grad": float("nan"),
            "stepped": False,
            "traces": traces,
        }
    grad_norm = 0.0
    stepped = False
    if used:
        trainable = [p for p in model.parameters() if p.requires_grad]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, clip))
        if math.isfinite(grad_norm):
            optimizer.step()
            stepped = True
        else:
            print("  non-finite gradient, skipping the update", flush=True)
            optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss_acc / used if used else float("nan"),
        "kl": kl_acc / used if used else float("nan"),
        "used": used,
        "grad": grad_norm,
        "stepped": stepped,
        "traces": traces,
    }


def load_policy_and_ref(ckpt, ref, device):
    """Load the policy and a frozen KL anchor. The anchor is not trained."""
    for label, path in (("policy", ckpt), ("KL anchor", ref)):
        if not Path(path).exists():
            raise SystemExit(f"missing {label} checkpoint: {path}")
    model, stoi = load_model(device, ckpt=ckpt)
    ref_model, _ = load_model(device, ckpt=ref)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()
    return model, ref_model, stoi


def rollout(model, text, stoi, device, temperature, sharpen):
    """One trajectory from the same sampler generate.py uses. Returns (T, 3) on CPU."""
    return rollout_batch(model, [text], stoi, device, temperature, sharpen)[0]


def rollout_batch(model, texts, stoi, device, temperature, sharpen):
    """Every text in one batched sampler loop. Each result is (T, 3) on CPU.

    `temperature` is one float, or one float per text.
    """
    training = model.training
    model.eval()
    seqs = sample_batch(
        model, texts, stoi, device,
        temperature=temperature, window_sharpen=sharpen,
    )
    if training:
        model.train()
    out = []
    for text, seq in zip(texts, seqs):
        seq = seq.detach().float().cpu()
        if seq.ndim != 2 or seq.shape[0] == 0:
            raise RuntimeError(f"model produced an empty trajectory for {text!r}")
        out.append(seq)
    return out


def load_prompts(path, stoi, min_chars, max_chars):
    vocab = set(stoi)
    prompts = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not (min_chars <= len(text) <= max_chars):
            continue
        if any(ch not in vocab for ch in text):
            continue
        prompts.append(text)
    if not prompts:
        raise SystemExit(
            f"no prompts in {path} with length in [{min_chars}, {max_chars}] "
            "and characters inside the model vocab"
        )
    return prompts


def resolve_prompts(text, path, stoi, min_chars, max_chars):
    if text:
        missing = sorted({ch for ch in text if ch not in stoi})
        if missing:
            raise SystemExit(f"--text has characters outside the vocab: {missing}")
        return [text]
    return load_prompts(path, stoi, min_chars, max_chars)


def word_prefixes(text):
    """Cumulative words of a line. 'john is' becomes ['john', 'john is']."""
    words = str(text).split()
    if not words:
        raise ValueError("prompt has no words")
    return [" ".join(words[:end]) for end in range(1, len(words) + 1)]


def select_top(rewards, keep):
    """Indices of the highest rewards. Ties keep the earlier index."""
    if keep < 1 or keep > len(rewards):
        raise ValueError("keep must be between 1 and the number of rewards")
    order = sorted(range(len(rewards)), key=lambda i: (-float(rewards[i]), i))
    return order[:keep]


def beam_factors(width):
    """Divisors of the beam width, from 1 up to the width itself.

    A factor f is one way to grow the next word: keep width/f prefixes and
    continue each of them f times. 16 gives 1, 2, 4, 8, 16. 9 gives 1, 3, 9.
    """
    width = int(width)
    if width < 1:
        raise ValueError("beam width must be at least 1")
    return [factor for factor in range(1, width + 1) if width % factor == 0]


def default_branches(n_words, width):
    """How many drawings each segment of a line splits into.

    A play line is about 9 words, so 16 finished lines can fork twice and
    still leave a few words in each segment. That tree is 4, then 2, then 2.
    A line under 6 words has room for one fork: 4 roots, each continued so
    the leaves still add up to the width. One word has no boundary to fork
    at, so every drawing is its own line.
    """
    n_words = int(n_words)
    width = int(width)
    if n_words < 1 or width < 1:
        raise ValueError("a tree needs at least one word and one drawing")
    if n_words == 1 or width == 1:
        return (width,)
    if n_words >= 6 and width % 8 == 0 and width >= 16:
        tail = width // 8
        if tail >= 2:
            return (4, 2, tail)
    if width % 4 == 0 and width // 4 >= 2:
        return (4, width // 4)
    root = _balanced_root(width)
    if root == width:
        return (width,)
    return (root, width // root)


def _balanced_root(width):
    """Divisor of width closest to its square root."""
    best = 1
    target = width ** 0.5
    for factor in range(1, width + 1):
        if width % factor:
            continue
        if abs(factor - target) < abs(best - target):
            best = factor
    return best


def even_ends(n_words, segments):
    """Word counts at which each segment finishes. The last one is the whole line."""
    n_words = int(n_words)
    segments = int(segments)
    if segments < 1 or segments > n_words:
        raise ValueError("segments must fit in the words")
    ends = []
    for index in range(1, segments):
        cut = (n_words * index) // segments
        if ends and cut <= ends[-1]:
            cut = ends[-1] + 1
        ends.append(cut)
    if ends and ends[-1] >= n_words:
        raise ValueError("the last segment would be empty")
    ends.append(n_words)
    return ends


def parse_tree(text):
    """'4x2x2' becomes (4, 2, 2). 'auto' and an empty string mean the default."""
    raw = str(text).strip().lower().replace("×", "x").replace("*", "x")
    if not raw or raw == "auto":
        return None
    try:
        branches = tuple(int(part) for part in raw.split("x"))
    except ValueError as exc:
        raise ValueError(f"tree must look like 4x2x2, got {text!r}") from exc
    if not branches or any(branch < 1 for branch in branches):
        raise ValueError(f"tree must look like 4x2x2, got {text!r}")
    return branches


def tree_plan(n_words, width, branches=None):
    """Segment ends and how many continuations each segment draws.

    Each entry is {"end", "copies"}. `end` is how many words are on the page
    when that segment finishes. The copies multiply to `width`.
    """
    n_words = int(n_words)
    width = int(width)
    if branches is None:
        branches = default_branches(n_words, width)
    else:
        branches = tuple(int(branch) for branch in branches)
    product = 1
    for branch in branches:
        if branch < 1:
            raise ValueError("each branch must be at least 1")
        product *= branch
    if product != width:
        raise ValueError(f"branches {branches} multiply to {product}, not {width}")
    if len(branches) > n_words:
        raise ValueError("more forks than words")
    ends = even_ends(n_words, len(branches))
    return [{"end": end, "copies": copies} for end, copies in zip(ends, branches)]


def tree_label(plan):
    return "x".join(str(stage["copies"]) for stage in plan)


def _leaves_under(node):
    children = node.get("children") or []
    if not children:
        yield node
        return
    for child in children:
        yield from _leaves_under(child)


def chunk_advantages(leaves):
    """Advantage of every node in a finished tree, keyed by id(node).

    The value of a node is the mean reward of the leaves under it. A chunk's
    advantage is that value minus the value of the prefix it grew from. A root
    is compared with the mean of every finished line. A leaf with no siblings
    therefore carries reward minus the group mean.
    """
    if not leaves:
        raise ValueError("no leaves")
    for leaf in leaves:
        if leaf.get("children"):
            raise ValueError("a leaf has children")
        if not math.isfinite(float(leaf["reward"])):
            raise ValueError("rewards must be finite")
    values = {}

    def value(node):
        key = id(node)
        if key in values:
            return values[key]
        children = node.get("children") or []
        if not children:
            values[key] = float(node["reward"])
            return values[key]
        total = 0.0
        count = 0
        for leaf in _leaves_under(node):
            total += float(leaf["reward"])
            count += 1
        if count == 0:
            raise ValueError("a node has no leaves")
        values[key] = total / count
        return values[key]

    seen = []
    marks = set()

    def walk(node):
        key = id(node)
        if key in marks:
            return
        marks.add(key)
        seen.append(node)
        parent = node.get("parent")
        if parent is not None:
            walk(parent)

    for leaf in leaves:
        walk(leaf)
    baseline = sum(float(leaf["reward"]) for leaf in leaves) / len(leaves)
    out = {}
    for node in seen:
        parent = node.get("parent")
        parent_value = baseline if parent is None else value(parent)
        out[id(node)] = value(node) - parent_value
    return out


# Full lines in the from-base log run near 30 pen steps per character.
FORK_STEPS_PER_CHAR = 30
# Spread a chunk boundary across this many pen steps.
ADV_RADIUS = 8
# Play lines of 16 leaves. (4, 2, 2) is the one to use most often.
MEDIUM_SHAPES = ((4, 2, 2), (4, 4), (2, 8), (16,))


def _product(branches):
    product = 1
    for branch in branches:
        product *= int(branch)
    return product


def segment_budgets(text, segments, steps_per_char=FORK_STEPS_PER_CHAR):
    """How many new pen steps each fork may draw.

    The last entry is None: that segment runs until the pen stops, which is
    how a line is drawn at deploy time. Earlier entries split an estimate of
    `steps_per_char` points per character into equal windows.
    """
    segments = int(segments)
    if segments < 1:
        raise ValueError("a line needs at least one segment")
    if segments == 1:
        return [None]
    estimate = max(segments, int(round(float(steps_per_char) * max(len(str(text)), 1))))
    window = max(1, estimate // segments)
    return [window] * (segments - 1) + [None]


def blend_token_advantages(advantage, length, parent=None, child=None, radius=ADV_RADIUS):
    """One advantage per pen step.

    The first steps fade in from the parent chunk. The last steps fade toward
    the mean of the children. A segment with no neighbor keeps `advantage`
    on every step, which is the flat-group case.
    """
    length = int(length)
    if length < 1:
        return []
    radius = max(int(radius), 0)
    own = float(advantage)
    out = []
    for t in range(length):
        value = own
        if parent is not None and radius > 0:
            weight = min(1.0, t / float(radius))
            value = (1.0 - weight) * float(parent) + weight * own
        if child is not None and radius > 0:
            weight = min(1.0, (length - 1 - t) / float(radius))
            value = weight * value + (1.0 - weight) * float(child)
        out.append(value)
    return out


def token_advantages(node, credited, radius=ADV_RADIUS):
    """Per-step advantages for one node, from the chunk values around it."""
    own = float(credited[id(node)])
    parent = node.get("parent")
    parent_adv = None if parent is None else float(credited[id(parent)])
    children = node.get("children") or []
    child_adv = None
    if children:
        child_adv = sum(float(credited[id(child)]) for child in children) / len(children)
    return blend_token_advantages(
        own, int(node["seq"].shape[0]),
        parent=parent_adv, child=child_adv, radius=radius,
    )


def class_budget(source, n_words, beam, pinned=None, easy_leaves=0, novel_leaves=64):
    """Leaf count and branch factors for one prompt.

    A play line gets the 16-leaf tree. Half of those episodes use 4×2×2, and
    the rest are 4×4, 2×8, or 16 separate lines. A dictionary line gets a few
    separate drawings. A novel line gets many separate drawings. `pinned`
    forces the play shape and is ignored for the other two.
    Returns (width, branches).
    """
    if source == "words":
        width = int(easy_leaves) if int(easy_leaves) > 0 else random.randint(4, 8)
        if width < 1:
            raise ValueError("a dictionary line needs at least one drawing")
        return width, (width,)
    if source == "novel":
        width = int(novel_leaves)
        if width < 1:
            raise ValueError("a novel line needs at least one drawing")
        return width, (width,)
    width = int(beam)
    if width < 1:
        raise ValueError("beam width must be at least 1")
    if pinned is not None:
        branches = tuple(int(branch) for branch in pinned)
        if _product(branches) != width:
            raise ValueError(
                f"branches {branches} multiply to {_product(branches)}, not {width}")
        return width, branches
    allowed = {shape for shape in MEDIUM_SHAPES if _product(shape) == width}
    bag = [shape for shape in ([(4, 2, 2)] * 3 + [(4, 4), (2, 8), (16,)]) if shape in allowed]
    if not bag:
        return width, default_branches(max(int(n_words), 1), width)
    return width, bag[random.randrange(len(bag))]


def pick_prompts(prompts, n):
    if len(prompts) >= n:
        return random.sample(prompts, n)
    return [random.choice(prompts) for _ in range(n)]


def novel_line(stoi, words=10, min_len=1, max_len=4, max_chars=48):
    """Ten short words of letters and digits from the model alphabet.

    A line like 'y142 OX7V Fz1M wsaY W5OF M3VV 8Zab c12d eF3g h4'.
    Each word is at most 4 characters. Every character is in `stoi`.
    Spaces only separate the words.
    """
    chars = [ch for ch in stoi if ch.isalnum()]
    if not chars:
        raise ValueError("vocab has no letters or digits")
    if words < 1:
        raise ValueError("a novel line needs at least one word")
    gaps = words - 1
    room = max_chars - gaps
    if room < words:
        raise ValueError("max_chars is too small for this many words")
    share = max(1, room // words)
    low = min(min_len, share)
    high = min(max_len, room - low * (words - 1))
    high = max(high, low)
    lengths = [random.randint(low, high) for _ in range(words)]
    while sum(lengths) + gaps > max_chars:
        index = max(range(words), key=lambda i: lengths[i])
        if lengths[index] <= low:
            break
        lengths[index] -= 1
    parts = ["".join(random.choice(chars) for _ in range(n)) for n in lengths]
    return " ".join(parts)


def load_dictionary(path, stoi, max_chars):
    """Lowercase words of letters only. Capitals, punctuation, and digits are dropped.

    A word longer than `max_chars` cannot be a training line on its own.
    """
    vocab = set(stoi)
    seen = set()
    words = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        word = raw.strip().lower()
        if len(word) < 2 or len(word) > max_chars:
            continue
        if not word.isascii() or not word.isalpha():
            continue
        if any(ch not in vocab for ch in word):
            continue
        if word in seen:
            continue
        seen.add(word)
        words.append(word)
    if not words:
        raise SystemExit(
            f"no lowercase words in {path} fit in {max_chars} characters "
            "and the model alphabet"
        )
    return words


def word_line(lexicon, max_chars=48):
    """Random dictionary words, all lowercase, no longer than a play line.

    Words are drawn until another one would pass `max_chars`. The joined
    line stays inside the same character cap as the Romeo prompts.
    """
    if max_chars < 1:
        raise ValueError("max_chars must leave room for a word")
    pool = [word for word in lexicon if 2 <= len(word) <= max_chars]
    if not pool:
        raise ValueError("no dictionary word fits in max_chars")
    by_length = {}
    for word in pool:
        by_length.setdefault(len(word), []).append(word)
    picked = []
    length = 0
    used = set()
    misses = 0
    while misses < 40:
        room = max_chars - length - (0 if not picked else 1)
        if room < 2:
            break
        lengths = [n for n in by_length if n <= room]
        if not lengths:
            break
        sizes = [len(by_length[n]) for n in lengths]
        draw = random.randrange(sum(sizes))
        chosen = lengths[0]
        covered = 0
        for n, size in zip(lengths, sizes):
            covered += size
            if draw < covered:
                chosen = n
                break
        bucket = by_length[chosen]
        word = bucket[random.randrange(len(bucket))]
        if word in used:
            misses += 1
            continue
        picked.append(word)
        used.add(word)
        length += len(word) if len(picked) == 1 else len(word) + 1
        misses = 0
    if not picked:
        raise ValueError("no dictionary word fits in max_chars")
    return " ".join(picked)


def choose_label(prompts, stoi, novel, max_chars=48, words=0.0, lexicon=None):
    """A play line, a random alphabet line, or lowercase dictionary words.

    `novel` and `words` are chances and share one draw, so they must add up
    to at most 1. The rest of the draws stay on the play. Returns
    `(text, source)` where source is `novel`, `words`, or `play`.
    """
    roll = random.random()
    if novel > 0 and roll < novel:
        return novel_line(stoi, max_chars=max_chars), "novel"
    if words > 0 and roll < novel + words:
        if not lexicon:
            raise ValueError("dictionary lines need a word list")
        return word_line(lexicon, max_chars=max_chars), "words"
    return pick_prompts(prompts, 1)[0], "play"


def _split_strokes(xy, pen):
    """Same cut as generate.save_plot: the pen-up point belongs to the stroke that ends."""
    strokes = []
    current = []
    for i, (x, y) in enumerate(xy):
        current.append((float(x), float(y)))
        if pen[i] >= 0.5 or i == len(xy) - 1:
            strokes.append(current)
            current = []
    return [s for s in strokes if s]


def _blank(height):
    from PIL import Image
    return Image.new("RGB", (height * 4, height), "white")


def strokes_to_image(seq, std, height=384, stroke_width=5, pad=28, supersample=2):
    """Render a normalised (T, 3) trajectory to a white line image.

    Data y grows up; image y grows down. Height matches TrOCR's input so the
    recognizer's resize does not have to invent resolution. A single-point
    stroke is drawn as a dot, otherwise i and j lose their marks.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit(
            "rendering handwriting needs pillow:\n  uv sync --extra rl\n"
            f"({exc})"
        ) from exc

    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[1] < 3 or len(seq) == 0 or not np.isfinite(seq).all():
        return _blank(height)

    xy = np.cumsum(seq[:, :2] * float(std), axis=0)
    strokes = _split_strokes(xy, seq[:, 2])
    if not strokes:
        return _blank(height)

    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span_x = max(maxx - minx, 1e-3)
    span_y = max(maxy - miny, 1e-3)

    ss = max(int(supersample), 1)
    height_s = height * ss
    pad_s = pad * ss
    width_px = max(stroke_width * ss, 1)
    inner_h = max(height_s - 2 * pad_s, 1)
    scale = inner_h / span_y
    width = int(math.ceil(span_x * scale)) + 2 * pad_s
    max_w = 2048 * ss
    if width > max_w:
        scale *= (max_w - 2 * pad_s) / max(span_x * scale, 1e-3)
        width = max_w
    width = max(width, height_s)

    img = Image.new("RGB", (width, height_s), "white")
    draw = ImageDraw.Draw(img)
    used_h = span_y * scale
    y_origin = pad_s + (inner_h - used_h) / 2.0

    def tx(x):
        return pad_s + (x - minx) * scale

    def ty(y):
        return y_origin + (maxy - y) * scale

    for stroke in strokes:
        pts = [(tx(x), ty(y)) for x, y in stroke]
        if len(pts) == 1:
            x, y = pts[0]
            r = max(width_px / 2.0, 1.0)
            draw.ellipse((x - r, y - r, x + r, y + r), fill="black")
        else:
            draw.line(pts, fill="black", width=width_px, joint="curve")

    if ss > 1:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        img = img.resize((max(1, width // ss), height), resample)
    return img


def strokes_to_images(seqs, std, **kwargs):
    """Render several trajectories at once. Each image is independent."""
    arrays = []
    for seq in seqs:
        if torch.is_tensor(seq):
            seq = seq.detach().cpu().numpy()
        arrays.append(np.asarray(seq))
    if len(arrays) <= 1:
        return [strokes_to_image(seq, std, **kwargs) for seq in arrays]
    workers = min(len(arrays), os.cpu_count() or 1, 8)

    def draw(seq):
        return strokes_to_image(seq, std, **kwargs)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(draw, arrays))


def consistency_sheet(images, text, captions, step=None, summary=""):
    """Stack every sample of one sentence, with a caption above each line.

    Used by the recognizer loop so a step's draws can be compared side by side
    as the run goes. Captions are the recognizer read and the CER.
    """
    from PIL import Image, ImageDraw, ImageFont

    if len(images) != len(captions):
        raise ValueError("captions must line up with images")

    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()

    def fit(img, max_w=980, max_h=150):
        scale = min(max_w / img.width, max_h / img.height, 1.0)
        if scale >= 1.0:
            return img
        resample = getattr(Image, "Resampling", Image).LANCZOS
        return img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))), resample)

    fitted = [fit(img) for img in images]
    margin, gap, cap_h, header_h = 16, 12, 26, 48
    row_w = max(img.width for img in fitted)
    width = margin * 2 + row_w
    height = margin * 2 + header_h + sum(cap_h + img.height + gap for img in fitted)
    sheet = Image.new("RGB", (max(width, 480), height), "white")
    draw = ImageDraw.Draw(sheet)
    title = text if step is None else f"step {step}   {text}"
    if summary:
        title = f"{title}   {summary}"
    draw.text((margin, margin), title, fill="black", font=font)
    y = margin + header_h
    for caption, img in zip(captions, fitted):
        draw.text((margin, y), caption, fill=(20, 20, 20), font=font)
        y += cap_h
        sheet.paste(img, (margin, y))
        draw.rectangle(
            (margin, y, margin + img.width - 1, y + img.height - 1),
            outline=(210, 210, 210))
        y += img.height + gap
    return sheet


def choice_sheet(images, text):
    """One image with the prompt and the numbered samples, left to right."""
    from PIL import Image, ImageDraw, ImageFont

    def fit(img, max_w=640, max_h=220):
        scale = min(max_w / img.width, max_h / img.height, 1.0)
        if scale >= 1.0:
            return img
        resample = getattr(Image, "Resampling", Image).LANCZOS
        return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), resample)

    fitted = [fit(img) for img in images]
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()

    margin, gap, header, label_h = 16, 16, 48, 36
    body_h = max(img.height for img in fitted)
    width = margin * 2 + sum(img.width for img in fitted) + gap * (len(fitted) - 1)
    height = margin * 2 + header + body_h + label_h
    sheet = Image.new("RGB", (max(width, 320), height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, margin), text, fill="black", font=font)
    x = margin
    top = margin + header
    for i, img in enumerate(fitted, start=1):
        y = top + (body_h - img.height) // 2
        sheet.paste(img, (x, y))
        draw.rectangle((x, y, x + img.width - 1, y + img.height - 1), outline=(210, 210, 210))
        draw.text((x + img.width / 2.0 - 6, top + body_h + 6), str(i), fill="black", font=font)
        x += img.width + gap
    return sheet


def reveal(path):
    """Open an image on macOS or Linux, and always print the path."""
    import subprocess
    import sys
    print(f"saved {path}", flush=True)
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class TrocrScorer:
    """TrOCR-Large handwritten as a CER reward.

    The first construction downloads the weights (about 1.5 GB) into the
    Hugging Face cache.
    """

    def __init__(self, device, model_name="microsoft/trocr-large-handwritten",
                 kind="exp", tau=0.1, beams=1, vcer_alpha=0.8, vcer_k=2, vcer_table=None,
                 confidence_weight=0.2):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise SystemExit(
                "Recognizer RL needs the rl extra:\n"
                "  uv sync --extra rl\n"
                f"({exc})"
            ) from exc
        print(f"loading {model_name} on {device}", flush=True)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.kind = kind
        self.tau = tau
        self.beams = beams
        self.confidence_weight = float(confidence_weight)
        tokenizer = self.processor.tokenizer
        self._skip_ids = {
            token_id for token_id in (
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                tokenizer.bos_token_id,
                getattr(self.model.config, "decoder_start_token_id", None),
            ) if token_id is not None
        }
        self._vcer = None
        self.vcer_config = None
        if kind in ("vcer", "both"):
            from vcer.reward import VcerReward
            self._vcer = VcerReward.load(vcer_table, k=vcer_k, alpha=vcer_alpha)
            self.vcer_config = self._vcer.config

    def _decode_limit(self, references):
        """Room for a 10–20 word line. TrOCR stops at its end token before this."""
        longest = max((len(str(text)) for text in references), default=0)
        return min(160, max(64, longest + 8))

    def _with_confidence(self, rewards, cers, details=None):
        confs = self.last_confidence
        weight = getattr(self, "confidence_weight", 0.0)
        if not confs or len(confs) != len(rewards) or len(cers) != len(rewards) or weight <= 0:
            return rewards
        if details is not None:
            for item, confidence in zip(details, confs):
                item["confidence"] = confidence
        return [mix_confidence(reward, confidence, weight)
                for reward, confidence in zip(rewards, confs)]

    @torch.no_grad()
    def read(self, images, max_new_tokens=64):
        # TrOCR-Large already batches the images it is given. A whole update
        # is 32 lines, which is more than this GPU comfortably encodes at once,
        # so the call is split into chunks that stay batched inside.
        if not images:
            self.last_confidence = []
            return []
        hyps = []
        confidences = []
        saw_confidence = False
        for start in range(0, len(images), 8):
            chunk = self._read_chunk(images[start:start + 8], max_new_tokens)
            hyps.extend(chunk)
            if self.last_confidence is not None:
                saw_confidence = True
                confidences.extend(self.last_confidence)
        self.last_confidence = confidences if saw_confidence else None
        return hyps

    def _read_chunk(self, images, max_new_tokens):
        encoded = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = encoded.pixel_values.to(self.device)
        generate_kwargs = {"num_beams": self.beams, "max_new_tokens": max_new_tokens}
        # Greedy decoding already has a logit row per chosen token. Beam search
        # does not line those rows up with the finished string, so confidence
        # stays off unless --beams 1.
        if self.beams == 1 and getattr(self, "confidence_weight", 0.0) > 0:
            outputs = self.model.generate(
                pixel_values,
                return_dict_in_generate=True,
                output_scores=True,
                **generate_kwargs,
            )
            self.last_confidence = token_confidence(
                outputs.scores, outputs.sequences, self._skip_ids)
            ids = outputs.sequences
        else:
            self.last_confidence = None
            ids = self.model.generate(pixel_values, **generate_kwargs)
        return self.processor.batch_decode(ids, skip_special_tokens=True)

    def score(self, images, references):
        self.last_confidence = None
        hyps = self.read(images, max_new_tokens=self._decode_limit(references))
        if self.kind == "cer":
            cers, rewards = [], []
            for ref, hyp in zip(references, hyps):
                raw, reward = string_reward(ref, hyp)
                cers.append(raw)
                rewards.append(reward)
            self.last_details = None
            return self._with_confidence(rewards, cers), cers, hyps
        vcer = getattr(self, "_vcer", None)
        if self.kind == "both":
            _pure, details = vcer.score_batch(references, hyps)
            self.last_details = details
            rewards = [
                blend_string_reward(item["cer"], item["wcer"])
                for item in details
            ]
            cers = [item["cer"] for item in details]
            return self._with_confidence(rewards, cers, details), cers, hyps
        if vcer is not None:
            rewards, details = vcer.score_batch(references, hyps)
            self.last_details = details
            cers = [item["cer"] for item in details]
            return self._with_confidence(rewards, cers, details), cers, hyps
        self.last_details = None
        cers, rewards = [], []
        for ref, hyp in zip(references, hyps):
            raw, reward = cer_and_reward(ref, hyp, kind=self.kind, tau=self.tau)
            cers.append(raw)
            rewards.append(reward)
        return self._with_confidence(rewards, cers), cers, hyps


def add_policy_args(parser, default_save):
    parser.add_argument("--ckpt", default=None,
                        help="Policy checkpoint to start from. Defaults depend on the script. "
                             f"The base model is {BASE_CKPT}.")
    parser.add_argument("--ref", default=BASE_CKPT,
                        help=f"Frozen KL anchor. Default: {BASE_CKPT}.")
    parser.add_argument("--save", default=default_save, help="Where to write the trained policy.")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature. The log-prob uses the same value.")
    parser.add_argument("--sharpen", type=float, default=2.0,
                        help="Window beta scale for sampling AND the log-prob. "
                             "2 matches generate.py. The numpy server uses 1.")
    parser.add_argument("--kl", type=float, default=0.1,
                        help="How hard the per-step KL pulls back toward --ref.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient norm clip.")
    parser.add_argument("--prompts", default="data/romeo_juliet.txt",
                        help="Line pool. Sampled at random. Default is the whole of "
                             "Romeo and Juliet, cut into lines of at most 48 characters.")
    parser.add_argument("--text", default=None,
                        help="Use this prompt every time instead of sampling --prompts.")
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=48,
                        help="Drop prompts longer than this. 48 keeps a line readable "
                             "after TrOCR resizes it to a square.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def short(text, n=60):
    text = str(text).replace("\n", " ")
    return text if len(text) <= n else text[:n - 1] + "…"
