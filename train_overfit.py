"""Overfit a SINGLE batch, over and over -- the classic learning sanity check.

If the model + loss + training loop are wired correctly, repeatedly fitting one fixed
batch should drive the loss down toward its floor and the model should reproduce those
exact samples. Use it to answer "can it learn at all?" before trusting a full run:

  - loss keeps dropping        -> pipeline learns; gradients flow.
  - loss stuck / NaN           -> a real bug (bad masking, exploding grads, dead inputs).
  - loss drops but samples are garbage -> the window/alignment can't even memorise this
                                  batch, which points at the conditioning path, not data.

A small batch is used so it overfits fast and visibly. Writes only to samples_overfit/
(and nothing else): it never touches best.pth. The eval text is taken straight from the
batch, so a healthy run should redraw that line almost exactly.
"""
import os
import torch
from model import HandwritingModel, sequence_loss, window_entropy
from data import get_loader
from generate import sample, save_plot, save_alignment

INIT_CKPT = "best-4.pth"      # warm-start from the trained model -> tests memorisation
                              # capacity. Set to None for a from-scratch pipeline test.
SAMPLE_DIR = "samples_overfit"
COLLECTED_DIR = ["collected", "collected2"]
INCLUDE_IAM = True

BATCH_SIZE = 4                # one SMALL batch, repeated -- overfits fast
STEPS = 2000                  # gradient steps on that single batch
LEARNING_RATE = 3e-4          # higher than a real run; constant (no scheduler)
ENTROPY_WEIGHT = 0.05
MAX_LEN = None

LOG_EVERY = 20
EVAL_EVERY = 200              # dump a sample of an in-batch line every N steps

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")


def decode(c_row, c_mask_row, itos):
    """One-hot text row -> string (the line the model should learn to reproduce)."""
    ids = c_row[c_mask_row.bool()].argmax(dim=-1).tolist()
    return "".join(itos[i] for i in ids)


def main():
    print(f"device: {DEVICE}   SINGLE-BATCH OVERFIT   init: {INIT_CKPT or 'scratch'}")
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    loader, std, stoi = get_loader(batch_size=BATCH_SIZE, max_len=MAX_LEN,
                                   collected_dir=COLLECTED_DIR, include_iam=INCLUDE_IAM)
    itos = {i: ch for ch, i in stoi.items()}

    # the ONE batch we overfit -- grabbed once, kept on device, reused every step.
    x, y, mask, c, c_mask = [t.to(DEVICE) for t in next(iter(loader))]
    eval_text = decode(c[0], c_mask[0], itos)
    print(f"overfitting {x.shape[0]} samples; eval text = {eval_text!r}")

    model = HandwritingModel(vocab_size=len(stoi)).to(DEVICE)
    if INIT_CKPT is not None:
        model.load_state_dict(torch.load(INIT_CKPT, map_location=DEVICE))
        print(f"warm-started from {INIT_CKPT}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for step in range(1, STEPS + 1):
        out, _, phis, _ = model(x, c, c_mask)
        loss = sequence_loss(out, y, mask) + ENTROPY_WEIGHT * window_entropy(phis, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0 or step == 1:
            print(f"  step {step}/{STEPS}  loss: {loss.item():.4f}", flush=True)

        if step % EVAL_EVERY == 0:
            model.eval()
            seq, phis_s = sample(model, eval_text, stoi, DEVICE, temperature=0.4)
            save_plot(seq, std, f"{SAMPLE_DIR}/step{step:05d}.png",
                      f"{eval_text!r}  step {step}  ({len(seq)} steps)")
            save_alignment(phis_s, eval_text, f"{SAMPLE_DIR}/step{step:05d}_align.png")
            model.train()

    print(f"done. final loss: {loss.item():.4f}   samples in {SAMPLE_DIR}/")


if __name__ == "__main__":
    main()
