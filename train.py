import os
import torch
from model import HandwritingModel, sequence_loss, window_entropy
from data import get_loader
from generate import sample, save_plot, save_alignment

INIT_CKPT = "models/pre-trained.pth"  # warm-start checkpoint; None = train from scratch
FREEZE_BACKBONE = False      # True = train only model.fc (output head)
ENTROPY_WEIGHT = 0           # window-sharpness regulariser strength (tune by win_H)

LEARNING_RATE = 1e-4
LR_MIN = 1e-5           # cosine decay target over the run
BATCH_SIZE = 32
EPOCHS = 20

# FULL RUN: train on whole lines (no truncation) for final quality.
# Set to e.g. 700 for fast iteration. None = use every point of every line.
MAX_LEN = None

# IAM + the hand-collected canvas samples, read straight from <dir>/*.json.
# Pass a list to pull from several collection folders.
COLLECTED_DIR = []

EVAL_EVERY = 2          # every N epochs, dump a sample + alignment plot to samples/
EVAL_TEXT = "extraordinary"   # a long word -- where the smear shows

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

LOG_EVERY = 20          # print a line every N steps so you can see it's alive


def diag_report(model, batch):
    """Diagnostics on a fixed batch, comparable epoch to epoch.

    Returns win_H: mean window entropy (LOWER = sharper attention; this is the
    number the entropy regulariser is driving down).
    """
    x, y, mask, c, c_mask = [t.to(DEVICE) for t in batch]
    with torch.no_grad():
        _, _, phis, _ = model(x, c, c_mask)
    return window_entropy(phis, mask).item()


def main():
    print(f"device: {DEVICE}   max_len: {MAX_LEN}   init: {INIT_CKPT or 'scratch'}")
    torch.backends.cudnn.benchmark = True
    os.makedirs("models", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    loader, std, stoi = get_loader(batch_size=BATCH_SIZE, max_len=MAX_LEN,
                                   collected_dir=COLLECTED_DIR)

    # save the vocab so generate.py encodes text with the SAME char->id mapping
    torch.save(stoi, "stoi.pth")

    model = HandwritingModel(vocab_size=len(stoi)).to(DEVICE)
    if INIT_CKPT is not None:
        sd = torch.load(INIT_CKPT, map_location=DEVICE)
        model.load_state_dict(sd)
        print(f"warm-started from {INIT_CKPT}")

    if FREEZE_BACKBONE:
        # Freeze the recurrent backbone, train only the output head -- refines the
        # pen/mixture readout without disturbing the learned dynamics or window.
        for p in model.parameters():
            p.requires_grad_(False)
        model.fc.weight.requires_grad_(True)
        model.fc.bias.requires_grad_(True)
        print("FROZEN backbone -- training only the output head (model.fc).")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE)
    # cosine decay LEARNING_RATE -> LR_MIN over the whole run (stepped per epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_MIN)

    # one fixed batch reused for the diagnostics, so the numbers are comparable
    # epoch to epoch (apples to apples).
    diag_batch = next(iter(loader))

    best_loss = float("inf")
    steps_per_epoch = len(loader)
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n = 0
        for x, y, mask, c, c_mask in loader:
            global_step += 1
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            c = c.to(DEVICE, non_blocking=True)
            c_mask = c_mask.to(DEVICE, non_blocking=True)

            out, _, phis, _ = model(x, c, c_mask)
            loss = sequence_loss(out, y, mask) + ENTROPY_WEIGHT * window_entropy(phis, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

            if n % LOG_EVERY == 0 or n == 1:
                print(f"  epoch {epoch+1}/{EPOCHS}  step {n}/{steps_per_epoch}  "
                      f"(global {global_step})  loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        avg = epoch_loss / n
        lr = scheduler.get_last_lr()[0]
        win_H = diag_report(model, diag_batch)
        print(f"epoch {epoch+1}/{EPOCHS}  avg_loss: {avg:.4f}  lr: {lr:.2e}  "
              f"win_H: {win_H:.3f}", flush=True)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), "models/post-trained.pth")
            print(f"  saved models/post-trained.pth (avg_loss {avg:.4f})", flush=True)

        # periodically SEE quality AND keep the weights, so the final model can be
        # chosen by EYE (alignment + sample) rather than by lowest loss alone -- loss
        # does not track visual quality (cf. best-2: good loss, collapsed window).
        if (epoch + 1) % EVAL_EVERY == 0:
            model.eval()
            seq, phis = sample(model, EVAL_TEXT, stoi, DEVICE, temperature=0.4)
            save_plot(seq, std, f"samples/epoch{epoch+1:02d}.png",
                      f"'{EVAL_TEXT}'  epoch {epoch+1}  ({len(seq)} steps)")
            save_alignment(phis, EVAL_TEXT, f"samples/epoch{epoch+1:02d}_align.png")
            torch.save(model.state_dict(), f"samples/epoch{epoch+1:02d}.pth")
            cap = 400 + 50 * len(EVAL_TEXT)        # matches sample_iter's adaptive guard
            print(f"  eval sample: {len(seq)} steps "
                  f"({'window-stopped' if len(seq) < cap else 'hit max_steps'})", flush=True)

    print(f"done. best avg_loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
