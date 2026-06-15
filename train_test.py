"""Fine-tune EXCLUSIVELY on the hand-collected strokes (collected/ + collected2/).

A sibling of train.py for isolating the collected data: no IAM lines in the training
set, so you can see what the adaptive-resampled canvas samples do to the model on their
own. std and the char->id vocab are still taken from IAM (via get_loader(include_iam=
False)) so the model stays compatible with the best.pth warm-start.

Writes to its OWN files -- best_test.pth and samples_test/ -- so it never clobbers the
main best.pth. The collected data goes through import_collected.file_to_array, which now
resamples each sample to IAM's points-per-character (the alignment-rate fix), so the
soft window shouldn't collapse the way it did with fixed-step resampling.
"""
import os
import shutil
import torch
from model import HandwritingModel, sequence_loss, window_entropy
from data import get_loader
from generate import sample, save_plot, save_alignment
from train import diag_report   # reuse the identical window-entropy diagnostic

INIT_CKPT = "best.pth"        # warm-start from the IAM model (training set is collected-only)
OUT_CKPT = "best_test.pth"    # separate output -- NEVER overwrites best.pth
SAMPLE_DIR = "samples_test"
COLLECTED_DIR = ["collected", "collected2"]

FREEZE_BACKBONE = False
ENTROPY_WEIGHT = 0         # window-sharpness regulariser (match train.py)

LEARNING_RATE = 1e-3
LR_MIN = 1e-5
BATCH_SIZE = 64
EPOCHS = 30                   # collected set is small (~170), so more passes

MAX_LEN = None

EVAL_EVERY = 1               # sample + checkpoint EVERY epoch
EVAL_TEXT = "extraordinary"

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

LOG_EVERY = 5


def main():
    print(f"device: {DEVICE}   COLLECTED-ONLY   init: {INIT_CKPT or 'scratch'}")
    torch.backends.cudnn.benchmark = True
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # collected-only: no IAM lines, but IAM std + vocab so best.pth stays compatible.
    loader, std, stoi = get_loader(batch_size=BATCH_SIZE, max_len=MAX_LEN,
                                   collected_dir=COLLECTED_DIR, include_iam=False)
    if len(loader.dataset.seqs) == 0:
        raise SystemExit("no collected samples found -- check collected/ and collected2/")
    torch.save(stoi, "stoi.pth")

    model = HandwritingModel(vocab_size=len(stoi)).to(DEVICE)
    if INIT_CKPT is not None:
        model.load_state_dict(torch.load(INIT_CKPT, map_location=DEVICE))
        print(f"warm-started from {INIT_CKPT}")

    if FREEZE_BACKBONE:
        for p in model.parameters():
            p.requires_grad_(False)
        model.fc.weight.requires_grad_(True)
        model.fc.bias.requires_grad_(True)
        print("FROZEN backbone -- training only the output head (model.fc).")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_MIN)

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

        # back up EVERY epoch's weights (timestamped by epoch) so no run is lost and you
        # can roll back to any point -- loss does not track visual quality here.
        epoch_ckpt = f"{SAMPLE_DIR}/epoch{epoch+1:02d}.pth"
        torch.save(model.state_dict(), epoch_ckpt)

        if avg < best_loss:
            best_loss = avg
            if os.path.exists(OUT_CKPT):                 # keep the previous best as a backup
                shutil.copy2(OUT_CKPT, OUT_CKPT + ".bak")
            torch.save(model.state_dict(), OUT_CKPT)
            print(f"  saved {OUT_CKPT} (avg_loss {avg:.4f})", flush=True)

        # sample EVERY epoch (EVAL_EVERY=1) so you can watch quality evolve by eye
        if (epoch + 1) % EVAL_EVERY == 0:
            model.eval()
            seq, phis = sample(model, EVAL_TEXT, stoi, DEVICE, temperature=0.4)
            save_plot(seq, std, f"{SAMPLE_DIR}/epoch{epoch+1:02d}.png",
                      f"'{EVAL_TEXT}'  epoch {epoch+1}  ({len(seq)} steps)")
            save_alignment(phis, EVAL_TEXT, f"{SAMPLE_DIR}/epoch{epoch+1:02d}_align.png")

    print(f"done. best avg_loss: {best_loss:.4f}  ->  {OUT_CKPT}")


if __name__ == "__main__":
    main()
