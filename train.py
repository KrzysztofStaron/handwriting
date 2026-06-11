import os
import torch
from model import HandwritingModel, sequence_loss
from data import get_loader
from generate import sample, save_plot, save_alignment

# Back to the recipe that actually broke through the plateau in the earlier run:
# small batch (noisy gradients escape the plateau), fp32 (precise LSTM updates),
# constant LR. The only addition is the sigma clamp in model.py, which prevents
# the end-of-training divergence the earlier run had.
LEARNING_RATE = 0.001
BATCH_SIZE = 64         # 8GB 4060 fits ~64 at full length; raise only with more VRAM
EPOCHS = 80             # final run: -1.788 was still descending at 40, so go further
GRAD_CLIP = 10

# FULL RUN: train on whole lines (no truncation) for final quality.
# Set back to 700 for fast iteration. None = use every point of every line.
MAX_LEN = None

EVAL_EVERY = 5          # every N epochs, dump a sample + alignment plot to samples/
EVAL_TEXT = "hello world"

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

LOG_EVERY = 20          # print a line every N steps so you can see it's alive


def main():
    print(f"device: {DEVICE}   max_len: {MAX_LEN}")
    torch.backends.cudnn.benchmark = True
    os.makedirs("samples", exist_ok=True)

    loader, std, stoi = get_loader(batch_size=BATCH_SIZE, max_len=MAX_LEN)

    # save the vocab so generate.py encodes text with the SAME char->id mapping
    torch.save(stoi, "stoi.pth")

    model = HandwritingModel(vocab_size=len(stoi)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float("inf")
    steps_per_epoch = len(loader)
    global_step = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n = 0
        for x, y, mask, c, c_mask in loader:
            global_step += 1
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            c = c.to(DEVICE, non_blocking=True)
            c_mask = c_mask.to(DEVICE, non_blocking=True)

            out, _, _ = model(x, c, c_mask)
            loss = sequence_loss(out, y, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

            if n % LOG_EVERY == 0 or n == 1:
                print(f"  epoch {epoch+1}/{EPOCHS}  step {n}/{steps_per_epoch}  "
                      f"(global {global_step})  loss: {loss.item():.4f}", flush=True)

        avg = epoch_loss / n
        print(f"epoch {epoch+1}/{EPOCHS}  avg_loss: {avg:.4f}", flush=True)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), "best.pth")
            print(f"  saved best.pth (avg_loss {avg:.4f})", flush=True)

        # periodically SEE quality form: dump a sample + the alignment diagonal
        if (epoch + 1) % EVAL_EVERY == 0:
            model.eval()
            seq, phis = sample(model, EVAL_TEXT, stoi, DEVICE, temperature=0.4)
            save_plot(seq, std, f"samples/epoch{epoch+1:02d}.png",
                      f"'{EVAL_TEXT}'  epoch {epoch+1}")
            save_alignment(phis, EVAL_TEXT, f"samples/epoch{epoch+1:02d}_align.png")
            model.train()

    print(f"done. best avg_loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
