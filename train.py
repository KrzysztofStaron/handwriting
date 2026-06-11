import torch
from model import HandwritingModel, sequence_loss
from data import get_loader

# Back to the recipe that actually broke through the plateau in the earlier run:
# small batch (noisy gradients escape the plateau), fp32 (precise LSTM updates),
# constant LR. The only addition is the sigma clamp in model.py, which prevents
# the end-of-training divergence the earlier run had.
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 40              # plenty: breakthrough happened ~600 steps = ~6 epochs before
GRAD_CLIP = 10

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

LOG_EVERY = 20          # print a line every N steps so you can see it's alive


def main():
    print(f"device: {DEVICE}")
    torch.backends.cudnn.benchmark = True

    loader, _, stoi = get_loader(batch_size=BATCH_SIZE)

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

    print(f"done. best avg_loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
