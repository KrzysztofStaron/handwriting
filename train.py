import torch
from model import HandwritingModel, sequence_loss
from data import get_loader

# --- tuned for a single RTX 4090 (24 GB) ---
# The model (3.3M params) is tiny, so the trick is to fill the GPU's parallel
# (batch) dimension and avoid CPU/data stalls. Time steps stay sequential no
# matter what, so huge batches are how you actually use the card.
LEARNING_RATE = 0.001    # sigma clamp removed the divergence, so back to a healthy LR
BATCH_SIZE = 128         # 12GB 4070 + full BPTT over ~1200 steps -> 256 OOMs; 128 is safe
EPOCHS = 80              # bigger batch = fewer updates/epoch, so give it more epochs
GRAD_CLIP = 10
USE_AMP = True           # bf16 autocast on Ada -> faster matmuls, safe exponent range

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {DEVICE}")

# let cuDNN pick the fastest kernels for our fixed-ish shapes
torch.backends.cudnn.benchmark = True

model = HandwritingModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# gently decay the LR over training for a stable, lower final loss
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

loader, _ = get_loader(batch_size=BATCH_SIZE)

amp = USE_AMP and DEVICE == "cuda"
best_loss = float("inf")
steps_per_epoch = len(loader)
LOG_EVERY = 5            # print a line every N steps so you can see it's alive
global_step = 0

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n = 0
    for x, y, mask in loader:
        global_step += 1
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            out, _ = model(x)
        # compute the MDN loss in fp32 for numerical safety (exp/log are touchy)
        loss = sequence_loss(out.float(), y, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        epoch_loss += loss.item()
        n += 1

        if n % LOG_EVERY == 0 or n == 1:
            print(f"  epoch {epoch+1}/{EPOCHS}  step {n}/{steps_per_epoch}  "
                  f"(global {global_step})  loss: {loss.item():.4f}", flush=True)

    scheduler.step()
    avg = epoch_loss / n
    lr = scheduler.get_last_lr()[0]
    print(f"epoch {epoch+1}/{EPOCHS}  avg_loss: {avg:.4f}  lr: {lr:.2e}", flush=True)

    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), "best.pth")

print(f"done. best avg_loss: {best_loss:.4f}")
