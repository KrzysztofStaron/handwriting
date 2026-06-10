import torch
from model import HandwritingModel, sequence_loss
from data import get_loader

# --- tuned for a single RTX 4090 (24 GB) ---
# The model (3.3M params) is tiny, so the trick is to fill the GPU's parallel
# (batch) dimension and avoid CPU/data stalls. Time steps stay sequential no
# matter what, so huge batches are how you actually use the card.
LEARNING_RATE = 0.001    # sigma clamp removed the divergence, so back to a healthy LR
BATCH_SIZE = 256         # 4x larger -> 4x more parallel work per step, fits easily in 24GB
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

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n = 0
    for x, y, mask in loader:
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

    scheduler.step()
    avg = epoch_loss / n
    lr = scheduler.get_last_lr()[0]
    print(f"epoch {epoch+1}/{EPOCHS}  avg_loss: {avg:.4f}  lr: {lr:.2e}")

    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), "best.pth")

print(f"done. best avg_loss: {best_loss:.4f}")
