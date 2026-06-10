import torch
from model import HandwritingModel, sequence_loss
from data import get_loader

LEARNING_RATE = 0.0005   # lower than before -> more stable, avoids the late divergence
BATCH_SIZE = 64
EPOCHS = 50              # the model needs far more than 10 epochs to converge

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

model = HandwritingModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loader, _ = get_loader(batch_size=BATCH_SIZE)

best_loss = float("inf")

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n = 0
    for x, y, mask in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        mask = mask.to(DEVICE)

        out, _ = model(x)
        loss = sequence_loss(out, y, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        epoch_loss += loss.item()
        n += 1

    avg = epoch_loss / n
    print(f"epoch {epoch+1}/{EPOCHS}  avg_loss: {avg:.4f}")

    # save only when this epoch beats the best so far -> 'best.pth' is never the diverged one
    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), "best.pth")
        print(f"  saved best.pth (avg_loss {avg:.4f})")

print(f"done. best avg_loss: {best_loss:.4f}")
