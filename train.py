import torch
import torch.optim as optim
from model import HandwritingModel, mdn_loss
from data import get_loader

device = "cpu"  # or "cuda" if you have GPU
model = HandwritingModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

loader, std = get_loader(batch_size=32)

num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0

    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        # forward pass
        out, _ = model(x)

        # compute loss
        loss = mdn_loss(out, y, mask)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (paper sec. 4.2: critical for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # update weights
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"epoch {epoch+1}/{num_epochs}  loss: {avg_loss:.4f}")

    # save checkpoint
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pt")

print("training done")
