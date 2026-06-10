import torch
from model import HandwritingModel, sequence_loss
from data import get_loader

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 1

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

model = HandwritingModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loader, _ = get_loader(batch_size=BATCH_SIZE)

step = 0
for epoch in range(EPOCHS):
    for x, y, mask in loader:
        step += 1


        x = x.to(DEVICE)
        y = y.to(DEVICE)
        mask = mask.to(DEVICE)
        
        out, _ = model(x)

        loss = sequence_loss(out, y, mask)
        if step % 10 == 0 or step==1:
            print(f"Step {step}, Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()