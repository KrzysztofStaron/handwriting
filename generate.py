import torch
import numpy as np
import matplotlib.pyplot as plt
from model import HandwritingModel, split_params
from data import get_loader
import math

# Load checkpoint
checkpoint_path = "10_752.pth"  # the lowest-loss checkpoint from training
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = Handwritingmodel().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

print(f"Loaded model from {checkpoint_path}")

# Get the normalization factor
_, std = get_loader()

def sample(model, steps=700, temperature=1.0, device="cpu"):
    """Generate a handwriting sequence by sampling from the model."""
    model.eval()
    points = []
    x = torch.zeros(1, 1, 3).to(device)
    hidden = None

    with torch.no_grad():
        for _ in range(steps):
            out, hidden = model(x, hidden)
            pi, mu_x, mu_y, sig_x, sig_y, rho, pen = split_params(out)

            # Pick a mixture component
            idx = torch.multinomial(pi[0, 0], 1).item()

            mx = mu_x[0, 0, idx].item()
            my = mu_y[0, 0, idx].item()
            sx = sig_x[0, 0, idx].item() * temperature
            sy = sig_y[0, 0, idx].item() * temperature
            r = rho[0, 0, idx].item()

            # Sample from bivariate normal via Cholesky
            u1 = torch.randn(1).item()
            u2 = torch.randn(1).item()
            dx = mx + sx * u1
            dy = my + sy * (r * u1 + math.sqrt(max(1 - r**2, 1e-8)) * u2)

            pen_up = 1.0 if torch.rand(1).item() < pen[0, 0].item() else 0.0
            points.append([dx, dy, pen_up])
            x = torch.tensor([[[dx, dy, pen_up]]]).to(device)

    return torch.tensor(points)

# Generate a sample
print("Generating handwriting...")
stroke_seq = sample(model, steps=700, temperature=0.9, device=device)

# Convert to absolute coordinates
xy = stroke_seq[:, :2].cpu().numpy()
xy = np.cumsum(xy * std, axis=0)  # denormalize and integrate
pen_up = stroke_seq[:, 2].cpu().numpy()

# Plot
fig, ax = plt.subplots(figsize=(12, 3))
start = 0
for j in range(len(stroke_seq)):
    if pen_up[j] == 1 or j == len(stroke_seq) - 1:
        ax.plot(xy[start:j+1, 0], xy[start:j+1, 1], 'k', linewidth=2)
        start = j + 1

ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f'Generated handwriting (temperature={0.5})')
plt.savefig('generated_sample.png', bbox_inches='tight', dpi=150)
print("Saved to generated_sample.png")
plt.close()

# Generate a few more with different temperatures
for temp in [0.3, 0.5, 0.8]:
    stroke_seq = sample(model, steps=700, temperature=temp, device=device)
    xy = np.cumsum(stroke_seq[:, :2].cpu().numpy() * std, axis=0)
    pen_up = stroke_seq[:, 2].cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 3))
    start = 0
    for j in range(len(stroke_seq)):
        if pen_up[j] == 1 or j == len(stroke_seq) - 1:
            ax.plot(xy[start:j+1, 0], xy[start:j+1, 1], 'k', linewidth=2)
            start = j + 1

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Generated handwriting (temperature={temp})')
    plt.savefig(f'sample_temp_{temp}.png', bbox_inches='tight', dpi=150)
    print(f"Saved to sample_temp_{temp}.png")
    plt.close()

print("Done!")
