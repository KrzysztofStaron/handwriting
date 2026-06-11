"""Export the trained model into torch-free serving artifacts:
  - weights.npz : every parameter as a float32 numpy array
  - stoi.json   : the char -> id vocab as plain JSON
  - std.json    : the normalisation constant (written if missing)

Run once with torch available (it isn't needed at serving time):  uv run export_weights.py
"""
import json
import numpy as np
import torch

CKPT = "best-4.pth"

sd = torch.load(CKPT, map_location="cpu")
np.savez("weights.npz", **{k: v.numpy().astype(np.float32) for k, v in sd.items()})

stoi = torch.load("stoi.pth")
json.dump(stoi, open("stoi.json", "w"), ensure_ascii=False)

try:
    json.load(open("std.json"))
except FileNotFoundError:
    from data import load_data
    _s, _sent, std = load_data()
    json.dump({"std": std}, open("std.json", "w"))

print(f"exported weights.npz ({len(sd)} tensors), stoi.json ({len(stoi)} chars) from {CKPT}")
