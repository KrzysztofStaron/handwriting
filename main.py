import numpy as np

data = np.load("data/strokes.npy", allow_pickle=True, encoding="latin1")

print(len(data))        # 6000  -> number of handwritten lines
print(data[0])          # the first line: a (739, 3) array
print(data[0].shape)    # (739, 3)  -> 739 pen points, 3 numbers each
print(data[0][:5])      # first 5 points