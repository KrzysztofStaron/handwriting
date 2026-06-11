"""Pure-numpy forward pass for the handwriting synthesis model.

Mirrors model.py exactly (LSTMCell layer 1 + window feedback, then nn.LSTM layers
2/3, MDN head) but with zero torch dependency -- so serving needs only numpy.
Weights come from a state_dict-style dict of numpy arrays (export once with torch).
"""
import numpy as np

M = 20  # output mixture components
K = 10  # window mixture components


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class NumpyHandwritingModel:
    def __init__(self, weights, hidden_size=400):
        self.w = weights
        self.H = hidden_size

    @classmethod
    def from_npz(cls, path="weights.npz", hidden_size=400):
        data = np.load(path)
        return cls({k: data[k].astype(np.float32) for k in data.files}, hidden_size)

    def _cell(self, x, h, c, prefix, l0):
        # PyTorch LSTM(Cell) math; gate order is (input, forget, cell, output)
        suf = "_l0" if l0 else ""
        g = (self.w[f"{prefix}.weight_ih{suf}"] @ x + self.w[f"{prefix}.bias_ih{suf}"]
             + self.w[f"{prefix}.weight_hh{suf}"] @ h + self.w[f"{prefix}.bias_hh{suf}"])
        H = self.H
        i = _sigmoid(g[:H]); f = _sigmoid(g[H:2*H])
        gg = np.tanh(g[2*H:3*H]); o = _sigmoid(g[3*H:4*H])
        c2 = f * c + i * gg
        return o * np.tanh(c2), c2

    def _window(self, h1, c, c_mask, kappa_prev):
        p = self.w["window_fc.weight"] @ h1 + self.w["window_fc.bias"]   # (3K,)
        alpha = np.exp(p[:K]); beta = np.exp(p[K:2*K])
        kappa = kappa_prev + np.exp(p[2*K:])
        u = np.arange(c.shape[0])
        phi = (alpha[:, None] * np.exp(-beta[:, None] * (kappa[:, None] - u[None, :])**2)).sum(0)
        phi = phi * c_mask
        return phi @ c, kappa, phi

    def forward(self, x, c, c_mask):
        """x: (T,3), c: (U,V), c_mask: (U,) -> out: (T,121), phis: (T,U)."""
        T, H, V = x.shape[0], self.H, c.shape[1]
        h1 = np.zeros(H); c1 = np.zeros(H); w = np.zeros(V); kappa = np.zeros(K)
        h1_seq, w_seq, phis = [], [], []
        for t in range(T):                                  # layer 1 + window (sequential)
            h1, c1 = self._cell(np.concatenate([x[t], w]), h1, c1, "lstm1", l0=False)
            w, kappa, phi = self._window(h1, c, c_mask, kappa)
            h1_seq.append(h1); w_seq.append(w); phis.append(phi)
        h1_seq = np.array(h1_seq); w_seq = np.array(w_seq)

        h2 = np.zeros(H); c2 = np.zeros(H); h2_seq = []      # layer 2
        for t in range(T):
            h2, c2 = self._cell(np.concatenate([x[t], h1_seq[t], w_seq[t]]), h2, c2, "lstm2", l0=True)
            h2_seq.append(h2)
        h2_seq = np.array(h2_seq)

        h3 = np.zeros(H); c3 = np.zeros(H); h3_seq = []      # layer 3
        for t in range(T):
            h3, c3 = self._cell(np.concatenate([x[t], h2_seq[t], w_seq[t]]), h3, c3, "lstm3", l0=True)
            h3_seq.append(h3)
        h3_seq = np.array(h3_seq)

        feat = np.concatenate([h1_seq, h2_seq, h3_seq], axis=1)        # (T, 3H)
        out = feat @ self.w["fc.weight"].T + self.w["fc.bias"]         # (T, 121)
        return out, np.array(phis)

    def sample_iter(self, c, c_mask, temperature=1.0, max_steps=2000, tail_steps=25, rng=None):
        """Autoregressive generation, one point at a time (mirrors generate.py).

        c: (U,V) one-hot text, c_mask: (U,). Yields (dx, dy, pen_up, phi) per step.
        """
        rng = rng if rng is not None else np.random.default_rng()
        H, V, U = self.H, c.shape[1], c.shape[0]
        z = lambda: np.zeros(H, np.float32)
        h1, c1, h2, c2, h3, c3 = z(), z(), z(), z(), z(), z()
        w = np.zeros(V, np.float32)
        kappa = np.zeros(K, np.float32)
        x = np.zeros(3, np.float32)
        reached_end_at = None

        for t in range(max_steps):
            h1, c1 = self._cell(np.concatenate([x, w]), h1, c1, "lstm1", l0=False)
            w, kappa, phi = self._window(h1, c, c_mask, kappa)
            h2, c2 = self._cell(np.concatenate([x, h1, w]), h2, c2, "lstm2", l0=True)
            h3, c3 = self._cell(np.concatenate([x, h2, w]), h3, c3, "lstm3", l0=True)
            out = self.w["fc.weight"] @ np.concatenate([h1, h2, h3]) + self.w["fc.bias"]

            rest = out[1:].reshape(M, 6)
            pi = _softmax(rest[:, 0]); pi /= pi.sum()        # guard exact-sum for rng.choice
            j = int(rng.choice(M, p=pi))
            sx = max(np.exp(rest[j, 3]), 1e-2) * temperature
            sy = max(np.exp(rest[j, 4]), 1e-2) * temperature
            r = np.tanh(rest[j, 5])
            u1, u2 = rng.standard_normal(2)
            dx = rest[j, 1] + sx * u1
            dy = rest[j, 2] + sy * (r * u1 + np.sqrt(max(1.0 - r * r, 1e-8)) * u2)
            pen_up = 1.0 if rng.random() < _sigmoid(out[0]) else 0.0
            x = np.array([dx, dy, pen_up], np.float32)
            yield float(dx), float(dy), pen_up, phi

            if reached_end_at is None and int(phi.argmax()) == U - 1:
                reached_end_at = t
            if reached_end_at is not None and t - reached_end_at >= tail_steps:
                break
