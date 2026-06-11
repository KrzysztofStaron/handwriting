import math
import torch
import torch.nn as nn

M = 20  # number of mixture components (paper sec. 4.2)
K = 10  # number of windows

class HandwritingModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=400):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # layer 1 sees: (dx,dy,pen)=3  +  the window w (size = vocab_size)
        self.lstm1 = nn.LSTMCell(3 + vocab_size, hidden_size)
        # upper layers see: input(3) + lower hidden + window   (skip connections, eq 53)
        self.lstm2 = nn.LSTMCell(3 + hidden_size + vocab_size, hidden_size)
        self.lstm3 = nn.LSTMCell(3 + hidden_size + vocab_size, hidden_size)

        self.window_fc = nn.Linear(hidden_size, 3 * K)   # eq 48: the ONLY window weights
        # output reads from all 3 layers (skip connections to output)
        self.fc = nn.Linear(3 * hidden_size, 1 + M * 6)

    def compute_window(self, h1, c, c_mask, kappa_prev):
        """Soft window over the text c (eq. 46-51).

        h1         : (B, H)      first layer hidden state
        c          : (B, U, V)   one-hot text
        c_mask     : (B, U)      1 = real char, 0 = padding
        kappa_prev : (B, K)      window location from the previous timestep

        returns w (B, V), kappa (B, K), phi (B, U).
        """
        p = self.window_fc(h1)                          # (B, 3K)  eq 48
        alpha = torch.exp(p[:, :K])                     # (B, K)   eq 49
        beta  = torch.exp(p[:, K:2 * K])                # (B, K)   eq 50
        kappa = kappa_prev + torch.exp(p[:, 2 * K:])    # (B, K)   eq 51 (slides forward)

        U = c.shape[1]
        u = torch.arange(U, device=c.device).float()    # (U,) character positions
        # phi(t,u) = sum_k alpha_k * exp(-beta_k (kappa_k - u)^2)   eq 46 (NOT normalized)
        phi = (alpha.unsqueeze(-1)
               * torch.exp(-beta.unsqueeze(-1) * (kappa.unsqueeze(-1) - u) ** 2)
               ).sum(dim=1)                             # (B, U)
        phi = phi * c_mask                              # ignore padded text positions
        w = torch.bmm(phi.unsqueeze(1), c).squeeze(1)   # (B, V)   eq 47
        return w, kappa, phi

    def forward(self, x, c, c_mask, hidden=None):
        B, T, _ = x.shape
        H, V = self.hidden_size, self.vocab_size
        dev = x.device

        if hidden is None:
            z = lambda: torch.zeros(B, H, device=dev)
            h1, c1, h2, c2, h3, c3 = z(), z(), z(), z(), z(), z()
            w = torch.zeros(B, V, device=dev)           # no window yet at t=0
            kappa = torch.zeros(B, K, device=dev)       # window starts at text position 0
        else:
            h1, c1, h2, c2, h3, c3, w, kappa = hidden

        outs, phis = [], []
        for t in range(T):
            xt = x[:, t]                                          # (B, 3)
            h1, c1 = self.lstm1(torch.cat([xt, w], 1), (h1, c1))  # layer 1 reads w_{t-1}
            w, kappa, phi = self.compute_window(h1, c, c_mask, kappa)  # new window w_t
            h2, c2 = self.lstm2(torch.cat([xt, h1, w], 1), (h2, c2))   # upper layers read w_t
            h3, c3 = self.lstm3(torch.cat([xt, h2, w], 1), (h3, c3))
            outs.append(self.fc(torch.cat([h1, h2, h3], 1)))     # (B, 121)
            phis.append(phi)

        out = torch.stack(outs, dim=1)                           # (B, T, 121)
        hidden = (h1, c1, h2, c2, h3, c3, w, kappa)
        return out, hidden, torch.stack(phis, dim=1)             # phis: (B, T, U)



# We take the last linear layer and split it into params of Gaussians (eq. 18-22).
def split_params(out):
    """Split raw outputs into mixture parameters (eq. 18-22)."""
    
    pen_logit = out[:, :, 0]
    rest = out[:, :, 1:].reshape(*out.shape[:2], M, 6)

    pi    = torch.softmax(rest[:, :, :, 0], dim=-1)  # weights, sum to 1
    mu_x  = rest[:, :, :, 1]                          # x means
    mu_y  = rest[:, :, :, 2]                          # y means
    sig_x = torch.exp(rest[:, :, :, 3]).clamp(min=1e-2)  # x std devs, > 0
    sig_y = torch.exp(rest[:, :, :, 4]).clamp(min=1e-2)  # y std devs, > 0
    rho   = torch.tanh(rest[:, :, :, 5])              # correlation, (-1, 1)
    pen   = torch.sigmoid(pen_logit)                  # pen_up probability

    return pi, mu_x, mu_y, sig_x, sig_y, rho, pen


# Given that the model predicted this Gaussian, how well does the actual point fit?
# N(x | mu, sigma, rho)
def gaussian_2d(dx, dy, mu_x, mu_y, sig_x, sig_y, rho):
    """Bivariate Gaussian probability density (equations 24 and 25)."""
    # x -> 1, y -> 2 
    
    z1 = ((dx - mu_x) ** 2) / (sig_x ** 2)
    z2 = ((dy - mu_y) ** 2) / (sig_y ** 2)
    z3 = (2*rho * (dx - mu_x) * (dy - mu_y)) / (sig_x * sig_y)
    Z = z1 + z2 - z3

    # Density prob
    mul1 = 1/(2*math.pi * sig_x * sig_y * torch.sqrt(1 - rho**2))
    mul2 = torch.exp(-Z / (2 * (1 - rho ** 2)))
    N = mul1 * mul2
    return N

def sequence_loss(out, y, mask):
    # y => (batch, time, 3)
    dx = y[:, :, 0].unsqueeze(-1)
    dy = y[:, :, 1].unsqueeze(-1)
    pen_up = y[:, :, 2]


    # ? j - the amount of gausians ig
    # e_t - pen
    # eveything but pen is an array of gausian paramneters
    pi, mu_x, mu_y, sig_x, sig_y, rho, pen = split_params(out)

    gauss = gaussian_2d(dx, dy, mu_x, mu_y, sig_x, sig_y, rho)
    mixture_prob = (pi * gauss).sum(dim=-1)

    eps = 1e-8
    nll_xy = -torch.log(mixture_prob + eps)
    nll_pen = -(pen_up * torch.log(pen + eps) + (1 - pen_up) * torch.log(1 - pen + eps))

    nll = (nll_xy + nll_pen) * mask
    loss = nll.sum() / mask.sum()

    return loss


# U -> Len of character sequence
# C -> Character sequence
# T -> Len of data sequence
# x -> data sequence 
