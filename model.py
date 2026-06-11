import math
import torch
import torch.nn as nn

M = 20  # number of mixture components (paper sec. 4.2)


class HandwritingModel(nn.Module):
    def __init__(self, hidden_size=400, num_layers=3):
        super().__init__()
        # input: (dx, dy, pen_up) = 3 numbers
        self.lstm = nn.LSTM(3, hidden_size, num_layers=num_layers, batch_first=True)


        # output: 121 raw numbers
        #   1 pen_up logit  +  M * (weight, mu_x, mu_y, sig_x, sig_y, rho)
        self.fc = nn.Linear(hidden_size, 1 + M * 6) 

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)   # out: (batch, time, hidden_size)
        out = self.fc(out)                   # out: (batch, time, 121)
        return out, hidden


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
