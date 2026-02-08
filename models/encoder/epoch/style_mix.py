import torch
import torch.nn as nn
from torch.distributions import Beta

from .base import BaseEpochEncoder
from .registry import register_epoch_encoder


class StyleMix(nn.Module):
    def __init__(self, alpha=0.1, p=0.5, eps=1e-6):
        super().__init__()
        self.alpha, self.p, self.eps = alpha, p, eps
        self.beta = Beta(alpha, alpha)

    def forward(self, x):
        if not self.training or torch.rand(1, device=x.device) > self.p:
            return x

        b = len(x)
        mu = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()

        perm = torch.randperm(b, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]

        lam = self.beta.sample((b, 1, 1)).to(x.device)

        x_normed = (x - mu) / sig
        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2

        return x_normed * sig_mix + mu_mix


@register_epoch_encoder("style_mix")
class StyleMixEncoder(BaseEpochEncoder):
    def _build(self, in_channels, out_channels, dropout):
        return nn.Sequential(
            # -------- Block 1 --------
            nn.Conv1d(in_channels, 64, kernel_size=49, stride=6, padding=24, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            StyleMix(p=0.1),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            nn.Dropout(dropout),
            # -------- Block 2 --------
            nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            StyleMix(p=0.1),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            # -------- Block 3 --------
            nn.Conv1d(128, 256, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            StyleMix(p=0.1),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            # -------- Block 4 --------
            nn.Conv1d(256, out_channels, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
        )
