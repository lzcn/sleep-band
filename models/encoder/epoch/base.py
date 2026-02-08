import torch
import torch.nn as nn


class BaseEpochEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.encoder = self._build(in_channels, out_channels, dropout, **kwargs)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _build(self, in_channels, out_channels, dropout, **kwargs) -> nn.Sequential:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.pool(x).squeeze(-1)
