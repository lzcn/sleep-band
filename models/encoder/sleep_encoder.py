import torch
import torch.nn as nn
from einops import rearrange

from .epoch import create_epoch_encoder
from .sequence import create_sequence_encoder


class SleepEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        dropout: float = 0.1,
        epoch_encoder: str = "conv",
        sequence_encoder: str = "transformer",
    ):
        super().__init__()

        self.epoch_encoder = create_epoch_encoder(
            mode=epoch_encoder,
            in_channels=in_channels,
            out_channels=d_model,
            dropout=dropout,
        )

        self.sequence_encoder = create_sequence_encoder(
            mode=sequence_encoder,
            d_model=d_model,
            dropout=dropout,
        )

        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = rearrange(x, "b n c t -> (b n) c t")
        x = self.epoch_encoder(x)
        x_epoch = rearrange(x, "(b n) c -> b n c", b=b)
        x_seq = self.sequence_encoder(x_epoch)
        h = self.proj(x_seq)
        return h
