import torch.nn as nn

from .base import BaseEpochEncoder
from .registry import register_epoch_encoder


@register_epoch_encoder("conv")
class ConvEncoder(BaseEpochEncoder):
    def _build(self, in_channels, out_channels, dropout):

        return nn.Sequential(
            # -------- Block 1 --------
            nn.Conv1d(in_channels, 64, kernel_size=49, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            nn.Dropout(dropout),
            # -------- Block 2 --------
            nn.Conv1d(64, 128, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            # -------- Block 3 --------
            nn.Conv1d(128, 256, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            # -------- Block 4 --------
            nn.Conv1d(256, out_channels, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
        )
