import torch
import torch.nn as nn

from .base import BaseEpochEncoder
from .registry import register_epoch_encoder


class SpectralIntegration1d(nn.Module):
    """depthwise -> pointwise -> channel-wise calibration"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
        reduction: int = 4,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        # band-wise temporal filtering
        self.filter = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

        # linear integration across bands
        self.mix = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

        # adaptive spectral calibration
        hidden_dim = max(out_channels // reduction, 4)
        self.calib = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.filter(x)
        x = self.mix(x)
        return x * self.calib(x)


@register_epoch_encoder("spectral")
class SpectralEncoder(BaseEpochEncoder):
    def _build(self, in_channels, out_channels, dropout):

        return nn.Sequential(
            # -------- Block 1 --------
            SpectralIntegration1d(in_channels, 64, kernel_size=49, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            nn.Dropout(dropout),
            # -------- Block 2 --------
            SpectralIntegration1d(64, 128, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            # -------- Block 3 --------
            SpectralIntegration1d(128, 256, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            # -------- Block 4 --------
            SpectralIntegration1d(256, out_channels, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
        )


class SpectralIntegrationBlock(nn.Module):
    """depthwise -> pointwise -> bn -> gelu -> channel-wise calibration"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
        reduction: int = 4,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        # band-wise temporal filtering
        self.filter = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

        # linear integration across bands
        self.mix = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()

        # adaptive spectral calibration
        hidden_dim = max(out_channels // reduction, 4)
        self.calib = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.filter(x)
        x = self.mix(x)
        x = self.bn(x)
        x = self.relu(x)
        return x * self.calib(x)


@register_epoch_encoder("spectral_post_calib")
class SpectralConvEncoder(BaseEpochEncoder):
    def _build(self, in_channels, out_channels, dropout):

        return nn.Sequential(
            # -------- Block 1 --------
            SpectralIntegrationBlock(in_channels, 64, kernel_size=49, stride=6, bias=False, padding=24),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            nn.Dropout(dropout),
            # -------- Block 2 --------
            SpectralIntegrationBlock(64, 128, kernel_size=9, stride=1, bias=False, padding=4),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            # -------- Block 3 --------
            SpectralIntegrationBlock(128, 256, kernel_size=9, stride=1, bias=False, padding=4),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            # -------- Block 4 --------
            SpectralIntegrationBlock(256, out_channels, kernel_size=9, stride=1, bias=False, padding=4),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
        )
