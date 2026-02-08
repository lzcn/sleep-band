import math

import torch
import torch.nn.functional as F
from torch import nn
from .base import BaseFrontend
from .registry import register_frontend


@register_frontend("sinc")
class SincConv1d(BaseFrontend):
    def __init__(
        self,
        in_channels,
        num_filters,
        filter_length=51,
        fs=100.0,
        min_cutoff_freq=0.5,
        min_bandwidth: float = 5.0,
        stride=1,
        normalize_filters=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.fs = fs
        self.min_cutoff_freq = min_cutoff_freq
        self.min_bandwidth = min_bandwidth
        self.stride = stride
        self.normalize_filters = normalize_filters

        # [F] initial low cutoff frequencies and bandwidths
        cutoff_freq = torch.linspace(0, 30.0 - min_bandwidth - min_cutoff_freq, num_filters)
        bandwidth = torch.full((num_filters,), min_bandwidth)

        # Parameters per input channel: shape [C, F]
        self.cutoff_freq = nn.Parameter(cutoff_freq.repeat(in_channels, 1))
        self.bandwidth = nn.Parameter(bandwidth.repeat(in_channels, 1))

        # Time vector n and Hamming window
        n = torch.linspace(0, filter_length - 1, steps=filter_length)
        n = (n - (filter_length - 1) / 2) / fs  # Centered time in seconds
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * (torch.arange(filter_length) / (filter_length - 1)))

        # Register as buffers (non-trainable, device aware)
        self.register_buffer("n", n.view(1, 1, -1))  # shape [1, 1, K]
        self.register_buffer("window", window.view(1, 1, -1))  # shape [1, 1, K]
        self.out_channels = in_channels * num_filters

    def _frequency_bounds(self):
        """Returns:
        Tuple of (low_freqs, high_freqs) as numpy arrays of shape [C, F]
        """
        low = self.min_cutoff_freq + torch.abs(self.cutoff_freq)
        high = torch.clamp(low + self.min_bandwidth + torch.abs(self.bandwidth), max=self.fs / 2)
        return low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped convolution with learnable band-pass sinc filters.

        Args:
            x (Tensor): Input signal of shape [B, C, T]

        Returns:
            Tensor: Output of shape [B, C * F, T]
        """
        assert x.shape[1] == self.in_channels, "Input channel mismatch"
        # Compute frequency bounds (low, high) in Hz: shape [C, F]
        low, high = self._frequency_bounds()
        # low = self.min_cutoff_freq + torch.abs(self.cutoff_freq)
        # high = torch.clamp(low + self.min_bandwidth + torch.abs(self.bandwidth), max=self.fs / 2)

        # Reshape for broadcasting with time axis: [C, F, 1]
        f1 = low.unsqueeze(-1)
        f2 = high.unsqueeze(-1)

        # Construct band-pass filters using windowed sinc: [C, F, K]
        bandwidth = torch.clamp(f2 - f1, min=1e-3)
        band_pass = (2 * f2 * torch.sinc(2 * f2 * self.n)) - (2 * f1 * torch.sinc(2 * f1 * self.n))
        band_pass *= self.window
        # band_pass /= 2 * bandwidth
        if self.normalize_filters:
            band_pass = band_pass / (band_pass.norm(p=2, dim=2, keepdim=True) + 1e-6)  # Normalize filters
        # Prepare filters for grouped conv: [C * F, 1, K]
        filters = band_pass.view(self.in_channels * self.num_filters, 1, self.filter_length)

        # Apply grouped 1D convolution
        # return F.conv1d(x, filters, stride=self.stride, groups=self.in_channels)
        return F.conv1d(x, filters, padding=self.filter_length // 2, stride=self.stride, groups=self.in_channels)

    @torch.no_grad()
    def get_filter_freqs(self):
        """Returns:
        Tuple of (low_freqs, high_freqs) as numpy arrays of shape [C, F]
        """
        low, high = self._frequency_bounds()
        low = low.cpu().numpy()
        high = high.cpu().numpy()
        return low, high

    @torch.no_grad()
    def get_frequency_response(self, n_fft=512, return_complex=False):
        """Compute frequency response of all filters.

        Returns:
            Tuple of (magnitude, frequencies) as numpy arrays
            - magnitude: shape [C, F, n_fft // 2]
            - frequencies: shape [n_fft // 2]
        """
        low, high = self._frequency_bounds()
        f1 = low.unsqueeze(-1)
        f2 = high.unsqueeze(-1)

        band_pass = 2 * f2 * torch.sinc(2 * f2 * self.n) - 2 * f1 * torch.sinc(2 * f1 * self.n)
        band_pass *= self.window
        band_pass /= 2 * f2 - 2 * f1

        fft = torch.fft.fft(band_pass, n=n_fft)  # [C, F, n_fft]
        freqs = torch.fft.fftfreq(n_fft, d=1.0 / self.fs)
        if return_complex:
            return fft.cpu().numpy(), freqs.cpu().numpy()
        magnitude = torch.abs(fft[..., : n_fft // 2]).cpu().numpy()
        freqs = freqs[: n_fft // 2].cpu().numpy()
        return magnitude, freqs
