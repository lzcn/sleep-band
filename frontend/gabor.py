import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import torch
from .base import BaseFrontend
import torch.nn.functional as F
from .registry import register_frontend


@register_frontend("gabor")
class GaborConv1d(BaseFrontend):
    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        filter_length: int = 101,
        fs: float = 100.0,
        min_freq: float = 0.5,
        max_freq: float = 50.0,
        min_sigma: float = 0.01,
        max_sigma: float = 0.1,
        log_scale: bool = True,
        output_envelope: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.fs = fs
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.output_envelope = output_envelope

        # Initialize frequencies
        if log_scale:
            init_freq = torch.linspace(math.log(min_freq), math.log(max_freq), num_filters).exp()
        else:
            init_freq = torch.linspace(min_freq, max_freq, num_filters)

        # Initialize sigma: proportional to 1/freq (constant Q factor)
        init_sigma = 1 / (init_freq * 2 * math.pi)
        init_sigma = torch.clamp(init_sigma, min=min_sigma, max=max_sigma)

        self.freq = nn.Parameter(init_freq.repeat(in_channels, 1))
        self.sigma = nn.Parameter(init_sigma.repeat(in_channels, 1))

        # time vector
        t = torch.linspace(0, filter_length - 1, steps=filter_length)
        t = (t - (filter_length - 1) / 2) / fs
        self.register_buffer("t", t.view(1, 1, -1))

        self.out_channels = in_channels * num_filters
        self.dropout = nn.Dropout1d(dropout)

    def _frequency_sigma_bounds(self):
        freq = torch.clamp(self.freq, min=self.min_freq, max=self.max_freq)
        sigma = torch.clamp(self.sigma, min=self.min_sigma, max=self.max_sigma)
        return freq, sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert C == self.in_channels, f"Input channel mismatch: {C} vs {self.in_channels}"

        freq, sigma = self._frequency_sigma_bounds()
        freq = freq.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)

        # Build Morlet filters
        gauss = torch.exp(-self.t**2 / (2 * sigma**2))
        sinusoid = torch.cos(2 * math.pi * freq * self.t)
        kernel = gauss * sinusoid

        # Normalize filters
        kernel = kernel / (kernel.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        # Reshape for grouped conv
        filters = kernel.view(self.in_channels * self.num_filters, 1, self.filter_length)

        # Apply grouped conv
        out = F.conv1d(x, filters, padding=self.filter_length // 2, groups=self.in_channels)

        if self.output_envelope:
            out = torch.abs(out)

        out = self.dropout(out)
        return out

    @torch.no_grad()
    def get_filter_params(self):
        freq, sigma = self._frequency_sigma_bounds()
        return freq.detach(), sigma.detach()

    @torch.no_grad()
    def get_frequency_response(self, n_fft=512, return_complex=False):
        freq, sigma = self._frequency_sigma_bounds()
        freq = freq.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)

        gauss = torch.exp(-self.t**2 / (2 * sigma**2))
        sinusoid = torch.cos(2 * math.pi * freq * self.t)
        signal = gauss * sinusoid
        signal = signal / (signal.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        fft = torch.fft.fft(signal, n=n_fft)
        freqs = torch.fft.fftfreq(n_fft, d=1.0 / self.fs)

        if return_complex:
            return fft.cpu().numpy(), freqs.cpu().numpy()
        magnitude = torch.abs(fft[..., : n_fft // 2]).cpu().numpy()
        freqs = freqs[: n_fft // 2].cpu().numpy()
        return magnitude, freqs


@register_frontend("constant_q")
class ConstantQGaborConv1d(BaseFrontend):
    """
    Strict constant-Q Morlet convolution (time-domain, truncated).

    sigma is NOT a free parameter:
        sigma = Q / (2*pi*f)

    This guarantees constant relative bandwidth across frequencies.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        filter_length: int = 101,
        fs: float = 100.0,
        min_freq: float = 0.5,
        max_freq: float = 50.0,
        q_init: float = 6.0,  # typical EEG-friendly value
        log_scale: bool = True,
        output_envelope: bool = True,
        learnable_Q: bool = True,  # keep False for cleanest design
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.fs = fs
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.output_envelope = output_envelope
        self.dropout = nn.Dropout1d(dropout)

        # Initialize center frequencies
        if log_scale:
            init_freq = torch.linspace(math.log(min_freq), math.log(max_freq), num_filters).exp()
        else:
            init_freq = torch.linspace(min_freq, max_freq, num_filters)

        self.freq = nn.Parameter(init_freq.repeat(in_channels, 1))

        # Q factor
        if learnable_Q:
            self.log_Q = nn.Parameter(torch.log(torch.tensor(q_init)))
        else:
            self.register_buffer("log_Q", torch.log(torch.tensor(q_init)))

        # Time axis
        t = torch.linspace(0, filter_length - 1, steps=filter_length)
        t = (t - (filter_length - 1) / 2) / fs
        self.register_buffer("t", t.view(1, 1, -1))

        self.out_channels = in_channels * num_filters

    def _bounded_freq(self):
        return torch.clamp(self.freq, self.min_freq, self.max_freq)

    def _compute_sigma(self, freq):
        Q = torch.exp(self.log_Q)
        sigma = Q / (2 * math.pi * freq)
        return sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        return: (B, C*K, T)
        """
        B, C, T = x.shape
        assert C == self.in_channels

        freq = self._bounded_freq().unsqueeze(-1)
        sigma = self._compute_sigma(freq)

        # Morlet kernel
        gauss = torch.exp(-self.t**2 / (2 * sigma**2))
        sinusoid = torch.cos(2 * math.pi * freq * self.t)
        morlet = gauss * sinusoid

        # L2 normalization
        morlet = morlet / (morlet.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        filters = morlet.view(self.in_channels * self.num_filters, 1, self.filter_length)

        out = F.conv1d(
            x,
            filters,
            padding=self.filter_length // 2,
            groups=self.in_channels,
        )

        if self.output_envelope:
            out = torch.abs(out)

        out = self.dropout(out)
        return out

    @torch.no_grad()
    def get_filter_params(self):
        freq = self._bounded_freq()
        sigma = self._compute_sigma(freq)
        return freq.detach(), sigma.detach()


@register_frontend("grouped_constant_q")
class GroupedConstantQConv1d(BaseFrontend):
    """
    Channel-wise + Group-wise Constant-Q Morlet convolution.

    For channel c and group g:
        sigma = Q[c, g] / (2 * pi * f)

    This constrains filters to a physiologically plausible
    frequency-scale manifold while allowing controlled flexibility.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        q_groups: list = (4, 1),  # e.g. [2, 1]
        filter_length: int = 101,
        fs: float = 100.0,
        min_freq: float = 0.5,
        max_freq: float = 50.0,
        q_init: float = 6.0,
        log_scale: bool = True,
        output_envelope: bool = True,
        learnable_Q: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert len(q_groups) == in_channels, "q_groups length must equal in_channels"

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.q_groups = q_groups
        self.filter_length = filter_length
        self.fs = fs
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.output_envelope = output_envelope
        self.dropout = nn.Dropout1d(dropout)
        # ------------------------------------------------------------
        # Initialize center frequencies (shared within each channel)
        # ------------------------------------------------------------
        if log_scale:
            init_freq = torch.linspace(math.log(min_freq), math.log(max_freq), num_filters).exp()
        else:
            init_freq = torch.linspace(min_freq, max_freq, num_filters)

        # (C, K)
        self.freq = nn.Parameter(init_freq.repeat(in_channels, 1))

        # ------------------------------------------------------------
        # Channel-wise, group-wise Q parameters
        # ------------------------------------------------------------
        self.log_Q = nn.ParameterList()

        for n_group in q_groups:
            q_init = torch.ones(n_group) * q_init
            if learnable_Q:
                self.log_Q.append(nn.Parameter(torch.log(q_init)))
            else:
                self.log_Q.append(nn.Parameter(torch.log(q_init), requires_grad=False))

        # ------------------------------------------------------------
        # Time axis (shared)
        # ------------------------------------------------------------
        t = torch.linspace(0, filter_length - 1, steps=filter_length)
        t = (t - (filter_length - 1) / 2) / fs
        self.register_buffer("t", t.view(1, 1, -1))

        self.out_channels = in_channels * num_filters

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _bounded_freq(self):
        return torch.clamp(self.freq, self.min_freq, self.max_freq)

    def _compute_sigma(self, freq):
        """
        freq: (C, K, 1)
        return sigma: (C, K, 1)
        """
        C, K, _ = freq.shape
        sigma = torch.zeros_like(freq)

        for c in range(C):
            n_group = self.q_groups[c]
            filters_per_group = K // n_group

            for g in range(n_group):
                q = torch.exp(self.log_Q[c][g])
                start = g * filters_per_group
                end = K if g == n_group - 1 else (g + 1) * filters_per_group
                sigma[c, start:end] = q / (2 * math.pi * freq[c, start:end])

        return sigma

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        return: (B, C*K, T)
        """
        B, C, T = x.shape
        assert C == self.in_channels

        freq = self._bounded_freq().unsqueeze(-1)  # (C, K, 1)
        sigma = self._compute_sigma(freq)  # (C, K, 1)

        # Morlet kernel
        gauss = torch.exp(-self.t**2 / (2 * sigma**2))
        sinusoid = torch.cos(2 * math.pi * freq * self.t)
        morlet = gauss * sinusoid

        # L2 normalization
        morlet = morlet / (morlet.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        filters = morlet.view(self.in_channels * self.num_filters, 1, self.filter_length)

        out = F.conv1d(
            x,
            filters,
            padding=self.filter_length // 2,
            groups=self.in_channels,
        )

        if self.output_envelope:
            out = torch.abs(out)
        out = self.dropout(out)
        return out


@register_frontend("complex_morlet")
class ComplexMorletConv1D(BaseFrontend):
    """
    Complex Morlet wavelet convolution in the time domain.

    Output = | x(t) * psi(t) |   if output_envelope=True
    Otherwise output = real( x * psi ).
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        filter_length: int = 201,
        fs: float = 100.0,
        min_freq: float = 0.5,
        max_freq: float = 50.0,
        min_sigma: float = 0.01,
        max_sigma: float = 0.1,
        log_scale: bool = True,
        output_envelope: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.fs = fs
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.output_envelope = output_envelope

        # ---------------------------------------------------------
        # Frequency initialization (Hz)
        # ---------------------------------------------------------
        if log_scale:
            init_freq = torch.linspace(math.log(min_freq), math.log(max_freq), num_filters).exp()
        else:
            init_freq = torch.linspace(min_freq, max_freq, num_filters)

        # Sigma initialization (seconds)
        init_sigma = 1 / (init_freq * 2 * math.pi)
        init_sigma = torch.clamp(init_sigma, min=min_sigma, max=max_sigma)

        # Learnable parameters (C, F)
        self.freq = nn.Parameter(init_freq.repeat(in_channels, 1))
        self.sigma = nn.Parameter(init_sigma.repeat(in_channels, 1))

        # Time vector (centered)
        t = torch.linspace(0, filter_length - 1, steps=filter_length)
        t = (t - (filter_length - 1) / 2) / fs
        self.register_buffer("t", t.view(1, 1, -1))

        self.out_channels = in_channels * num_filters

    # ---------------------------------------------------------
    # Clamp parameters into allowed ranges
    # ---------------------------------------------------------
    def _frequency_sigma_bounds(self):
        freq = torch.clamp(self.freq, min=self.min_freq, max=self.max_freq)
        sigma = torch.clamp(self.sigma, min=self.min_sigma, max=self.max_sigma)
        return freq, sigma

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert C == self.in_channels, f"Input channel mismatch: {C} vs {self.in_channels}"

        freq, sigma = self._frequency_sigma_bounds()
        freq = freq.unsqueeze(-1)  # (C, F, 1)
        sigma = sigma.unsqueeze(-1)  # (C, F, 1)

        # -----------------------------------------------------
        # Build analytic (complex) Morlet filters
        # -----------------------------------------------------
        gauss = torch.exp(-self.t**2 / (2 * sigma**2))  # (C, F, L)
        sinusoid = torch.exp(1j * 2 * math.pi * freq * self.t)  # complex

        # Zero-mean correction
        kappa = torch.exp(-((2 * math.pi * freq * sigma) ** 2) / 2)
        sinusoid = sinusoid - kappa  # enforce zero mean

        morlet = gauss * sinusoid  # complex wavelet

        # Normalize L2
        morlet = morlet / (morlet.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        # -----------------------------------------------------
        # Grouped convolution with complex filters
        # -----------------------------------------------------
        real_f = morlet.real.view(self.out_channels, 1, self.filter_length)
        imag_f = morlet.imag.view(self.out_channels, 1, self.filter_length)

        xr = F.conv1d(x, real_f, padding=self.filter_length // 2, groups=self.in_channels)
        xi = F.conv1d(x, imag_f, padding=self.filter_length // 2, groups=self.in_channels)

        y = torch.complex(xr, xi)  # (B, C*F, T)

        # Envelope or real output
        if self.output_envelope:
            return torch.abs(y)
        return y.real

    # ---------------------------------------------------------
    # Retrieve learned parameters
    # ---------------------------------------------------------
    @torch.no_grad()
    def get_filter_params(self):
        freq, sigma = self._frequency_sigma_bounds()
        return freq.cpu().numpy(), sigma.cpu().numpy()

    # ---------------------------------------------------------
    # Frequency response of filters
    # ---------------------------------------------------------
    @torch.no_grad()
    def get_frequency_response(self, n_fft=512, return_complex=False):
        freq, sigma = self._frequency_sigma_bounds()
        freq = freq.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)

        gauss = torch.exp(-self.t**2 / (2 * sigma**2))
        sinusoid = torch.exp(1j * 2 * math.pi * freq * self.t)
        kappa = torch.exp(-((2 * math.pi * freq * sigma) ** 2) / 2)
        sinusoid = sinusoid - kappa
        signal = gauss * sinusoid
        signal = signal / (signal.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        fft = torch.fft.fft(signal, n=n_fft)
        freqs = torch.fft.fftfreq(n_fft, d=1.0 / self.fs)

        if return_complex:
            return fft.cpu().numpy(), freqs.cpu().numpy()

        magnitude = torch.abs(fft[..., : n_fft // 2]).cpu().numpy()
        freqs = freqs[: n_fft // 2].cpu().numpy()
        return magnitude, freqs
