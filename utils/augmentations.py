import torch
import torch.fft
import torch.nn as nn


class FourierAugmentor(nn.Module):
    def __init__(self, max_lambda: float = 0.5, random_lambda: bool = True, pair_dim: str = "batch"):
        super().__init__()
        self.max_lambda = max_lambda
        self.random_lambda = random_lambda
        self.pair_dim = pair_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, E, C, T], real-valued.
        Returns:
            Tensor: Augmented tensor of shape [B, E, C, T].
        """
        B, E, C, T = x.shape
        device = x.device

        # -------- Pairing strategy --------
        if self.pair_dim == "batch":
            x1 = x.reshape(B, -1, T)
            x2 = x[torch.randperm(B, device=device)].reshape(B, -1, T)

        elif self.pair_dim == "epoch":
            x_perm = x.permute(1, 0, 2, 3)
            x1 = x_perm.reshape(E, -1, T)
            x2 = x_perm[torch.randperm(E, device=device)].reshape(E, -1, T)

        elif self.pair_dim == "channel":
            x_perm = x.permute(2, 0, 1, 3)
            x1 = x_perm.reshape(C, -1, T)
            x2 = x_perm[torch.randperm(C, device=device)].reshape(C, -1, T)

        elif self.pair_dim == "all":
            x1 = x.reshape(-1, T)
            x2 = x1[torch.randperm(x1.shape[0], device=device)]

        else:
            raise ValueError(f"Unsupported pair_dim: {self.pair_dim}")

        x1 = x1.reshape(-1, T)
        x2 = x2.reshape(-1, T)

        # -------- Real FFT --------
        fft_1 = torch.fft.rfft(x1, dim=-1)
        fft_2 = torch.fft.rfft(x2, dim=-1)

        A1, A2 = torch.abs(fft_1), torch.abs(fft_2)
        P1 = torch.angle(fft_1)

        # -------- Sample λ --------
        if self.random_lambda:
            lambdas = torch.rand(A1.shape[0], 1, device=device) * self.max_lambda
        else:
            lambdas = torch.full((A1.shape[0], 1), self.max_lambda, device=device)

        # -------- Amplitude mixing --------
        A_mix = (1.0 - lambdas) * A1 + lambdas * A2

        # -------- Reconstruction --------
        fft_new = A_mix * torch.exp(1j * P1)
        x_aug = torch.fft.irfft(fft_new, n=T, dim=-1)

        return x_aug.view(B, E, C, T)


class FrequencyBandAugmentor(nn.Module):
    """
    Frequency domain data augmentation for EEG/EOG signals.

    Performs band-specific augmentation including:
    - Spectral mixing (MixUp in frequency domain)
    - Phase jittering for temporal variation
    - Random masking for robustness

    Args:
        fs (int): Sampling frequency in Hz
        phase_jitter_std (float): Standard deviation for phase jittering
        mask_prob (float): Probability of masking frequency components
        lambda_max (float): Maximum mixing coefficient for spectral mixing
    """

    def __init__(self, fs=100, phase_jitter_std=0.02, mask_prob=0.05, max_lambda=0.5):
        super().__init__()
        self.fs = fs
        self.phase_jitter_std = phase_jitter_std
        self.mask_prob = mask_prob
        self.max_lambda = max_lambda

        # Frequency bands (Hz)
        self.bands = [
            [(0.5, 4), (4, 8), (8, 12), (11, 16), (13, 30)],  # EEG
            [(0.5, 2), (0.5, 4), (0.5, 8), (0.5, 10)],  # EOG
        ]

    def forward(self, x):
        """
        Apply frequency-domain augmentation to input signals.

        Args:
            x (torch.Tensor): Input tensor [B, E, C, T] where:
                - B: batch size
                - E: number of epochs/windows
                - C: number of channels (EEG, EOG)
                - T: time samples per epoch

        Returns:
            torch.Tensor: Augmented tensor with same shape [B, E, C, T]
        """
        B, E, C, T = x.shape
        device = x.device
        x = x.reshape(B * E, C, T)
        # Transform to frequency domain [*, C, F]
        X_f = torch.fft.rfft(x, dim=-1)
        A_f = torch.abs(X_f)
        P_f = torch.angle(X_f)
        freqs = torch.fft.rfftfreq(T, d=1 / self.fs).to(device)

        X_f_aug = torch.zeros_like(X_f, device=device)
        A_f_ref = A_f.clone()

        # Process each channel separately
        for c in range(C):
            for f_lo, f_hi in self.bands[c]:
                band = (freqs >= f_lo) & (freqs <= f_hi)
                if not band.any():
                    continue
                idx = band.nonzero(as_tuple=False).squeeze(1)

                # Extract amplitude and phase
                A_b = A_f[:, c, idx]
                P_b = P_f[:, c, idx]
                perm = torch.randperm(B * E, device=device)
                A_b_ref = A_f_ref[perm][:, c, idx]

                # Spectral mixing: mix with random permutation
                lam = torch.rand(B * E, 1, device=device) * self.max_lambda
                A_b_mix = (1 - lam) * A_b + lam * A_b_ref

                # Phase jittering for temporal variation
                P_jitter = P_b + torch.randn_like(P_b) * self.phase_jitter_std

                # Random masking for robustness
                mask = (torch.rand_like(A_b_mix) > self.mask_prob).float()
                A_b_mix = A_b_mix * mask

                # Reconstruct complex spectrum
                X_b = A_b_mix * torch.exp(1j * P_jitter)
                X_f_aug[:, c, idx] = X_b

        # Transform back to the time domain
        x_aug = torch.fft.irfft(X_f_aug, n=T, dim=-1).real
        x_aug = x_aug.reshape(B, E, C, T)
        return x_aug
