import torch.nn as nn


class SleepDecoder(nn.Module):
    """
    Decode latent representations back into time-domain physiological signals.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_channels),
            representing latent feature embeddings for each epoch.

    Returns:
        torch.Tensor: Reconstructed signal of shape
            (batch_size, seq_len, out_channels, n_samples),
            where `n_samples` corresponds to the number of time points
            per epoch (e.g., 30 s x 100 Hz = 3000 for PSG data).
    """

    def __init__(self, in_channels=512, out_channels=2, dropout=0.1):
        super(SleepDecoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=5, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(512, 256, kernel_size=10, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(256, 128, kernel_size=10, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(128, 64, kernel_size=10, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(64, 64, kernel_size=5, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(64, out_channels, kernel_size=49, stride=15, padding=17, bias=False),
        )
        self.out_channels = out_channels

    def forward(self, x):
        b, n, d = x.shape
        x = x.view(b * n, d, 1)
        x = self.upsample(x)
        return x.view(b, n, self.out_channels, -1)
