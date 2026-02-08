import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decoder import SleepDecoder
from models.encoder import SleepEncoder
from datasets import SleepDataModule
from methods import DomainContext
import utils
from einops import rearrange


class SleepModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 512,
        dropout: float = 0.1,
        num_classes: int = 5,
        frontend: str = None,
        epoch_encoder: str = "conv",
        use_decoder=False,
    ):
        super().__init__()
        self.num_classes = num_classes

        out_channels = in_channels
        if frontend is not None:
            self.frontend = utils.build_frontend(frontend)
            in_channels = self.frontend.out_channels
        else:
            self.frontend = None
            in_channels = in_channels

        self.encoder = SleepEncoder(in_channels, d_model, dropout, epoch_encoder=epoch_encoder)
        self.decoder = SleepDecoder(d_model, out_channels, dropout) if use_decoder else None
        self.classifier = nn.Linear(d_model, num_classes)

    def filter(self, x):
        if self.frontend is not None:
            b = len(x)
            x = rearrange(x, "b n c t -> (b n) c t")
            x = self.frontend(x)
            x = rearrange(x, "(b n) c t -> b n c t", b=b)

        return x

    def forward(self, x):

        if self.frontend is not None:
            b = len(x)
            x = rearrange(x, "b n c t -> (b n) c t")
            x = self.frontend(x)
            x = rearrange(x, "(b n) c t -> b n c t", b=b)

        feat = self.encoder(x)  # (B, N, D)
        logits = self.classifier(feat)  # (B, N, C)

        # ---- optional reconstruction ----
        if self.decoder is not None:
            recon = self.decoder(feat)
        else:
            recon = None
        return logits, feat, recon


class SleepBase(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 512,
        dropout: float = 0.1,
        num_classes: int = 5,
        frontend: str = None,
        epoch_encoder: str = "conv",
        label_smoothing: float = 0.1,
        # for domain generalization method
        domain_method: str = None,
        recon_weight=0.0,
        data_module: SleepDataModule = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.model = SleepModel(
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
            num_classes=num_classes,
            frontend=frontend,
            epoch_encoder=epoch_encoder,
            use_decoder=(recon_weight > 0),
        )

        if domain_method is not None:
            self.domain_method = utils.build_domain_method(domain_method, num_domains=data_module.num_source_domains)
        else:
            self.domain_method = None

        # Loss functions
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.recon_weight = recon_weight

    def forward(self, x, y, domains=None, context: DomainContext | None = None):
        """x: [B, N, C, T], y: [B, N], z: [B]"""

        logits, h, recon = self.model(x)

        total_loss, metrics = 0.0, {}

        # ---- classification loss ----
        loss = self.criterion(logits.transpose(1, 2), y)
        metrics["Loss/ce"] = loss.item()
        total_loss += loss

        # ---- optional reconstruction ----
        if self.recon_weight > 0.0:
            recon_loss = F.mse_loss(recon, x)
            total_loss += self.recon_weight * recon_loss
            metrics["Loss/recon"] = recon_loss.item()

        # ---- optional domain method ----
        if self.domain_method is not None:
            domain_loss, domain_metrics = self.domain_method(
                model=self.model,
                features=h,
                logits=logits,
                labels=y,
                domain_labels=domains,
                context=context,
            )
            total_loss += domain_loss
            metrics.update(domain_metrics)

        metrics["Loss/total"] = total_loss.item()
        return total_loss, metrics

    @torch.no_grad()
    def inference(self, x):
        logits, _, _ = self.model(x)
        return logits


if __name__ == "__main__":

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    def print_model_stats(name, model):
        total = count_parameters(model)
        encoder = count_parameters(model.encoder.epoch_encoder)
        ratio = encoder / total * 100
        print(f"{name} Total Parameters: {total}")
        print(f"{name} Encoder Parameters: {encoder}")
        print(f"{name} Encoder-to-Total Ratio: {ratio:.2f}%\n")
        return total, encoder

    # Instantiate models
    sleep_base = SleepModel()
    sleep_band = SleepModel(
        frontend={"name": "constant_q", "num_filters": 20, "in_channels": 2},
        epoch_encoder="spectral",
    )

    # Print stats for both models and get their parameters
    sleep_base_total, sleep_base_encoder = print_model_stats("Standard Model", sleep_base)
    sleep_band_total, sleep_band_encoder = print_model_stats("Sleep-Band Model", sleep_band)

    # Calculate the relative ratios of Sleep-Band parameters compared to Sleep-Base
    total_ratio = (sleep_band_total / sleep_base_total) * 100
    encoder_ratio = (sleep_band_encoder / sleep_base_encoder) * 100

    print(f"Sleep-Band Model Total Parameters as a Percentage of Sleep-Base Model: {total_ratio:.2f}%")
    print(f"Sleep-Band Model Encoder Parameters as a Percentage of Sleep-Base Model: {encoder_ratio:.2f}%")
