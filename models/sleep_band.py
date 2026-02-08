import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from datasets import SleepDataModule
from models.sleep_base import SleepModel

logger = utils.logger.get_logger(__name__)


class SleepBand(nn.Module):

    def __init__(
        self,
        # Model architecture
        in_channels=2,
        d_model=512,
        dropout=0.1,
        num_classes=5,
        # -- Filter --
        frontend=None,
        # -- Epoch Encoder --
        epoch_encoder="spectral",
        # -- Augmentation configuration ---
        data_augmentation="fourier",
        phase_jitter_std=0.02,
        mask_prob=0.05,
        max_lambda=0.5,
        # Knowledge distillation configuration
        init_copy=False,
        consistency="kl",
        T=10.0,  # Temperature for knowledge distillation
        beta=2.0,  # Weight for consistency loss
        momentum=0.999,  # Momentum for teacher updates
        # loss
        recon_weight=0.0,
        data_module: SleepDataModule = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        # co-teacher distillation
        self.consistency = consistency
        self.T = T
        self.beta = beta
        self.momentum = momentum

        self.recon_weight = recon_weight

        self.student = SleepModel(
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
            num_classes=num_classes,
            frontend=frontend,
            epoch_encoder=epoch_encoder,
            use_decoder=(recon_weight > 0),
        )
        self.teacher = SleepModel(
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
            num_classes=num_classes,
            frontend=frontend,
            epoch_encoder=epoch_encoder,
            use_decoder=(recon_weight > 0),
        )
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters(), strict=False):
            if init_copy:  # Disabling it slightly improves performance
                t_param.data.copy_(s_param.data)
            t_param.requires_grad = False

        # Data augmentation module
        self.data_augmentation = data_augmentation
        if data_augmentation == "bandwise":
            self.augmentor = utils.augmentations.FrequencyBandAugmentor(
                phase_jitter_std=phase_jitter_std,
                mask_prob=mask_prob,
                max_lambda=max_lambda,
            )
        else:
            self.augmentor = utils.augmentations.FourierAugmentor(max_lambda=max_lambda)
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y, *args, **kwargs):
        """
        Args:
            x: [B, N, C, T] input signals.
            y: [B, N] labels.
        Returns:
            total_loss, metrics
        """

        total_loss = 0.0
        metrics = {}

        # Generate augmented version of input
        x_ori, x_aug = x, self.augmentor(x)

        # Student predictions on original and augmented data
        logits_s_ori, _, recon_s_ori = self.student(x_ori)
        logits_s_aug, _, recon_s_aug = self.student(x_aug)

        # Supervised classification loss (student only)
        loss_sup_ori = self.criterion(logits_s_ori.transpose(1, 2), y)
        loss_sup_aug = self.criterion(logits_s_aug.transpose(1, 2), y)
        loss_sup = 0.5 * (loss_sup_ori + loss_sup_aug)

        metrics["Loss/sup"] = loss_sup.item()
        total_loss += loss_sup

        if self.recon_weight > 0.0:
            ae_ori = F.mse_loss(recon_s_ori, x_ori, reduction="mean")
            ae_aug = F.mse_loss(recon_s_aug, x_aug, reduction="mean")
            ae_loss = 0.5 * (ae_ori + ae_aug)
            total_loss += self.recon_weight * ae_loss
            metrics["Loss/ae"] = ae_loss.item()

        with torch.no_grad():
            logits_t_ori, _, _ = self.teacher(x_ori)
            logits_t_aug, _, _ = self.teacher(x_aug)
        loss_kd_aug = self.consistency_loss(logits_s_ori, logits_t_aug)
        loss_kd_ori = self.consistency_loss(logits_s_aug, logits_t_ori)
        loss_consistency = 0.5 * (loss_kd_aug + loss_kd_ori)

        total_loss += self.beta * loss_consistency
        metrics["Loss/consistency"] = loss_consistency.item()

        metrics["Loss/total"] = total_loss.item()

        return total_loss, metrics

    def consistency_loss(self, student_logits, teacher_logits):
        if self.consistency == "mse":
            return self._mse_loss(student_logits, teacher_logits)
        elif self.consistency == "kl":
            return self._kd_loss(student_logits, teacher_logits)
        else:
            raise ValueError(f"Unknown consistency loss type: {self.consistency}")

    def _mse_loss(self, student_logits, teacher_logits):
        """MSE loss between student and teacher logits."""
        return F.mse_loss(student_logits, teacher_logits, reduction="mean")

    def _kd_loss(self, student_logits, teacher_logits):
        """KL divergence loss between student and teacher."""
        T = self.T
        p_s = F.log_softmax(student_logits / T, dim=-1)
        p_t = F.softmax(teacher_logits / T, dim=-1)
        kl = F.kl_div(p_s, p_t, reduction="none").sum(dim=-1).mean()
        return T**2 * kl

    @staticmethod
    def _corr_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Self-similarity (correlation) consistency loss."""
        s = F.normalize(student_logits, dim=-1)  # [B, N, C]
        t = F.normalize(teacher_logits, dim=-1)  # [B, N, C]

        corr_s = torch.einsum("bnc,bmc->bnm", s, s)
        corr_t = torch.einsum("bnc,bmc->bnm", t, t)

        return F.mse_loss(corr_s, corr_t)

    @torch.no_grad()
    def post_step(self):
        """Momentum update for teacher network."""
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters(), strict=False):
            t_param.data = self.momentum * t_param.data + (1 - self.momentum) * s_param.data

    @torch.no_grad()
    def inference(self, x):
        """Inference using student network."""
        logits, _, _ = self.student(x)
        return logits
