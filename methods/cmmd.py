import torch
from typing import Dict, Tuple

from .base import DomainMethod
from .registry import register_domain_method


# ======================================================
# Gaussian MMD
# ======================================================

def _rbf_mmd(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x: (M, D)
    y: (N, D)
    unbiased RBF MMD
    """
    if x.size(0) < 2 or y.size(0) < 2:
        return x.new_zeros(())

    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)

    bandwidth = torch.cat([xx.flatten(), yy.flatten(), xy.flatten()]).mean().detach()
    gamma = 1.0 / (bandwidth + eps)

    Kxx = torch.exp(-gamma * xx)
    Kyy = torch.exp(-gamma * yy)
    Kxy = torch.exp(-gamma * xy)

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


# ======================================================
# Class-Conditional MMD
# ======================================================

@register_domain_method("cmmd")
class ClassConditionalMMD(DomainMethod):
    """
    Class-Conditional MMD

    Aligns:
        P(h | y=c, domain=d1)  ~  P(h | y=c, domain=d2)

    This avoids class mixing and is much stronger than vanilla MMD.
    """

    def __init__(
        self,
        domain_weight: float = 1.0,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(domain_weight=domain_weight)
        self.normalize = normalize

    # ======================================================
    # forward
    # ======================================================
    def forward(
        self,
        *,
        features: torch.Tensor | None = None,      # [B, N, D]
        labels: torch.Tensor | None = None,        # [B, N]
        domain_labels: torch.Tensor | None = None, # [B]
        **_,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        if features is None or labels is None or domain_labels is None:
            zero = torch.tensor(0.0, device=features.device if features is not None else "cpu")
            return zero, {}

        B, N, D = features.shape

        # -----------------------------------------
        # flatten
        # -----------------------------------------
        x = features.reshape(B * N, D)
        y = labels.reshape(-1)
        z = domain_labels.unsqueeze(1).expand(B, N).reshape(-1)

        domains = torch.unique(z)
        classes = torch.unique(y)

        if domains.numel() < 2:
            zero = x.new_zeros(())
            return zero, {"Loss/cmmd": 0.0}

        loss = x.new_zeros(())
        count = 0

        # -----------------------------------------
        # class-conditional pairwise alignment
        # -----------------------------------------
        for c in classes:

            mask_c = (y == c)

            feats_by_domain = []

            for d in domains:
                mask = mask_c & (z == d)
                if mask.sum() > 1:
                    feats_by_domain.append(x[mask])

            if len(feats_by_domain) < 2:
                continue

            for i in range(len(feats_by_domain)):
                for j in range(i + 1, len(feats_by_domain)):
                    loss += _rbf_mmd(feats_by_domain[i], feats_by_domain[j])
                    count += 1

        if count > 0:
            loss = loss / count

        if self.normalize:
            loss = loss / D

        weighted_loss = self.apply_weight(loss)

        metrics = {"Loss/cmmd": loss.item()}

        return weighted_loss, metrics
