import torch
from typing import Dict, Tuple

from .base import DomainMethod
from .registry import register_domain_method


def _covariance(x: torch.Tensor) -> torch.Tensor:
    """
    x: (M, D)
    return: (D, D)
    """
    if x.size(0) < 2:
        return x.new_zeros(x.size(1), x.size(1))
    x = x - x.mean(dim=0, keepdim=True)
    return (x.T @ x) / (x.size(0) - 1)


@register_domain_method("coral")
class CORAL(DomainMethod):
    """
    CORAL loss for (B, N, D) features with sample-level domain labels.

    Aligns mean and covariance of features across domains.
    """

    def __init__(self, domain_weight: float = 1.0, normalize: bool = True, **kwargs):
        super().__init__(domain_weight=domain_weight)
        self.normalize = normalize

    def forward(
        self,
        *,
        features: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        **_,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        if features is None or domain_labels is None:
            zero = torch.tensor(0.0, device=features.device if features is not None else "cpu")
            return zero, {}

        B, N, D = features.shape

        # -------- flatten temporal dimension --------
        x = features.reshape(B * N, D)
        z = domain_labels.unsqueeze(1).expand(B, N).reshape(-1)

        domains = torch.unique(z)
        if domains.numel() < 2:
            zero = x.new_tensor(0.0)
            return zero, {"Loss/coral": 0.0}

        means, covs = [], []

        for d in domains:
            xd = x[z == d]
            if xd.size(0) < 2:
                continue
            means.append(xd.mean(dim=0))
            covs.append(_covariance(xd))

        if len(means) < 2:
            zero = x.new_tensor(0.0)
            return zero, {"Loss/coral": 0.0}

        raw_loss = 0.0
        count = 0

        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                raw_loss += (means[i] - means[j]).pow(2).mean()
                raw_loss += (covs[i] - covs[j]).pow(2).mean()
                count += 1

        raw_loss = raw_loss / count

        if self.normalize:
            raw_loss = raw_loss / (D * D)

        weighted_loss = self.apply_weight(raw_loss)

        metrics = {"Loss/coral": raw_loss.item()}

        return weighted_loss, metrics
