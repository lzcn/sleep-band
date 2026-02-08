import torch
from typing import Dict, Tuple

from .base import DomainMethod
from .registry import register_domain_method


def _mmd(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    n1, n2 = source.size(0), target.size(0)

    total = torch.cat([source, target], dim=0)

    # use cdist (more stable + less memory)
    dist2 = torch.cdist(total, total, p=2).pow(2)

    bandwidth = dist2.mean()

    K = torch.exp(-dist2 / (bandwidth + 1e-12))

    XX = K[:n1, :n1].mean()
    YY = K[n1:, n1:].mean()
    XY = K[:n1, n1:].mean()

    return XX + YY - 2 * XY


@register_domain_method("mmd")
class MMD(DomainMethod):
    """
    MMD domain alignment loss for features shaped (B, N, D) with domain labels shaped (B).

    - Flattens temporal dimension: (B, N, D) -> (B*N, D)
    - Broadcasts domain labels: (B,) -> (B*N,)
    - Computes average pairwise MMD^2 across all domain pairs
    """

    def __init__(
        self,
        domain_weight: float = 1.0,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: float | None = None,
        normalize: bool = False,
        eps: float = 1e-12,
        min_samples_per_domain: int = 2,
        **kwargs,
    ):
        super().__init__(domain_weight=domain_weight)
        if kernel_mul <= 0:
            raise ValueError("kernel_mul must be > 0")
        if kernel_num < 1:
            raise ValueError("kernel_num must be >= 1")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if min_samples_per_domain < 1:
            raise ValueError("min_samples_per_domain must be >= 1")

        self.kernel_mul = float(kernel_mul)
        self.kernel_num = int(kernel_num)
        self.fix_sigma = fix_sigma
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.min_samples_per_domain = int(min_samples_per_domain)

    def forward(
        self,
        *,
        features: torch.Tensor | None = None,  # (B, N, D)
        domain_labels: torch.Tensor | None = None,  # (B,)
        **_,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if features is None or domain_labels is None:
            device = features.device if features is not None else "cpu"
            zero = torch.tensor(0.0, device=device)
            return zero, {}

        B, N, D = features.shape

        # Flatten temporal axis
        x = features.reshape(B * N, D)

        # Broadcast domain labels to sample-level
        z = domain_labels.reshape(B, 1).expand(B, N).reshape(-1).to(device=x.device)

        domains = torch.unique(z)
        if domains.numel() < 2:
            zero = x.new_tensor(0.0)
            return zero, {"Loss/mmd": 0.0}

        # Collect domain feature sets
        feats = []
        for d in domains:
            xd = x[z == d]
            if xd.size(0) >= self.min_samples_per_domain:
                feats.append(xd)

        if len(feats) < 2:
            zero = x.new_tensor(0.0)
            return zero, {"Loss/mmd": 0.0}

        raw = x.new_zeros(())
        count = 0
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                raw = raw + _mmd(
                    feats[i],
                    feats[j],
                    kernel_mul=self.kernel_mul,
                    kernel_num=self.kernel_num,
                    fix_sigma=self.fix_sigma,
                    eps=self.eps,
                )
                count += 1

        raw = raw / count

        if self.normalize:
            # Optional heuristic scaling; keep False by default
            raw = raw / float(D)

        weighted = self.apply_weight(raw)

        metrics = {
            "Loss/mmd": float(raw.detach().item()),
        }
        return weighted, metrics
