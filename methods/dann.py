import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Tuple

from .base import DomainMethod, DomainContext
from .registry import register_domain_method


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


@register_domain_method("dann")
class DANN(DomainMethod):
    def __init__(
        self, d_model: int = 512, num_domains: int = 4, lambda_max: float = 1.0, domain_weight: float = 1.0, **kwargs
    ):
        super().__init__(domain_weight=domain_weight)
        self.lambda_max = float(lambda_max)

        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_domains),
        )

    def _lambda_schedule(self, context):
        """
        Standard DANN schedule:
            lambda = 2 / (1 + exp(-10 p)) - 1
        """
        if context is None:
            return self.lambda_max

        p = context.step / context.total_steps
        return self.lambda_max * (2.0 / (1.0 + torch.exp(-10 * torch.tensor(p))) - 1.0)

    def forward(
        self,
        *,
        features: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        context: DomainContext = None,
        **_,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # ---- global feature ----
        h = features.mean(dim=1)  # (B, D)

        # ---- compute lambda from context ----
        lambda_ = self._lambda_schedule(context)

        # ---- gradient reversal ----
        h = GradientReversalFunction.apply(h, lambda_)

        # ---- domain classifier ----
        domain_logits = self.domain_classifier(h)
        domain_loss = F.cross_entropy(domain_logits, domain_labels)

        metrics = {"Loss/domain": domain_loss.item()}
        return self.apply_weight(domain_loss), metrics
