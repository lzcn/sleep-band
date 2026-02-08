import torch
from torch import nn
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainContext:
    step: int
    epoch: int
    total_steps: int
    training: bool


class DomainMethod(nn.Module):
    def __init__(self, domain_weight=1.0, **kwargs):
        super(DomainMethod, self).__init__()
        self.domain_weight = domain_weight

    def forward(
        self,
        *,
        model: nn.Module | None = None, 
        features: torch.Tensor | None = None, # [B, N, D]
        logits: torch.Tensor | None = None, # [B, N, C]
        domain_labels: torch.Tensor | None = None, # [B]
        context: DomainContext | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Must return:
            loss: torch.Tensor (scalar)
            metrics: dict[str, float]
        """
        raise NotImplementedError

    def apply_weight(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.domain_weight
