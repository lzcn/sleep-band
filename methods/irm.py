import torch
import torch.nn.functional as F

from .base import DomainMethod, DomainContext
from .registry import register_domain_method


def _grad_norm_sq(grads):
    """Sum of squared L2 norms of parameter gradients."""
    loss = 0.0
    for g in grads:
        if g is not None:
            loss = loss + g.pow(2).sum()
    return loss


@register_domain_method("irm")
class IRM(DomainMethod):
    """
    Invariant Risk Minimization (IRMv1) penalty.

    Penalty = sum_e || ∇_w L_e ||^2
    where w are the classifier parameters.

    Expected shapes:
        features: (B, N, D)   [unused here]
        logits:   (B, N, C)
        labels:   (B, N)
        domains:  (B,)
    """

    def __init__(self, domain_weight: float = 1000.0, min_group_size: int = 2, warmup_epochs: int = 5, **kwargs):
        super().__init__(domain_weight=domain_weight)
        self.min_group_size = min_group_size
        self.warmup_epochs = warmup_epochs

    def forward(
        self,
        *,
        model,
        logits: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor | None = None,
        context: DomainContext = None,
        **_,
    ):
        # Optional warmup: no IRM penalty in early steps
        if context is not None and self.warmup_epochs > 0 and context.epoch < self.warmup_epochs:
            zero = logits.new_tensor(0.0)
            return zero, {}

        classifier = model.classifier
        if classifier is None:
            raise ValueError("IRM requires model.classifier")

        B, N, C = logits.shape
        logits_flat = logits.reshape(B * N, C)
        labels_flat = labels.reshape(B * N)

        # If no domain labels: treat as single environment
        if domain_labels is None:
            loss = F.cross_entropy(logits_flat, labels_flat)
            grads = torch.autograd.grad(
                loss,
                classifier.parameters(),
                create_graph=True,
                allow_unused=False,
            )
            penalty = _grad_norm_sq(grads)
            return penalty, {"Loss/irm": penalty.item()}

        # Expand domain labels to match (B*N)
        domains_flat = domain_labels.repeat_interleave(N)

        unique_domains = torch.unique(domains_flat)
        if unique_domains.numel() < 2:
            zero = logits.new_tensor(0.0)
            return zero, {}

        penalty = 0.0
        count = 0

        for d in unique_domains:
            idx = (domains_flat == d).nonzero(as_tuple=True)[0]
            if idx.numel() < self.min_group_size:
                continue

            loss_d = F.cross_entropy(logits_flat[idx], labels_flat[idx])

            grads_d = torch.autograd.grad(loss_d, classifier.parameters(), create_graph=True, allow_unused=False)

            penalty = penalty + _grad_norm_sq(grads_d)
            count += 1

        penalty = logits.new_tensor(0.0) if count == 0 else penalty / count

        weighted_penalty = self.apply_weight(penalty)
        return weighted_penalty, {"Loss/irm": penalty.item()}
