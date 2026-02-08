import torch

from .base import DomainMethod

from .registry import register_domain_method


@register_domain_method("relcoral")
class RelCORAL(DomainMethod):
    """SleepDG: A feature-level domain generalization method that aligns mean, covariance,
    and relational feature structures across domains."""

    def __init__(self, domain_weight=0.5, num_domains=4, **kwargs):
        super().__init__(domain_weight=domain_weight)
        self.num_domains = num_domains

    def forward(
        self,
        *,
        features: torch.Tensor,  # (B, N, D)
        domain_labels: torch.Tensor,  # (B,)
        **_,
    ):
        if domain_labels is None:
            zero = features.new_tensor(0.0)
            return zero, {"Loss/domain": 0.0}

        loss = self._compute_alignment(features, domain_labels)
        return loss, {"Loss/domain": loss.item()}

    def _compute_alignment(self, features: torch.Tensor, domains: torch.Tensor):
        d = features.shape[-1]
        unique_domains = torch.unique(domains)

        if unique_domains.numel() < 2:
            return features.new_tensor(0.0)

        means, covariances, relations = [], [], []

        for i in unique_domains:
            idx = torch.where(domains == i)[0]
            if idx.numel() == 0:
                continue

            feats = features[idx]
            mean, cov = self.compute_mean_covariance(feats.view(-1, d))
            rel = self.compute_relation_matrix(feats)

            means.append(mean)
            covariances.append(cov)
            relations.append(rel)

        means = torch.stack(means, dim=0)
        covariances = torch.stack(covariances, dim=0)
        relations = torch.stack(relations, dim=0)

        pairs = torch.combinations(torch.arange(len(means), device=features.device))

        mean_pairs = means[pairs]
        cov_pairs = covariances[pairs]
        rel_pairs = relations[pairs]

        loss_epoch = self.metric_diff(mean_pairs, cov_pairs)
        loss_rel = self.relation_diff(rel_pairs)
        return loss_epoch + loss_rel

    def compute_mean_covariance(self, x):
        mean = x.mean(dim=0, keepdim=True)
        x_centered = x - mean
        cov = x_centered.T @ x_centered / (x.shape[0] - 1)
        return mean.squeeze(0), cov

    def compute_relation_matrix(self, x):
        mean_feature = x.mean(2, keepdim=True)
        cent_feature = x - mean_feature
        var = torch.norm(cent_feature, p=2, dim=2).unsqueeze(2)
        relation = torch.bmm(cent_feature, cent_feature.transpose(1, 2)) / (torch.bmm(var, var.transpose(1, 2)))
        return relation.mean(0)

    def metric_diff(self, means_pairs, covas_pairs):
        mean_diff = (means_pairs[:, 0] - means_pairs[:, 1]).pow(2).mean(1).sum()
        cova_diff = (covas_pairs[:, 0] - covas_pairs[:, 1]).pow(2).mean(1).mean(1).sum()
        return mean_diff + cova_diff

    def relation_diff(self, relations_pairs):
        return (relations_pairs[:, 0] - relations_pairs[:, 1]).pow(2).mean(0).sum()
