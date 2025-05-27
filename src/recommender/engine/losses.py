import torch


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor):
    """Compute BPR loss: log-sigmoid of score difference."""
    return -torch.log(
        torch.sigmoid(pos_scores - neg_scores)
    ).mean()