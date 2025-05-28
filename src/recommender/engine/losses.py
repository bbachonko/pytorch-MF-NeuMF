"""Module containing implementation of custom loss functions used in models' training."""
import torch
import torch.nn.functional as F


# def bpr_loss(
#     pos_scores: torch.Tensor,
#     neg_scores: torch.Tensor,
# ) -> torch.Tensor:
#     """Computes the Bayesian Personalized Ranking (BPR) loss.

#     BPR loss is used in implicit feedback recommendation systems 
#     to optimize the relative ranking between positive and negative items.
#     It maximizes the probability that a user prefers a positive item over a negative one,
#     using the log-sigmoid of the score difference.

#     Args:
#         pos_scores (torch.Tensor): Tensor of predicted scores for positive (preferred) items 
#                                    for a batch of user-item pairs; shape [batch_size].
#         neg_scores (torch.Tensor): Tensor of predicted scores for negative (non-preferred) items 
#                                    for the same batch of users; shape [batch_size].

#     Returns:
#         torch.Tensor: Scalar tensor representing the mean BPR loss over the batch.
#     """
#     return -torch.log(
#         torch.sigmoid(pos_scores - neg_scores)
#     ).mean()



def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """Computes the Bayesian Personalized Ranking (BPR) loss.

    BPR loss is used in implicit feedback recommendation systems 
    to optimize the relative ranking between positive and negative items.
    It maximizes the probability that a user prefers a positive item over a negative one,
    using the log-sigmoid of the score difference.

    Args:
        pos_scores (torch.Tensor): Tensor of predicted scores for positive (preferred) items 
            for a batch of user-item pairs; shape [batch_size].
        neg_scores (torch.Tensor): Tensor of predicted scores for negative (non-preferred) items 
            for the same batch of users; shape [batch_size].

    Returns:
        torch.Tensor: Scalar tensor representing the mean BPR loss over the batch.
    """
    if pos_scores.dim() == 1 and neg_scores.dim() == 2:
        pos_scores = pos_scores.unsqueeze(1)  # (B,) -> (B, 1) for broadcasting

    # Smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive.
    loss = F.softplus(-(pos_scores - neg_scores))  # stable −log σ(x) = softplus(−x)

    return loss.mean()
