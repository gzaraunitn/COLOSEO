import torch
from torch import nn
from itertools import permutations
from .misc import gather, get_rank
import torch.nn.functional as F


class COPLoss(nn.Module):
    def __init__(self, n_clips=3):
        super().__init__()
        self.perms = torch.tensor(list(permutations(range(n_clips))))
        self.loss = nn.L1Loss()

    def forward(self, pred, label):
        predicted_perm_indices = pred.argmax(dim=1)
        predicted_perm = self.perms[predicted_perm_indices].float()
        gt_perm = self.perms[label].float()
        return self.loss(predicted_perm, gt_perm)


def simclr_loss_func_old(
    z: torch.Tensor,
    indexes: torch.Tensor,
    mask: torch.Tensor = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.
    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).
    Return:
        torch.Tensor: SimCLR loss.
    """

    # print("{} z {}".format(z.device, z.size()))
    # print("{} indices {}".format(indexes.device, indexes.size()))

    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)

    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)

    gathered_indexes = gather(indexes)

    # print("{} gathered_z {}".format(gathered_z.device, gathered_z.size()))
    # print("{} gathered_indices {}".format(gathered_indexes.device, gathered_indexes.size()))

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)

    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    # print("POS MASK BEFORE {}".format(pos_mask))
    # print("NEG MASK BEFORE {}".format(neg_mask))

    # apply acceptance mask
    if mask is not None:
        pos_mask[~mask, :] = 0
        neg_mask[~mask, :] = 0
        mask = gather(mask)
        pos_mask[:, ~mask] = 0
        neg_mask[:, ~mask] = 0

    # print("POS MASK AFTER {}".format(pos_mask))
    # print("NEG MASK AFTER {}".format(neg_mask))

    # print("{} pos_mask {}".format(pos_mask.device, pos_mask.size()))
    # print("{} neg_mask {}".format(neg_mask.device, neg_mask.size()))

    has_pos = pos_mask.sum(dim=1) > 0
    has_neg = neg_mask.sum(dim=1) > 0
    final_mask = torch.logical_and(has_pos, has_neg)

    # print("{} has_pos {}".format(has_pos.device, has_pos.size()))
    # print("{} has_neg {}".format(has_neg.device, has_neg.size()))

    pos = torch.sum(sim[final_mask] * pos_mask[final_mask], 1, keepdim=True)
    neg = torch.sum(sim[final_mask] * neg_mask[final_mask], 1, keepdim=True)

    # print("{} pos {}".format(pos.device, pos.size()))
    # print("{} neg {}".format(neg.device, neg.size()))

    print("{} Loss: {}".format(pos.device, -(torch.mean(torch.log(pos / (pos + neg))))))

    if pos_mask[final_mask].sum() == 0:
        print("Returning 0?")
        loss = torch.tensor(0.0, device=z.device)
    else:
        print(
            "{} Returning {}?".format(
                pos.device, -(torch.mean(torch.log(pos / (pos + neg))))
            )
        )
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

    print("{} Returning {}!!!!!!".format(loss.device, loss))
    return loss


def simclr_loss_func(
    z: torch.Tensor,
    indexes: torch.Tensor,
    mask: torch.Tensor = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.
    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).
    Return:
        torch.Tensor: SimCLR loss.
    """

    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)

    sim = torch.exp(torch.einsum("if, jf -> ij", gathered_z, gathered_z) / temperature)

    gathered_indexes = gather(indexes)
    gathered_indexes = gathered_indexes.unsqueeze(0)

    # positives
    pos_mask = gathered_indexes.t() == gathered_indexes
    pos_mask.fill_diagonal_(0)
    # negatives
    neg_mask = gathered_indexes.t() != gathered_indexes

    # apply acceptance mask
    if mask is not None:
        mask = gather(mask).bool()
        pos_mask[~mask, :] = 0
        neg_mask[~mask, :] = 0
        pos_mask[:, ~mask] = 0
        neg_mask[:, ~mask] = 0

    has_pos = pos_mask.sum(dim=1) > 0
    has_neg = neg_mask.sum(dim=1) > 0
    final_mask = torch.logical_and(has_pos, has_neg)

    pos = torch.sum(sim[final_mask] * pos_mask[final_mask], 1, keepdim=True)
    neg = torch.sum(sim[final_mask] * neg_mask[final_mask], 1, keepdim=True)

    if pos_mask[final_mask].sum() == 0:
        loss = torch.tensor(0.0, device=z.device)
    else:
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

    return loss
