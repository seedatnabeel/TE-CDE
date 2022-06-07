import torch
from torch.nn import functional as F


def compute_norm_mse_loss(ground_truth_outputs, predictions, active_entries, norm=1150):
    """
    Computes normed MSE Loss

    Args:
    outputs (torch.tensor): list of true outputs (ground_truth)
    predictions (torch.tensor): list of model predictions
    active_entries (torch.tensor): list of active entries
    norm (int): normalization constant

    Returns:
    mse_loss (float): normed mse loss value
    """
    mse_loss = torch.mean(
        (ground_truth_outputs - (predictions) / norm).pow(2) * active_entries,
    )
    return mse_loss


def compute_cross_entropy_loss(outputs, predictions, active_entries):
    """
    Computes cross entropy lossLoss

    Args:
    outputs (torch.tensor): list of true outputs (ground_truth)
    predictions (torch.tensor): list of model predictions
    active_entries (torch.tensor): list of active entries

    Returns:
    ce_loss (float): cross entropy value
    """

    ce_loss = torch.mean(
        -torch.sum(predictions * torch.log(F.softmax(outputs, dim=1)), dim=1),
    )
    return ce_loss
