import torch
import torch.nn.functional as F


@torch.jit.script
def modified_margin_loss(
    x1: torch.Tensor,
    x2: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    margin: float = 1.5,
    alpha: float = 50.0,
    gamma: float = 3.0,
) -> torch.Tensor:
    """We use margin ranking loss but add a penalty term to encourage
    diversity in the output."""

    x1 = x1.flatten()
    x2 = x2.flatten()

    y_true = torch.where(y1 > y2, 1, -1)
    loss = F.margin_ranking_loss(x1, x2, y_true, margin=margin, reduction="none")

    # penalizes the model for having similar outputs
    penalty_mask = (y1 != y2).float()
    diff = x1 - x2
    penalty_term = torch.exp(-alpha * torch.pow(diff, 2)) * gamma
    final_loss = torch.mean(loss + penalty_term * penalty_mask)

    return final_loss


@torch.jit.script
def composite_margin_loss_with_huber(
    x: torch.Tensor,
    y: torch.Tensor,
    margin: float,
    alpha: float,
    gamma: float,
    delta: float,
    max_combinations: int = 4,
) -> torch.Tensor:
    with torch.no_grad():
        combination = torch.combinations(torch.arange(x.shape[0]), 2)

        # randomly sample combinations
        if combination.shape[0] > max_combinations:
            combination = combination[
                torch.randperm(combination.shape[0])[:max_combinations]
            ]

    x1 = x[combination[:, 0]]
    x2 = x[combination[:, 1]]
    y1 = y[combination[:, 0]]
    y2 = y[combination[:, 1]]
    #
    margin_loss = modified_margin_loss(x1, x2, y1, y2, margin, alpha, gamma)
    # huber = F.huber_loss(x.flatten(), y**0.5)

    # return delta * margin_loss + (1 - delta) * huber
    return huber
