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
    return margin_loss


# @torch.jit.script
def listMLE(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    sorted_y_true, indices = torch.sort(y_true, descending=True, dim=-1)
    sorted_y_pred = torch.gather(y_pred, dim=-1, index=indices)

    sorted_y_pred = sorted_y_pred - sorted_y_pred.max(dim=-1, keepdim=True)[0]

    # Calculate ListMLE loss
    log_fact = torch.cumsum(
        torch.log(
            sorted_y_true.new_tensor(range(1, sorted_y_true.shape[-1] + 1)).float()
        ),
        dim=0,
    )
    listmle_loss = -torch.sum(log_fact - sorted_y_pred)

    return listmle_loss


def listMLEalt(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    indices = y_true_shuffled.argsort(descending=True, dim=-1)

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(
        preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1
    ).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    return torch.mean(torch.sum(observation_loss, dim=1))
