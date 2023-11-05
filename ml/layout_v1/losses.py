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


def listMLE(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # Ensure the predictions and targets have the same shape
    assert y_pred.shape == y_true.shape

    # Sort true labels to get the correct permutation for predicted labels
    _, indices = torch.sort(y_true, descending=True, dim=-1)
    sorted_y_pred = torch.gather(y_pred, dim=-1, index=indices)

    # Prevent overflow by normalization (Log-Sum-Exp trick)
    max_pred = sorted_y_pred.max(dim=-1, keepdim=True)[0]
    sorted_y_pred -= max_pred

    # Compute the cumsum of log for numerical stability
    log_cumsum = torch.cumsum(
        torch.log(
            torch.arange(
                1, sorted_y_pred.size(-1) + 1, device=sorted_y_pred.device
            ).float()
        ),
        dim=0,
    )

    # Calculate the ListMLE loss
    listmle_loss = torch.sum(log_cumsum + torch.logsumexp(-sorted_y_pred, dim=-1))

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


@torch.jit.script
def get_combinations(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    n_permutations: int = 8,
    ease_rate: float = 1.0,
) -> torch.Tensor:
    combinations = torch.combinations(torch.arange(y_pred.shape[0]), 2)

    y1 = y_true[combinations[:, 0]]
    y2 = y_true[combinations[:, 1]]

    y_true = torch.where(y1 > y2, 1, -1)

    # calculate ease
    ease = torch.nn.functional.softmax(y1 - y2)

    # take the average of the ease and a uniform distribution
    uniform = torch.ones_like(ease) / ease.shape[0]
    final_probs = ease_rate * ease + (1 - ease_rate) * uniform

    # randomly sample combinations weighted by ease
    if combinations.shape[0] > n_permutations:
        combinations = combinations[
            torch.multinomial(final_probs, n_permutations, replacement=False)
        ]

    return combinations


@torch.jit.script
def margin_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    margin: float,
    n_permutations: int = 8,
) -> torch.Tensor:
    with torch.no_grad():
        combinations = torch.combinations(torch.arange(y_pred.shape[0]), 2)

        # randomly sample combinations
        if combinations.shape[0] > n_permutations:
            combinations = combinations[
                torch.randperm(combinations.shape[0])[:n_permutations]
            ]

    y_pred = y_pred - y_pred.max(dim=-1, keepdim=True)[0]  # idk
    x1 = y_pred[combinations[:, 0]]
    x2 = y_pred[combinations[:, 1]]

    y1 = y_true[combinations[:, 0]]
    y2 = y_true[combinations[:, 1]]

    y_true = torch.where(y1 > y2, 1, -1)
    loss = F.margin_ranking_loss(x1, x2, y_true, margin=margin, reduction="mean")

    return loss
