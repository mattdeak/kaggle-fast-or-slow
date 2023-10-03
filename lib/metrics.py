from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl


def xla_slowdown_from_runtime_preds(
    file_ids: list[str] | pl.Series,
    y_true: pl.Series | npt.NDArray[np.float_],
    y_pred: pl.Series | npt.NDArray[np.float_],
) -> float:
    """Compute the XLA slowdown metric.

    This function assumes y_pred is a predicted regression on y_true.

    2 - (
            Best Runtime in the top 5 predicted configurations
            /
            True Best Runtime
        )


    Args:
        file_ids: The file ids.
        y_true: The true execution time.
        y_pred: The predicted execution time.

    Returns:
        The XLA slowdown metric.
    """
    if len(file_ids) != len(y_true) != len(y_pred):
        raise ValueError("file_ids, y_true, and y_pred must have the same length.")

    K = 5
    comparison_df = pl.DataFrame(
        {
            "file_id": file_ids,
            "label": y_true,
            "preds": y_pred,
        }
    )

    gp = (
        comparison_df.groupby("file_id")
        .agg(
            # get the rank accuracy
            pl.col("preds").rank(method="ordinal").alias("preds_rank"),
            pl.col("label").min().alias("best_runtime"),
            pl.col("label"),
        )
        .explode(pl.col("preds_rank"), pl.col("label"))
        # get the top K
        .filter(pl.col("preds_rank") < (K + 1))
        .group_by("file_id")
        .agg(
            (
                2
                - (
                    pl.col("label").min().alias("best_of_chosen_runtime")
                    / pl.col("best_runtime").first()  # all the same
                )
            ).alias("slowdown")
        )
    )
    result = gp["slowdown"].mean()  # get the mean slowdown
    return cast(float, result)
