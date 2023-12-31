from typing import Any

import polars as pl
import typer

import wandb
from lib.transforms.node_sum_pooling_with_graph_features_v1 import get_data
from ml.xla_v1.sweep import SWEEP_CONFIGURATION
from ml.xla_v1.train import build_model, train, validate

app = typer.Typer()


@app.command()
def sweep(max_sweeps: int = 30):
    sweep_id = wandb.sweep(sweep=SWEEP_CONFIGURATION, project="kaggle-fast-or-slow")
    wandb.agent(sweep_id, _train_and_eval, count=max_sweeps)


@app.command()
def retrain_from_sweep(sweep_id: str, model_id: str, max_configs_per_file: int = 700):
    api = wandb.Api()
    sweep = api.sweep(f"kaggle-fast-or-slow/{sweep_id}")
    best_run = sweep.best_run(order="val_perf")
    best_config = best_run.config

    wandb.init(
        project="kaggle-fast-or-slow",
        config={
            "data_pipeline": "node_sum_pooling_with_graph_features_v1",
            "model": "xgboost_regressor",
        },
        notes="This is a full train on train + validation data, using the best hyperparameters from the sweep.",
        job_type="retrain_from_sweep",
    )

    train_data = get_data("train", max_configs_per_file=max_configs_per_file)
    val_data = get_data("valid")

    # Join train and validation data, since we're retraining on the full dataset
    data = pl.concat([train_data, val_data])
    model = build_model(**best_config)

    train_metrics = train(model, data)
    print(train_metrics)

    model.save_model(f"models/{model_id}.xgb")

    art = wandb.Artifact(
        model_id,
        type="model",
        description="XGBoost model trained on the full dataset, using the best hyperparameters from the sweep.",
    )
    art.add_file(f"models/{model_id}.xgb")
    wandb.log_artifact(art)

    wandb.join()
    wandb.finish()


@app.command()
def train_and_eval(sample: bool):
    _train_and_eval(sample=sample)


def _train_and_eval(
    sample: bool = False,
    max_configs_per_file: int = 700,
    **hyperparameters: dict[str, Any],
):
    with wandb.init(  # type: ignore
        project="kaggle-fast-or-slow",
        config={
            "data_pipeline": "node_sum_pooling_with_graph_features_v1",
            "model": "xgboost_regressor",
        },
        notes="This approach simply sums all node features for the graph aggregation. We compute a few basic graph features, and then join on the config features. XGBoost is tuned on a log-transformed runtime value.",
        job_type="train_and_eval" if not sample else "debug",
    ):
        model = build_model(**hyperparameters)

        train_data = get_data("train", max_configs_per_file=max_configs_per_file)
        val_data = get_data("valid")  # no max_configs_per_file here

        if sample:
            train_data = train_data.limit(100)
            val_data = val_data.limit(100)

        train_metrics = train(model, train_data)
        print(train_metrics)
        wandb.log(train_metrics)

        val_metrics = validate(model, val_data)
        print(val_metrics)
        wandb.log(val_metrics)

        wandb.join()
        wandb.finish()


if __name__ == "__main__":
    app()
