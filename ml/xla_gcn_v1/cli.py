import typer

import wandb
from ml.xla_gcn_v1.sweep import SWEEP_CONFIGURATION
from ml.xla_gcn_v1.train import train_gat

app = typer.Typer()


@app.command()
def sweep(max_sweeps: int = 10):
    sweep_id = wandb.sweep(sweep=SWEEP_CONFIGURATION, project="kaggle-fast-or-slow")
    wandb.agent(sweep_id, train_gat, count=max_sweeps)


@app.command()
def retrain_from_sweep(sweep_id: str, model_id: str, max_configs_per_file: int = 700):
    # api = wandb.Api()
    # sweep = api.sweep(f"kaggle-fast-or-slow/{sweep_id}")
    # best_run = sweep.best_run(order="val_perf")
    # best_config = best_run.config
    #
    # wandb.init(
    #     project="kaggle-fast-or-slow",
    #     config={
    #         "data_pipeline": "node_sum_pooling_with_graph_features_v1",
    #         "model": "xgboost_regressor",
    #     },
    #     notes="This is a full train on train + validation data, using the best hyperparameters from the sweep.",
    #     job_type="retrain_from_sweep",
    # )
    #
    # train_data = get_data("train", max_configs_per_file=max_configs_per_file)
    # val_data = get_data("valid")
    #
    # # Join train and validation data, since we're retraining on the full dataset
    # data = pl.concat([train_data, val_data])
    # model = build_model(**best_config)
    #
    # train_metrics = train(model, data)
    # print(train_metrics)
    #
    # model.save_model(f"models/{model_id}.xgb")
    #
    # art = wandb.Artifact(
    #     model_id,
    #     type="model",
    #     description="XGBoost model trained on the full dataset, using the best hyperparameters from the sweep.",
    # )
    # art.add_file(f"models/{model_id}.xgb")
    # wandb.log_artifact(art)
    #
    # wandb.join()
    # wandb.finish()
    raise NotImplementedError()


if __name__ == "__main__":
    app()
