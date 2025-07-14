import mlflow
import pandas as pd
from functools import lru_cache


def get_best_runs(client, experiment, max=10):
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],  # or fetch the actual experiment ID
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.eval_f1 DESC"],
        max_results=max,
    )
    return runs


def get_best_model(client, experiment, runs, max=1):
    pipeline = None
    for run in runs:
        best_run_id = run.info.run_id
        model = client.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            filter_string="attributes.status = 'READY'",
            order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
            max_results=max,
        )
        if model:
            model = model[0]
            print(best_run_id, model, model.metrics[0])
            pipeline = mlflow.transformers.load_model(model.model_uri)
            break

    if not pipeline:
        raise Exception("no model found, ensure you have the training pipeline run first")

    return run, pipeline


mapping = None


@lru_cache(maxsize=None)
def label_to_emoji(label) -> tuple:
    if label.startswith("LABEL_"):
        label = int(label.split("_")[1])
    global mapping
    if mapping is None:
        mapping = pd.read_csv("data/raw/Mapping.csv")
    row = mapping[mapping["number"] == label]
    if not row.empty:
        emoji = row.iloc[0]["emoticons"]
        return label, emoji
    else:
        return label, None
