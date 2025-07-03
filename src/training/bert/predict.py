import mlflow
from mlflow.tracking import MlflowClient
from .train import label_to_emoji

client = MlflowClient()
experiment = client.get_experiment_by_name("emoji-suggester-bert")


def get_best_runs(experiment, max=10):
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],  # or fetch the actual experiment ID
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.eval_f1 DESC"],
        max_results=max,
    )
    return runs


def get_best_model(experiment, runs, max=1):
    pipeline = None
    for run in runs:
        best_run_id = run.info.run_id
        model = client.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            filter_string="attributes.status = 'READY'",
            max_results=1,
        )
        print(model)
        if model:
            print(best_run_id, model)
            model = model[0]
            pipeline = mlflow.transformers.load_model(model.model_uri)
            break

    if not pipeline:
        raise Exception("no model found")

    return run, pipeline


runs = get_best_runs()
best_run_id, pipeline = get_best_model(experiment, runs)


def predict(text):
    result = pipeline(text)[0]
    label = result["label"]
    score = result["score"]

    print(f"Suggested emoji label: {label_to_emoji(label)} (confidence: {score:.2f})")


if __name__ == "___main__":
    text = None
    while text != "exit":
        text = input("Enter your text: ")
        predict(text)
