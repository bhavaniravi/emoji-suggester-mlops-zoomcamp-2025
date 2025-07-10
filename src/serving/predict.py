from mlflow.tracking import MlflowClient
from src.training.bert.utils import get_best_model, get_best_runs, label_to_emoji

client = MlflowClient()
experiment = client.get_experiment_by_name("emoji-suggester-bert")


runs = get_best_runs(client, experiment)
best_run_id, pipeline = get_best_model(client, experiment, runs)


def predict(text):
    result = pipeline(text)[0]
    label = result["label"]
    score = result["score"]

    print(f"Suggested emoji label: {label_to_emoji(label)} (confidence: {score:.2f})")


if __name__ == "__main__":
    print ("\n\n Welcome to emoji predictor! Try these texts I am happy, I am loved...")
    text = None
    while text != "exit":
        text = input("Enter your text: ")
        predict(text)
