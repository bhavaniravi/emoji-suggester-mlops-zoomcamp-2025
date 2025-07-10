import pandas as pd
from mlflow.tracking import MlflowClient
from src.training.bert.utils import get_best_model, get_best_runs, label_to_emoji
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

client = MlflowClient()
experiment = client.get_experiment_by_name("emoji-suggester-bert")


runs = get_best_runs(client, experiment)
best_run_id, pipeline = get_best_model(client, experiment, runs)


def batch_predict(texts, pipeline, label_to_emoji_fn, batch_size=100, position=0):
    results = []
    for i in tqdm(
        range(0, len(texts), batch_size), desc="Predicting", position=position
    ):
        batch = texts[i : i + batch_size]
        predictions = pipeline(batch)
        for result in predictions:
            label = result["label"]
            score = result["score"]
            mapped_label, emoji = label_to_emoji_fn(label)
            results.append((mapped_label, score, emoji))
    return results


def predict(text):
    result = pipeline(text)[0]
    label = result["label"]
    score = result["score"]
    label, emoji = label_to_emoji(label)
    return label, score, emoji


def train():
    df = pd.read_csv("./data/processed/train.csv")
    print("train predicting....")
    results = batch_predict(df["text"].tolist(), pipeline, label_to_emoji, position=0)
    df[["predicted_label", "score", "emoji"]] = pd.DataFrame(results, index=df.index)
    df.to_csv("data/predictions/bert_train.csv", header=True, index=None)
    print("train complete")


def test():
    df = pd.read_csv("./data/processed/test.csv")
    print("test predicting....")
    results = batch_predict(df["text"].tolist(), pipeline, label_to_emoji, position=1)
    df[["predicted_label", "score", "emoji"]] = pd.DataFrame(results, index=df.index)
    df.to_csv("data/predictions/bert_test.csv", header=True, index=None)
    print("test complete")


def mock():
    df = pd.read_csv("./data/mock/test.csv")
    print("mock predicting....")
    results = batch_predict(df["text"].tolist(), pipeline, label_to_emoji, position=2)
    df[["predicted_label", "score", "emoji"]] = pd.DataFrame(results, index=df.index)
    df.to_csv("data/predictions/bert_mock.csv", header=True, index=None)
    print("mock complete")


def main():
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(train): "train",
            executor.submit(test): "test",
            executor.submit(mock): "mock",
        }

        for future in as_completed(futures):
            task_name = futures[future]
            try:
                future.result()  # Will raise exceptions if any occurred
                print(f"{task_name} completed successfully.")
            except Exception as e:
                print(f"{task_name} failed with exception: {e}")


if __name__ == "__main__":
    main()
