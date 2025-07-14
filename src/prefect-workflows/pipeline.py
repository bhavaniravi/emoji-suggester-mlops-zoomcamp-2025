from prefect import flow, task
import boto3
import json
import logging
import os
import subprocess
import mlflow
from mlflow.tracking import MlflowClient

# Setup logging format and level
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"

S3_BUCKET = "emoji-predictor-bucket"
RAW_PREFIX = "data/raw/"
ARCHIVE_PREFIX = "data/archive/"
PREPROCESSED_PREFIX = "data/preprocessed/"
STATE_PREPROCESS = "tracking/last_preprocessed.json"
STATE_TRAIN = "tracking/last_trained.json"
LOCALSTACK_ENDPOINT = "http://localhost:4566"
TRACKING_URI = "http://localhost:5002"

s3 = boto3.client("s3", endpoint_url=LOCALSTACK_ENDPOINT)


@task
def get_state(key: str):
    print(f"Fetching state for key: {key}")
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        state = json.loads(obj["Body"].read())
        version = state.get("version")
        print(f"State found for {key}: version={version}")
        print(f"[get_state] Found version: {version} for key: {key}")
        return version
    except s3.exceptions.NoSuchKey:
        print(f"No state found at {key}")
        print(f"[get_state] No state found at {key}")
        return None


@task
def update_state(key: str, version: str):
    data = {"version": version}
    print(f"Updating state at {key} to version {version}")
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(data))
    print(f"[update_state] Updated state {key} to version {version}")


@task
def find_complete_dataset(last_version):
    print(
        f"Listing raw dataset files from S3 bucket '{S3_BUCKET}' with prefix '{RAW_PREFIX}'"
    )
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=RAW_PREFIX)
    files = [
        obj["Key"]
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".csv")
    ]
    print(f"Found {len(files)} raw CSV files.")

    import re

    train_files = [f for f in files if re.match(rf"{RAW_PREFIX}Train(_.*)?\.csv", f)]
    test_files = [f for f in files if re.match(rf"{RAW_PREFIX}Test(_.*)?\.csv", f)]
    mapping_files = [
        f for f in files if re.match(rf"{RAW_PREFIX}Mapping(_.*)?\.csv", f)
    ]

    print(f"Train files: {train_files}")
    print(f"Test files: {test_files}")
    print(f"Mapping files: {mapping_files}")

    if not (train_files and test_files and mapping_files):
        print("Incomplete dataset in raw folder.")
        print("[find_complete_dataset] Dataset incomplete, returning None")
        return None, []

    latest_train = sorted(train_files)[-1]
    latest_test = sorted(test_files)[-1]
    latest_mapping = sorted(mapping_files)[-1]

    file_set = sorted([latest_train, latest_test, latest_mapping])
    version = "_".join(file_set)

    print(f"Latest dataset files: {file_set}")
    print(f"Computed dataset version: {version}")

    if last_version is None or version > last_version:
        print(f"New dataset version found: {version} (previous: {last_version})")
        print(f"[find_complete_dataset] New dataset version: {version}")
        return version, file_set

    print("No new dataset version found.")
    print("[find_complete_dataset] No new dataset version found")
    return None, []


@task
def preprocess_data():
    command = [
        "uv",
        "run",
        "python",
        "./src/preprocess/data_cleanup.py",
        "--source",
        "localstack",
        "--endpoint",
        LOCALSTACK_ENDPOINT,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Script failed: {result.stderr}")
        raise RuntimeError("Preprocessing failed")
    print(f"Preprocessing output: {result.stdout}")


@task
def train_model():
    command = [
        "uv",
        "run",
        "python",
        "./src/training/bert/train.py",
        "--source",
        "localstack",
        "--bucket",
        "emoji-predictor-bucket",
        "--endpoint",
        LOCALSTACK_ENDPOINT,
        "--tracking-uri",
        TRACKING_URI,
        "--output-dir",
        "models/bert_output/",
        # "--checkpoint-uri",
        # "models/bert_output/checkpoint-69000",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Training failed: {result.stderr}")
        raise RuntimeError(f"Training failed=={result.stderr}")
    print(result.stdout)

    print("Model training completed.")


@task
def move_files_to_archive(files):
    print(f"[move_files_to_archive] Moving files: {files}")
    for file_key in files:
        archive_key = file_key.replace(RAW_PREFIX, ARCHIVE_PREFIX)
        print(f"Copying {file_key} to {archive_key}")
        s3.copy_object(
            Bucket=S3_BUCKET,
            CopySource={"Bucket": S3_BUCKET, "Key": file_key},
            Key=archive_key,
        )
        s3.delete_object(Bucket=S3_BUCKET, Key=file_key)
        print(f"[move_files_to_archive] Moved {file_key} to {archive_key}")


@task
def register_best_model():
    """Register the best model found in MLflow."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("emoji-suggester-bert")
    from src.training.bert.utils import get_best_runs, get_best_model

    runs = get_best_runs(client, experiment)
    best_run, pipeline = get_best_model(client, experiment, runs)
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "emoji-suggester-bert"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Registered model: {result.name}, version: {result.version}")


@task
def evaluate_registered_model():
    """Evaluate the registered model by running eval.py script."""
    eval_script = "./src/eval/eval.py"
    command = [
        "python",
        eval_script,
        "--mapping1",
        "data/raw/Mapping.csv",
        "--mapping2",
        "data/mock/mapping.csv",
        "--pred1",
        "data/predictions/bert_train.csv",
        "--pred2",
        "data/predictions/bert_mock.csv",
        "--output",
        "data/evidently/eval.html",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Evaluation failed: {result.stderr}")
        raise RuntimeError("Evaluation failed")
    print(f"Evaluation output: {result.stdout}")


# ===== PIPELINE 1: Preprocessing =====


@flow(name="preprocessing-pipeline", log_prints=True)
def preprocessing_pipeline():
    print("=== Starting Preprocessing Pipeline ===")
    last_preprocessed_version = get_state(STATE_PREPROCESS)
    version, files = find_complete_dataset(last_preprocessed_version)

    if version and (
        last_preprocessed_version is None or version > last_preprocessed_version
    ):
        print(f"New raw data version {version} detected; beginning preprocessing.")
        preprocess_data()
        update_state(STATE_PREPROCESS, version)
        print("Preprocessing done, triggering training pipeline.")
    else:
        print("No new raw data to preprocess.")

    training_pipeline()


# ===== PIPELINE 2: Training =====


@flow(name="training-pipeline", log_prints=True)
def training_pipeline():
    print("=== Starting Training Pipeline ===")
    last_preprocessed_version = get_state(STATE_PREPROCESS)
    last_trained_version = get_state(STATE_TRAIN)

    print(f"Last preprocessed version: {last_preprocessed_version}")
    print(f"Last trained version: {last_trained_version}")
    print(f"[training_pipeline] Last preprocessed version: {last_preprocessed_version}")

    # Only train if preprocessing done & training not done yet
    if last_preprocessed_version and (
        last_trained_version is None or last_preprocessed_version > last_trained_version
    ):
        print("Conditions met for training. Starting training...")
        train_model()
        update_state(STATE_TRAIN, last_preprocessed_version)

        # Archive raw files only after successful training
        _, files = find_complete_dataset(None)  # get latest files ignoring last_version
        if files:
            print("Archiving raw files after successful training.")
            move_files_to_archive(files)
        else:
            print("No raw files found to archive.")

        # Register the best model and evaluate it
        register_best_model()
        evaluate_registered_model()
    else:
        print("No new data to train.")


if __name__ == "__main__":
    preprocessing_pipeline()
