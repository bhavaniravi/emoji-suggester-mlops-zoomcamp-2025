from prefect import flow, task
import boto3
import json
import logging
import os
import subprocess

# Setup logging format and level
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
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

s3 = boto3.client("s3", endpoint_url=LOCALSTACK_ENDPOINT)


@task
def get_state(key: str):
    logging.info(f"Fetching state for key: {key}")
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        state = json.loads(obj["Body"].read())
        version = state.get("version")
        logging.info(f"State found for {key}: version={version}")
        print(f"[get_state] Found version: {version} for key: {key}")
        return version
    except s3.exceptions.NoSuchKey:
        logging.info(f"No state found at {key}")
        print(f"[get_state] No state found at {key}")
        return None


@task
def update_state(key: str, version: str):
    data = {"version": version}
    logging.info(f"Updating state at {key} to version {version}")
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(data))
    print(f"[update_state] Updated state {key} to version {version}")


@task
def find_complete_dataset(last_version):
    logging.info(f"Listing raw dataset files from S3 bucket '{S3_BUCKET}' with prefix '{RAW_PREFIX}'")
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=RAW_PREFIX)
    files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".csv")]
    logging.info(f"Found {len(files)} raw CSV files.")

    import re

    train_files = [f for f in files if re.match(rf"{RAW_PREFIX}Train(_.*)?\.csv", f)]
    test_files = [f for f in files if re.match(rf"{RAW_PREFIX}Test(_.*)?\.csv", f)]
    mapping_files = [f for f in files if re.match(rf"{RAW_PREFIX}Mapping(_.*)?\.csv", f)]

    logging.info(f"Train files: {train_files}")
    logging.info(f"Test files: {test_files}")
    logging.info(f"Mapping files: {mapping_files}")

    if not (train_files and test_files and mapping_files):
        logging.info("Incomplete dataset in raw folder.")
        print("[find_complete_dataset] Dataset incomplete, returning None")
        return None, []

    latest_train = sorted(train_files)[-1]
    latest_test = sorted(test_files)[-1]
    latest_mapping = sorted(mapping_files)[-1]

    file_set = sorted([latest_train, latest_test, latest_mapping])
    version = "_".join(file_set)

    logging.info(f"Latest dataset files: {file_set}")
    logging.info(f"Computed dataset version: {version}")

    if last_version is None or version > last_version:
        logging.info(f"New dataset version found: {version} (previous: {last_version})")
        print(f"[find_complete_dataset] New dataset version: {version}")
        return version, file_set

    logging.info("No new dataset version found.")
    print("[find_complete_dataset] No new dataset version found")
    return None, []


@task
def preprocess_data():
    command = [
        "uv", "run", "python", "./src/preprocess/data_cleanup.py",
        "--source", "localstack",
        "--endpoint", LOCALSTACK_ENDPOINT
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Script failed: {result.stderr}")
        raise RuntimeError("Preprocessing failed")
    logging.info(f"Preprocessing output: {result.stdout}")


@task
def train_model():
    command = [
        "uv", "run", "python", "./src/training/bert/train.py",
        "--source", "localstack",
        "--bucket", "emoji-predictor-bucket",
        "--endpoint", "http://localhost:4566",
        "--tracking-uri", "http://localhost:5002",
        "--output-dir", "models/bert_output/",
        "--checkpoint-uri", "models/bert_output/checkpoint-69000",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Training failed: {result.stderr}")
        raise RuntimeError("Training failed")
    logging.info(result.stdout)

    logging.info("Model training completed.")


@task
def move_files_to_archive(files):
    logging.info(f"Moving files to archive: {files}")
    print(f"[move_files_to_archive] Moving files: {files}")
    for file_key in files:
        archive_key = file_key.replace(RAW_PREFIX, ARCHIVE_PREFIX)
        logging.info(f"Copying {file_key} to {archive_key}")
        s3.copy_object(Bucket=S3_BUCKET, CopySource={'Bucket': S3_BUCKET, 'Key': file_key}, Key=archive_key)
        s3.delete_object(Bucket=S3_BUCKET, Key=file_key)
        logging.info(f"Moved {file_key} to {archive_key}")
        print(f"[move_files_to_archive] Moved {file_key} to {archive_key}")


# ===== PIPELINE 1: Preprocessing =====

@flow(name="preprocessing-pipeline")
def preprocessing_pipeline():
    logging.info("=== Starting Preprocessing Pipeline ===")
    print("\n=== Starting Preprocessing Pipeline ===")
    last_preprocessed_version = get_state(STATE_PREPROCESS)
    version, files = find_complete_dataset(last_preprocessed_version)

    if version and (last_preprocessed_version is None or version > last_preprocessed_version):
        logging.info(f"New raw data version {version} detected; beginning preprocessing.")
        print(f"[preprocessing_pipeline] New version detected: {version}, files: {files}")
        preprocess_data()
        update_state(STATE_PREPROCESS, version)
        logging.info("Preprocessing done, triggering training pipeline.")
        print("[preprocessing_pipeline] Preprocessing complete, triggering training pipeline.")
        training_pipeline()
    else:
        logging.info("No new raw data to preprocess.")
        print("[preprocessing_pipeline] No new raw data to preprocess.")


# ===== PIPELINE 2: Training =====

@flow(name="training-pipeline")
def training_pipeline():
    logging.info("=== Starting Training Pipeline ===")
    print("\n=== Starting Training Pipeline ===")
    last_preprocessed_version = get_state(STATE_PREPROCESS)
    last_trained_version = get_state(STATE_TRAIN)

    logging.info(f"Last preprocessed version: {last_preprocessed_version}")
    logging.info(f"Last trained version: {last_trained_version}")
    print(f"[training_pipeline] Last preprocessed version: {last_preprocessed_version}")
    print(f"[training_pipeline] Last trained version: {last_trained_version}")

    # Only train if preprocessing done & training not done yet
    if last_preprocessed_version and (last_trained_version is None or last_preprocessed_version > last_trained_version):
        logging.info("Conditions met for training. Starting training...")
        print("[training_pipeline] Training conditions met, training model...")
        train_model()
        update_state(STATE_TRAIN, last_preprocessed_version)

        # Archive raw files only after successful training
        _, files = find_complete_dataset(None)  # get latest files ignoring last_version
        if files:
            logging.info("Archiving raw files after successful training.")
            print("[training_pipeline] Archiving raw files...")
            move_files_to_archive(files)
        else:
            logging.info("No raw files found to archive.")
            print("[training_pipeline] No raw files to archive.")
    else:
        logging.info("No new data to train.")
        print("[training_pipeline] No new data to train.")


if __name__ == "__main__":
    print("=== Running preprocessing pipeline ===")
    preprocessing_pipeline()

    print("=== Running training pipeline ===")
    training_pipeline()
