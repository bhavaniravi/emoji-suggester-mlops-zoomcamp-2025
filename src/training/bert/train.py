import os
import click
import mlflow
import mlflow.transformers
from mlflow import MlflowClient
from mlflow.server import get_app_client
import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    TextClassificationPipeline,
)
logging.info ("loading tokenizer")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
logging.info ("loading complete")

def _storage_opts(source, endpoint):
    if source == "s3":
        return {}
    elif source == "localstack":
        return {
            "key": "test",
            "secret": "test",
            "client_kwargs": {"endpoint_url": endpoint},
        }
    return None

os.environ["TOKENIZERS_PARALLELISM"] = "false"


columns = {"TEXT": "text", "Label": "label", "id": "id"}

@click.command()
@click.option("--tracking-uri", default="http://127.0.0.1:5002", help="MLflow tracking URI")
@click.option("--experiment", default="emoji-suggester-bert", help="MLflow experiment name")
@click.option("--output-dir", default="models/bert_output", help="Output directory")
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda", "mps"]), help="Device to train on")
@click.option("--checkpoint-uri", default=None, help="Resume from checkpoint if available")
@click.option(
    "--source",
    type=click.Choice(["local", "s3", "localstack"]),
    default="local",
    help="Data source: local, s3, or localstack",
)
@click.option("--endpoint", default="http://localhost:4566", help="Data prefix path")
@click.option("--bucket", default="emoji-predictor-bucket", help="S3 bucket name")
@click.option("--prefix", default="data/processed", help="Data prefix path")
@click.option("--data-path", default="train.csv", help="Training data CSV")
def train(tracking_uri, experiment, output_dir, device, checkpoint_uri, endpoint, source, bucket, prefix, data_path):
    
    # Decide base path
    if source == "local":
        base = prefix
    elif source == "s3":
        base = f"s3://{bucket}/{prefix}"
    elif source == "localstack":
        base = f"s3://{bucket}/{prefix}"
    logging.info(
        f"base path ={base}" 
    )
    storage_options = _storage_opts(source, endpoint)
    logging.info ("reading data")
    df = pd.read_csv(f"{base}/{data_path}", nrows=100,  storage_options=storage_options)
    df["label"].value_counts(normalize=True)
    num_labels = df["label"].nunique()
    logging.info(f"✅ Number of labels in dataset: {num_labels}")

    logging.info(f"setting tracking uri {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    logging.info(mlflow.get_tracking_uri())
    logging.info("getting experiment")
    mlflow.set_experiment(experiment)

    logging.info("loading tokeinzer")
    
    ds = tokenize(df)

    with mlflow.start_run():
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )
        model.gradient_checkpointing_enable()
        logging.info(f"output dir {output_dir}")
        training_args = TrainingArguments(
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=1,
            eval_strategy="steps",
            eval_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=1e-4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=100,
            max_grad_norm=1.0,
            num_train_epochs=10,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            compute_metrics=compute_metrics,
        )

        pipe = TextClassificationPipeline(
            model=model, tokenizer=tokenizer, top_k=1, task="text-classification"
        )
        logging.info (f"training starts, {device} {checkpoint_uri}")
        trainer.model.to(device)
        trainer.train(resume_from_checkpoint=checkpoint_uri if checkpoint_uri else False)
        logging.info ("training complete")
        model.save_pretrained(f"{output_dir}/model")
        tokenizer.save_pretrained(f"{output_dir}/tokenizer")

        preds_output = trainer.predict(ds["test"])
        preds = preds_output.predictions.argmax(axis=1)
        labels = preds_output.label_ids
        acc = accuracy_score(labels, preds)
        logging.info(f"✅ Accuracy: {acc:.4f}")

        mlflow.log_param("model", "bert-base-uncased")
        mlflow.log_param("num_labels", num_labels)
        mlflow.log_param("epochs", training_args.num_train_epochs)
        mlflow.log_param("device", device)
        mlflow.log_metric("accuracy", acc)

        mlflow.transformers.log_model(
            transformers_model=pipe,
            name="emoji-distilbert-pipeline",
            input_example="I'm so excited today",
        )
        mlflow.log_artifacts(f"{output_dir}/tokenizer", artifact_path="tokenizer")

    logging.info("🎉 Training complete and logged to MLflow.")


def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)


def tokenize(df):
    if "text" not in df.columns or "label" not in df.columns:
        raise Exception("Wrong data format: needs columns ['text', 'label']")
    ds = Dataset.from_pandas(df[["text", "label"]])
    ds = ds.map(tokenize_batch)
    return ds.train_test_split(test_size=0.2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


if __name__ == "__main__":
    train()
