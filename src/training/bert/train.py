# src/modeling/bert_finetune.py

import mlflow.transformers
import pandas as pd
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizerFast
from transformers import TextClassificationPipeline
from sklearn.metrics import f1_score
from transformers import DistilBertForSequenceClassification
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlflow

mlflow.set_experiment("emoji-suggester-bert")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def read_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    df["label"].value_counts(normalize=True)
    return df


def label_to_emoji(label, mapping=None):
    if not mapping:
        path = "data/raw/Mapping.csv"
        mapping = pd.read_csv(path)
    mapping = read_data(mapping)
    num = int(label.split("_")[1])
    return mapping[mapping["number"] == num]["emoticons"]


def tokenize_batch(batch):
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=128
    )


def tokenize(df):
    if "text" not in df.columns or "label" not in df.columns:
        raise Exception("Wrong data format")
    ds = Dataset.from_pandas(df[["text", "label"]])
    ds = ds.map(tokenize_batch)
    ds = ds.train_test_split(test_size=0.2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def train():
    df = read_data("data/processed/train.csv")
    df = preprocess_data(df)
    num_labels = df["label"].nunique()
    print("number of labels in dataset =", num_labels)
    ds = tokenize(df)

    with mlflow.start_run():
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )
        model.gradient_checkpointing_enable()

        # Training config
        training_args = TrainingArguments(
            output_dir="models/bert_output",  # where to save
            save_strategy="steps",  # or "epoch"
            save_steps=1000,  # 👈 how often to save
            save_total_limit=1,  # keep only 2 most recent
            eval_strategy="steps",  # optional, helps with best model logic
            eval_steps=1000,  # match save_steps if you want
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
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            compute_metrics=compute_metrics,
        )
        pipe = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            return_all_scores=False,
            task="text-classification",
        )
        trainer.model.to("mps")
        trainer.train(resume_from_checkpoint=True)

        # Save model and encoder
        model.save_pretrained("models/bert_output/model")

        tokenizer.save_pretrained("models/bert_output/tokenizer")

        # Get predictions on eval set
        preds_output = trainer.predict(ds["test"])
        preds = preds_output.predictions.argmax(axis=1)
        labels = preds_output.label_ids

        acc = accuracy_score(labels, preds)
        print(f"Accuracy: {acc:.4f}")

        mlflow.log_param("model", "bert-base-uncased")
        mlflow.log_param("num_labels", num_labels)
        mlflow.log_param("epochs", training_args.num_train_epochs)
        mlflow.log_metric("accuracy", acc)
        mlflow.transformers.log_model(
            transformers_model=pipe,
            name="emoji-distilbert-pipeline",
            input_example="I'm so excited today",
        )
        mlflow.log_artifacts("models/bert_output/tokenizer", artifact_path="tokenizer")
