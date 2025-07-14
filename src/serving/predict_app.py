import streamlit as st
from mlflow.tracking import MlflowClient
from src.training.bert.utils import get_best_model, get_best_runs, label_to_emoji

client = MlflowClient()
experiment = client.get_experiment_by_name("emoji-suggester-bert")
runs = get_best_runs(client, experiment)


def predict(text):
    best_run_id, pipeline = get_best_model(client, experiment, runs)
    result = pipeline(text)[0]
    label = result["label"]
    score = result["score"]
    return label_to_emoji(label), score


# Streamlit app
st.title("Emoji Predictor")
st.write("Try texts like: 'I am happy', 'I am loved', ...")

text = st.text_input("Enter your text:")

if text:
    emoji, score = predict(text)
    st.write(f"**Suggested emoji:** {emoji}")
    st.write(f"**Confidence:** {score:.2f}")
