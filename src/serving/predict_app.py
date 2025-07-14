import streamlit as st
from mlflow.tracking import MlflowClient
from src.training.bert.utils import get_best_model, get_best_runs, label_to_emoji
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TextClassificationPipeline,
)

def predict(text, local=True):
    if not local:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("emoji-suggester-bert")
        runs = get_best_runs(client, experiment)
        best_run_id, pipeline = get_best_model(client, experiment, runs)
        result = pipeline(text)[0]
    else:
        # Load the model and tokenizer from the local directory
        output_dir = "models/bert_output"
        model = DistilBertForSequenceClassification.from_pretrained(f"{output_dir}/model")
        tokenizer = DistilBertTokenizerFast.from_pretrained(f"{output_dir}/tokenizer")
        pipeline = TextClassificationPipeline(
            model=model, tokenizer=tokenizer, top_k=1, task="text-classification"
        )
        result = pipeline(text)[0][0]

    
    print (f"Prediction result: {result}")
    label = result["label"]
    score = result["score"]
    return label_to_emoji(label), score


# Streamlit app
st.title("Emoji Predictor")
st.write("Try texts like: 'I am happy', 'I am loved', ...")
st.selectbox("Select a model", options=["Local Model", "MLflow Model(needs local training)"], index=0, key="model_choice")
local = st.session_state.model_choice == "Local Model"

text = st.text_input("Enter your text:")

if text:
    emoji, score = predict(text, local=local)
    st.write(f"**Suggested emoji:** {emoji}")
    st.write(f"**Confidence:** {score:.2f}")
