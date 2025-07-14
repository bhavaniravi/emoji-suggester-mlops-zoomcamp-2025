# Emoji Suggester - MLOps Zoomcamp Project

## Problem Description

Ever seen those emoji's popping up right after you type something. Eg., 'Cool' and you keyboard would go 😎. I've always found that to be fascinating. 
The goal of this project is to develop a machine learning model that can automatically predict and suggest the most contextually relevant emoji(s) based on the text input provided by a user. 
Such a model would enhance user experience by streamlining communication, reducing typing effort, and ensuring emotional tone is accurately conveyed in digital conversations.
An emoji predictor based off [hugging face emoji predictor dataset](https://www.kaggle.com/datasets/hariharasudhanas/twitter-emoji-prediction).

## Tools and Models

- Model - The distillBERT model is fine-tuned with the training data to do the prediction
- Experiment Tracking - MLFlow
- Workflow Orchestration - Airflow 3.0
- Model Deployment - Dockerized
- Model Monitoring - EvidentlyAI

## Setup Guide

## Requirements

You would need `python>=3.12` and `uv` in your machine to run these pipelines

1. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)

### Local setup

```
make init
```

### Training the model

Training will take sometime, the pipeline will 
1. Train the model
2. Register the best model
3. Run eval script using evidently

Before starting the pipeline we need prefect server running parallel terminal

```
make run-prefect
```


```
make train
```

### Running the application

```
make build
make run-app
```

### Running the Dockerized app

```
docker build -t emoji-extractor:0.0.1 .
docker run emoji-extractor:0.0.1
```

### Running Tests

```
make pytest
```

### Running evals

```
make run-eval
```