# Emoji Suggester - MLOps Zoomcamp Project

## Problem Description

Ever seen those emoji's popping up right after you type something. Eg., 'Cool' and you keyboard would go 😎. I've always found that to be fascinating. Hence this project. An emoji predictor based off [hugging face emoji predictor dataset](https://www.kaggle.com/datasets/hariharasudhanas/twitter-emoji-prediction). 

- Model - The distillBERT model is fine-tuned with the training data to do the prediction
- Experiment Tracking - MLFlow
- Workflow Orchestration - Airflow 3.0
- Model Deployment - Dockerized
- Model Monitoring - EvidentlyAI

## Setup Guide

The project is split into two main folders

1. `src` - Where you will find all model related stuff
2. `workflows` - where you will find airflow workflow related stuffs

We are separating these two because often Airflow dependencies conflict with other dependencies of the project, stopping us from using the latest version of certain libraries. To mitigate this, we are going to have two separate docker envs, one for workflow, one for the model


To connect kind cluster to docker network
```
docker network connect kind-net kind-control-plane
```

```
awslocal s3api create-bucket --bucket emoji-predictor-bucket
```

```
awslocal s3api put-object \
  --bucket emoji-predictor-bucket \
  --key data/raw/Train.csv

awslocal s3api put-object \
  --bucket emoji-predictor-bucket \
  --key data/raw/Test.csv

awslocal s3api put-object \
  --bucket emoji-predictor-bucket \
  --key data/raw/Mapping.csv

awslocal s3 cp data/raw/ s3://emoji-predictor-bucket/data/raw --recursive

```

```
docker buildx build --platform=linux/amd64 -t emoji-extractor:latest --cache-to type=inline . --cache-from type=inline .

The last 2 hrs has been painful to get the kind cluster to pull the image. 
Once that's fixed the script has to be updated

```
docker build -t emoji-extractor:0.0.1 .
kind load docker-image emoji-extractor:0.0.1
```


```
# localstack setup
awslocal s3api create-bucket --bucket emoji-predictor-bucket
awslocal s3 cp data/raw/ s3://emoji-predictor-bucket/data/raw --recursive
awslocal s3api delete-objects \
  --bucket emoji-predictor-bucket \
  --delete "$(awslocal s3api list-objects \
      --bucket emoji-predictor-bucket \
      --prefix data/processed/ \
      --query='{Objects: Contents[].{Key: Key}}')"

```