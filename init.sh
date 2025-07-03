#!/bin/bash

# Init project
mkdir emoji-suggester
cd emoji-suggester

# Create folder structure
mkdir -p data/raw data/processed data/evidently
mkdir -p models src/notebooks src/data src/training src/serving src/monitoring
mkdir -p scripts tests logs .github/workflows

# Init project with uv
uv venv
uv pip install --upgrade pip
uv pip install mlflow apache-airflow streamlit evidently scikit-learn pandas numpy matplotlib
uv pip install torch torchvision torchaudio  # If using PyTorch
uv pip install transformers  # If using HuggingFace

# Freeze initial requirements
uv pip freeze > requirements.txt

# Create a basic README with checklist
cat <<EOL > README.md
# Emoji Suggester - MLOps Zoomcamp Project

Suggest emojis based on input text using a text classification model trained on tweets.

## ✅ Project Checklist

### 🔧 Project Setup
- [x] `uv` virtual environment
- [ ] Requirements saved in `requirements.txt`
- [ ] Data folders created

### 🧪 ML Pipeline
- [ ] Data ingestion script
- [ ] Data preprocessing
- [ ] Model training (with MLflow logging)
- [ ] Evaluation + metrics
- [ ] Register model with MLflow

### 🔁 Orchestration (Airflow)
- [ ] DAGs for preprocessing, training, and batch inference
- [ ] DAG tested and running locally

### 🚀 Model Serving
- [ ] Streamlit UI to accept text and show predicted emoji(s)
- [ ] Use registered model from MLflow

### ⚙️ CI/CD (GitHub Actions)
- [ ] Lint + test workflow
- [ ] Docker build + push
- [ ] Deploy Streamlit app

### 🧪 Testing
- [ ] Unit tests for components
- [ ] Integration test (end-to-end DAG or script)
- [ ] CLI/Makefile support

### 📊 Monitoring
- [ ] Use Evidently to track data drift
- [ ] Setup Prometheus + Grafana to monitor API or pipeline
- [ ] Log model inputs/outputs

### 📄 Docs
- [ ] Update README with clear setup instructions
- [ ] Add model usage examples

---

echo "✅ Project scaffolded. You're ready to build!"
