# Emoji Suggester - MLOps Zoomcamp Project

Suggest emojis based on input text using a text classification model trained on tweets.

## ✅ Project Checklist

### 🔧 Project Setup
- [x]  virtual environment
- [ ] Requirements saved in 
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
