lint-fix:
	uv run ruff check --fix

lint:
	uv run ruff check 

format:
	uv run ruff format

init:
	uv venv
	uv sync
	# moving data to s3
	cd mlflow && docker-compose up -d && cd ..
	cd localstack && docker-compose up -d && cd ..
	uv run awslocal s3api create-bucket --bucket emoji-predictor-bucket
	uv run awslocal s3 cp data/raw/ s3://emoji-predictor-bucket/data/raw --recursive

run-prefect:
	uv run prefect server start

train:
	make reset-workflow
	uv run python src/prefect-workflows/pipeline.py

build:
	docker build -t emoji-extractor:0.0.1 .


serve:
	export PYTHONPATH=. && uv run streamlit run src/serving/predict_app.py


setup-all:
	make local-setup
	make start-prefect
	make run-workflow

reset-workflow:
	uv run awslocal s3 cp data/raw/ s3://emoji-predictor-bucket/data/raw --recursive
	uv run awslocal s3 rm s3://emoji-predictor-bucket/tracking/ --recursive

run-eval:
	uv run python src/eval/eval.py 
	open data/evidently/eval.html