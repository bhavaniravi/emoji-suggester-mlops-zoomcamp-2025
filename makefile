lint-fix:
	uv run ruff check --fix

lint:
	uv run ruff check 

format:
	uv run ruff format

local-setup:
	uv venv
	uv sync
	# moving data to s3
	cd mlflow && docker-compose up && cd ..
	cd localstack && docker-compose up && cd ..
	uv run awslocal s3 cp data/raw/ s3://emoji-predictor-bucket/data/raw --recursive

start-prefect:
	prefect server start

run-workflow:
	uv run python src/prefect-workflows/pipeline.py

build:
	docker build -t emoji-extractor:0.0.1 .

run:
	docker run emoji-extractor:0.0.1 src/prefect-workflows/pipeline.py