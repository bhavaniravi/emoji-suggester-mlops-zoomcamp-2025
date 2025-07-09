from airflow.sdk import DAG, Asset, AssetWatcher, task
from airflow.triggers.base import BaseEventTrigger
from airflow.providers.amazon.aws.triggers.s3 import S3KeyTrigger


base_path = "s3://emoji-predictor-bucket"


class S3EventTrigger(S3KeyTrigger, BaseEventTrigger):
    pass

trigger = S3EventTrigger(
    bucket_name="emoji-predictor-bucket", bucket_key="data/raw/Train*.csv", wildcard_match=True, aws_conn_id='localstack-s3'
)
raw_asset = Asset(
    "raw-emoji-data", watchers=[AssetWatcher(name="raw_data_watcher", trigger=trigger)]
)
outlets=[Asset(f"{base_path}/data/preprocessed/train.csv"), Asset(f"{base_path}/data/preprocessed/test.csv"), Asset(f"{base_path}/data/preprocessed/mapping.csv")]
inlets=[raw_asset]


cmds=["echo $(pwd) &&", "uv", "run", "python", "src/preprocess/data_cleanup.py", "--source", "localstack", ]
with DAG(
    dag_id="preprocess_emoji_data",
    schedule=[raw_asset],
    catchup=False,
    tags=['emoji'],
    max_active_runs=1
):

    @task.bash(inlets=inlets, outlets=outlets)
    def preprocess_data(*args, **kwargs):
        return " ".join(cmds)


    preprocess_data_task = preprocess_data()