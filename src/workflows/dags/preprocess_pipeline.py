from airflow.sdk import DAG, Asset, AssetWatcher
from airflow.triggers.base import BaseEventTrigger
from airflow.providers.amazon.aws.triggers.s3 import S3KeyTrigger
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.utils.pod_manager import OnFinishAction


base_path = "s3://emoji-predictor-bucket"


class S3EventTrigger(S3KeyTrigger, BaseEventTrigger):
    pass

trigger = S3EventTrigger(
    "emoji-predictor-bucket", f"{base_path}/data/raw/Train*.csv", wildcard_match=True, aws_conn_id='localstack-s3'
)
raw_asset = Asset(
    "raw-emoji-data", watchers=[AssetWatcher(name="raw_data_watcher", trigger=trigger)]
)


with DAG(
    dag_id="preprocess_emoji_data",
    schedule=[raw_asset],
    catchup=False,
    tags=['emoji'],
    max_active_runs=1
):
    preprocess_data = KubernetesPodOperator(
        name="preprocess_data",
        image="emoji-extractor:0.0.1", # Do not use :latest and IfNotPresent together
        cmds=["uv", "run", "python", "./src/preprocess/data_cleanup.py", "--source", "localstack"],
        task_id="preprocess_data",
        outlets=[Asset(f"{base_path}/data/preprocessed/train.csv"), Asset(f"{base_path}/data/preprocessed/test.csv"), Asset(f"{base_path}/data/preprocessed/mapping.csv")],
        do_xcom_push=True,
        kubernetes_conn_id='kubernetes-kind',
        on_finish_action=OnFinishAction("keep_pod"),
        image_pull_policy='Never'
    )
