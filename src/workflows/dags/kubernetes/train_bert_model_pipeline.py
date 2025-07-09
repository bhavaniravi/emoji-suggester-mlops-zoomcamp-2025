from airflow.sdk import DAG, Asset, AssetWatcher
from airflow.triggers.base import BaseEventTrigger
from airflow.providers.amazon.aws.triggers.s3 import S3KeyTrigger
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.utils.pod_manager import OnFinishAction
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import V1Volume, V1VolumeMount, V1HostPathVolumeSource
from kubernetes.client import V1ResourceRequirements

model_volume = V1Volume(
    name="model-folder",
    host_path=V1HostPathVolumeSource(path="/mnt/models", type="Directory"),
)

model_volume_mount = V1VolumeMount(name="model-folder", mount_path="/mnt/model")
base_path = "s3://emoji-predictor-bucket"


class S3EventTrigger(S3KeyTrigger, BaseEventTrigger):
    pass


trigger = S3EventTrigger(
    bucket_name="emoji-predictor-bucket",
    bucket_key="data/preprocessed/Train*.csv",
    wildcard_match=True,
    aws_conn_id="localstack-s3",
)
preprocessed_asset = Asset(
    "preprocessed-emoji-data",
    watchers=[AssetWatcher(name="raw_data_watcher", trigger=trigger)],
)
env_vars = {"GIT_PYTHON_REFRESH": "quiet"}


with DAG(
    dag_id="train_emoji_model",
    schedule=[preprocessed_asset],
    catchup=False,
    tags=["emoji"],
    max_active_runs=1,
):
    train_model_task = KubernetesPodOperator(
        name="train_model",
        image="emoji-extractor:0.0.1",
        cmds=[
            "uv",
            "run",
            "python",
            "./src/training/bert/train.py",
            "--source",
            "localstack",
            "--bucket",
            "emoji-predictor-bucket",
            "--endpoint",
            "http://host.docker.internal:4566",
            "--tracking-uri",
            "http://host.docker.internal:5002",
            "--output-dir",
            "/mnt/model/bert_output/",
            "--checkpoint-uri",
            "/mnt/model/bert_output/checkpoint-69000",
        ],
        task_id="train_model",
        outlets=[
            Asset(f"{base_path}/models/model.pkl"),
            Asset(f"{base_path}/mlruns/"),
            Asset(f"{base_path}/logs/train.log"),
        ],
        do_xcom_push=True,
        kubernetes_conn_id="kubernetes-kind",
        on_finish_action=OnFinishAction("keep_pod"),
        image_pull_policy="Never",
        volumes=[model_volume],
        volume_mounts=[model_volume_mount],
        container_logs=True,
        logging_interval=10,
        env_vars=env_vars,
        container_resources=V1ResourceRequirements(
                requests={"cpu": "500m", "memory": "512Mi"},
                limits={"cpu": "1", "memory": "1Gi"}
            )
    )
