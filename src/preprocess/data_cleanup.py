import pandas as pd
import click
import os

columns = {"TEXT": "text", "Label": "label", "id": "id"}

@click.command()
@click.option(
    "--source",
    type=click.Choice(["local", "s3", "localstack"]),
    default="local",
    help="Data source: local, s3, or localstack",
)
@click.option("--bucket", default="emoji-predictor-bucket", help="S3 bucket name")
@click.option("--prefix", default="data/raw", help="Data prefix path")
def main(source, bucket, prefix):
    # Decide base path
    if source == "local":
        base = prefix
    elif source == "s3":
        base = f"s3://{bucket}/{prefix}"
    elif source == "localstack":
        base = f"s3://{bucket}/{prefix}"
        os.environ["AWS_ENDPOINT_URL"] = "http://localhost:4566"

    # Read data
    emoji_mapping = pd.read_csv(f"{base}/Mapping.csv", storage_options=_storage_opts(source))
    train_data = pd.read_csv(f"{base}/Train.csv", usecols=["TEXT", "Label"], storage_options=_storage_opts(source))
    test_data = pd.read_csv(f"{base}/Test.csv", usecols=["TEXT", "id"], storage_options=_storage_opts(source))

    # Ensure all labels are in mappings
    emojis = emoji_mapping["number"]
    assert all(train_data["Label"].isin(emojis))

    # Clean
    train_data_clean = clean_data(train_data)
    test_data_clean = clean_data(test_data)

    # Write
    output_prefix = "data/processed" if source == "local" else f"s3://{bucket}/data/processed"
    train_data_clean.to_csv(f"{output_prefix}/train.csv", index=False, header=True, storage_options=_storage_opts(source))
    test_data_clean.to_csv(f"{output_prefix}/test.csv", index=False, header=True, storage_options=_storage_opts(source))
    emoji_mapping.to_csv(f"{output_prefix}/mapping.csv", index=False, header=True, storage_options=_storage_opts(source))

    print("✅ Done.")


def _storage_opts(source):
    if source == "s3":
        return {}
    elif source == "localstack":
        return {
            "key": "test",
            "secret": "test",
            "client_kwargs": {"endpoint_url": "http://localhost:4566"},
        }
    return None


def clean_data(df):
    df["TEXT"] = df["TEXT"].str.strip("\n")
    df["TEXT"] = (
        df["TEXT"]
        .str.split()
        .apply(
            lambda words: " ".join(
                [
                    w.replace(".", "")
                    for w in words
                    if len(w) >= 3 and not (w.startswith("@") or w.startswith("#"))
                ]
            )
        )
    )
    df = df[df["TEXT"].notna() & (df["TEXT"].str.strip() != "")]
    df = df.rename(columns=columns)
    return df


if __name__ == "__main__":
    main()
