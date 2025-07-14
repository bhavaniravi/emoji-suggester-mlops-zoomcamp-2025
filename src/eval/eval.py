import os
import pandas as pd
import click
from evidently import Dataset, DataDefinition, Report
from evidently import MulticlassClassification
from evidently.presets import DataSummaryPreset, DataDriftPreset, ClassificationQuality

@click.command()
@click.option('--mapping1', default="data/raw/Mapping.csv", help='Path to first mapping CSV.')
@click.option('--mapping2', default="data/mock/mapping.csv", help='Path to second mapping CSV.')
@click.option('--pred1', default="data/predictions/bert_train.csv", help='Path to first predictions CSV.')
@click.option('--pred2', default="data/predictions/bert_mock.csv", help='Path to second predictions CSV.')
@click.option('--output', default="data/evidently/eval.html", help='Output HTML file for evaluation.')
def main(mapping1, mapping2, pred1, pred2, output):
    mdf1 = pd.read_csv(mapping1)
    mdf2 = pd.read_csv(mapping2)
    mdf = pd.concat([mdf1, mdf2], ignore_index=True)

    schema = DataDefinition(
        text_columns=["text"],
        categorical_columns=["label", "predicted_label"],
        classification=[
            MulticlassClassification(
                target="label",
                prediction_labels="predicted_label",
                labels=mdf.to_dict()["emoticons"],
            )
        ],
    )

    df1 = pd.read_csv(pred1)
    df2 = pd.read_csv(pred2)

    eval_data_1 = Dataset.from_pandas(df1, data_definition=schema)
    eval_data_2 = Dataset.from_pandas(df2, data_definition=schema)

    report = Report(
        metrics=[DataSummaryPreset(), DataDriftPreset(), ClassificationQuality()]
    )

    my_eval = report.run(eval_data_1, eval_data_2)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    my_eval.save_html(output)
    print(f"Evaluation report saved to {output}")

if __name__ == "__main__":
    print("Starting evaluation...")
    main()

    