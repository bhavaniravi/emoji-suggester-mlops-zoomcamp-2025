import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently import MulticlassClassification
from evidently.presets import DataSummaryPreset, DataDriftPreset, ClassificationQuality

mdf1 = pd.read_csv("data/processed/mapping.csv")
mdf2 = pd.read_csv("data/mock/mapping.csv")
mdf = pd.concat([mdf1, mdf2], ignore_index=True)

# Define the schema for your dataset
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

# Load your datasets
df1 = pd.read_csv("data/predictions/bert_train.csv")
df2 = pd.read_csv("data/predictions/bert_mock.csv")

# Create Dataset objects
eval_data_1 = Dataset.from_pandas(df1, data_definition=schema)
eval_data_2 = Dataset.from_pandas(df2, data_definition=schema)

# Create a report with the desired metrics
report = Report(
    metrics=[DataSummaryPreset(), DataDriftPreset(), ClassificationQuality()]
)

# Run the report
my_eval = report.run(eval_data_1, eval_data_2)

# Save the evaluation results to an HTML file
my_eval.save_html("data/evidently/eval.html")
