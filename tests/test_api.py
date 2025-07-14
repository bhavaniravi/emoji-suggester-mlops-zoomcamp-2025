# mock before importing predict_app to avoid circular import issues
from unittest.mock import patch
from src.serving import predict_app


def mock_pipeline(text):
    return [{"label": "LABEL_0", "score": 0.99}]


@patch(
    "src.serving.predict_app.get_best_model",
    new=lambda client, experiment, runs: (None, mock_pipeline),
)
@patch(
    "src.serving.predict_app.get_local_pipeline",
    new=lambda: mock_pipeline,
)
def test_predict():
    text = "I am happy"
    print(predict_app.predict(text))
    emoji, score = predict_app.predict(text)
    label, emoji = emoji
    assert isinstance(label, int)
    assert label == 0
    assert isinstance(emoji, str)
    assert len(emoji) > 0
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
