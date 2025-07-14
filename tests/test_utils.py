from src.training.bert.utils import label_to_emoji


def test_label_to_emoji_valid_label():
    # Assuming mapping.csv contains a row with number 0 and a valid emoji
    label, emoji = label_to_emoji("LABEL_0")
    assert isinstance(label, int)
    assert isinstance(emoji, str)
    assert len(emoji) > 0


def test_label_to_emoji_invalid_label():
    # Assuming mapping.csv does not contain number 9999
    label, emoji = label_to_emoji("LABEL_9999")
    assert isinstance(label, int)
    assert emoji is None
