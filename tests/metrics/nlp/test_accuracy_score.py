import pytest

from metrics.nlp.accuracy_score import AccuracyScore


def test_accuracy_score():
    score = AccuracyScore()
    ground_truths = ["cat", "dog", "fish"]
    predictions = ["cat", "dog", "fish"]
    result = score.get_score(ground_truths, predictions)
    assert result == {"accuracy": 1.0}

    predictions = ["cat", "dog", "bird"]
    result = score.get_score(ground_truths, predictions)
    assert result == {"accuracy": 2 / 3}


def test_mismatched_lengths():
    score = AccuracyScore()
    ground_truths = ["cat", "dog"]
    predictions = ["cat"]
    with pytest.raises(ValueError):
        score.get_score(ground_truths, predictions)
