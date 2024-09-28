import pytest

from metrics.nlp.bert_score import BertScore


def test_bert_score():
    score = BertScore()
    ground_truths = ["the cat is on the mat", "there is a cat"]
    predictions = ["the cat is on the mat", "there is a dog"]
    result = score.get_score(ground_truths, predictions)
    assert "bert_f1" in result
    assert isinstance(result["bert_f1"], float)
