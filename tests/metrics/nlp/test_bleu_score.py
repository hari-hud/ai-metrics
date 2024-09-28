from metrics.nlp.bleu_score import BLEUScore


def test_bleu_score():
    score = BLEUScore()
    ground_truths = ["the cat is on the mat", "there is a cat"]
    predictions = ["the cat is on the mat", "there is a dog"]
    result = score.get_score(ground_truths, predictions)
    assert "bleu_score" in result
    assert isinstance(result["bleu_score"], float)
