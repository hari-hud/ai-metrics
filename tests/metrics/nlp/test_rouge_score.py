from metrics.nlp.rouge_score import ROUGEScores


def test_rouge_scores():
    score = ROUGEScores()
    ground_truths = ["the cat is on the mat", "there is a cat"]
    predictions = ["the cat is on the mat", "there is a dog"]
    result = score.get_score(ground_truths, predictions)
    expected_keys = ["rouge_1_score", "rouge_2_score", "rouge_3_score", "rouge_L_score"]
    assert all(key in result for key in expected_keys)
    assert all(isinstance(result[key], float) for key in expected_keys)
