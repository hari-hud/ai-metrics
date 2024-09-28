from typing import Dict, List

from sacrebleu import BLEU

from metrics.nlp.base_score import Score


class BLEUScore(Score):
    def __init__(self):
        self.scorer = BLEU()

    def get_score(
        self, ground_truths: List[str], predictions: List[str]
    ) -> Dict[str, float]:
        """Calculate BLEU score."""
        bleu_score = self.scorer.corpus_score(
            predictions, [[truth] for truth in ground_truths]
        ).score
        return {"bleu_score": bleu_score}
