from typing import Dict, List

from metrics.nlp.base_score import Score


class AccuracyScore(Score):
    def get_score(
        self, ground_truths: List[str], predictions: List[str]
    ) -> Dict[str, float]:
        """Calculate accuracy score."""
        if len(ground_truths) != len(predictions):
            raise ValueError(
                "The number of ground truths and predictions must be the same."
            )

        corrects = sum(pred == label for pred, label in zip(predictions, ground_truths))
        val_acc = corrects / len(ground_truths)
        return {"accuracy": val_acc}
