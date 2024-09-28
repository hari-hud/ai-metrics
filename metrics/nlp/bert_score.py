from typing import Dict, List

from bert_score import score as bert_score

from metrics.nlp.base_score import Score


class BertScore(Score):
    def get_score(
        self, ground_truths: List[str], predictions: List[str]
    ) -> Dict[str, float]:
        """Calculate BERT score F1."""
        # Getting F1 score
        _, _, f1 = bert_score(
            predictions,
            ground_truths,
            lang="en",
            model_type="bert-base-uncased",
            batch_size=64,
        )

        # Convert to numpy and get the mean
        scores = {"bert_f1": f1.mean().item()}
        return scores
