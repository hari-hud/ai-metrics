from typing import Dict, List

import numpy as np
from rouge_score import rouge_scorer

from metrics.nlp.base_score import Score


class ROUGEScores(Score):
    def __init__(self):
        self.rscorers = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rouge3", "rougeL"], use_stemmer=True
        )

    def get_score(
        self, ground_truths: List[str], predictions: List[str]
    ) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        all_rouges = []

        for gt, pred in zip(ground_truths, predictions):
            scores = self.rscorers.score(pred, gt)
            all_rouges.append(
                [
                    scores["rouge1"].fmeasure,
                    scores["rouge2"].fmeasure,
                    scores["rouge3"].fmeasure,
                    scores["rougeL"].fmeasure,
                ]
            )

        mean_rouges = np.mean(np.array(all_rouges), axis=0).tolist()

        return {
            "rouge_1_score": mean_rouges[0],
            "rouge_2_score": mean_rouges[1],
            "rouge_3_score": mean_rouges[2],
            "rouge_L_score": mean_rouges[3],
        }
