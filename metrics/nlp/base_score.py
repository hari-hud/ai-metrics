from abc import ABC, abstractmethod
from typing import Dict, List


class Score(ABC):
    @abstractmethod
    def get_score(
        self, ground_truths: List[str], predictions: List[str]
    ) -> Dict[str, float]:
        """Calculate the score based on ground truths and predictions."""
        pass
