"""Model evaluation rubric system."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class EvaluationRubric:
    """Rubric for evaluating HAR model performance."""

    # Thresholds for different quality levels
    THRESHOLDS = {
        "accuracy": {"excellent": 0.90, "good": 0.80, "acceptable": 0.70},
        "f1_score": {"excellent": 0.90, "good": 0.80, "acceptable": 0.70},
        "ece": {"excellent": 0.05, "good": 0.10, "acceptable": 0.15},
        "robustness": {"excellent": 0.85, "good": 0.70, "acceptable": 0.60},
    }

    def __init__(self) -> None:
        """Initialize evaluation rubric."""
        pass

    def evaluate(
        self,
        accuracy: float,
        f1_score: float,
        ece: float,
        avg_robustness: float,
    ) -> Dict[str, str]:
        """Evaluate model performance against rubric.

        Args:
            accuracy: Model accuracy.
            f1_score: F1 score.
            ece: Expected Calibration Error.
            avg_robustness: Average robustness score.

        Returns:
            Dictionary of verdicts for each metric.
        """
        verdicts = {}

        # Accuracy verdict
        verdicts["accuracy"] = self._get_verdict(
            accuracy, self.THRESHOLDS["accuracy"]
        )

        # F1 score verdict
        verdicts["f1_score"] = self._get_verdict(
            f1_score, self.THRESHOLDS["f1_score"]
        )

        # ECE verdict (lower is better)
        verdicts["ece"] = self._get_verdict_inverse(
            ece, self.THRESHOLDS["ece"]
        )

        # Robustness verdict
        verdicts["robustness"] = self._get_verdict(
            avg_robustness, self.THRESHOLDS["robustness"]
        )

        # Overall verdict
        verdicts["overall"] = self._get_overall_verdict(verdicts)

        return verdicts

    def _get_verdict(self, value: float, thresholds: Dict[str, float]) -> str:
        """Get verdict for a metric (higher is better).

        Args:
            value: Metric value.
            thresholds: Dictionary of thresholds.

        Returns:
            Verdict string.
        """
        if value >= thresholds["excellent"]:
            return "Excellent ✓"
        elif value >= thresholds["good"]:
            return "Good"
        elif value >= thresholds["acceptable"]:
            return "Acceptable"
        else:
            return "Poor ✗"

    def _get_verdict_inverse(
        self, value: float, thresholds: Dict[str, float]
    ) -> str:
        """Get verdict for a metric (lower is better).

        Args:
            value: Metric value.
            thresholds: Dictionary of thresholds.

        Returns:
            Verdict string.
        """
        if value <= thresholds["excellent"]:
            return "Excellent ✓"
        elif value <= thresholds["good"]:
            return "Good"
        elif value <= thresholds["acceptable"]:
            return "Acceptable"
        else:
            return "Poor ✗"

    def _get_overall_verdict(self, verdicts: Dict[str, str]) -> str:
        """Compute overall verdict from individual verdicts.

        Args:
            verdicts: Dictionary of individual verdicts.

        Returns:
            Overall verdict string.
        """
        # Count excellent, good, acceptable, poor
        excellent_count = sum(1 for v in verdicts.values() if "Excellent" in v)
        poor_count = sum(1 for v in verdicts.values() if "Poor" in v)

        if poor_count > 0:
            return "Poor ✗"
        elif excellent_count >= 3:
            return "Excellent ✓"
        else:
            return "Good"
