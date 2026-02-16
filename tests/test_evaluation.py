"""Tests for evaluation metrics."""

import numpy as np
import pytest

from mv_coach.evaluation.metrics import calculate_ece
from mv_coach.evaluation.rubric import EvaluationRubric


def test_calculate_ece():
    """Test ECE calculation."""
    logits = np.random.randn(100, 6)
    labels = np.random.randint(0, 6, 100)

    ece = calculate_ece(logits, labels)

    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_evaluation_rubric():
    """Test evaluation rubric."""
    rubric = EvaluationRubric()

    verdicts = rubric.evaluate(
        accuracy=0.85,
        f1_score=0.82,
        ece=0.08,
        avg_robustness=0.75,
    )

    assert "accuracy" in verdicts
    assert "f1_score" in verdicts
    assert "ece" in verdicts
    assert "robustness" in verdicts
    assert "overall" in verdicts
