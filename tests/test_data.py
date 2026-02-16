"""Tests for data loading."""

import numpy as np
import pytest

from mv_coach.data.loader import HARDataset


def test_har_dataset():
    """Test HAR dataset creation."""
    X = np.random.randn(100, 128, 6).astype(np.float32)
    y = np.random.randint(0, 6, 100).astype(np.int64)
    subject_ids = np.random.randint(1, 5, 100).astype(np.int64)

    dataset = HARDataset(X, y, subject_ids)

    assert len(dataset) == 100
    x_sample, y_sample = dataset[0]
    assert x_sample.shape == (128, 6)
    assert isinstance(y_sample.item(), int)
