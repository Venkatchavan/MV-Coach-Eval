"""Tests for models."""

import pytest
import torch

from mv_coach.models.backbone import CNN1DBackbone, TCNBackbone
from mv_coach.models.registry import ModelRegistry


def test_tcn_backbone():
    """Test TCN backbone forward pass."""
    model = TCNBackbone(
        input_channels=6,
        num_classes=6,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
    )

    x = torch.randn(8, 128, 6)
    output = model(x)

    assert output.shape == (8, 6)


def test_cnn1d_backbone():
    """Test 1D CNN backbone forward pass."""
    model = CNN1DBackbone(
        input_channels=6,
        num_classes=6,
        hidden_dim=32,
        num_layers=2,
        dropout=0.3,
    )

    x = torch.randn(8, 128, 6)
    output = model(x)

    assert output.shape == (8, 6)


def test_model_registry():
    """Test model registry."""
    models = ModelRegistry.list_models()
    assert "tcn" in models
    assert "cnn1d" in models

    model = ModelRegistry.build(
        "tcn",
        input_channels=6,
        num_classes=6,
        hidden_dim=64,
        dropout=0.2,
        num_layers=2,
    )

    assert isinstance(model, torch.nn.Module)
