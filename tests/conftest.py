"""Test configuration."""

import pytest


@pytest.fixture
def device():
    """Device fixture."""
    import torch

    return torch.device("cpu")
