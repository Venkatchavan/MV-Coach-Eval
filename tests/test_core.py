"""Tests for core modules."""

import pytest
import torch

from mv_coach.core.device import DeviceManager
from mv_coach.core.version import __version__


def test_version():
    """Test version is available."""
    assert __version__ is not None
    assert isinstance(__version__, str)


def test_device_manager_cpu():
    """Test device manager with CPU."""
    dm = DeviceManager(device_override="cpu")
    device = dm.get_device()
    assert device.type == "cpu"


def test_device_manager_auto():
    """Test device manager with auto-detection."""
    dm = DeviceManager()
    device = dm.get_device()
    assert device.type in ["cpu", "cuda"]
