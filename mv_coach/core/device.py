"""Device management for CPU/GPU support."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and configuration for training and inference."""

    def __init__(self, device_override: Optional[str] = None) -> None:
        """Initialize device manager.

        Args:
            device_override: Optional device string to override auto-detection (e.g., 'cpu', 'cuda:0').
        """
        self.device = self._configure_device(device_override)
        self._log_device_info()

    def _configure_device(self, device_override: Optional[str]) -> torch.device:
        """Configure and return the device to use.

        Args:
            device_override: Optional device override string.

        Returns:
            Configured torch device.
        """
        if device_override:
            return torch.device(device_override)

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def _log_device_info(self) -> None:
        """Log device information."""
        logger.info(f"Using device: {self.device}")

        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(self.device)
            cuda_version = torch.version.cuda
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"CUDA Version: {cuda_version}")
        else:
            logger.info("Running on CPU")

    def get_device(self) -> torch.device:
        """Get the configured device.

        Returns:
            Configured torch device.
        """
        return self.device
