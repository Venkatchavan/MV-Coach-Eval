"""Uncertainty quantification using Monte Carlo Dropout."""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MCDropoutModel(nn.Module):
    """Wrapper for Monte Carlo Dropout uncertainty estimation."""

    def __init__(self, backbone: nn.Module, mc_passes: int = 30) -> None:
        """Initialize MC Dropout model.

        Args:
            backbone: Base model architecture.
            mc_passes: Number of forward passes for MC sampling.
        """
        super().__init__()
        self.backbone = backbone
        self.mc_passes = mc_passes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output logits.
        """
        return self.backbone(x)

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict with uncertainty estimation using MC Dropout.

        Args:
            x: Input tensor of shape (batch, ...).

        Returns:
            Tuple of:
                - mean_probs: Mean predicted probabilities (batch, num_classes)
                - variance: Predictive variance (batch, num_classes)
                - predictions: Final class predictions (batch,)
        """
        self.train()  # Enable dropout during inference

        all_logits = []

        with torch.no_grad():
            for _ in range(self.mc_passes):
                logits = self.backbone(x)
                all_logits.append(logits)

        # Stack logits: (mc_passes, batch, num_classes)
        all_logits = torch.stack(all_logits, dim=0)

        # Convert to probabilities
        all_probs = F.softmax(all_logits, dim=-1)

        # Compute mean and variance
        mean_probs = all_probs.mean(dim=0)
        variance = all_probs.var(dim=0)

        # Final predictions
        predictions = mean_probs.argmax(dim=-1)

        return mean_probs, variance, predictions


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration."""

    def __init__(self) -> None:
        """Initialize temperature scaling."""
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling.

        Args:
            logits: Input logits.

        Returns:
            Calibrated logits.
        """
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> None:
        """Fit temperature parameter on validation set.

        Args:
            logits: Validation logits.
            labels: Validation labels.
            lr: Learning rate.
            max_iter: Maximum iterations.
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        logger.info(f"Temperature scaling fitted: T = {self.temperature.item():.4f}")
