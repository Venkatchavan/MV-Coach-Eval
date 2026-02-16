"""Robustness testing engine with noise perturbations."""

import logging
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from mv_coach.core.exceptions import EvaluationError

logger = logging.getLogger(__name__)


class RobustnessTester:
    """Test model robustness to sensor noise and perturbations."""

    def __init__(
        self,
        gaussian_noise_std: List[float],
        axis_dropout_prob: List[float],
        modality_dropout_prob: List[float],
    ) -> None:
        """Initialize robustness tester.

        Args:
            gaussian_noise_std: List of gaussian noise standard deviations to test.
            axis_dropout_prob: List of axis dropout probabilities to test.
            modality_dropout_prob: List of modality dropout probabilities to test.
        """
        self.gaussian_noise_std = gaussian_noise_std
        self.axis_dropout_prob = axis_dropout_prob
        self.modality_dropout_prob = modality_dropout_prob

    def add_gaussian_noise(
        self, x: torch.Tensor, std: float
    ) -> torch.Tensor:
        """Add Gaussian noise to input data.

        Args:
            x: Input tensor of shape (batch, time, features).
            std: Standard deviation of Gaussian noise.

        Returns:
            Noisy tensor.
        """
        noise = torch.randn_like(x) * std
        return x + noise

    def apply_axis_dropout(
        self, x: torch.Tensor, dropout_prob: float
    ) -> torch.Tensor:
        """Randomly drop entire axes (features) from input.

        Args:
            x: Input tensor of shape (batch, time, features).
            dropout_prob: Probability of dropping each axis.

        Returns:
            Tensor with dropped axes.
        """
        batch_size, time_steps, num_features = x.shape

        # Create dropout mask for features
        mask = (
            torch.bernoulli(
                torch.ones(batch_size, 1, num_features) * (1 - dropout_prob)
            )
            .to(x.device)
        )

        return x * mask

    def apply_modality_dropout(
        self, x: torch.Tensor, dropout_prob: float, accel_dim: int = 3
    ) -> torch.Tensor:
        """Drop entire sensor modalities (e.g., accelerometer or gyroscope).

        Args:
            x: Input tensor of shape (batch, time, features).
            dropout_prob: Probability of dropping a modality.
            accel_dim: Number of accelerometer features (first N features).

        Returns:
            Tensor with dropped modality.
        """
        batch_size, time_steps, num_features = x.shape

        x_perturbed = x.clone()

        # Randomly drop accelerometer or gyroscope for each sample
        for i in range(batch_size):
            if torch.rand(1).item() < dropout_prob:
                # Randomly choose which modality to drop
                if torch.rand(1).item() < 0.5:
                    # Drop accelerometer (first accel_dim features)
                    x_perturbed[i, :, :accel_dim] = 0
                else:
                    # Drop gyroscope (remaining features)
                    x_perturbed[i, :, accel_dim:] = 0

        return x_perturbed

    def evaluate_robustness(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        clean_accuracy: float,
    ) -> Dict[str, float]:
        """Evaluate model robustness under various perturbations.

        Args:
            model: Trained model.
            dataloader: Test dataloader.
            device: Device to run evaluation on.
            clean_accuracy: Accuracy on clean data.

        Returns:
            Dictionary of robustness scores.
        """
        model.eval()
        results = {}

        # Gaussian noise robustness
        for std in self.gaussian_noise_std:
            acc = self._evaluate_with_perturbation(
                model,
                dataloader,
                device,
                perturbation_fn=lambda x: self.add_gaussian_noise(x, std),
            )
            robustness_score = acc / clean_accuracy if clean_accuracy > 0 else 0.0
            results[f"gaussian_noise_std_{std}"] = {
                "accuracy": acc,
                "robustness_score": robustness_score,
            }
            logger.info(
                f"Gaussian noise (std={std}): "
                f"Acc={acc:.4f}, Robustness={robustness_score:.4f}"
            )

        # Axis dropout robustness
        for prob in self.axis_dropout_prob:
            acc = self._evaluate_with_perturbation(
                model,
                dataloader,
                device,
                perturbation_fn=lambda x: self.apply_axis_dropout(x, prob),
            )
            robustness_score = acc / clean_accuracy if clean_accuracy > 0 else 0.0
            results[f"axis_dropout_prob_{prob}"] = {
                "accuracy": acc,
                "robustness_score": robustness_score,
            }
            logger.info(
                f"Axis dropout (prob={prob}): "
                f"Acc={acc:.4f}, Robustness={robustness_score:.4f}"
            )

        # Modality dropout robustness
        for prob in self.modality_dropout_prob:
            if prob == 0.0:
                continue  # Skip zero probability
            acc = self._evaluate_with_perturbation(
                model,
                dataloader,
                device,
                perturbation_fn=lambda x: self.apply_modality_dropout(x, prob),
            )
            robustness_score = acc / clean_accuracy if clean_accuracy > 0 else 0.0
            results[f"modality_dropout_prob_{prob}"] = {
                "accuracy": acc,
                "robustness_score": robustness_score,
            }
            logger.info(
                f"Modality dropout (prob={prob}): "
                f"Acc={acc:.4f}, Robustness={robustness_score:.4f}"
            )

        return results

    def _evaluate_with_perturbation(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        perturbation_fn,
    ) -> float:
        """Evaluate model accuracy with perturbation.

        Args:
            model: Model to evaluate.
            dataloader: Test dataloader.
            device: Device.
            perturbation_fn: Function to apply perturbation.

        Returns:
            Accuracy.
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Apply perturbation
                inputs_perturbed = perturbation_fn(inputs)

                # Forward pass
                outputs = model(inputs_perturbed)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return accuracy
