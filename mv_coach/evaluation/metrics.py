"""Evaluation metrics for HAR models."""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate.
        dataloader: DataLoader for evaluation.
        device: Device to run evaluation on.

    Returns:
        Tuple of:
            - accuracy: Overall accuracy
            - metrics: Dictionary of metrics (precision, recall, F1)
            - all_labels: True labels
            - all_predictions: Predicted labels
    """
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels, all_predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        all_labels, all_predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        all_labels, all_predictions, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return accuracy, metrics, all_labels, all_predictions


def calculate_ece(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE).

    Args:
        logits: Model logits of shape (n_samples, n_classes).
        labels: True labels of shape (n_samples,).
        n_bins: Number of bins for calibration.

    Returns:
        ECE score.
    """
    # Convert logits to probabilities
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

    # Get confidence (max probability) and predictions
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)

    # Calculate accuracy for each sample
    accuracies = (predictions == labels).astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def collect_logits(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect model logits and labels for calibration.

    Args:
        model: Model to evaluate.
        dataloader: DataLoader.
        device: Device.

    Returns:
        Tuple of (logits, labels).
    """
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_logits, all_labels
