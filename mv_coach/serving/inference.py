"""Model serving and inference interface."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from mv_coach.core.device import DeviceManager
from mv_coach.core.exceptions import ModelError
from mv_coach.models.registry import ModelRegistry
from mv_coach.models.uncertainty import MCDropoutModel

logger = logging.getLogger(__name__)


class ModelVersionRegistry:
    """Registry for versioned model storage and retrieval."""

    def __init__(self, registry_dir: str = "./model_registry") -> None:
        """Initialize model version registry.

        Args:
            registry_dir: Root directory for model registry.
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        model_name: str,
        version: str,
        model_state: dict,
        metadata: dict,
    ) -> Path:
        """Register a model with version.

        Args:
            model_name: Name of the model.
            version: Semantic version (e.g., '1.0.0').
            model_state: Model state dict.
            metadata: Model metadata.

        Returns:
            Path to registered model.
        """
        model_dir = self.registry_dir / model_name / f"version_{version}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.pt"
        torch.save(model_state, model_path)

        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model registered: {model_name} v{version} at {model_dir}")
        return model_dir

    def load_model(
        self,
        model_name: str,
        version: str,
    ) -> Tuple[dict, dict]:
        """Load a registered model.

        Args:
            model_name: Name of the model.
            version: Version to load.

        Returns:
            Tuple of (model_state, metadata).

        Raises:
            ModelError: If model not found.
        """
        model_dir = self.registry_dir / model_name / f"version_{version}"
        model_path = model_dir / "model.pt"
        metadata_path = model_dir / "metadata.json"

        if not model_path.exists():
            raise ModelError(
                f"Model not found: {model_name} v{version} at {model_path}"
            )

        # Load model state
        model_state = torch.load(model_path, map_location="cpu")

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.info(f"Model loaded: {model_name} v{version}")
        return model_state, metadata

    def list_versions(self, model_name: str) -> list[str]:
        """List all versions of a model.

        Args:
            model_name: Name of the model.

        Returns:
            List of version strings.
        """
        model_dir = self.registry_dir / model_name

        if not model_dir.exists():
            return []

        versions = []
        for version_dir in model_dir.glob("version_*"):
            version = version_dir.name.replace("version_", "")
            versions.append(version)

        return sorted(versions)


class InferenceEngine:
    """Inference engine for serving models."""

    def __init__(
        self,
        model: torch.nn.Module,
        device_manager: DeviceManager,
        mc_passes: int = 30,
    ) -> None:
        """Initialize inference engine.

        Args:
            model: Trained model.
            device_manager: Device manager.
            mc_passes: Number of MC passes for uncertainty.
        """
        self.model = model
        self.device = device_manager.get_device()
        self.model.to(self.device)
        self.mc_passes = mc_passes

        # Wrap with MC Dropout
        self.mc_model = MCDropoutModel(self.model, mc_passes)

    def predict(
        self,
        x: np.ndarray,
        return_uncertainty: bool = True,
    ) -> Dict[str, any]:
        """Run inference on input data.

        Args:
            x: Input data of shape (batch, time, features).
            return_uncertainty: Whether to compute uncertainty.

        Returns:
            Dictionary with predictions, confidence, and optional uncertainty.
        """
        # Convert to tensor
        x_tensor = torch.FloatTensor(x).to(self.device)

        if return_uncertainty:
            # MC Dropout inference
            mean_probs, variance, predictions = self.mc_model.predict_with_uncertainty(
                x_tensor
            )

            # Compute confidence and uncertainty
            confidence = mean_probs.max(dim=-1).values
            uncertainty = variance.sum(dim=-1)  # Total variance

            result = {
                "predictions": predictions.cpu().numpy(),
                "probabilities": mean_probs.cpu().numpy(),
                "confidence": confidence.cpu().numpy(),
                "uncertainty": uncertainty.cpu().numpy(),
                "variance": variance.cpu().numpy(),
            }

        else:
            # Standard inference
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x_tensor)
                probs = torch.softmax(logits, dim=-1)
                confidence, predictions = probs.max(dim=-1)

            result = {
                "predictions": predictions.cpu().numpy(),
                "probabilities": probs.cpu().numpy(),
                "confidence": confidence.cpu().numpy(),
            }

        return result

    def predict_single(
        self,
        x: np.ndarray,
        activity_labels: Optional[Dict[int, str]] = None,
    ) -> Dict[str, any]:
        """Predict for a single sample with detailed output.

        Args:
            x: Input sample of shape (time, features).
            activity_labels: Optional mapping of class indices to labels.

        Returns:
            Detailed prediction result.
        """
        # Add batch dimension
        x_batch = np.expand_dims(x, axis=0)

        # Run prediction
        result = self.predict(x_batch, return_uncertainty=True)

        # Extract single sample results
        prediction = int(result["predictions"][0])
        confidence = float(result["confidence"][0])
        uncertainty = float(result["uncertainty"][0])
        probs = result["probabilities"][0]

        # Format output
        output = {
            "predicted_class": prediction,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "class_probabilities": probs.tolist(),
        }

        if activity_labels:
            output["predicted_label"] = activity_labels.get(prediction, "Unknown")

        return output
