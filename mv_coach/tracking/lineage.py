"""Model lineage tracking."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from mv_coach.core.version import __version__

logger = logging.getLogger(__name__)


class LineageTracker:
    """Track model lineage and provenance."""

    def __init__(self) -> None:
        """Initialize lineage tracker."""
        pass

    def create_lineage(
        self,
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_hash: str,
        git_commit: Optional[str] = None,
        parent_model: Optional[str] = None,
        parent_version: Optional[str] = None,
    ) -> dict:
        """Create lineage information for a model.

        Args:
            model_name: Name of the model.
            model_version: Model version.
            dataset_name: Name of dataset used.
            dataset_hash: Hash of dataset.
            git_commit: Git commit hash.
            parent_model: Parent model name (if fine-tuned).
            parent_version: Parent model version.

        Returns:
            Lineage dictionary.
        """
        lineage = {
            "model_name": model_name,
            "model_version": model_version,
            "dataset_name": dataset_name,
            "dataset_hash": dataset_hash,
            "framework_version": __version__,
            "timestamp": datetime.now().isoformat(),
            "git_commit": git_commit,
        }

        if parent_model and parent_version:
            lineage["parent"] = {
                "model_name": parent_model,
                "model_version": parent_version,
            }

        return lineage

    def save_lineage(self, lineage: dict, output_path: Path) -> None:
        """Save lineage to file.

        Args:
            lineage: Lineage dictionary.
            output_path: Output file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(lineage, f, indent=2)

        logger.info(f"Lineage saved to {output_path}")

    def load_lineage(self, lineage_path: Path) -> dict:
        """Load lineage from file.

        Args:
            lineage_path: Path to lineage file.

        Returns:
            Lineage dictionary.
        """
        with open(lineage_path, "r") as f:
            lineage = json.load(f)

        return lineage
