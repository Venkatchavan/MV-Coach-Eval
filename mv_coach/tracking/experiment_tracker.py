"""JSON-based experiment tracking."""

import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Lightweight JSON-based experiment tracker."""

    def __init__(self, experiment_dir: Path) -> None:
        """Initialize experiment tracker.

        Args:
            experiment_dir: Directory to save experiment artifacts.
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.experiment_dir / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment tracker initialized: {self.run_dir}")

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration.

        Args:
            config: Configuration dictionary.
        """
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration saved to {config_path}")

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics.

        Args:
            metrics: Metrics dictionary.
        """
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")

    def log_robustness(self, robustness_results: Dict[str, Any]) -> None:
        """Log robustness testing results.

        Args:
            robustness_results: Robustness results dictionary.
        """
        robustness_path = self.run_dir / "robustness.json"
        with open(robustness_path, "w") as f:
            json.dump(robustness_results, f, indent=2)

        logger.info(f"Robustness results saved to {robustness_path}")

    def log_lineage(self, lineage: Dict[str, Any]) -> None:
        """Log experiment lineage.

        Args:
            lineage: Lineage information dictionary.
        """
        lineage_path = self.run_dir / "lineage.json"
        with open(lineage_path, "w") as f:
            json.dump(lineage, f, indent=2)

        logger.info(f"Lineage saved to {lineage_path}")

    def save_model(self, model: torch.nn.Module, metadata: Dict[str, Any]) -> None:
        """Save model checkpoint and metadata.

        Args:
            model: Model to save.
            metadata: Model metadata.
        """
        # Save model state
        model_path = self.run_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Save metadata
        metadata_path = self.run_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")

    def generate_report(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        robustness_results: Optional[Dict[str, Any]] = None,
        verdicts: Optional[Dict[str, str]] = None,
    ) -> None:
        """Generate markdown report for the experiment.

        Args:
            config: Configuration.
            metrics: Metrics.
            robustness_results: Robustness results.
            verdicts: Evaluation verdicts.
        """
        report_path = self.run_dir / "report.md"

        lines = []
        lines.append("# Experiment Report\n\n")
        lines.append(f"**Run ID:** {self.run_id}\n")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Configuration
        lines.append("## Configuration\n\n")
        lines.append(f"- **Dataset:** {config.get('data', {}).get('name', 'N/A')}\n")
        lines.append(f"- **Model:** {config.get('model', {}).get('name', 'N/A')}\n")
        lines.append(
            f"- **Epochs:** {config.get('training', {}).get('epochs', 'N/A')}\n"
        )
        lines.append(
            f"- **Seed:** {config.get('training', {}).get('seed', 'N/A')}\n\n"
        )

        # Metrics
        lines.append("## Performance Metrics\n\n")
        lines.append(f"- **Accuracy:** {metrics.get('accuracy', 0):.4f}\n")
        lines.append(f"- **Precision:** {metrics.get('precision', 0):.4f}\n")
        lines.append(f"- **Recall:** {metrics.get('recall', 0):.4f}\n")
        lines.append(f"- **F1 Score:** {metrics.get('f1_score', 0):.4f}\n")
        lines.append(f"- **ECE:** {metrics.get('ece', 0):.4f}\n\n")

        # Robustness
        if robustness_results:
            lines.append("## Robustness Results\n\n")
            for test_name, result in robustness_results.items():
                if isinstance(result, dict):
                    lines.append(f"- **{test_name}:**\n")
                    lines.append(
                        f"  - Accuracy: {result.get('accuracy', 0):.4f}\n"
                    )
                    lines.append(
                        f"  - Robustness Score: {result.get('robustness_score', 0):.4f}\n"
                    )
            lines.append("\n")

        # Verdicts
        if verdicts:
            lines.append("## Evaluation Verdicts\n\n")
            for metric, verdict in verdicts.items():
                lines.append(f"- **{metric}:** {verdict}\n")
            lines.append("\n")

        # Write report
        with open(report_path, "w") as f:
            f.writelines(lines)

        logger.info(f"Report generated: {report_path}")

    def get_git_commit_hash(self) -> Optional[str]:
        """Get current git commit hash.

        Returns:
            Git commit hash or None if not available.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not retrieve git commit hash")
            return None

    def compute_data_hash(self, data_dir: str) -> str:
        """Compute hash of data directory for lineage.

        Args:
            data_dir: Path to data directory.

        Returns:
            MD5 hash of data directory.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            return "N/A"

        # Compute hash based on directory structure
        hasher = hashlib.md5()
        hasher.update(str(data_path).encode())
        return hasher.hexdigest()[:16]
