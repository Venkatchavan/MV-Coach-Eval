"""Model comparison utilities."""

import logging
from pathlib import Path
from typing import Dict, List

import json

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare multiple models and generate comparison reports."""

    def __init__(self) -> None:
        """Initialize model comparator."""
        pass

    def compare(
        self,
        model_results: Dict[str, Dict],
    ) -> Dict[str, any]:
        """Compare multiple model results.

        Args:
            model_results: Dictionary mapping model names to their results.

        Returns:
            Comparison results.
        """
        comparison = {
            "models": list(model_results.keys()),
            "metrics": {},
            "best_model": {},
            "deltas": {},
        }

        # Extract metrics for each model
        metrics_list = ["accuracy", "f1_score", "ece", "avg_robustness"]

        for metric in metrics_list:
            comparison["metrics"][metric] = {}
            values = []

            for model_name, results in model_results.items():
                value = results.get(metric, 0.0)
                comparison["metrics"][metric][model_name] = value
                values.append((model_name, value))

            # Determine best model for this metric
            if metric == "ece":
                # Lower is better for ECE
                best = min(values, key=lambda x: x[1])
            else:
                # Higher is better for other metrics
                best = max(values, key=lambda x: x[1])

            comparison["best_model"][metric] = best[0]

        # Calculate deltas (compared to first model)
        if len(model_results) >= 2:
            baseline_name = list(model_results.keys())[0]
            baseline = model_results[baseline_name]

            for model_name, results in model_results.items():
                if model_name == baseline_name:
                    continue

                comparison["deltas"][model_name] = {}

                for metric in metrics_list:
                    baseline_val = baseline.get(metric, 0.0)
                    current_val = results.get(metric, 0.0)
                    delta = current_val - baseline_val
                    comparison["deltas"][model_name][metric] = delta

        return comparison

    def generate_markdown_report(
        self,
        comparison: Dict,
        output_path: Path,
    ) -> None:
        """Generate markdown comparison report.

        Args:
            comparison: Comparison results.
            output_path: Path to save report.
        """
        lines = []
        lines.append("# Model Comparison Report\n")
        lines.append(f"Models: {', '.join(comparison['models'])}\n")
        lines.append("## Metrics Comparison\n")

        # Create table
        lines.append("| Metric | " + " | ".join(comparison["models"]) + " |\n")
        lines.append("|" + "---|" * (len(comparison["models"]) + 1) + "\n")

        for metric, values in comparison["metrics"].items():
            row = [metric]
            for model in comparison["models"]:
                value = values[model]
                row.append(f"{value:.4f}")
            lines.append("| " + " | ".join(row) + " |\n")

        # Best models
        lines.append("\n## Best Model per Metric\n")
        for metric, best_model in comparison["best_model"].items():
            lines.append(f"- **{metric}**: {best_model}\n")

        # Deltas
        if comparison["deltas"]:
            lines.append("\n## Performance Deltas\n")
            baseline = comparison["models"][0]
            lines.append(f"Baseline: {baseline}\n")

            for model_name, deltas in comparison["deltas"].items():
                lines.append(f"\n### {model_name} vs {baseline}\n")
                for metric, delta in deltas.items():
                    sign = "+" if delta >= 0 else ""
                    lines.append(f"- {metric}: {sign}{delta:.4f}\n")

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.writelines(lines)

        logger.info(f"Comparison report saved to {output_path}")

    def save_comparison_json(
        self,
        comparison: Dict,
        output_path: Path,
    ) -> None:
        """Save comparison results as JSON.

        Args:
            comparison: Comparison results.
            output_path: Path to save JSON.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Comparison JSON saved to {output_path}")
