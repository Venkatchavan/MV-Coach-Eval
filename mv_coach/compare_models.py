"""Script to compare multiple trained models."""

import argparse
import json
import logging
from pathlib import Path

from mv_coach.core.logging import setup_logger
from mv_coach.evaluation.comparator import ModelComparator

logger = logging.getLogger(__name__)


def load_experiment_results(experiment_dir: Path) -> dict:
    """Load results from an experiment directory.

    Args:
        experiment_dir: Path to experiment directory.

    Returns:
        Dictionary with experiment results.
    """
    metrics_file = experiment_dir / "metrics.json"
    robustness_file = experiment_dir / "robustness.json"
    config_file = experiment_dir / "config.json"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics not found in {experiment_dir}")

    # Load metrics
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # Load config
    config = {}
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)

    # Load robustness if available
    robustness = None
    if robustness_file.exists():
        with open(robustness_file, "r") as f:
            robustness = json.load(f)

    return {
        "metrics": metrics,
        "config": config,
        "robustness": robustness,
    }


def main():
    """Compare multiple trained models."""
    parser = argparse.ArgumentParser(
        description="Compare multiple trained models"
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        type=str,
        help="Paths to experiment directories to compare",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./comparison_report",
        help="Output directory for comparison report",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger("mv_coach_compare")

    logger.info("=" * 80)
    logger.info("MV-Coach-Eval Model Comparison")
    logger.info("=" * 80)

    # Load results from all experiments
    model_results = {}

    for exp_dir_str in args.experiment_dirs:
        exp_dir = Path(exp_dir_str)
        if not exp_dir.exists():
            logger.warning(f"Experiment directory not found: {exp_dir}")
            continue

        try:
            results = load_experiment_results(exp_dir)
            model_name = results["config"].get("model", {}).get("name", exp_dir.name)
            model_results[model_name] = results["metrics"]

            logger.info(f"Loaded results for: {model_name}")

        except Exception as e:
            logger.error(f"Error loading {exp_dir}: {e}")

    if len(model_results) < 2:
        logger.error("Need at least 2 models to compare")
        return

    # Run comparison
    logger.info(f"\nComparing {len(model_results)} models...")

    comparator = ModelComparator()
    comparison = comparator.compare(model_results)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Comparison Summary")
    logger.info("=" * 80)

    for metric, values in comparison["metrics"].items():
        logger.info(f"\n{metric.upper()}:")
        for model_name, value in values.items():
            logger.info(f"  {model_name}: {value:.4f}")
        best = comparison["best_model"][metric]
        logger.info(f"  → Best: {best}")

    # Generate reports
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    md_report = output_dir / "comparison_report.md"
    comparator.generate_markdown_report(comparison, md_report)
    logger.info(f"\nMarkdown report saved to: {md_report}")

    # JSON report
    json_report = output_dir / "comparison.json"
    comparator.save_comparison_json(comparison, json_report)
    logger.info(f"JSON report saved to: {json_report}")

    logger.info("\n" + "=" * 80)
    logger.info("Comparison complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
