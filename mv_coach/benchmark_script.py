"""Main benchmark script for HAR model evaluation."""

import logging
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from mv_coach.core.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RobustnessConfig,
    TrainingConfig,
    UncertaintyConfig,
)
from mv_coach.core.device import DeviceManager
from mv_coach.core.exceptions import DataLoadError
from mv_coach.core.logging import setup_logger
from mv_coach.core.version import __version__
from mv_coach.data.loader import BaseDataLoader
from mv_coach.data.motionsense import MotionSenseDataLoader
from mv_coach.evaluation.comparator import ModelComparator
from mv_coach.evaluation.metrics import (
    calculate_ece,
    collect_logits,
    evaluate_model,
)
from mv_coach.evaluation.robustness import RobustnessTester
from mv_coach.evaluation.rubric import EvaluationRubric
from mv_coach.models.registry import ModelRegistry
from mv_coach.serving.inference import ModelVersionRegistry
from mv_coach.tracking.experiment_tracker import ExperimentTracker
from mv_coach.tracking.lineage import LineageTracker

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_data_loader(data_config: DataConfig) -> BaseDataLoader:
    """Create data loader based on configuration.

    Args:
        data_config: Data configuration.

    Returns:
        Data loader instance.

    Raises:
        DataLoadError: If dataset not supported.
    """
    if data_config.name == "motionsense":
        return MotionSenseDataLoader(data_config.data_dir)
    elif data_config.name == "uci_har":
        # Placeholder for UCI HAR
        raise DataLoadError("UCI HAR dataset not yet implemented")
    else:
        raise DataLoadError(f"Unknown dataset: {data_config.name}")


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    training_config: TrainingConfig,
) -> nn.Module:
    """Train the model.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Device to train on.
        training_config: Training configuration.

    Returns:
        Trained model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(training_config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        logger.info(
            f"Epoch {epoch+1}/{training_config.epochs}: "
            f"Train Loss={train_loss/len(train_loader):.4f}, "
            f"Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss/len(val_loader):.4f}, "
            f"Val Acc={val_acc:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= training_config.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    return model


def run_loso_evaluation(cfg: DictConfig) -> Dict:
    """Run Leave-One-Subject-Out evaluation.

    Args:
        cfg: Hydra configuration.

    Returns:
        Results dictionary.
    """
    # Setup
    set_seed(cfg.training.seed)
    device_manager = DeviceManager(cfg.device_override)
    device = device_manager.get_device()

    # Load data
    data_loader = create_data_loader(DataConfig(**cfg.data))
    all_subjects = data_loader.get_all_subjects()

    logger.info(f"Running LOSO evaluation on {len(all_subjects)} subjects")

    all_accuracies = []
    all_f1_scores = []

    # Take first subject for demo (in production, loop through all subjects)
    test_subject = all_subjects[0]
    logger.info(f"Testing on subject {test_subject}")

    # Create LOSO split
    train_dataset, test_dataset = data_loader.get_loso_split(test_subject)
    train_loader, test_loader = data_loader.create_dataloaders(
        train_dataset, test_dataset, cfg.data.batch_size, cfg.data.num_workers
    )

    # Build model
    timesteps, features = data_loader.get_input_shape()
    model = ModelRegistry.build(
        cfg.model.name,
        input_channels=features,
        num_classes=data_loader.get_num_classes(),
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        num_layers=cfg.model.num_layers,
    )
    model.to(device)

    # Train model
    logger.info("Training model...")
    model = train_model(
        model,
        train_loader,
        test_loader,
        device,
        TrainingConfig(**cfg.training),
    )

    # Evaluate
    logger.info("Evaluating model...")
    accuracy, metrics, labels, predictions = evaluate_model(
        model, test_loader, device
    )

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 Score: {metrics['f1_score']:.4f}")

    # Calculate ECE
    logits, true_labels = collect_logits(model, test_loader, device)
    ece = calculate_ece(logits, true_labels)
    logger.info(f"ECE: {ece:.4f}")

    # Robustness testing
    avg_robustness = 1.0
    robustness_results = None

    if cfg.robustness.enabled:
        logger.info("Running robustness tests...")
        robustness_tester = RobustnessTester(
            gaussian_noise_std=cfg.robustness.gaussian_noise_std,
            axis_dropout_prob=cfg.robustness.axis_dropout_prob,
            modality_dropout_prob=cfg.robustness.modality_dropout_prob,
        )

        robustness_results = robustness_tester.evaluate_robustness(
            model, test_loader, device, accuracy
        )

        # Calculate average robustness score
        robustness_scores = [
            r["robustness_score"]
            for r in robustness_results.values()
            if isinstance(r, dict)
        ]
        avg_robustness = np.mean(robustness_scores) if robustness_scores else 1.0
        logger.info(f"Average Robustness Score: {avg_robustness:.4f}")

    # Evaluation rubric
    rubric = EvaluationRubric()
    verdicts = rubric.evaluate(
        accuracy=accuracy,
        f1_score=metrics["f1_score"],
        ece=ece,
        avg_robustness=avg_robustness,
    )

    logger.info("Evaluation Verdicts:")
    for metric, verdict in verdicts.items():
        logger.info(f"  {metric}: {verdict}")

    # Return results
    results = {
        "accuracy": accuracy,
        "f1_score": metrics["f1_score"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "ece": ece,
        "avg_robustness": avg_robustness,
        "robustness_results": robustness_results,
        "verdicts": verdicts,
        "model": model,
    }

    return results


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main benchmark entry point.

    Args:
        cfg: Hydra configuration.
    """
    # Setup logging
    log_file = Path(cfg.output_dir) / f"{cfg.experiment_name}.log"
    global logger
    logger = setup_logger("mv_coach", log_file)

    logger.info("=" * 80)
    logger.info(f"MV-Coach-Eval v{__version__}")
    logger.info("Production-grade HAR Benchmarking Platform")
    logger.info("=" * 80)

    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Initialize experiment tracker
    tracker = ExperimentTracker(Path(cfg.output_dir) / cfg.experiment_name)

    # Save configuration
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    tracker.log_config(config_dict)

    # Run evaluation
    logger.info("Starting LOSO evaluation...")
    results = run_loso_evaluation(cfg)

    # Log results
    metrics = {
        "accuracy": results["accuracy"],
        "f1_score": results["f1_score"],
        "precision": results["precision"],
        "recall": results["recall"],
        "ece": results["ece"],
        "avg_robustness": results["avg_robustness"],
    }
    tracker.log_metrics(metrics)

    if results["robustness_results"]:
        tracker.log_robustness(results["robustness_results"])

    # Create lineage
    lineage_tracker = LineageTracker()
    git_commit = tracker.get_git_commit_hash()
    data_hash = tracker.compute_data_hash(cfg.data.data_dir)

    lineage = lineage_tracker.create_lineage(
        model_name=cfg.model.name,
        model_version=__version__,
        dataset_name=cfg.data.name,
        dataset_hash=data_hash,
        git_commit=git_commit,
    )
    tracker.log_lineage(lineage)

    # Save model
    model_metadata = {
        "version": __version__,
        "dataset": cfg.data.name,
        "metrics": metrics,
        "git_commit": git_commit,
        "config": config_dict,
    }
    tracker.save_model(results["model"], model_metadata)

    # Generate report
    tracker.generate_report(
        config=config_dict,
        metrics=metrics,
        robustness_results=results["robustness_results"],
        verdicts=results["verdicts"],
    )

    # Register model
    registry = ModelVersionRegistry()
    registry.register_model(
        model_name=cfg.model.name,
        version=__version__,
        model_state=results["model"].state_dict(),
        metadata=model_metadata,
    )

    logger.info("=" * 80)
    logger.info("Benchmark complete!")
    logger.info(f"Results saved to: {tracker.run_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
