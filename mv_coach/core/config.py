"""Configuration management using Hydra."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    """Dataset configuration."""

    name: str
    batch_size: int
    num_workers: int
    data_dir: str


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str
    input_channels: int
    num_classes: int
    hidden_dim: int
    dropout: float
    num_layers: int


@dataclass
class UncertaintyConfig:
    """Uncertainty quantification configuration."""

    enabled: bool
    mc_passes: int
    temperature_scaling: bool


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    seed: int


@dataclass
class RobustnessConfig:
    """Robustness testing configuration."""

    enabled: bool
    gaussian_noise_std: List[float]
    axis_dropout_prob: List[float]
    modality_dropout_prob: List[float]


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    data: DataConfig
    model: ModelConfig
    uncertainty: UncertaintyConfig
    training: TrainingConfig
    robustness: RobustnessConfig
    device_override: Optional[str]
    experiment_name: str
    output_dir: str
