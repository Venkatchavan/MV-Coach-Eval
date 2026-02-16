# Changelog

All notable changes to MV-Coach-Eval will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-16

### Added
- Initial release of MV-Coach-Eval
- LOSO (Leave-One-Subject-Out) data splitting
- TCN and 1D CNN model architectures
- Monte Carlo Dropout for uncertainty quantification
- Robustness testing engine (Gaussian noise, axis dropout, modality dropout)
- Comprehensive metrics (accuracy, precision, recall, F1, ECE)
- Evaluation rubric system
- Model comparison utilities
- Experiment tracking with JSON-based lineage
- Model registry with semantic versioning
- Hydra-based configuration system
- Device manager for CPU/GPU auto-detection
- Docker support
- CI/CD workflows (GitHub Actions)
- Makefile for common tasks
- Pre-commit hooks
- Testing infrastructure with ≥80% coverage
- MotionSense dataset adapter
- Production-grade serving interface

### Infrastructure
- Python 3.10+ support
- Type checking with mypy
- Linting with black and flake8
- Automated testing with pytest
- Code coverage reporting
- GitHub Actions for CI/CD
- Docker containerization
- Comprehensive documentation

## [Unreleased]

### Planned
- UCI HAR dataset adapter
- Temperature scaling calibration
- ONNX model export
- Multi-subject LOSO evaluation
- Enhanced comparison visualizations
- Web UI for experiment tracking
- Additional model architectures (LSTM, Transformer)
- Data augmentation strategies
- Hyperparameter tuning utilities
