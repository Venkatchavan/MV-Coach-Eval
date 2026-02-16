# MV-Coach-Eval

[![CI](https://github.com/yourusername/mv-coach-eval/workflows/CI/badge.svg)](https://github.com/yourusername/mv-coach-eval/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Multimodal Virtual Coach Evaluation Harness** - Production-grade HAR benchmarking platform

## Overview

MV-Coach-Eval is a reproducible, production-ready evaluation harness for Human Activity Recognition (HAR) that provides:

- ✅ **Accuracy & F1 Score** evaluation
- ✅ **Expected Calibration Error (ECE)** measurement
- ✅ **Robustness** to sensor noise testing
- ✅ **Leave-One-Subject-Out (LOSO)** generalization
- ✅ **Cross-model benchmarking**
- ✅ **Uncertainty quantification** via Monte Carlo Dropout

This is not a model script. This is a **modular ML evaluation system**.

## Features

- 🏗️ **Clean Architecture** with SOLID principles
- 🔧 **Config-driven** system using Hydra
- 📊 **Comprehensive metrics** including calibration and robustness
- 🔄 **Reproducible** experiments with deterministic training
- 📦 **Model registry** with semantic versioning
- 🐳 **Docker support** for containerized execution
- 🔬 **Experiment tracking** with JSON-based lineage
- 🧪 **Production-grade testing** with ≥80% coverage

## Installation

### From source

```bash
# Clone repository
git clone https://github.com/yourusername/mv-coach-eval.git
cd mv-coach-eval

# Install dependencies
make install

# Or manually
pip install -e ".[dev]"
```

### Using Docker

```bash
# Build Docker image
make docker-build

# Run benchmark
make docker-run
```

## Quick Start

### Run benchmark

```bash
# Run with default configuration
make train

# Or directly
python mv_coach/benchmark_script.py

# Override configuration
python mv_coach/benchmark_script.py model.dropout=0.3 training.epochs=20
```

### Run model comparison

```bash
make compare

# Or manually
python mv_coach/benchmark_script.py model=tcn
python mv_coach/benchmark_script.py model=cnn1d
```

## Project Structure

```
mv_coach/
├── core/              # Core functionality (config, device, logging)
├── data/              # Data loading with LOSO splitting
├── models/            # Model architectures and uncertainty
├── evaluation/        # Metrics, robustness, rubric, comparison
├── tracking/          # Experiment tracking and lineage
├── serving/           # Inference interface and model registry
└── benchmark_script.py  # Main entry point

configs/               # Hydra configuration files
tests/                 # Test suite
.github/workflows/     # CI/CD pipelines
```

## Configuration

All configuration uses Hydra. Override via CLI:

```bash
python mv_coach/benchmark_script.py \
    model=tcn \
    model.dropout=0.3 \
    training.epochs=50 \
    training.seed=42 \
    robustness.enabled=true
```

## Datasets

Supported datasets:
- ✅ **MotionSense** - Smartphone IMU data
- 🚧 **UCI HAR** - Coming soon

All datasets use **LOSO (Leave-One-Subject-Out)** splitting. No random splits allowed.

## Development

### Run tests

```bash
make test
```

### Linting and type checking

```bash
make lint
make type
```

### Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```

## CI/CD

GitHub Actions workflows:
- **CI**: Runs on push/PR (lint, type check, tests, coverage)
- **Release**: Triggers on version tags (build, Docker push, GitHub release)

## Model Registry

Models are stored with semantic versioning:

```
model_registry/
    model_name/
        version_0.1.0/
            model.pt
            metadata.json
```

Metadata includes:
- Dataset used
- Metrics (accuracy, F1, ECE)
- Git commit hash
- Configuration snapshot

## Versioning

Follows semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking architecture changes
- **MINOR**: New feature additions
- **PATCH**: Bug fixes

## License

MIT License - see [LICENSE](LICENSE) file

## Citation

If you use MV-Coach-Eval in your research, please cite:

```bibtex
@software{mvcoacheval2026,
  title={MV-Coach-Eval: Production-grade HAR Benchmarking Platform},
  author={MV-Coach-Eval Team},
  year={2026},
  url={https://github.com/yourusername/mv-coach-eval}
}
```

## Contact

For questions or issues, please open a GitHub issue.

---

**MV-Coach-Eval** - Production ML Evaluation Platform
