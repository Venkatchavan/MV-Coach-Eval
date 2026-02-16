# MV-Coach-Eval Quick Reference

## Installation

```bash
git clone <repo-url>
cd Medical_Project_Agent
make install
```

## Common Commands

### Training

```bash
# Default training
make train

# With custom config
python mv_coach/benchmark_script.py model=tcn training.epochs=50

# With robustness disabled
python mv_coach/benchmark_script.py robustness=disabled
```

### Model Comparison

```bash
# Train and compare
make compare

# Compare existing experiments
python mv_coach/compare_models.py \
    experiments/exp1/run_TIMESTAMP \
    experiments/exp2/run_TIMESTAMP
```

### Testing & Quality

```bash
make test           # Run tests
make lint           # Format code
make type           # Type check
make clean          # Clean artifacts
```

### Docker

```bash
make docker-build   # Build image
make docker-run     # Run CPU
make docker-run-gpu # Run GPU
```

## Configuration Overrides

```bash
# Model
model=tcn|cnn1d

# Dataset
data=motionsense|uci_har

# Training
training.epochs=N
training.learning_rate=X
training.seed=N

# Model params
model.dropout=X
model.hidden_dim=N
model.num_layers=N

# Robustness
robustness=default|disabled
```

## Project Structure

```
mv_coach/
├── core/          # Config, device, logging
├── data/          # Data loading + LOSO
├── models/        # Architectures + registry
├── evaluation/    # Metrics + robustness
├── tracking/      # Experiments + lineage
└── serving/       # Inference + registry

configs/           # Hydra configs
tests/             # Test suite
```

## Key Files

- `benchmark_script.py` - Main training script
- `compare_models.py` - Model comparison
- `configs/config.yaml` - Main config
- `pyproject.toml` - Dependencies
- `Makefile` - Common commands

## Experiment Outputs

After training, find in `experiments/<name>/run_<timestamp>/`:
- `config.json` - Resolved config
- `metrics.json` - Performance metrics
- `robustness.json` - Robustness results
- `model.pt` - Trained model
- `report.md` - Human-readable report

## Metrics

- **Accuracy**: Overall accuracy
- **F1 Score**: Weighted F1
- **ECE**: Expected Calibration Error
- **Robustness**: Avg score under perturbations

## Rubric Thresholds

| Metric | Excellent | Good | Acceptable |
|--------|-----------|------|------------|
| Accuracy | ≥0.90 | ≥0.80 | ≥0.70 |
| F1 Score | ≥0.90 | ≥0.80 | ≥0.70 |
| ECE | ≤0.05 | ≤0.10 | ≤0.15 |
| Robustness | ≥0.85 | ≥0.70 | ≥0.60 |

## Python API

```python
# Data loading
from mv_coach.data.motionsense import MotionSenseDataLoader
loader = MotionSenseDataLoader("./data/motionsense")
train_ds, test_ds = loader.get_loso_split(subject_id)

# Model building
from mv_coach.models.registry import ModelRegistry
model = ModelRegistry.build("tcn", input_channels=6, num_classes=6)

# Inference
from mv_coach.serving.inference import InferenceEngine
engine = InferenceEngine(model, device_manager, mc_passes=30)
result = engine.predict_single(x)

# Model registry
from mv_coach.serving.inference import ModelVersionRegistry
registry = ModelVersionRegistry()
registry.register_model(name, version, state_dict, metadata)
```

## Troubleshooting

**CUDA out of memory**
→ Reduce batch size: `data.batch_size=32`

**Data not found**
→ Extract MotionSense to `./data/motionsense/`

**Import errors**
→ Run `pip install -e .`

**Test failures**
→ Check dependencies: `pip install -e ".[dev]"`

## Resources

- [README.md](README.md) - Full overview
- [GETTING_STARTED.md](GETTING_STARTED.md) - Tutorial
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
- [Project_Spec.md](Project_Spec.md) - Specification

---

**Version**: 0.1.0 | **Date**: Feb 2026
