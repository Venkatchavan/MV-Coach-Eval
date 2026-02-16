# Getting Started with MV-Coach-Eval

This guide will help you get started with the MV-Coach-Eval platform.

## Prerequisites

- Python 3.10 or higher
- Git
- Optional: Docker (for containerized execution)
- Optional: CUDA-capable GPU (for accelerated training)

## Installation

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/Venkatchavan/MV-Coach-Eval.git
cd MV-Coach-Eval

# Install dependencies
make install

# Or manually
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Option 2: Docker

```bash
# Build Docker image
make docker-build

# Or manually
docker build -t mv-coach-eval:latest .
```

## Preparing Your Data

### MotionSense Dataset

1. Download the MotionSense dataset
2. Extract to `./data/motionsense/`
3. Expected structure:
   ```
   data/motionsense/
       A_DeviceMotion_data/
           sub_1_dws.csv
           sub_1_ups.csv
           ...
   ```

## Running Your First Benchmark

### 1. Basic Training

```bash
# Run with default configuration
python mv_coach/benchmark_script.py
```

This will:
- Load the MotionSense dataset
- Train a TCN model with LOSO evaluation
- Run robustness tests
- Generate experiment report
- Save model to registry

### 2. Custom Configuration

```bash
# Override specific parameters
python mv_coach/benchmark_script.py \
    model=cnn1d \
    model.dropout=0.3 \
    training.epochs=30 \
    training.seed=123
```

### 3. Using Makefile

```bash
# Run training
make train

# Run with different models
make compare
```

## Understanding the Output

After running a benchmark, you'll find:

```
experiments/
    har_benchmark/
        run_TIMESTAMP/
            config.json          # Resolved configuration
            metrics.json         # Performance metrics
            robustness.json      # Robustness test results
            lineage.json         # Model lineage
            model.pt            # Trained model
            model_metadata.json # Model metadata
            report.md           # Human-readable report
```

### Key Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 score across classes
- **ECE**: Expected Calibration Error (calibration quality)
- **Avg Robustness**: Average robustness score across perturbations

### Evaluation Verdicts

The rubric evaluates models as:
- **Excellent ✓**: Exceeds production quality
- **Good**: Production-ready
- **Acceptable**: Usable but needs improvement
- **Poor ✗**: Not production-ready

## Comparing Models

### Method 1: Sequential Training

```bash
# Train model A
python mv_coach/benchmark_script.py model=tcn experiment_name=exp_tcn

# Train model B
python mv_coach/benchmark_script.py model=cnn1d experiment_name=exp_cnn

# Compare
python mv_coach/compare_models.py \
    experiments/exp_tcn/run_TIMESTAMP \
    experiments/exp_cnn/run_TIMESTAMP \
    --output comparison_results
```

### Method 2: Using Makefile

```bash
make compare
```

This will train multiple models and generate comparison reports.

## Advanced Usage

### LOSO Evaluation

The system automatically uses Leave-One-Subject-Out (LOSO) cross-validation:

```python
# In your custom script
from mv_coach.data.motionsense import MotionSenseDataLoader

loader = MotionSenseDataLoader("./data/motionsense")
subjects = loader.get_all_subjects()

for test_subject in subjects:
    train_dataset, test_dataset = loader.get_loso_split(test_subject)
    # Train and evaluate...
```

### Uncertainty Quantification

Models automatically support MC Dropout for uncertainty:

```python
from mv_coach.serving.inference import InferenceEngine
from mv_coach.core.device import DeviceManager

device_manager = DeviceManager()
engine = InferenceEngine(model, device_manager, mc_passes=30)

result = engine.predict_single(x_sample)
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Uncertainty: {result['uncertainty']:.4f}")
```

### Model Registry

Access trained models programmatically:

```python
from mv_coach.serving.inference import ModelVersionRegistry

registry = ModelVersionRegistry("./model_registry")

# Register a model
registry.register_model(
    model_name="tcn_har",
    version="0.1.0",
    model_state=model.state_dict(),
    metadata={"accuracy": 0.92, "dataset": "motionsense"}
)

# Load a model
model_state, metadata = registry.load_model("tcn_har", "0.1.0")

# List versions
versions = registry.list_versions("tcn_har")
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_models.py -v

# Check coverage
pytest --cov=mv_coach --cov-report=html
```

### Code Quality

```bash
# Format code
make lint

# Type checking
make type

# Run all checks
make lint type test
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Docker Usage

### Basic Usage

```bash
# Build image
make docker-build

# Run with CPU
make docker-run

# Run with GPU
make docker-run-gpu
```

### Custom Docker Run

```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/experiments:/app/experiments \
    -e HYDRA_FULL_ERROR=1 \
    mv-coach-eval:latest \
    python mv_coach/benchmark_script.py model=tcn
```

## Configuration System

### Configuration Files

Located in `configs/`:
- `config.yaml`: Main configuration
- `data/`: Dataset configurations
- `model/`: Model configurations
- `robustness/`: Robustness test configurations

### Override Patterns

```bash
# Single override
python mv_coach/benchmark_script.py training.epochs=50

# Multiple overrides
python mv_coach/benchmark_script.py \
    training.epochs=50 \
    model.dropout=0.3 \
    training.learning_rate=0.0001

# Switch configuration group
python mv_coach/benchmark_script.py model=cnn1d

# Disable robustness
python mv_coach/benchmark_script.py robustness=disabled
```

## Troubleshooting

### Common Issues

**Issue**: `DataLoadError: MotionSense data directory not found`
- Solution: Download and extract MotionSense dataset to `./data/motionsense/`

**Issue**: CUDA out of memory
- Solution: Reduce batch size: `python mv_coach/benchmark_script.py data.batch_size=32`

**Issue**: Import errors
- Solution: Install in editable mode: `pip install -e .`

**Issue**: Pre-commit hooks failing
- Solution: Run `make lint` to auto-format code

### Getting Help

- Check the [README](README.md) for overview
- Review [Project_Spec.md](Project_Spec.md) for architecture details
- Open an issue on GitHub for bugs
- Check logs in `experiments/` directories

## Next Steps

1. **Train your first model**: Follow the basic training example
2. **Compare models**: Train multiple architectures and compare results
3. **Customize configuration**: Modify configs for your use case
4. **Add new datasets**: Implement adapter following `BaseDataLoader`
5. **Extend models**: Add custom architectures to registry
6. **Deploy models**: Use inference engine for serving

## Resources

- [Project Specification](Project_Spec.md)
- [README](README.md)
- [CHANGELOG](CHANGELOG.md)
- [API Documentation](docs/api.md) (coming soon)

---

Happy benchmarking! 🚀
