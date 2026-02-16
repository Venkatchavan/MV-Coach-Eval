# PROJECT_SPEC.md
# MV-Coach-Eval (Enterprise Edition)

Multimodal Virtual Coach Evaluation Harness  
Production-grade HAR benchmarking platform

---

# 1. Vision

Build a reproducible, production-ready evaluation harness for Human Activity Recognition (HAR) that evaluates:

- Accuracy
- F1 Score
- Expected Calibration Error (ECE)
- Robustness to sensor noise
- Leave-One-Subject-Out (LOSO) generalization
- Cross-model benchmarking
- Uncertainty reliability

This is not a model script.

This is a modular ML evaluation system.

---

# 2. Dataset (MANDATORY)

Use Adapter Pattern to load:

- MotionSense
- UCI HAR

Strict Requirements:

- Time-series IMU data only
- LOSO splitting
- No random split allowed
- Split by subject_id

---

# 3. Architecture Principles

- Clean Architecture
- SOLID
- Dependency Injection
- Strict typing (MyPy compliant)
- Google-style docstrings
- No notebooks
- Deterministic training
- Config-driven system
- Reproducible experiments

---

# 4. Project Structure

```
mv_coach/
├── core/
│   ├── config.py
│   ├── device.py
│   ├── logging.py
│   ├── version.py
│   └── exceptions.py
├── data/
│   ├── loader.py
│   └── motionsense.py
├── models/
│   ├── backbone.py
│   ├── uncertainty.py
│   └── registry.py
├── evaluation/
│   ├── metrics.py
│   ├── robustness.py
│   ├── rubric.py
│   └── comparator.py
├── tracking/
│   ├── experiment_tracker.py
│   └── lineage.py
├── serving/
│   └── inference.py
└── benchmark_script.py

configs/
    config.yaml
    model/
    data/
    robustness/

experiments/
model_registry/

.github/workflows/
Dockerfile
Makefile
pyproject.toml
.pre-commit-config.yaml
```

---

# 5. Hydra-Based Configuration System

All configuration must use Hydra.

config.yaml must define:

- dataset
- model parameters
- dropout rate
- number of MC passes
- robustness settings
- training hyperparameters
- seed
- device override

Support CLI overrides:

```
python benchmark_script.py model.dropout=0.3 training.epochs=20
```

All experiment runs must snapshot final resolved config.

---

# 6. Deterministic Reproducibility

Enforce:

- torch.manual_seed
- numpy random seed
- cudnn deterministic mode
- Log seed in experiment metadata

Two identical runs with same config must produce same metrics.

---

# 7. GPU / CPU Auto Detection

Implement device manager:

```
if torch.cuda.is_available():
    use GPU
else:
    fallback to CPU
```

Log:

- Device used
- GPU name
- CUDA version (if available)

Allow override via config.

---

# 8. Model Layer

Backbone:
- 1D CNN / TCN

Uncertainty:
- Monte Carlo Dropout
- N forward passes
- Mean prediction
- Predictive variance

Optional:
- Temperature scaling calibration

---

# 9. Model Registry Pattern

Implement `ModelRegistry`:

Directory structure:

```
model_registry/
    model_name/
        version_x.y.z/
            model.pt
            metadata.json
```

Metadata must include:

- Semantic version
- Dataset used
- Metrics
- Git commit hash
- Timestamp
- Config snapshot

Support:

- register_model()
- load_model()
- list_versions()

---

# 10. Semantic Versioning Strategy

Follow:

MAJOR.MINOR.PATCH

- MAJOR: breaking architecture changes
- MINOR: new feature additions
- PATCH: bug fixes

Version stored in:

core/version.py

CI must tag releases automatically.

---

# 11. Release Workflow (GitHub Actions)

On Git Tag (v*.*.*):

- Run CI
- Build Docker image
- Push Docker image
- Create GitHub Release
- Upload experiment artifacts (optional)

Version must match core/version.py.

---

# 12. Robustness Engine

Support:

- Gaussian jitter
- Axis dropout
- Full modality dropout

Add robustness score:

Robustness = Accuracy_under_noise / Accuracy_clean

---

# 13. Benchmark Comparison Mode

Add comparison capability:

```
python benchmark_script.py mode=compare model=a model=b
```

Output:

| Model | Clean Acc | ECE | Robustness | Verdict |
|-------|----------|-----|------------|---------|

Generate:

- Side-by-side Markdown report
- JSON comparison artifact

Comparator must compute:

- Delta Accuracy
- Delta ECE
- Stability score

---

# 14. Lightweight Experiment Tracking

Implement JSON-based tracker.

Each run:

```
experiments/run_TIMESTAMP/
    config.json
    metrics.json
    robustness.json
    lineage.json
    model.pt
    report.md
```

Lineage must track:

- Parent model (if fine-tuned)
- Previous version
- Dataset hash

---

# 15. Serving Interface

Add inference module:

- Load registered model
- Run MC Dropout
- Return:
  - predicted class
  - confidence
  - uncertainty
  - rubric verdict

Optional:
- Export ONNX model

---

# 16. CI/CD

CI must enforce:

- Lint (black, flake8)
- Type check (mypy)
- Tests
- Coverage ≥ 80%
- Build Docker
- Validate import of package

Release workflow on tags.

---

# 17. Makefile

Include:

```
make install
make lint
make type
make test
make train
make compare
make docker-build
make docker-run
make release
```

Primary:

```
make train
```

---

# 18. Pre-Commit Hooks

Must block commit if:

- Formatting fails
- Typing fails
- Lint fails

---

# 19. Dockerfile

- Python 3.10-slim
- Non-root user
- Install dependencies
- Expose logs
- Default command runs benchmark

Must support GPU runtime if available.

---

# 20. Definition of Done

Project is production-grade when:

- CI passing
- Coverage ≥ 80%
- Deterministic runs
- Hydra config working
- Model registry functional
- Comparison mode operational
- Docker build successful
- Release workflow tested
- Version tagging enforced
- `make train` runs full pipeline
- `make compare` runs comparison
- All experiments logged
- No notebooks
- No manual steps

Everything must be automated.

---

# 21. Execution Order for Copilot

1. Data layer (LOSO first)
2. Hydra config
3. Model + uncertainty
4. Robustness engine
5. Metrics + rubric
6. Device manager
7. Experiment tracker
8. Model registry
9. Comparison mode
10. Benchmark script
11. Tests
12. CI
13. Release workflow
14. Docker
15. Makefile
16. Pre-commit

Prioritize:

- Readability
- Modularity
- Reproducibility
- Strict typing
- Clean architecture
- Automation-first design

This is a production ML evaluation platform — not a demo.
