.PHONY: help install lint type test train compare docker-build docker-run clean release

help:
	@echo "MV-Coach-Eval Makefile"
	@echo "======================"
	@echo "Available targets:"
	@echo "  install       - Install dependencies"
	@echo "  lint          - Run linting (black, flake8)"
	@echo "  type          - Run type checking (mypy)"
	@echo "  test          - Run tests with coverage"
	@echo "  train         - Run benchmark training"
	@echo "  compare       - Run model comparison"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo "  clean         - Clean build artifacts"
	@echo "  release       - Create release package"

install:
	pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

lint:
	black mv_coach tests
	flake8 mv_coach tests --max-line-length=88 --extend-ignore=E203

type:
	mypy mv_coach --ignore-missing-imports

test:
	pytest tests/ --cov=mv_coach --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

train:
	python mv_coach/benchmark_script.py

compare:
	python mv_coach/benchmark_script.py model=tcn
	python mv_coach/benchmark_script.py model=cnn1d

docker-build:
	docker build -t mv-coach-eval:latest .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/experiments:/app/experiments mv-coach-eval:latest

docker-run-gpu:
	docker run --rm --gpus all -v $(PWD)/data:/app/data -v $(PWD)/experiments:/app/experiments mv-coach-eval:latest

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf experiments/run_*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

release:
	python -m build
	@echo "Package built in dist/"
	@echo "To upload to PyPI: twine upload dist/*"
