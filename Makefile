.PHONY: install install-dev test lint format clean help

help:
	@echo "Available commands:"
	@echo "  install       Install package dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run tests with pytest"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Remove build artifacts and cache files"
	@echo "  train         Run training script"
	@echo "  notebook      Start Jupyter notebook server"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src tests
	black --check src tests
	isort --check-only src tests

format:
	black src tests scripts
	isort src tests scripts

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml

train:
	python scripts/train.py --config configs/default_config.yaml

notebook:
	jupyter notebook notebooks/
