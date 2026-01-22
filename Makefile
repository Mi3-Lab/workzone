.PHONY: help install install-dev train infer test lint format clean

help:
	@echo "WorkZone - AI Construction Zone Detection System"
	@echo ""
	@echo "Available commands:"
	@echo "  make install           Install package dependencies"
	@echo "  make install-dev       Install with development dependencies"
	@echo "  make train             Train YOLO model"
	@echo "  make infer             Run inference on video"
	@echo "  make app               Run the main Streamlit app (Phase 2.1 Evaluation)"
	@echo "  make streamlit         Alias for 'make app'"
	@echo "  make app-semantic-fusion  Run the Semantic Fusion Streamlit app"
	@echo "  make app-basic-detection  Run the Basic Detection Streamlit app"
	@echo "  make app-advanced-scoring Run the Advanced Scoring Streamlit app"
	@echo "  make workzone         Run the Jetson Orin Launcher (GUI)"
	@echo "  make test              Run all tests"
	@echo "  make test-coverage     Run tests with coverage report"
	@echo "  make lint              Run linting checks"
	@echo "  make format            Format code with black and isort"
	@echo "  make clean             Clean cache and build files"
	@echo "  make docs              Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

train:
	python -m src.workzone.cli.train_yolo --device 0

infer:
	python -m src.workzone.cli.infer_video \
		--video data/Construction_Data/EYosemiteAve_NightFog.mp4 \
		--model weights/best.pt \
		--output results/output.mp4

app-semantic-fusion:
	streamlit run src/workzone/apps/streamlit/app_semantic_fusion.py

app-basic-detection:
	streamlit run src/workzone/apps/streamlit/app_basic_detection.py

app-advanced-scoring:
	streamlit run src/workzone/apps/streamlit/app_advanced_scoring.py

app:
	@echo "🚀 Launching WorkZone Streamlit App..."
	@bash scripts/launch_streamlit.sh

streamlit: app

workzone:
	@echo "🚀 Launching WorkZone Controller (Jetson Orin)..."
	@python3 scripts/jetson_launcher.py

test:
	pytest tests/ -v

test-coverage:
	pytest tests/ -v --cov=src/workzone --cov-report=html
	@echo "Coverage report generated: htmlcov/index.html"

lint:
	flake8 src/ tests/
	mypy src/
	pylint src/

format:
	black src/ tests/ --line-length 88
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache build/ dist/ *.egg-info/
	rm -rf htmlcov/ .coverage

docs:
	@echo "Building documentation..."
	cd docs && sphinx-build -b html . _build/

.DEFAULT_GOAL := help
