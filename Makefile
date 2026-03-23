.PHONY: install dev test lint format run clean docker

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

run:
	streamlit run src/viz/app.py

demo:
	python examples/demo.py

cli-price:
	python -m src.cli price --coupon 0.05 --maturity 5 --ytm 0.05

cli-risk:
	python -m src.cli risk --coupon 0.05 --maturity 5 --ytm 0.05

cli-curve:
	python -m src.cli curve

cli-scenario:
	python -m src.cli scenario --coupon 0.05 --maturity 5 --shift 50

docker:
	docker build -t fixed-income-toolkit .

docker-run:
	docker run -p 8501:8501 fixed-income-toolkit

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .mypy_cache/
