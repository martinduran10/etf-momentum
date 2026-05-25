.PHONY: help install test results clean lint

help:
	@echo "Available targets:"
	@echo "  install     Install Python dependencies"
	@echo "  test        Run the pytest suite"
	@echo "  results     Regenerate every figure and table in reports/"
	@echo "  clean       Remove caches and generated outputs"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

results:
	python scripts/generate_results.py

clean:
	rm -rf __pycache__ .pytest_cache .ipynb_checkpoints
	rm -rf src/__pycache__ tests/__pycache__
	rm -f reports/figures/*.png reports/tables/*.csv
