# Makefile for omero-annotate-ai

.PHONY: help install test test-unit test-integration test-omero clean docs lint format

help:		## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:	## Install package in development mode
	pip install -e .[dev,test,docs]

test:		## Run all tests (unit + integration)
	python -m pytest tests/ -v

test-unit:	## Run only unit tests
	python -m pytest tests/ -v -m "not (omero or integration or docker)"

test-integration:	## Run integration tests with Docker OMERO
	@echo "Starting OMERO server with Docker Compose..."
	cd tests && docker-compose up -d
	@echo "Waiting for OMERO server to be ready..."
	sleep 60
	@echo "Running integration tests..."
	python -m pytest tests/test_omero_integration.py -v -m "omero or integration" || true
	@echo "Stopping OMERO server..."
	cd tests && docker-compose down -v

test-omero:	## Run OMERO-specific tests only
	python -m pytest tests/ -v -m "omero"

start-omero:	## Start OMERO server with Docker Compose
	cd tests && docker-compose up -d
	@echo "OMERO server starting... wait ~60 seconds for full startup"
	@echo "OMERO.web will be available at http://localhost:5080"
	@echo "OMERO server ports: 6063, 6064"

stop-omero:	## Stop OMERO server
	cd tests && docker-compose down -v

clean:		## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:		## Build documentation
	cd docs && make html
	@echo "Documentation built in docs/_build/html/"

docs-serve:	## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

lint:		## Run linting checks
	flake8 src/omero_annotate_ai
	black --check src/omero_annotate_ai
	isort --check-only src/omero_annotate_ai

format:		## Format code
	black src/omero_annotate_ai
	isort src/omero_annotate_ai

check:		## Run all checks (lint + test-unit)
	$(MAKE) lint
	$(MAKE) test-unit