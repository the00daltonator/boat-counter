# Boat Counter Makefile

# Python settings
PYTHON = python3
VENV = venv
PIP = $(VENV)/bin/pip
PYTHON_VENV = $(VENV)/bin/python

# Path settings
SRC_DIR = .
TEST_DIR = tests
COV_REPORT_DIR = coverage-report

.PHONY: all setup clean test lint coverage run help

all: help

help:
	@echo "Boat Counter Make Commands:"
	@echo "  setup        Create virtual environment and install dependencies"
	@echo "  test         Run all tests"
	@echo "  coverage     Run tests with coverage report"
	@echo "  lint         Run linting checks"
	@echo "  clean        Remove build artifacts and temporary files"
	@echo "  run          Run the boat counter with default settings"

setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

test: setup
	@echo "Running tests..."
	@$(PYTHON_VENV) -m pytest -v $(TEST_DIR)

coverage: setup
	@echo "Running tests with coverage..."
	@mkdir -p $(COV_REPORT_DIR)
	@$(PYTHON_VENV) -m pytest --cov=$(SRC_DIR) --cov-report=html:$(COV_REPORT_DIR) --cov-report=term $(TEST_DIR)
	@echo "Coverage report generated in $(COV_REPORT_DIR)"

lint: setup
	@echo "Running linting..."
	@$(PYTHON_VENV) -m pylint $(SRC_DIR)/boat_counter.py || true
	@$(PYTHON_VENV) -m pylint $(TEST_DIR) || true

clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV)
	@rm -rf $(COV_REPORT_DIR)
	@rm -rf __pycache__
	@rm -rf .pytest_cache
	@rm -rf $(TEST_DIR)/__pycache__
	@rm -rf $(TEST_DIR)/.pytest_cache
	@rm -rf .coverage
	@echo "Cleanup complete!"

run: setup
	@echo "Running Boat Counter..."
	@$(PYTHON_VENV) boat_counter.py --display 