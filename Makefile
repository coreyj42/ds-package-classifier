# Variables
PYTHON := poetry run python
POETRY := poetry

# Targets
.PHONY: all install lint test format clean

all: install format lint test

install:
	$(POETRY) install

format:
	$(PYTHON) -m black ds_package_classifier tests

lint:
	$(POETRY) run flake8 ds_package_classifier tests

test:
	$(PYTHON) -m pytest tests

build:
	$(POETRY) build

clean:
	$(POETRY) env remove $(shell $(POETRY) env info -p)