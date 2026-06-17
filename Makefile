.PHONY: format format-check lint type static test all

format:
	python -m ruff format .

format-check:
	python -m ruff format --check .

lint:
	python -m ruff check .

type:
	python -m mypy

static: format-check lint type

test:
	python -m pytest

all: format-check lint type test