.PHONY: setup test lint

setup:
	uv sync --group dev --group train

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
