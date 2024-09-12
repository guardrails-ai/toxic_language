.PHONY: test dev lint type qa

dev:
	pip install -e ".[dev]"

lint:
	ruff check .

type:
	pyright validator

test:
	pytest -v tests

qa:
	make lint
	make type
	make test
