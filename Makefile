.PHONY: test

test:
	uv run --no-project --with-requirements fuzz/requirements-property.txt python -m pytest -q
