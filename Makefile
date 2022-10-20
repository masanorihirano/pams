PROJECT_NAME := pams
RUN := poetry run

.PHONY: check
check: test lint mypy

.PHONY: install
install:
	@poetry install

.PHONY: test
test: pytest

.PHONY: pytest
pytest:
	$(RUN) pytest

.PHONY: test-cov
test-cov:
	$(RUN) pytest --cov=$(PROJECT_NAME) --cov-report=xml

.PHONY: lint
lint: lint-black lint-isort

.PHONY: lint-black
lint-black:
	$(RUN) black --check --diff --quiet --skip-magic-trailing-comma .

.PHONY: lint-isort
lint-isort:
	$(RUN) isort --check --force-single-line-imports --quiet .

.PHONY: mypy
mypy:
	$(RUN) mypy $(PROJECT_NAME)

.PHONY: format
format: format-black format-isort

.PHONY: format-black
format-black:
	$(RUN) black --quiet --skip-magic-trailing-comma .

.PHONY: format-isort
format-isort:
	$(RUN) isort --force-single-line-imports --quiet .

.PHONY: publish
publish:
	@poetry publish --build

.PHONY: test-publish
test-publish:
	@poetry publish --build -r testpypi

.PHONY: doc
doc:
	@cd docs && make html