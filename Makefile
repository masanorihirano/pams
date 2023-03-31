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
lint: lint-black lint-isort lint-flake8

.PHONY: lint-black
lint-black:
	$(RUN) black --check --diff --quiet .

.PHONY: lint-isort
lint-isort:
	$(RUN) isort --check --quiet .

.PHONY: lint-flake8
lint-flake8:
	$(RUN) pflake8 .


.PHONY: mypy
mypy:
	$(RUN) mypy $(PROJECT_NAME)

.PHONY: format
format: format-black format-isort

.PHONY: format-black
format-black:
	$(RUN) black --quiet .

.PHONY: format-isort
format-isort:
	$(RUN) isort --quiet .

.PHONY: publish
publish:
	@poetry publish --build

.PHONY: test-publish
test-publish:
	@poetry publish --build -r testpypi

.PHONY: doc-intl
doc-intl:
	@cd docs && make gettext && sphinx-intl update -p build/gettext -l ja && \
	 echo "please check and modify docs/source/locale/ja/LC_MESSAGES/index.po"

.PHONY: doc
doc:
	@cd docs && rm -rf build/html/ build/html-ja/ build/html-en/ && \
	 rm -r source/reference/generated && git checkout source/reference/generated/.gitignore && \
	 make -e SPHINXOPTS="-a" html && mv build/html/ build/html-en/ && \
	 make -e SPHINXOPTS="-D language='ja' -a" html &&  mv build/html/ build/html-ja && \
	 mkdir -p build/html/ && mv build/html-en/ build/html/en/ &&  mv build/html-ja/ build/html/ja/ && \
	 cp source/_index.html build/html/index.html