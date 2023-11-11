.DEFAULT_GOAL := help

LD_LIBRARY_PATH := $(shell python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
SPHINX_BUILD := 1

# Setups the python environment with poetry. Assumes there is an active venv already.
setup: 
	poetry install --sync --no-root --with dev --with docs
	maturin develop

# Runs rust unit tests
rust-test:
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH)" cargo test --no-default-features --workspace

# Runs python unit tests
pytest:
	USE_SYMENGINE=1 pytest tests

# Runs all unit tests
test: rust-test pytest

# Generates the readme from example jupyter notebook. Needs fixing manually because of github markdown format
gen-readme:
	jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --to markdown notebooks/Example.ipynb

tag-python:
	tomlq -t -i --arg tag "$(LIESYM_VERSION)" '.tool.poetry.version |= $$tag' pyproject.toml 
	tomlq -t -i --arg tag "$(LIESYM_VERSION)" '.project.version |= $$tag' pyproject.toml 

tag-rust:
	tomlq -t -i --arg tag "$(LIESYM_VERSION)" '.package.version |= $$tag' Cargo.toml 
	tomlq -t -i --arg tag "$(LIESYM_VERSION)" '.package.version |= $$tag' ./rootsystem/Cargo.toml 

# Tags both rust and python version with $LIESYM_VERSION
tag: tag-python tag-rust

# Builds the sphinx docs with warnings as errors
html:
	cd docs && $(MAKE) html SPHINXOPTS="-W"

# Update poetry deps to general python deps
conv-deps:
	@#!/bin/bash; \
	tomlq -i -t --arg i "$$dep" '.project.dependencies |=  [ ]' pyproject.toml; \
	poetry export --without-hashes  | while IFS= read -r dep; \
	do \
		tomlq -i -t --arg i "$$dep" '.project.dependencies |= . + [$$i]' pyproject.toml; \
	done

help:
	@echo "Available targets:"
	@echo "=================="
	@awk '/^#/{c=substr($$0,3);next}c&&/^[[:alpha:]][[:alnum:]_-]+:/{print substr($$1,1,index($$1,":")),c}1{c=0}' $(MAKEFILE_LIST) | sort | column  -s: -t 

.PHONY: setup rust-test pytest test gen-readme help
