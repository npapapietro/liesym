
.DEFAULT_GOAL := help

gen-readme:
	jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --to markdown notebooks/Example.ipynb


help:
	@echo "Available targets:"
	@echo "  gen-readme	- Generates the markdown for the readme from the example notebook."
	@echo "  help   	- Show this help message"

.PHONY: gen-readme