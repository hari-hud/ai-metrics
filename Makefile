.PHONY: install test lint build clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/

lint:
	pre-commit run --all-files

build:
	python -m build

upload: build
	twine upload dist/*

clean:
	rm -rf dist build *.egg-info
