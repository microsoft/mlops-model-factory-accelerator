SHELL:=/bin/bash

setup:
	python -m venv /tmp/fridge_env
	. /tmp/fridge_env/bin/activate
	python -m pip install --upgrade pip
	python -m pip install -r ./devops/pipelines/requirements/build_validation_requirements.txt

lint: setup
	flake8 .