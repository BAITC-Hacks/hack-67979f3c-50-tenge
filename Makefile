SHELL := /bin/sh
.DEFAULT_GOAL := run

PYTHON ?= python3
VENV ?= .venv
PORT ?= 8501
VENV_PYTHON := $(VENV)/bin/python

.PHONY: run setup pipeline ui test docker-up docker-pipeline docker-down help

# Install once, and again when either requirements file changes.
$(VENV_PYTHON):
	"$(PYTHON)" -m venv "$(VENV)"

$(VENV)/.requirements-installed: requirements.txt starter/requirements.txt $(VENV_PYTHON)
	"$(VENV_PYTHON)" -m pip install --quiet -r requirements.txt
	touch "$@"

setup: $(VENV)/.requirements-installed

pipeline: setup
	"$(VENV_PYTHON)" run.py --data data --out out

# The UI must start only after the pipeline succeeds, including with make -j.
run: pipeline
	"$(VENV_PYTHON)" -m streamlit run viz/app.py --server.port "$(PORT)"

ui: setup
	"$(VENV_PYTHON)" -m streamlit run viz/app.py --server.port "$(PORT)"

test: setup
	"$(VENV_PYTHON)" -m pytest -q

docker-pipeline:
	docker compose run --build --rm pipeline

docker-up: docker-pipeline
	docker compose up --build -d dashboard
	@echo "Dashboard: http://localhost:8501"

docker-down:
	docker compose down

help:
	@echo "make / make run       Install dependencies, compute results, start the UI"
	@echo "make setup            Prepare the local Python environment"
	@echo "make pipeline         Compute all outputs without starting the UI"
	@echo "make ui               Open the UI using existing outputs"
	@echo "make test             Run the test suite"
	@echo "make docker-up        Compute results and start the dashboard in Docker"
	@echo "make docker-pipeline  Compute results in Docker"
	@echo "make docker-down      Stop the Docker services"
	@echo "Options: PYTHON=python3.12 PORT=8502 VENV=.venv"
