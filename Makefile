.PHONY: install check lint format typecheck test cov demo db-up db-down db-ps image image-smoke \
        cluster-dry

COMPOSE = docker compose -f infrastructure/docker/docker-compose.yml --env-file .env
TAG ?= latest
EXTRAS ?= databases

install:            ## create .venv from the lockfile (+ FAISS)
	uv sync --locked --extra faiss

check: lint typecheck test   ## everything CI runs

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

typecheck:
	uv run mypy

test:
	uv run pytest -q

cov:
	uv run pytest -q --cov=elevate_rag --cov-report=term-missing

demo:               ## offline end-to-end demo (synthetic data, no credentials)
	uv run elevate-rag demo --out demo_output

db-up:              ## reference DBs (Qdrant + Neo4j); PROFILE=all starts every engine
	$(COMPOSE) --profile $${PROFILE:-reference} up -d

db-down:            ## stop the DB containers (data volumes are kept)
	$(COMPOSE) --profile all down

db-ps:              ## status of the DB containers
	$(COMPOSE) --profile all ps

image:              ## application image; EXTRAS="databases local-models" adds torch (~2 GB)
	docker build -t elevate-rag:$(TAG) --build-arg EXTRAS="$(EXTRAS)" .

image-smoke:        ## offline demo inside the image (no credentials, no network)
	docker run --rm --network none elevate-rag:$(TAG) demo --out /data/demo

DRY_ROOT ?= /tmp/elevate-dry
cluster-dry:        ## print the SLURM jobs `submit.sh smoke` would submit (no cluster needed)
	ELEVATE_ROOT=$(DRY_ROOT) DRY_RUN=1 infrastructure/cluster/submit.sh smoke
