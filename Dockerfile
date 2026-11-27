# Elevate-RAG application image (CLI). Databases run separately: infrastructure/docker.
#
#   docker build -t elevate-rag .                                   # CPU, all DB clients
#   docker build -t elevate-rag --build-arg EXTRAS="databases local-models" .   # + torch (~2 GB)
#   docker run --rm --env-file .env --network host -v /mnt/elevate:/data elevate-rag env
#
# Dependencies come only from uv.lock (`--locked`), so the image matches CI and the lockfile.

ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm

# ---------------------------------------------------------------------------------- builder
FROM ghcr.io/astral-sh/uv:0.11.8 AS uv
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv

ARG EXTRAS="databases"
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app
# 1) third-party dependencies only (cached until pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN set -eu; extras=""; for e in ${EXTRAS}; do extras="$extras --extra $e"; done; \
    uv sync --locked --no-dev --no-install-project $extras
# 2) the project itself
COPY src ./src
RUN set -eu; extras=""; for e in ${EXTRAS}; do extras="$extras --extra $e"; done; \
    uv sync --locked --no-dev --no-editable $extras
# 3) tiktoken downloads its vocabulary on first use; bake it in so the image runs offline.
#    o200k_base serves both Elevate-RAG and LightRAG's default (gpt-4o-mini).
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN /opt/venv/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

# ---------------------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE}
RUN useradd --create-home --uid 10001 elevate \
 && mkdir -p /data && chown elevate:elevate /data

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/tiktoken /opt/tiktoken
WORKDIR /app
COPY configs ./configs
COPY scripts ./scripts

# All mutable state lives under /data (mount an EBS volume there). Relative paths inside
# configs/ resolve against /app; the storage locations below come from the environment.
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    ELEVATE_DATA_DIR=/data/data \
    ELEVATE_CACHE_DIR=/data/cache \
    ELEVATE_INDEX_DIR=/data/indexes \
    ELEVATE_RUNS_DIR=/data/runs \
    HF_HOME=/data/hf-cache \
    ELEVATE_JSON_LOGS=1

USER elevate
VOLUME ["/data"]
ENTRYPOINT ["elevate-rag"]
CMD ["--help"]
