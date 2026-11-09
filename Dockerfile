# ABOUTME: Image for the Semvec x Neo4j demo: seed script, interactive demo and test suite.
# ABOUTME: CPU-only torch keeps the image small; configuration and secrets come from .env at runtime.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU build of torch first, so sentence-transformers does not pull the multi-GB CUDA wheels.
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install ".[dev]" && rm -rf build

COPY docs ./docs
COPY schema ./schema
COPY healthcare-fixtures ./healthcare-fixtures
COPY scripts ./scripts
COPY tests ./tests

RUN useradd --create-home --uid 1000 app \
    && mkdir -p "$HF_HOME" \
    && chown -R app:app /app /home/app
USER app

CMD ["python", "scripts/interactive_demo.py"]
