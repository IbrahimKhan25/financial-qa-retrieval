# Retrieval API image: ONNX Runtime + tokenizers + BM25, no PyTorch.
# Build the model and index files first (they are baked in, so one image = one model version):
#   uv run python scripts/build_serving_artifacts.py && uv run python scripts/check_serving_parity.py
#   docker build -t finqa-serve .
#   docker run --rm -p 8000:8000 --cpus 4 finqa-serve
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Dependencies first, from the lock file, so this layer is cached until the lock changes.
# --only-group serve installs the serving packages and nothing else (no torch, no pandas).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group serve --no-cache

COPY src/finqa ./src/finqa
COPY serving_artifacts ./serving_artifacts

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    FINQA_ARTIFACTS=/app/serving_artifacts \
    FINQA_ORT_THREADS=4

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"

CMD ["uvicorn", "finqa.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
