FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY resources/vectors_q8.npy ./resources/vectors_q8.npy
COPY resources/vectors_f16.npy ./resources/vectors_f16.npy
COPY resources/labels.npy ./resources/labels.npy
COPY resources/centroids.npy ./resources/centroids.npy
COPY resources/cluster_indices.npy ./resources/cluster_indices.npy
COPY resources/cluster_offsets.npy ./resources/cluster_offsets.npy
COPY resources/tree.npz ./resources/tree.npz

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
