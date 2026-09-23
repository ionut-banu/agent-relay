FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    RELAY_DATABASE_URL=sqlite:////data/agent-relay.db

WORKDIR /app

# Dependencies first so code edits don't invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY main.py database.py storage.py schemas.py errors.py worker.py dashboard.py dashboard.html ./

RUN useradd --system --uid 1000 relay && mkdir /data && chown relay /data
USER relay
VOLUME /data
EXPOSE 8000

# 0.0.0.0 so the published port is reachable from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
