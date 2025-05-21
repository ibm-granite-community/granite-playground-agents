FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /bin/

WORKDIR /app
COPY . /app

RUN uv sync --no-dev --no-cache --locked --link-mode copy

FROM python:3.11-slim-bookworm

ENV LLM_MODEL=ibm-granite/granite-3.3-8b-instruct
ENV LLM_API_BASE=unused
ENV LLM_API_KEY=unused

ENV PRODUCTION_MODE=True
ENV PATH="/app/.venv/bin:$PATH"

RUN useradd -m appuser
USER appuser

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

WORKDIR /app

CMD ["python", "granite_chat/agent.py"]
