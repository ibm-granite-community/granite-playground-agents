FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /bin/

ENV UV_LINK_MODE=copy \
    PRODUCTION_MODE=true

ADD . /app
WORKDIR /app

RUN uv sync --no-cache --locked --link-mode copy

FROM python:3.11-slim-bookworm

ENV LLM_MODEL=ibm-granite/granite-3.3-8b-instruct
ENV LLM_API_BASE=http://localhost:8333/api/v1/llm

ENV PRODUCTION_MODE=True \
    PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY --from=builder /app /app

# Set working directory
WORKDIR /app

CMD ["python", "granite_chat/agent.py"]