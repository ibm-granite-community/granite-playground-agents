FROM registry.access.redhat.com/ubi9/python-311 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/

WORKDIR /app
USER root

COPY . /app/

ENV PRODUCTION_MODE=True

RUN uv --directory acp sync --no-dev --no-cache --locked --link-mode copy

ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/tmp

CMD ["uv", "--directory", "acp", "run", "--no-sync", "-m", "acp_agent.agent"]
