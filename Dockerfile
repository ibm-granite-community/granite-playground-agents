FROM registry.access.redhat.com/ubi9/python-311 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /bin/

WORKDIR /app
USER root

COPY . /app/

ENV PRODUCTION_MODE=True

RUN uv sync --no-dev --no-cache --locked --link-mode copy

RUN chown 1001:1001 -R /app

USER 1001

ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/tmp

CMD ["uv", "run", "--no-sync", "granite_chat/agent.py"]