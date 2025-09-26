FROM registry.access.redhat.com/ubi9/python-311 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/

WORKDIR /app
USER root

COPY granite_core /app/granite_core
COPY acp /app/acp

ENV PRODUCTION_MODE=True

RUN uv --directory acp sync --no-dev --no-cache --locked --link-mode copy --no-editable

FROM registry.access.redhat.com/ubi9/python-311 AS runner
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/

WORKDIR /app
USER root

COPY --from=builder /app/acp /app

ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/tmp

CMD ["uv", "run", "--no-sync", "-m", "acp_agent.agent"]
