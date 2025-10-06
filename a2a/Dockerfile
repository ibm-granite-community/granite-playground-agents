# Base
FROM registry.access.redhat.com/ubi9 AS base
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/

WORKDIR /app
USER root

RUN dnf -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    dnf -y install python3.13 && \
    dnf clean all

# Build
FROM base AS builder

COPY granite_core /app/granite_core
COPY a2a /app/a2a

ENV PRODUCTION_MODE=True

RUN uv --directory a2a sync --no-dev --no-cache --locked --link-mode copy --no-editable

# Run
FROM base AS runner

USER 1001

COPY --from=builder --chown=1001:1001 /app/a2a /app

ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/tmp

CMD ["uv", "run", "--no-sync", "-m", "a2a_agents.agent_chat"]
