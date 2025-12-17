# Base
FROM registry.access.redhat.com/ubi10-minimal AS base

WORKDIR /app
USER root

ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/tmp

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /bin/
RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-10.noarch.rpm && \
    microdnf -y install python3.13

# Build
FROM base AS builder

COPY granite_core /app/granite_core
COPY a2a /app/a2a

RUN uv --directory a2a --no-managed-python sync --no-dev --no-cache --locked --link-mode copy --no-editable

# Run
FROM base AS runner

USER 1001

COPY --from=builder --chown=1001:1001 /app/a2a /app

CMD ["uv", "run", "--no-sync", "-m", "a2a_agents.agents.search.agent_search"]
