FROM registry.access.redhat.com/ubi9 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/

WORKDIR /app
USER root

RUN dnf -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    dnf -y install python3.13

COPY granite_core /app/granite_core
COPY a2a /app/a2a

ENV PRODUCTION_MODE=True

RUN uv --directory a2a sync --no-dev --no-cache --locked --link-mode copy --no-editable

FROM registry.access.redhat.com/ubi9/python-311 AS runner
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/

WORKDIR /app
USER root

COPY --from=builder /app/a2a /app

ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/tmp

CMD ["uv", "run", "--no-sync", "-m", "a2a_agents.agent_chat"]
