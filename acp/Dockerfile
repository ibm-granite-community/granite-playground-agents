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
COPY acp /app/acp

RUN microdnf -y install git && \
    uv --directory acp --no-managed-python sync --no-dev --no-cache --locked --link-mode copy --no-editable

# Pre-download NLTK punkt resource
RUN python -m nltk.downloader punkt_tab -d /app/acp/nltk_data

# Run
FROM base AS runner

USER 1001

COPY --from=builder --chown=1001:1001 /app/acp /app

# Set environment variable so NLTK finds it
ENV NLTK_DATA=/app/nltk_data

CMD ["uv", "run", "--no-sync", "-m", "acp_agent.agent"]
