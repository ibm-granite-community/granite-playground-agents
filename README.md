# Granite Chat

Implemented using beeai-framework and ACP, providing a variety of different agents

- basic chat with Granite (no external sources)
- chat plus search (uses an external search source such as Google/Tavily)
- deep research (uses external search with additional planning and recursion)

## Pre-requisites

- Python 3.11 or higher
- UV package manager: https://docs.astral.sh/uv/

### Development

Install the pre-commit hooks prior to modifying the code:

```sh
pre-commit install
```

## Installation

Install dependencies using `uv sync`

Activate your virtual environment `source .venv/bin/activate`

## Configuration

The agent is designed to use models from Watsonx directly. Copy the `.env.template` file to `.env` and fill in the missing secrets.

```bash
cp .env.template .env
# then edit the file
```

## Running the agent

Run the agent locally.

```sh
uv run -m granite_chat.agent.py
```

## Using the agent

You can chat with the agent via the BeeAI Platform ui using the [beeai-cli](https://docs.beeai.dev/how-to/cli-reference)

```sh
beeai ui
```

Go to `Agents` and look for `granite-chat`.

## Containerisation

### Build

```bash
podman build -t beeai-platform-granite-chat:latest .
```

### Run

```shell
podman run --env-file .env --name beeai-platform-granite-chat -p 8000:8000 --rm localhost/beeai-platform-granite-chat:latest
```


## Tests

This repository currently has a stubbed out test in the `tests` directory.

Tests can be run with:

```bash
uv run pytest
```
