# Granite Chat

ACP agent that supports basic chat with Granite. Implemented using beeai-framework.

## Pre-requisites

- Python 3.11 or higher
- UV package manager: https://docs.astral.sh/uv/

### Development

Install the pre-commit hooks prior to modifying the code:

```sh
pre-commit install
```

## Installation

1. Install dependencies using `uv sync`

2. Activate venv using `source .venv/bin/activate`

## Configuration

Usage or RITS is recommended through BeeAI platform. The platform will abstract away any complexity of using RITS as standard OpenAI API compatible endpoints.

1. Make sure you setup RITS an inference provider in the platform. (Select Other and enter RITS Model inference endpoint and RITS_API_KEY)
2. You need to copy `.env.beeai` and rename to `.env` and as long as the platform in running on http://localhost:8333 the agent should work. Keep in mind API key is then unused as it's provided by the platform

If you want to run the agent on RITS without involving beeai.

1. Make a copy of `.env.rits`.
2. Set `LLM_API_BASE` to your RITS Model inference endpoint. Include `/v1` at the end.
3. Fill in your RITS_API_KEY in `LLM_API_KEY` and `LLM_API_HEADERS`

## Running the agent

Run the agent locally.

```sh
uv run granite_chat/agent.py
```

## Using the agent

You can chat with the agent via the beeai ui.

```sh
beeai ui
```

Go to `Agents` and look for `granite-chat`.

You can also run the agent using the client.

```sh
uv run granite_chat/client.py
```

## Containerisation

### Build

```bash
podman build -t beeai-platform-granite-chat:latest .
```

### Run

```shell
podman run --env-file .env --name beeai-platform-granite-chat -p 8000:8000 --rm localhost/beeai-platform-granite-chat:latest
```
