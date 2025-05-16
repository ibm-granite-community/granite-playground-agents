# Granite Chat

ACP agent that supports basic chat with Granite. Implemented using beeai-framework.

## Pre-requisites

- Python 3.11 or higher
- UV package manager: https://docs.astral.sh/uv/

## Installation

1. Install dependencies using `uv sync`

2. Activate venv using `source .venv/bin/activate`

## Configuration

Example RITS configuration can be found in `.env.example`

Rename to `.env` and fill in the `OPENAI_API_KEY` with your `RITS_API_KEY`.

You also need to fill in the `OPENAI_API_HEADERS` with your `RITS_API_KEY`.

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