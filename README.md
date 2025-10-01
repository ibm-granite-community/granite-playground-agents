# Granite Chat

Implemented using beeai-framework with agents implemented using ACP and A2a:

- basic chat with Granite (no external sources)
- chat plus search (uses an external search source such as Google/Tavily)
- chat with Thinking (no external sources)
- deep research (uses external search with additional planning and recursion)

## Pre-requisites

- Python (version range specified in individual `pyproject.toml` files)
- UV package manager: https://docs.astral.sh/uv/

### Development

Install the pre-commit hooks prior to modifying the code:

```sh
pre-commit install
```

#### Debugging with vscode

Due to the monorepo layout used in this repository, each agent can be used with the Python debugger in vscode by creating a `launch.json` file similar to the following example. Please note, the important piece to change is the `cwd` specification that should point to the sub-directory of the agent to be executed in the debugger:

```json
{
  "name": "Python Debugger: Current File",
  "type": "debugpy",
  "request": "launch",
  "program": "${file}",
  "console": "integratedTerminal",
  "cwd": "${workspaceFolder}/acp"
}
```

## Configuration

The core library is designed to use Granite models that can be served from a variety of back ends. To configure the library, ensure environment variables are in place when running the code (this can be done via a `.env` file). The configuration options available are documented in the [granite_core config.py](granite_core/granite_core/config.py) file where you will find a brief description of each option, the data type it expects, potential limitations on values and a default value.

The agents are configured in a similar way to the core, via environment variables (that can also be set via a `.env` file). The configurations are in the relevant `config.py` files for each agent. The agents will start without any additional configuration by adopting default values such as using Granite models served via a local Ollama and search provided by simple a DuckDuckGo implementation. This is sufficient for initial/early experimental usage. However, you are encouraged to explore the options to achieve better accuracy and throughput.

## Running and using the agents

Run the agents locally.

### ACP

Run the agent

```sh
uv --directory acp run -m acp_agent.agent
```

Use the agent

```sh
curl -X POST \
  --url http://localhost:8000/runs \
  -H 'content-type: application/json' \
  --data '{
  "agent_name": "granite-chat",
  "input": [
    {
      "role": "user",
      "parts": [
        {
          "content": "Hello"
        }
      ]
    }
  ],
  "mode": "stream"
}'
```

### A2A

Run an agent

```sh
uv --directory a2a run -m a2a_agents.agent_chat
```

You can chat with the agent via the BeeAI Platform ui using the [beeai-cli](https://docs.beeai.dev/how-to/cli-reference)

```sh
beeai agent run "Granite Chat
```

## Containerisation

### Build

```bash
# ACP Agent
podman build -t beeai-platform-granite-chat:latest -f acp/Dockerfile .

# A2A Agent
podman build -t beeai-platform-granite-chat:latest -f a2a/Dockerfile .
```

### Run

```shell
podman run --env-file .env --name beeai-platform-granite-chat -p 8000:8000 --rm beeai-platform-granite-chat:latest
```

## Tests

This repository currently has a stubbed out test in the `tests` directory.

Tests can be run with:

```bash
uv run pytest
```
