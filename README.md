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

The agent is designed to use models from Watsonx directly. Copy the `.env.template` file to `.env` and fill in the missing secrets.

```bash
cp .env.template .env
# then edit the file
```

## Running and using the agents

Run the agent locally.

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

Run the agent

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
podman run --env-file .env --name beeai-platform-granite-chat -p 8000:8000 --rm localhost/beeai-platform-granite-chat:latest
```

## Tests

This repository currently has a stubbed out test in the `tests` directory.

Tests can be run with:

```bash
uv run pytest
```
