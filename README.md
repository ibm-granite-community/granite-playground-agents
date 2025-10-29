# Granite Playground Agents

These are the agents used to power the IBM [Granite Playground](https://www.ibm.com/granite/playground).

# Agents Overview

Implemented using [beeai-framework](github.com/i-am-bee/beeai-framework/), the agents can be exposed with the [A2A](https://a2a-protocol.org) (or the deprecated ACP) protocol:

- basic **chat** with Granite (no external sources)
- chat with **thinking** (no external sources, the LLM will perform additional reasoning steps)
- chat plus **search** (uses an external search source DuckDuckGo|Google|Tavily)
- deep **research** (uses external search with additional planning and recursion)

## Pre-requisites

- Python (version range specified in individual `pyproject.toml` files)
- UV package manager: https://docs.astral.sh/uv/
- An A2A client e.g. [BeeAI Platform](https://github.com/i-am-bee/beeai-platform) recommended
- Access to an LLM e.g. [Ollama](https://ollama.com/)

# Getting Started

This guide will get you started with the out-of-the-box agent configuration. This may not be optimally configured for your desired use case and as such, we recommend looking at the [configuration](#configuration) _after_ you have successfully run the agents using the defaults. The configuration options can be overridden with environment variables (including via a `.env` file).

## Setup Ollama

The default LLM provider is a local Ollama server. You need to have Ollama downloaded and running on your machine with the Granite 4 model. Follow these steps:

1. Go to [Ollama](https://ollama.com/) and download the installer for your system
1. Start the Ollama server
1. Pull the `ibm/granite4` and the `nomic-embed-text` models
   ```sh
   ollama pull ibm/granite4:latest
   ollama pull nomic-embed-text:latest
   ```

## Setup the BeeAI platform

An A2A client is required to use the agents. The agents are designed to work with the BeeAI Platform and take advantage of several A2A extensions offered by the platform. Follow these steps:

1. Refer to the [BeeAI quick start guide](https://docs.beeai.dev/introduction/quickstart) to download and install the platform
1. Run the platform

   ```sh
   beeai platform start
   # wait for the platform to fully start before moving on
   ```

There is a [beeai-cli](https://docs.beeai.dev/how-to/cli-reference) reference you can use for further commands but the above is sufficient to get started.

## Run the agents

Select which agent you would like to run and start the agent:

```sh
# pick one of these
uv --directory a2a run -m a2a_agents.agent_chat
uv --directory a2a run -m a2a_agents.agent_search
uv --directory a2a run -m a2a_agents.agent_research
uv --directory a2a run -m a2a_agents.agent
```

After starting the agent, you will see lots of log output. If you're running the BeeAI Platform then the agent will register itself with the platform and you will see the following log message that indicates success:

```
INFO     | beeai_sdk    | Agent registered successfully
```

You can now interact with the agents via the BeeAI Platform user interface in your web browser. Run the following to start your web browser at the appropriate page:

```sh
beeai ui
```

The UI will start in your web browser. Select the ☰ hamburger menu (top left) and click on the Granite agent that you are running. Once selected, you can type your prompt into the input box to run the agent.

> [!TIP]
> The first time you start the BeeAI Platform UI, you will need to select an LLM back end. Ensure your local Ollama server is running. Use the arrow keys on your keyboard to select `Ollama`. Use the default settings that you're presented with when the platform checks the Ollama connection.

> [!NOTE]
> BeeAI Platform provides agents with an A2A extension that allows them access to the LLM models provided via the platform. The agents in this repository do not use this functionality since they make their own direct connection to an LLM via their [configuration](#configuration).

<details>
<summary>Run the ACP agent</summary>

We do not recommend using the ACP version of the agents since the ACP protocol has been deprecated. Access to the ACP agents is via direct HTTP connection since these agents will not work with the BeeAI Platform.

Instructions for running and connecting to the ACP agents are available below:

```sh
uv --directory acp run -m acp_agent.agent
```

Use the agent via an HTTP GET request

```sh
curl -X POST \
  --url http://localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{
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

</details>

# Configuration

The core library is designed to use Granite models that can be served from a variety of back ends. To configure the library, ensure environment variables are in place when running the code (this can be done via a `.env` file). The full configuration options available are documented in the [granite_core config.py](granite_core/granite_core/config.py) file where you will find a brief description of each option, the data type it expects, potential limitations on values and a default value.

The agents are configured in a similar way to the core, via environment variables (that can also be set via a `.env` file). The configurations are in the relevant `config.py` files for each agent. The agents will start without any additional configuration by adopting default values such as using Granite models served via a local Ollama and search provided by simple a DuckDuckGo implementation. This is sufficient for initial/early experimental usage. However, you are encouraged to explore the options to achieve better performance for your use case.

The following table illustrates some of the main options:

| Option              | Default                  | Notes                                                                    |
| ------------------- | ------------------------ | ------------------------------------------------------------------------ |
| OLLAMA_BASE_URL     | `http://localhost:11434` | Update this if running Ollama on a non standard port or alternate host   |
| LLM_PROVIDER        | `ollama`                 | Alternate providers are `watsonx` or `openai`.                           |
| LLM_MODEL           | `ibm/granite4`           | Update to the ID required by the LLM_PROVIDER. Granite 3 also supported. |
| EMBEDDINGS_PROVIDER | `ollama`                 | Alternate providers are `watsonx` or `openai`.                           |
| EMBEDDINGS_MODEL    | `nomic-embed-text`       | Use an appropriate long context embedding model from your provider.      |
| RETRIEVER           | `duckduckgo`             | Alternate retrievers are `google` and `tavily`. Used in search/research. |
| LOG_LEVEL           | `INFO`                   | You can get more verbose logs by setting to `DEBUG`.                     |

Retriever options don't have default values but must be used if configuring an alternate retriever:

| Option         | Type   | Notes                                    |
| -------------- | ------ | ---------------------------------------- |
| GOOGLE_API_KEY | Secret | The API key used to access Google search |
| GOOGLE_CX_KEY  | Secret | The CX key used to access Google search  |
|                |        |                                          |
| TAVILY_API_KEY | Secret | The API key used to access Tavily search |

LLM provider options don't have default values but must be used if configuring an alternate LLM provider:

| Option                     | Type   | Notes                                                   |
| -------------------------- | ------ | ------------------------------------------------------- |
| WATSONX_API_BASE           | URL    | Watsonx URL e.g. `https://us-south.ml.cloud.ibm.com`    |
| WATSONX_REGION             | Str    | Watsonx Region e.g. `us-south`                          |
| WATSONX_PROJECT_ID         | Secret | Required if setting LLM_PROVIDER to `watsonx`           |
| WATSONX_API_KEY            | Secret | Required if setting LLM_PROVIDER to `watsonx`           |
|                            |        |                                                         |
| LLM_API_BASE               | URL    | OpenAI base URL for access to the LLM                   |
| LLM_API_KEY                | Secret | Required if the LLM_API_BASE is authenticated           |
|                            |        |                                                         |
| EMBEDDINGS_OPENAI_API_BASE | URL    | OpenAI base URL for access to the embedding model       |
| EMBEDDINGS_OPENAI_API_KEY  | Secret | Required if EMBEDDINGS_OPENAI_API_BASE is authenticated |

> [!NOTE]
> The embeddings provider will use the same watsonx credentials as the LLM if configured to use watsonx.

# Development

For development work on the agents, you must install [pre-commit](https://pre-commit.com/) and the pre-commit hooks prior to modifying the code:

```sh
pre-commit install
```

All pre-commit hooks must be run and pass before code is accepted into the repository.

## Containerisation

### Build

```bash
podman build -t granite-playground-agents:latest -f a2a/Dockerfile .
```

### Run

```sh
podman run --env-file .env --name granite-playground-agents -p 8000:8000 --rm granite-playground-agents:latest
```
