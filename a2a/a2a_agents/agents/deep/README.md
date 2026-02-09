# Deep Agent

A prototype A2A (Agent-to-Agent) agent with internet access. This agent uses the [deepagents](https://github.com/langchain-ai/deepagents) framework and integrates with the Granite Core MCP service for web search functionality.

## Installation

1. Install dependencies:
```bash
uv --directory a2a sync
```

2. Set up environment variables

## Configuration

The agent supports multiple LLM providers and is ideally run using an OpenAI-compatible endpoint. Configure the following environment variables (typically in project root `.env`):

### OpenAI-Compatible Endpoint (Recommended)

```bash
# LLM Provider Configuration
LLM_PROVIDER=openai
LLM_MODEL=<your-model-name>
LLM_API_BASE=<your-openai-compatible-endpoint>
LLM_API_KEY=<your-api-key>

# Optional: Custom headers (JSON format)
# LLM_API_HEADERS='{"Authorization": "Bearer token"}'
```

## Usage

### 1. Start the MCP Service

First, start the Granite Core MCP service that provides the internet_search tool:

```bash
# Run with default settings (port 8001, streamable-http transport)
uv --directory granite_core_mcp run -m granite_core_mcp.internet_search
```

The MCP service must be running on `http://localhost:8001/mcp` before starting the agent.

### 2. Run the Agent

Once the MCP service is running, start the deep agent:

```bash
uv --directory a2a run python -m a2a_agents.agents.deep.agent
```

The agent will start an A2A server on the configured host and port (default: `http://0.0.0.0:8000`).


### Extending the Agent

To add additional tools or modify behavior:

1. **Add Tools**: Pass additional tools to `create_deep_agent()` in the `tools` parameter
2. **Custom Middleware**: Add middleware to the `middleware` list in `create_deep_agent()`
3. **Modify System Prompt**: Update the prompt in [`prompts.py`](prompts.py)

## Troubleshooting

### MCP Service Connection Issues

If the agent fails to connect to the MCP service:

1. Verify the MCP service is running: `curl http://localhost:8001/mcp`
2. Check the port configuration matches in both services
3. Ensure no firewall is blocking localhost connections

### Tool Call Errors

The agent includes `PatchInvalidToolCallsMiddleware` to handle malformed tool calls. If you encounter persistent tool call issues:

1. Check the model's tool calling capabilities
2. Review the tool schemas in the MCP service
3. Enable debug mode: `create_deep_agent(..., debug=True)`
