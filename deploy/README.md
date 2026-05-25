# Part 4 — Deployment & MCP client integration

**Owner:** _assign one teammate_
**Goal:** stand up the full stack (API + MCP server + OpenWebUI) with one command, and show an LLM in OpenWebUI calling the predict tool.

## Stack
- `api` — FastAPI predictor (port 8000)
- `mcp_server` — FastMCP server, streamable HTTP transport (port 8001)
- `openwebui` — chat UI configured to talk to the MCP server (port 3000)

## Run
```bash
docker compose -f deploy/docker-compose.yml up --build
```
Then open `http://localhost:3000`, pick a model, and ask:
> "I have a Dell laptop, Intel i7, 16GB RAM, 512GB storage, 14-inch screen,
>  NVIDIA RTX 3060, Windows 11, 1.8kg, 10h battery, 2-year warranty.
>  Use the predict tool and tell me if $1500 is fair."

## Alternative — local Claude Desktop / Claude Code
Copy `mcp_client_config.example.json` into your client config (path varies
by client) so it spawns `mcp_server` over stdio.
