# Handover — Part 4: Deploy + OpenWebUI integration

You own packaging the whole stack with docker-compose and getting an LLM in OpenWebUI to call our MCP tool. ~3 hours. **This is the riskiest part of the project — start immediately.**

## Your single deliverable
Run one command, get a chat UI where an LLM successfully invokes our `predict_laptop_price` tool. Record a short demo.

```bash
docker compose -f deploy/docker-compose.yml up --build
# open http://localhost:3000, prove the tool gets called
```

## Setup (10 min)
```bash
cd <repo root>
docker --version && docker compose version   # both must work
pip install -r requirements.txt              # so you can sanity-test the API locally
```

## You don't need to wait for ML or API to be done
- The API has a built-in **stub model** — it returns plausible prices even before Part 1 finishes. So you can bring up the whole stack from minute one.
- When ML drops `model.pkl` into `ml/artifacts/`, restart the `api` container — the volume mount picks it up.

## Step-by-step

### 1. Stack up against the stub (30 min)
```bash
docker compose -f deploy/docker-compose.yml up --build
```
Verify:
- `curl http://localhost:8000/health` → returns JSON (status may be `degraded` until ML lands — that's fine).
- `curl http://localhost:3000` → OpenWebUI loads.

### 2. Wire OpenWebUI to the MCP server (the hard part — 1–2h)
OpenWebUI's MCP support has shifted across versions. Check the running version's docs first. The compose file has `MCP_SERVERS` set as a starting point; you may need to switch to:
- `TOOL_SERVER_CONNECTIONS` env var, or
- the in-UI Tools / Functions menu pointing at `http://mcp_server:8001`, or
- the mcpo bridge (`open-webui/mcpo`) if the native integration doesn't fit.

**Time-box this to ~90 minutes.** If you're not getting tool calls through, jump to Plan B below.

### 3. Demo conversation
In OpenWebUI, pick a model that supports tool use and send:
> I have a Dell laptop, Intel i7, 16GB RAM, 512GB storage, 14-inch screen,
> NVIDIA RTX 3060, Windows 11, 1.8kg, 10h battery, 2-year warranty.
> Use the predict tool and tell me whether $1500 is fair.

You should see the LLM call `predict_laptop_price` and respond with the returned number.

### 4. Record the demo (15 min)
Screen-record the full chat. Save as `deploy/demo.mp4` (or upload and link in the root README).

## Plan B — if OpenWebUI fights you past 90 min
The brief allows **any MCP-compatible client**. Pivot to **Claude Desktop** or **Claude Code** over stdio:
1. Edit `deploy/mcp_client_config.example.json` — fill in absolute paths.
2. Drop it into the client's MCP config (Claude Desktop: `~/Library/Application Support/Claude/claude_desktop_config.json` on Mac; check the docs for your OS).
3. Restart the client, send the same prompt, screen-record.

Document the pivot in `deploy/README.md` so the panel knows it was a deliberate choice. Containerizing the API + MCP server in compose still counts toward the "nice to have" architecture goal even without OpenWebUI in the loop.

## Useful commands
```bash
docker compose -f deploy/docker-compose.yml logs api          # API logs
docker compose -f deploy/docker-compose.yml logs mcp_server   # MCP logs
docker compose -f deploy/docker-compose.yml logs openwebui    # OpenWebUI logs
docker compose -f deploy/docker-compose.yml restart api       # pick up new model.pkl
docker compose -f deploy/docker-compose.yml down -v           # nuke (including OpenWebUI data)
```

## Done when
- [ ] `docker compose up` brings up the full stack and all containers stay healthy
- [ ] At least one MCP client (OpenWebUI **or** Claude Desktop/Code) successfully calls `predict_laptop_price`
- [ ] Demo recording saved
- [ ] `deploy/README.md` updated to reflect whatever wiring actually worked

## If you're blocked
- Container won't start → `docker compose logs <service>`, paste into AI.
- OpenWebUI loads but doesn't see the tool → curl `http://localhost:8001` from inside the openwebui container (`docker compose exec openwebui sh`) to confirm networking before debugging the integration.
- API returns 500 → check `MODEL_PATH` env var and the volume mount.
- Anything blocking past 30 min → don't dig solo, ping Matei.
