# Handover — Parts 2 & 3: FastAPI service + FastMCP server

You own the glue between the trained model and the LLM client. Two deliverables, ~3 hours total. Covers everything in `api/` and `mcp_server/`.

## Your two deliverables
1. **FastAPI service** (`api/`) — `POST /predict` returns a price, `GET /health` reports model status.
2. **FastMCP server** (`mcp_server/`) — exposes the predictor as an MCP **tool** that an LLM can call. Also exposes a dataset stats **resource** and a prompt template.

Both are already scaffolded. Your job is to verify they work, fix anything broken, and add edge cases.

## Setup (10 min)
```bash
cd <repo root>
pip install -r requirements.txt
python -c "from shared.schema import LaptopFeatures; print(LaptopFeatures.model_json_schema()['required'])"
# expect a list of required fields like ['brand', 'processor', 'ram_gb', ...]
```

## You don't need to wait for ML
The API has a built-in **stub model** in `api/app/model_loader.py` — if `ml/artifacts/model.pkl` doesn't exist yet, the API still works and returns plausible numbers (`/health` reports `degraded`). Build everything against the stub. When Matei drops the real `model.pkl`, just restart the API server — it auto-picks up the new model.

## The contract — DO NOT BREAK
- All request/response shapes live in `shared/schema.py`. **Don't touch that file.** Field renames cascade into Matei's training code and the deploy person's docker-compose.
- The API is the only thing that reads `ml/artifacts/model.pkl`. The MCP server calls the API over HTTP, not the model directly.

---

## Step-by-step

### Part 2 — FastAPI (~1h)

1. **Run it.**
   ```bash
   uvicorn api.app.main:app --reload --port 8000
   ```
2. **Smoke test.**
   ```bash
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/predict -H 'content-type: application/json' -d '{
     "brand":"Dell","processor":"Intel i7","ram_gb":16,"storage_gb":512,
     "screen_size_inches":14.0,"graphics_card":"NVIDIA RTX 3060",
     "operating_system":"Windows 11","weight_kg":1.8,"battery_life_hours":10,
     "warranty_years":2}'
   ```
   Expect a JSON with `predicted_price_usd`.
3. **Open the auto-generated docs** at `http://localhost:8000/docs` — useful to share with the deploy person and to demo to the panel.
4. **Harden a bit (optional but cheap wins):**
   - Add a `/predict/batch` endpoint that accepts a list (if you have time — the brief only requires single).
   - Add a try/except around `predict_fn` returning HTTP 400 with the validation error, so bad inputs don't 500.
   - Add 2–3 pytest cases in `api/tests/test_predict.py` against the stub model.

### Part 3 — FastMCP (~1.5h)

1. **Run the MCP server in stdio mode** (talks to Claude Code / Claude Desktop):
   ```bash
   API_URL=http://localhost:8000 python -m mcp_server.server
   ```
   (Keep the FastAPI from Part 2 running in another terminal.)
2. **Test from Claude Code or Claude Desktop.**
   - Copy `deploy/mcp_client_config.example.json`, fill in absolute paths for `cwd`, `PYTHONPATH`, and `DATASET_PATH`.
   - Drop it into the client's MCP config:
     - **Claude Desktop (Mac):** `~/Library/Application Support/Claude/claude_desktop_config.json`
     - **Claude Desktop (Linux):** `~/.config/Claude/claude_desktop_config.json`
     - **Claude Code:** project `.mcp.json` or `~/.claude.json` (whichever your setup uses)
   - Restart the client. Ask: *"Use the laptop predict tool to estimate the price of a Dell with i7, 16GB RAM..."*
   - Confirm the tool gets called and a number comes back.
3. **Also test HTTP transport** (this is what OpenWebUI will use in Part 4):
   ```bash
   API_URL=http://localhost:8000 python -m mcp_server.server --http
   # default port 8001
   ```
4. **Verify the resource and prompt** show up — `dataset://stats` should return JSON when fetched by the client.

---

## Done when
- [ ] `POST /predict` returns a price for a valid payload
- [ ] `GET /health` reflects whether `model.pkl` is loaded (vs. stub)
- [ ] MCP server runs in **both** stdio and `--http` modes without crashing
- [ ] You successfully called `predict_laptop_price` from Claude Code or Claude Desktop end-to-end
- [ ] You pinged the deploy person to confirm the MCP server is ready for OpenWebUI integration

## If you're blocked
- Pydantic validation errors on `/predict` → check the request JSON against `shared/schema.py`. The enum string values are case-sensitive (`"Intel i7"`, not `"intel i7"`).
- MCP server starts but client doesn't see tools → check the client's MCP log. 90% of the time it's a wrong absolute path in `mcp_client_config.example.json` (use `pwd` from the repo root).
- MCP HTTP server unreachable from another container → bind to `0.0.0.0`, not `localhost`. FastMCP defaults to `0.0.0.0` for streamable-http, but double-check the logs.
- Tool gets called but returns an error → trace through: client → MCP server → `httpx.post` → API. Print at each boundary; usually it's a serialization mismatch (enum value vs. enum name).
- Anything blocking past 30 min → don't dig solo, ping Matei.
