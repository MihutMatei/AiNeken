# AiNeken — Laptop Price Predictor (Internship Project)

End-to-end ML → API → MCP → LLM-client demo.
Track: **Option B (regression)** on `data/laptop_price_dataset.csv`. Brief in `docs/`.

## Layout
```
shared/      Pydantic schema — the contract every layer uses.
data/        The CSV.
ml/          Part 1 — train.py → ml/artifacts/model.pkl
api/         Part 2 — FastAPI /predict
mcp_server/  Part 3 — FastMCP tool wrapping the API
deploy/      Part 4 — docker-compose with OpenWebUI
```

---

## 8-hour plan (linear, AI-driven)

The dataset is tiny (3k rows) and the contract is already in `shared/schema.py`, so don't burn time on design. Build serially, test after each step, never go back.

| H | Step | Done when |
|---|------|-----------|
| 0:00–0:30 | Setup: `pip install -r requirements.txt`, sanity-check the CSV loads. | `python -c "from ml.src.preprocess import load_dataset; print(load_dataset('data/laptop_price_dataset.csv')[0].shape)"` prints `(3000, 10)` |
| 0:30–1:30 | **Part 1.** Run `python -m ml.src.train --data data/laptop_price_dataset.csv --out ml/artifacts/`. If MAE > $500, ask AI to swap the estimator (try `HistGradientBoostingRegressor`, `RandomForestRegressor`). | `ml/artifacts/model.pkl` exists, `model_card.json` shows reasonable metrics |
| 1:30–3:00 | **Part 2.** `uvicorn api.app.main:app --reload`. Curl `/predict` with a real payload. Add a couple of edge-case tests if time allows. | `/health` returns `status: ok`; `/predict` returns a number |
| 3:00–5:00 | **Part 3.** `python -m mcp_server.server` over stdio. Test from **Claude Code or Claude Desktop** using `deploy/mcp_client_config.example.json` — fill in absolute paths. Ask the LLM "predict price for X" and verify it calls the tool. | Tool call visible in the client, prediction returned |
| 5:00–7:00 | **Part 4.** `docker compose -f deploy/docker-compose.yml up --build`. Wire OpenWebUI to the MCP server. **This is the riskiest step** — start it the moment Part 3 works. | OpenWebUI chat invokes the tool and shows the result |
| 7:00–8:00 | Record demo, take screenshots, write 1-page summary, push to repo. | Submission ready |

### Hard rules for the day
1. **Train first, integrate second.** Once `model.pkl` exists, everything downstream is real, not stubbed.
2. **One tab, one feature.** No branching, no PRs. Commit to `main` after each green step so you can roll back the *last* step if something breaks.
3. **Plan B for OpenWebUI:** if the OpenWebUI ↔ MCP wiring fights you past ~1h, demo from **Claude Desktop / Claude Code over stdio** instead (already configured via `deploy/mcp_client_config.example.json`). The brief allows any MCP-compatible client; OpenWebUI is "nice to have".
4. **Let AI write boilerplate, you write the prompts.** Paste `shared/schema.py` + the failing error into the AI; do not hand-debug Pydantic / FastAPI / Docker issues. But always read the diff before pasting back.
5. **Don't touch `shared/schema.py` after H1.** Field renames cascade everywhere.

---

## Quickstart (run it once end-to-end to prove the whole chain)

```bash
pip install -r requirements.txt

python -m ml.src.train --data data/laptop_price_dataset.csv --out ml/artifacts/

uvicorn api.app.main:app --port 8000 &
curl -X POST http://localhost:8000/predict -H 'content-type: application/json' -d '{
  "brand":"Dell","processor":"Intel i7","ram_gb":16,"storage_gb":512,
  "screen_size_inches":14.0,"graphics_card":"NVIDIA RTX 3060",
  "operating_system":"Windows 11","weight_kg":1.8,"battery_life_hours":10,
  "warranty_years":2}'

python -m mcp_server.server --http &           # or no flag for stdio
docker compose -f deploy/docker-compose.yml up --build
```

## Definition of done
- [ ] `ml/artifacts/model.pkl` trained
- [ ] `POST /predict` returns a price
- [ ] MCP tool callable from at least one client (OpenWebUI or Claude Desktop/Code)
- [ ] Short demo video / screenshots saved
