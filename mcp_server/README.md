# Part 3 — FastMCP server

**Owner:** _assign one teammate (can be same as Part 2)_
**Goal:** expose the predictor as an MCP **tool** so an LLM can call it.

Uses the official Python MCP SDK (`mcp` package, `FastMCP` API).

## What is exposed
- **Tool** `predict_laptop_price(features: LaptopFeatures) -> PredictionResponse`
  Calls the FastAPI service at `API_URL` (default `http://localhost:8000`).
- **Resource** `dataset://stats` — JSON summary of the training dataset
  (counts, ranges, brand/processor distributions). Lets the LLM reason about
  what's plausible without re-loading the CSV.
- **Prompt** `interpret_prediction` — template the LLM can pull to format
  recommendations around a returned price.

## Run
```bash
export API_URL=http://localhost:8000
python -m mcp_server.server          # stdio transport for desktop clients
# or
python -m mcp_server.server --http   # HTTP transport for OpenWebUI
```
