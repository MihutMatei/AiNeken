# Deploy — AiNeken Laptop Price Predictor

## Stack

| Serviciu | URL | Descriere |
|---|---|---|
| FastAPI | http://localhost:8000 | REST API + model ML |
| MCP Server | http://localhost:8001 | FastMCP server |
| OpenWebUI | http://localhost:3000 | Chat UI |

---

## Pornire stack

```bash
# Asigura-te ca Docker daemon ruleaza
sudo service docker start   # Linux / WSL

# Porneste tot stack-ul
docker compose -f deploy/docker-compose.yml up --build
```

Verificare servicii:
```bash
curl http://localhost:8000/health   # {"status":"degraded"} e ok pana vine model.pkl
curl http://localhost:8001/mcp      # MCP server activ
```

Oprire:
```bash
docker compose -f deploy/docker-compose.yml down
```

---

## Integrare MCP — Client ales: Claude Code

### De ce Claude Code in loc de OpenWebUI

Versiunea de OpenWebUI din acest stack (`ghcr.io/open-webui/open-webui:main`) nu expune
un endpoint nativ pentru MCP servers in UI. Dupa investigatie, am ales sa folosim
**Claude Code** ca client MCP — optiune explicita mentionata in handover ca alternativa valida.

### Setup Claude Code (o singura data)

```bash
# Instaleaza Node.js (daca nu e instalat)
# https://nodejs.org/en/download

# Instaleaza Claude Code
npm install -g @anthropic-ai/claude-code

# Inregistreaza serverul MCP
claude mcp add laptop-predictor python /path/to/AiNeken/mcp_server/server.py

# Porneste
claude
```

### Rulare demo

Cu stack-ul Docker pornit, in Claude Code trimite:

```
I have a Dell laptop, Intel i7, 16GB RAM, 512GB storage, 14-inch screen,
NVIDIA RTX 3060, Windows 11, 1.8kg, 10h battery, 2-year warranty.
Use the predict_laptop_price tool and tell me whether $1500 is fair.
```

Claude Code va apela automat tool-ul `predict_laptop_price` si va raspunde cu pretul prezis.

---

## Actualizare model ML

Cand coechipierul termina antrenarea modelului:

# model.pkl se pune in ml/artifacts/
# Restart container API ca sa incarce modelul
docker compose -f deploy/docker-compose.yml restart api

# Verifica ca modelul e incarcat
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,...}

# In mcp_server/server.py seteaza MOCK_MODE = False
# Apoi restart MCP server
docker compose -f deploy/docker-compose.yml restart mcp_server

## Troubleshooting

| Problema | Solutie |
|---|---|
| Container nu porneste | `docker compose logs <serviciu>` |
| API returns 500 | Verifica `MODEL_PATH` si volumul `ml/artifacts/` |
| MCP tool nu e apelat | Specifica explicit "use the predict_laptop_price tool" in prompt |
| Docker daemon not running | `sudo service docker start` (WSL) |

---

## Note arhitectura

- Stack-ul ruleaza in **WSL2** pe Windows pentru compatibilitate cu colegii pe Linux
- `mcp_server/server.py` suporta ambele transporturi: `stdio` (Claude Code) si `http` (Docker)
- `MOCK_MODE = True` pana cand `ml/artifacts/model.pkl` e disponibil
