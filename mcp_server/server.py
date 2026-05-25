"""
mcp_server/server.py - Server FastMCP pentru predictia preturilor de laptopuri.
"""

import os
import httpx
from fastmcp import FastMCP

# ─────────────────────────────────────────────
# CONFIGURARE
# ─────────────────────────────────────────────

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# Seteaza False cand coechipierul termina FastAPI-ul
MOCK_MODE = False

# ─────────────────────────────────────────────
# SERVER MCP
# ─────────────────────────────────────────────

mcp = FastMCP("laptop-predictor")


# ─────────────────────────────────────────────
# TOOL: predict_laptop_price
# ─────────────────────────────────────────────

@mcp.tool()
def predict_laptop_price(
    brand: str,
    processor: str,
    ram_gb: int,
    storage_gb: int,
    screen_size_inches: float,
    graphics_card: str,
    operating_system: str,
    weight_kg: float,
    battery_life_hours: float,
    warranty_years: int
) -> dict:
    """
    Prezice pretul unui laptop in dolari ($) pe baza specificatiilor tehnice.

    Parametri:
    - brand: Producatorul laptopului (ex: Asus, Dell, Apple, MSI)
    - processor: Tipul de procesor (ex: Intel i7, AMD Ryzen 5)
    - ram_gb: Memoria RAM in GB (ex: 8, 16, 32)
    - storage_gb: Capacitatea de stocare in GB (ex: 256, 512, 1024)
    - screen_size_inches: Diagonala ecranului in inch (ex: 15.6)
    - graphics_card: Placa video (ex: NVIDIA GTX 1650, Intel UHD)
    - operating_system: Sistemul de operare (ex: Windows 11, Linux, macOS)
    - weight_kg: Greutatea in kilograme (ex: 1.8)
    - battery_life_hours: Autonomia bateriei in ore (ex: 8.0)
    - warranty_years: Perioada de garantie in ani (ex: 1, 2, 3)
    """

    payload = {
        "brand": brand,
        "processor": processor,
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "screen_size_inches": screen_size_inches,
        "graphics_card": graphics_card,
        "operating_system": operating_system,
        "weight_kg": weight_kg,
        "battery_life_hours": battery_life_hours,
        "warranty_years": warranty_years
    }

    if MOCK_MODE:
        mock_price = 800 + ram_gb * 30 + (storage_gb / 512) * 100
        return {
            "predicted_price": round(mock_price, 2),
            "currency": "USD",
            "mode": "MOCK - inlocuieste cu FastAPI real cand e gata",
            "input": payload
        }

    try:
        response = httpx.post(f"{FASTAPI_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        return {
            "predicted_price": result["predicted_price_usd"],
            "currency": "USD",
            "input": payload
        }
    except httpx.RequestError as e:
        return {
            "error": f"Nu pot contacta FastAPI la {FASTAPI_URL}. Detalii: {str(e)}",
            "tip": "Verifica ca serverul FastAPI ruleaza si ca FASTAPI_URL e corect."
        }
    except httpx.HTTPStatusError as e:
        return {
            "error": f"FastAPI a returnat eroare {e.response.status_code}.",
            "detalii": e.response.text
        }


# ─────────────────────────────────────────────
# RESOURCE: statistici dataset
# ─────────────────────────────────────────────

@mcp.resource("laptop://dataset/stats")
def get_dataset_stats() -> str:
    """
    Returneaza statistici generale despre dataset-ul de laptopuri.
    Utile pentru LLM ca sa interpreteze daca un pret e rezonabil.
    """
    return """
    Statistici dataset laptopuri (3000 inregistrari):
    - Pret mediu:     ~1800 USD
    - Pret minim:     ~400 USD
    - Pret maxim:     ~4000 USD
    - RAM cel mai frecvent: 16 GB
    - Branduri: Asus, Dell, Apple, MSI, Lenovo, HP, Acer
    - Procesoare: Intel i5/i7/i9, AMD Ryzen 5/7/9

    Actualizeaza aceste valori dupa ce coechipierul antreneaza modelul
    si calculeaza statisticile reale din dataset.
    """


# ─────────────────────────────────────────────
# PORNIRE SERVER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--http" in sys.argv:
        mcp.run(transport="http", host="0.0.0.0", port=8001)
    else:
        mcp.run(transport="stdio")
