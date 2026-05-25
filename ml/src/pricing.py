"""Hand-picked pricing tables — the domain knowledge baked into the model.

Edit these freely. After every edit, retrain:

    python -m ml.src.train --data data/laptop_price_dataset.csv --out ml/artifacts/

The data only calibrates a *residual* on top of these values, so changing the
numbers here visibly moves predictions in the expected direction.

All values in USD. These are rough mid-market reference prices, not
research-grade — adjust to taste.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Component prices — what a bare CPU / GPU would cost on the open market.
# A laptop's "fair component floor" is roughly cpu + gpu + ram + storage cost.
# ---------------------------------------------------------------------------

CPU_PRICE_USD: dict[str, float] = {
    "Intel i5":     220.0,
    "Intel i7":     380.0,
    "Intel i9":     580.0,
    "AMD Ryzen 5":  200.0,
    "AMD Ryzen 7":  360.0,
    "AMD Ryzen 9":  550.0,
}

GPU_PRICE_USD: dict[str, float] = {
    "Intel UHD":         0.0,    # integrated, no extra cost
    "AMD Radeon":       60.0,    # mostly integrated, small premium
    "NVIDIA GTX 1650": 180.0,
    "NVIDIA RTX 3060": 330.0,
    "NVIDIA RTX 3070": 480.0,
}

# Per-GB component prices.
PER_GB_RAM_USD: float = 8.0          # rough laptop RAM markup
PER_GB_STORAGE_USD: float = 0.20     # SSD, rough

# Fixed assembly / motherboard / chassis / margin floor.
ASSEMBLY_BASE_USD: float = 200.0


# ---------------------------------------------------------------------------
# Brand markup — what each brand charges *on top of* the component floor.
# Apple gets the famous tax; Acer is the budget reference.
# These are PRIORS — Ridge will learn a residual correction from the data.
# ---------------------------------------------------------------------------

BRAND_MARKUP_USD: dict[str, float] = {
    "Apple":   450.0,    # the Apple tax
    "Razer":   300.0,    # premium gaming
    "MSI":     150.0,
    "Asus":    100.0,
    "Dell":     80.0,
    "HP":       50.0,
    "Lenovo":   50.0,
    "Acer":     20.0,    # budget baseline
}


# ---------------------------------------------------------------------------
# OS premium — license cost + bundled software polish.
# ---------------------------------------------------------------------------

OS_PREMIUM_USD: dict[str, float] = {
    "macOS":      100.0,
    "Windows 11":  40.0,
    "Windows 10":  20.0,
    "Linux":        0.0,    # free, often discounted SKUs
}


# ---------------------------------------------------------------------------
# Sanity helpers (used by pricing_model.py — kept here so all numbers and
# the math that consumes them live in one file).
# ---------------------------------------------------------------------------

def component_floor_usd(
    processor: str, graphics_card: str, ram_gb: float, storage_gb: float
) -> float:
    """Bare-bones component cost — before any brand or OS markup."""
    return (
        CPU_PRICE_USD.get(processor, 0.0)
        + GPU_PRICE_USD.get(graphics_card, 0.0)
        + PER_GB_RAM_USD * ram_gb
        + PER_GB_STORAGE_USD * storage_gb
        + ASSEMBLY_BASE_USD
    )


def prior_price_usd(
    brand: str,
    processor: str,
    graphics_card: str,
    operating_system: str,
    ram_gb: float,
    storage_gb: float,
) -> float:
    """The full prior: components + brand markup + OS premium.

    This is what the model predicts *before* the data-driven Ridge residual
    is added. Use it as the explainable backbone.
    """
    return (
        component_floor_usd(processor, graphics_card, ram_gb, storage_gb)
        + BRAND_MARKUP_USD.get(brand, 0.0)
        + OS_PREMIUM_USD.get(operating_system, 0.0)
    )
