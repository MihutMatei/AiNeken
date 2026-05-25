"""Mapping between CSV column names and the snake_case fields used everywhere else.

The CSV has columns with spaces and units (e.g. "RAM (GB)"). We normalize once,
here, and the rest of the codebase only sees the snake_case names defined in
shared/schema.py.
"""

CSV_TO_FIELD = {
    "Brand": "brand",
    "Model": "model_name",
    "Processor": "processor",
    "RAM (GB)": "ram_gb",
    "Storage (GB)": "storage_gb",
    "Screen Size (inches)": "screen_size_inches",
    "Graphics Card": "graphics_card",
    "Operating System": "operating_system",
    "Weight (kg)": "weight_kg",
    "Battery Life (hours)": "battery_life_hours",
    "Price ($)": "price_usd",
    "Warranty (years)": "warranty_years",
}

TARGET = "price_usd"

# Columns excluded from the feature matrix during training.
DROP_FROM_FEATURES = ("model_name",)
