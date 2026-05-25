FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY mcp_server/ ./mcp_server/
COPY data/ ./data/

ENV PYTHONPATH=/app
ENV API_URL=http://api:8000
ENV DATASET_PATH=/app/data/laptop_price_dataset.csv

EXPOSE 8001
CMD ["python", "-m", "mcp_server.server", "--http"]
