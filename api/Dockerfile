FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY api/ ./api/

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/ml/artifacts/model.pkl
ENV MODEL_CARD_PATH=/app/ml/artifacts/model_card.json

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
