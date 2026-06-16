# Real Estate Fraud Detection — Dockerfile (Railway optimized)
# FastAPI service

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY configs/    configs/
COPY src/        src/
COPY api/        api/
COPY models/     models/
COPY data/processed/feature_engineer.pkl  data/processed/feature_engineer.pkl
COPY data/processed/stacking_meta.pkl     data/processed/stacking_meta.pkl

RUN mkdir -p logs reports/plots data/splits

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV GIT_PYTHON_REFRESH=quiet

EXPOSE $PORT

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
