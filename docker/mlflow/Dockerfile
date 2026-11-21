FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN pip install --no-cache-dir mlflow==3.17.1

# Lokasi metadata (SQLite) dan artefak model, di-mount ke named volume
RUN mkdir -p /mlflow/db /mlflow/artifacts

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]