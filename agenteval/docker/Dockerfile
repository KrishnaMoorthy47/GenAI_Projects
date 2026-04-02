FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY agenteval/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agenteval/src/ ./src/
COPY agenteval/data/ ./data/
COPY rate_limit.py .

RUN useradd -m -u 1000 agenteval && chown -R agenteval:agenteval /app
USER agenteval

EXPOSE 8003

ENV PYTHONPATH=/app/src:/app

CMD ["uvicorn", "agenteval.main:app", "--host", "0.0.0.0", "--port", "8003"]
