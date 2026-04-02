FROM python:3.11-slim

WORKDIR /app

# System deps: libgomp1 required by faiss-cpu; gcc for some Python package builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (layer-cached)
COPY chatbot/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source, scripts, and shared middleware
COPY chatbot/src/ ./src/
COPY chatbot/scripts/ ./scripts/
COPY rate_limit.py .

# Data directories (volume-mounted in docker-compose)
RUN mkdir -p data/input data/indexed

# Non-root user
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8002

ENV PYTHONPATH=/app/src:/app

CMD ["uvicorn", "chatbot.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
