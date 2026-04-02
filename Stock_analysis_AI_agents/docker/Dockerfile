FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, finagent app (shared graph), and rate limiter
COPY src/ ./src/
COPY ../finagent/app/ ./app/
COPY ../rate_limit.py ./rate_limit.py

# Non-root user
RUN useradd -m -u 1000 stockagent && chown -R stockagent:stockagent /app
USER stockagent

ENV PYTHONPATH=/app/src

EXPOSE 8005

CMD ["uvicorn", "stockagent.main:app", "--host", "0.0.0.0", "--port", "8005"]
