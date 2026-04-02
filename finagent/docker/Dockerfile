FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY finagent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and shared middleware
COPY finagent/src/ ./src/
COPY rate_limit.py .

# Non-root user
RUN useradd -m -u 1000 finagent && chown -R finagent:finagent /app
USER finagent

EXPOSE 8000

ENV PYTHONPATH=/app/src:/app

CMD ["uvicorn", "finagent.main:app", "--host", "0.0.0.0", "--port", "8000"]
