FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY codesentinel/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY codesentinel/src/ ./src/
COPY rate_limit.py .

ENV PYTHONPATH=/app/src:/app

# Non-root user
RUN useradd -m -u 1000 sentinel && chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8001

CMD ["uvicorn", "codesentinel.main:app", "--host", "0.0.0.0", "--port", "8001"]
