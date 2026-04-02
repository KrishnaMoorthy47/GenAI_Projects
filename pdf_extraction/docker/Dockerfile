FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and shared rate limiter
COPY src/ ./src/
COPY ../rate_limit.py ./rate_limit.py

# Non-root user
RUN useradd -m -u 1000 pdfextract && chown -R pdfextract:pdfextract /app
USER pdfextract

ENV PYTHONPATH=/app/src

EXPOSE 8008

CMD ["uvicorn", "pdfextract.main:app", "--host", "0.0.0.0", "--port", "8008"]
