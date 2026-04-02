FROM python:3.12-slim

WORKDIR /app

# System deps: chromium for browser-use automation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and shared rate limiter
COPY src/ ./src/
COPY ../rate_limit.py ./rate_limit.py

# Non-root user
RUN useradd -m -u 1000 jobsearch && chown -R jobsearch:jobsearch /app
USER jobsearch

ENV PYTHONPATH=/app/src
ENV CHROME_PATH=/usr/bin/chromium

EXPOSE 8007

CMD ["uvicorn", "jobsearch.main:app", "--host", "0.0.0.0", "--port", "8007"]
