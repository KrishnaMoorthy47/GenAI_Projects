FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY voiceagent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY voiceagent/src/ ./src/
COPY rate_limit.py .

RUN useradd -m -u 1000 voiceagent && chown -R voiceagent:voiceagent /app
USER voiceagent

EXPOSE 8004

ENV PYTHONPATH=/app/src:/app

CMD ["uvicorn", "voiceagent.main:app", "--host", "0.0.0.0", "--port", "8004"]
