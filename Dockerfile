# Dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (psycopg builds, etc.)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code
COPY . .

# Optional: not required on Render, but harmless
EXPOSE 8000

# Bind to Render's assigned $PORT; add sane timeouts and logs
CMD bash -lc 'gunicorn \
  -k uvicorn.workers.UvicornWorker \
  -w ${WEB_CONCURRENCY:-2} \
  -b 0.0.0.0:${PORT:-8000} \
  --timeout 60 --graceful-timeout 30 --keep-alive 5 \
  --access-logfile - --error-logfile - \
  api.app.main:app'